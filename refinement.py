"""
V.V.A.L.T PyTorch Native Implementation - Refinement & Verification

Logic refinement and consistency verification modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass

from .frame_encoders import ForwardMode


@dataclass
class RefinementTrace:
    """Trace for refinement/verification step."""
    component_name: str
    input_stats: Dict[str, float]
    output_stats: Dict[str, float]
    refinement_magnitude: float
    has_nan_inf: bool
    clipped_values: int
    processing_time_ms: float

    def to_dict(self) -> Dict:
        return {
            "component_name": self.component_name,
            "input_stats": self.input_stats,
            "output_stats": self.output_stats,
            "refinement_magnitude": self.refinement_magnitude,
            "has_nan_inf": self.has_nan_inf,
            "clipped_values": self.clipped_values,
            "processing_time_ms": self.processing_time_ms,
        }


class LogicRefinementUnit(nn.Module):
    """
    Bounded logic refinement module.

    Two-layer network with tanh activations and residual connection.
    Ensures bounded output in [-1, 1] range.
    """

    def __init__(self, frame_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()

        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2 * frame_dim

        # Two-layer refinement network
        self.layer1 = nn.Linear(frame_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, frame_dim)

        # Residual blend parameter (learnable or fixed)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Initialize
        nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x: torch.Tensor, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """
        Apply bounded logic refinement.

        Args:
            x: Input tensor (batch_size, frame_dim) or (frame_dim,)
            mode: Forward execution mode

        Returns:
            Refined tensor in [-1, 1]
        """
        # Layer 1 with tanh
        h = torch.tanh(self.layer1(x))

        # Layer 2 with tanh
        refined = torch.tanh(self.layer2(h))

        # Residual connection with learned blending
        alpha = torch.clamp(self.alpha, 0.0, 1.0)  # Ensure alpha in [0, 1]
        output = alpha * torch.tanh(x) + (1 - alpha) * refined

        # Ensure strictly bounded
        output = torch.clamp(output, -1.0, 1.0)

        return output

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, RefinementTrace]:
        """Forward with detailed tracing."""
        import time
        start = time.perf_counter()

        output = self.forward(x, mode=ForwardMode.FAST)

        # Compute refinement magnitude
        refinement_mag = (output - torch.tanh(x)).norm(p=2).item()
        input_norm = torch.tanh(x).norm(p=2).item()
        relative_change = refinement_mag / (input_norm + 1e-8)

        trace = RefinementTrace(
            component_name="LogicRefinementUnit",
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(output),
            refinement_magnitude=relative_change,
            has_nan_inf=False,  # Verified later by ConsistencyVerifier
            clipped_values=0,  # Clipping happens in verifier
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace

    def _compute_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute tensor statistics."""
        return {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "min": x.min().item(),
            "max": x.max().item(),
            "norm_l2": x.norm(p=2).item(),
        }

    def get_refinement_magnitude(self, x: torch.Tensor) -> float:
        """Measure how much refinement changes input."""
        with torch.no_grad():
            refined = self.forward(x)
            original = torch.tanh(x)
            diff = (refined - original).norm(p=2).item()
            baseline = original.norm(p=2).item()
            return diff / (baseline + 1e-8)


class ConsistencyVerifier(nn.Module):
    """
    Safety validation and consistency verification.

    Detects NaN/Inf values and clips to safe bounds.
    """

    def __init__(self, safe_bounds: Tuple[float, float] = (-10.0, 10.0)):
        super().__init__()

        self.safe_min = safe_bounds[0]
        self.safe_max = safe_bounds[1]

        # Register as buffers (not trainable parameters)
        self.register_buffer('min_bound', torch.tensor(self.safe_min))
        self.register_buffer('max_bound', torch.tensor(self.safe_max))

    def forward(self, x: torch.Tensor, strict: bool = False, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """
        Verify and sanitize tensor.

        Args:
            x: Input tensor
            strict: If True, raise on invalid values
            mode: Forward execution mode

        Returns:
            Safe tensor
        """
        # Check for NaN/Inf
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()

        if has_nan or has_inf:
            if strict:
                raise ValueError(f"Invalid values detected: NaN={has_nan}, Inf={has_inf}")

            # Sanitize
            x = torch.nan_to_num(x, nan=0.0, posinf=self.safe_max, neginf=self.safe_min)

        # Clip to safe range
        x_safe = torch.clamp(x, self.safe_min, self.safe_max)

        return x_safe

    def forward_with_trace(self, x: torch.Tensor, strict: bool = False) -> Tuple[torch.Tensor, RefinementTrace]:
        """Forward with detailed tracing."""
        import time
        start = time.perf_counter()

        # Check validity
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()

        # Count values that need clipping
        needs_upper_clip = (x > self.safe_max).sum().item()
        needs_lower_clip = (x < self.safe_min).sum().item()
        total_clipped = needs_upper_clip + needs_lower_clip

        # Apply verification
        x_safe = self.forward(x, strict=strict, mode=ForwardMode.FAST)

        trace = RefinementTrace(
            component_name="ConsistencyVerifier",
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(x_safe),
            refinement_magnitude=(x_safe - x).norm(p=2).item(),
            has_nan_inf=has_nan or has_inf,
            clipped_values=total_clipped,
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return x_safe, trace

    def _compute_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute tensor statistics (handling NaN/Inf)."""
        # Use nanmean, etc. to handle invalid values
        x_valid = torch.nan_to_num(x, nan=0.0, posinf=self.safe_max, neginf=self.safe_min)

        return {
            "mean": x_valid.mean().item(),
            "std": x_valid.std().item(),
            "min": x_valid.min().item(),
            "max": x_valid.max().item(),
            "norm_l2": x_valid.norm(p=2).item(),
        }

    def check_validity(self, x: torch.Tensor) -> Dict[str, bool]:
        """Check tensor validity."""
        return {
            "has_nan": torch.isnan(x).any().item(),
            "has_inf": torch.isinf(x).any().item(),
            "in_range": ((x >= self.safe_min) & (x <= self.safe_max)).all().item(),
        }

    def verify_bounds(self, x: torch.Tensor) -> bool:
        """Check if tensor is within safe bounds."""
        validity = self.check_validity(x)
        return validity["in_range"] and not validity["has_nan"] and not validity["has_inf"]
