"""
V.V.A.L.T PyTorch Native Implementation - Frame Encoders

Modular, inspectable frame encoders as separate nn.Module components.
Each encoder can be individually analyzed, visualized, and debugged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ForwardMode(Enum):
    """Forward pass execution modes."""
    FAST = "fast"  # Optimized, no tracing
    FINE = "fine"  # Basic tracing
    DIAGNOSTIC = "diagnostic"  # Full detailed tracing + hooks


@dataclass
class FrameTrace:
    """Detailed trace for a single frame."""
    frame_name: str
    input_stats: Dict[str, float]
    output_stats: Dict[str, float]
    activation_pattern: torch.Tensor
    gradient_norm: Optional[float] = None
    processing_time_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "frame_name": self.frame_name,
            "input_stats": self.input_stats,
            "output_stats": self.output_stats,
            "activation_pattern": self.activation_pattern.tolist() if isinstance(self.activation_pattern, torch.Tensor) else self.activation_pattern,
            "gradient_norm": self.gradient_norm,
            "processing_time_ms": self.processing_time_ms,
        }


class BaseFrameEncoder(nn.Module):
    """Base class for all frame encoders with tracing capabilities."""

    def __init__(self, input_dim: int, frame_dim: int, frame_name: str):
        super().__init__()
        self.input_dim = input_dim
        self.frame_dim = frame_dim
        self.frame_name = frame_name

        # Hook storage
        self.pre_hooks = []
        self.post_hooks = []

    def register_pre_hook(self, hook):
        """Register pre-forward hook."""
        self.pre_hooks.append(hook)

    def register_post_hook(self, hook):
        """Register post-forward hook."""
        self.post_hooks.append(hook)

    def _run_pre_hooks(self, x: torch.Tensor):
        """Execute pre-forward hooks."""
        for hook in self.pre_hooks:
            hook(self, x)

    def _run_post_hooks(self, x: torch.Tensor, output: torch.Tensor):
        """Execute post-forward hooks."""
        for hook in self.post_hooks:
            hook(self, x, output)

    def _compute_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for tensor."""
        return {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "min": x.min().item(),
            "max": x.max().item(),
            "norm_l2": x.norm(p=2).item(),
            "sparsity": (x.abs() < 0.01).float().mean().item(),
        }

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, FrameTrace]:
        """Forward pass with detailed tracing."""
        raise NotImplementedError


class SemanticFrameEncoder(BaseFrameEncoder):
    """
    Semantic Frame Encoder - Meaning-based representation.

    Uses tanh activation to capture semantic content with bounded outputs.
    Architecture: Linear -> Tanh
    """

    def __init__(self, input_dim: int, frame_dim: int):
        super().__init__(input_dim, frame_dim, "semantic")

        self.projection = nn.Linear(input_dim, frame_dim)
        self.activation = nn.Tanh()

        # Initialize with orthogonal weights for stability
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """Standard forward pass."""
        if mode == ForwardMode.DIAGNOSTIC:
            self._run_pre_hooks(x)

        z = self.projection(x)
        output = self.activation(z)

        if mode == ForwardMode.DIAGNOSTIC:
            self._run_post_hooks(x, output)

        return output

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, FrameTrace]:
        """Forward pass with detailed tracing."""
        import time
        start = time.perf_counter()

        # Compute
        z = self.projection(x)
        output = self.activation(z)

        # Build trace
        trace = FrameTrace(
            frame_name=self.frame_name,
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(output),
            activation_pattern=output.detach().cpu()[:min(10, output.shape[-1])],  # First 10 activations
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace


class StructuralFrameEncoder(BaseFrameEncoder):
    """
    Structural Frame Encoder - Pattern-based representation.

    Uses sinusoidal encoding to capture structural patterns.
    Architecture: Linear -> Split -> [Sin, Cos] -> Concat
    """

    def __init__(self, input_dim: int, frame_dim: int):
        super().__init__(input_dim, frame_dim, "structural")

        self.projection = nn.Linear(input_dim, frame_dim)
        self.half_dim = frame_dim // 2

        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """Standard forward pass."""
        if mode == ForwardMode.DIAGNOSTIC:
            self._run_pre_hooks(x)

        z = self.projection(x)

        # Split and apply sin/cos
        sin_part = torch.sin(z[..., :self.half_dim])
        cos_part = torch.cos(z[..., self.half_dim:self.half_dim*2])

        # Handle odd dimensions
        if self.frame_dim % 2 == 1:
            remainder = z[..., -1:]
            output = torch.cat([sin_part, cos_part, remainder], dim=-1)
        else:
            output = torch.cat([sin_part, cos_part], dim=-1)

        if mode == ForwardMode.DIAGNOSTIC:
            self._run_post_hooks(x, output)

        return output

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, FrameTrace]:
        """Forward pass with detailed tracing."""
        import time
        start = time.perf_counter()

        output = self.forward(x, mode=ForwardMode.FAST)

        trace = FrameTrace(
            frame_name=self.frame_name,
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(output),
            activation_pattern=output.detach().cpu()[:min(10, output.shape[-1])],
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace


class CausalFrameEncoder(BaseFrameEncoder):
    """
    Causal Frame Encoder - Cause-effect representation.

    Uses gradient-like operations to capture causal relationships.
    Architecture: Linear -> Diff -> Tanh
    """

    def __init__(self, input_dim: int, frame_dim: int):
        super().__init__(input_dim, frame_dim, "causal")

        self.projection = nn.Linear(input_dim, frame_dim)
        self.activation = nn.Tanh()

        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """Standard forward pass."""
        if mode == ForwardMode.DIAGNOSTIC:
            self._run_pre_hooks(x)

        z = self.projection(x)

        # Gradient-like operation (temporal derivative approximation)
        # For batched input: (batch, dim) -> diff along dim
        if z.dim() == 1:
            causal_signal = torch.cat([z[:1], z[1:] - z[:-1]])
        else:
            first = z[..., :1]
            diffs = z[..., 1:] - z[..., :-1]
            causal_signal = torch.cat([first, diffs], dim=-1)

        output = self.activation(causal_signal)

        if mode == ForwardMode.DIAGNOSTIC:
            self._run_post_hooks(x, output)

        return output

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, FrameTrace]:
        """Forward pass with detailed tracing."""
        import time
        start = time.perf_counter()

        output = self.forward(x, mode=ForwardMode.FAST)

        trace = FrameTrace(
            frame_name=self.frame_name,
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(output),
            activation_pattern=output.detach().cpu()[:min(10, output.shape[-1])],
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace


class RelationalFrameEncoder(BaseFrameEncoder):
    """
    Relational Frame Encoder - Connection-based representation.

    Uses L2 normalization to capture relational structure.
    Architecture: Linear -> L2Normalize
    """

    def __init__(self, input_dim: int, frame_dim: int):
        super().__init__(input_dim, frame_dim, "relational")

        self.projection = nn.Linear(input_dim, frame_dim)

        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """Standard forward pass."""
        if mode == ForwardMode.DIAGNOSTIC:
            self._run_pre_hooks(x)

        z = self.projection(x)
        output = F.normalize(z, p=2, dim=-1, eps=1e-8)

        if mode == ForwardMode.DIAGNOSTIC:
            self._run_post_hooks(x, output)

        return output

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, FrameTrace]:
        """Forward pass with detailed tracing."""
        import time
        start = time.perf_counter()

        output = self.forward(x, mode=ForwardMode.FAST)

        trace = FrameTrace(
            frame_name=self.frame_name,
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(output),
            activation_pattern=output.detach().cpu()[:min(10, output.shape[-1])],
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace


class GraphFrameEncoder(BaseFrameEncoder):
    """
    Graph Frame Encoder - Topology-aligned representation.

    Applies graph convolution when adjacency matrix is provided.
    Architecture: Linear -> GraphConv (optional) -> Tanh
    """

    def __init__(self, input_dim: int, frame_dim: int):
        super().__init__(input_dim, frame_dim, "graph")

        self.projection = nn.Linear(input_dim, frame_dim)
        self.activation = nn.Tanh()

        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization of adjacency matrix."""
        # Add self-loops
        adj_with_loops = adj + torch.eye(adj.shape[0], device=adj.device)

        # Degree matrix
        degree = adj_with_loops.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # D^-0.5 @ A @ D^-0.5
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        normalized = D_inv_sqrt @ adj_with_loops @ D_inv_sqrt

        return normalized

    def forward(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        mode: ForwardMode = ForwardMode.FAST
    ) -> torch.Tensor:
        """Standard forward pass."""
        if mode == ForwardMode.DIAGNOSTIC:
            self._run_pre_hooks(x)

        z = self.projection(x)

        # Apply graph convolution if adjacency provided
        if graph_adj is not None:
            norm_adj = self._normalize_adjacency(graph_adj)
            if z.dim() == 1:
                z = norm_adj @ z
            else:
                # Batched: (batch, dim) x (dim, dim) -> (batch, dim)
                z = z @ norm_adj.T

        output = self.activation(z)

        if mode == ForwardMode.DIAGNOSTIC:
            self._run_post_hooks(x, output)

        return output

    def forward_with_trace(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, FrameTrace]:
        """Forward pass with detailed tracing."""
        import time
        start = time.perf_counter()

        output = self.forward(x, graph_adj, mode=ForwardMode.FAST)

        trace = FrameTrace(
            frame_name=self.frame_name,
            input_stats=self._compute_stats(x),
            output_stats=self._compute_stats(output),
            activation_pattern=output.detach().cpu()[:min(10, output.shape[-1])],
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace


class MultiFrameEncoder(nn.Module):
    """
    Unified multi-frame encoder module.

    Encapsulates all five frame encoders as inspectable submodules.
    """

    def __init__(self, input_dim: int, frame_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.frame_dim = frame_dim

        # All frame encoders as separate modules
        self.semantic_encoder = SemanticFrameEncoder(input_dim, frame_dim)
        self.structural_encoder = StructuralFrameEncoder(input_dim, frame_dim)
        self.causal_encoder = CausalFrameEncoder(input_dim, frame_dim)
        self.relational_encoder = RelationalFrameEncoder(input_dim, frame_dim)
        self.graph_encoder = GraphFrameEncoder(input_dim, frame_dim)

        self.frame_names = ["semantic", "structural", "causal", "relational", "graph"]

    def forward(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        mode: ForwardMode = ForwardMode.FAST
    ) -> Dict[str, torch.Tensor]:
        """
        Encode input into all five frames.

        Args:
            x: Input tensor (batch_size, input_dim) or (input_dim,)
            graph_adj: Optional adjacency matrix (frame_dim, frame_dim)
            mode: Forward execution mode

        Returns:
            Dictionary of frame tensors
        """
        frames = {
            "semantic": self.semantic_encoder(x, mode=mode),
            "structural": self.structural_encoder(x, mode=mode),
            "causal": self.causal_encoder(x, mode=mode),
            "relational": self.relational_encoder(x, mode=mode),
            "graph": self.graph_encoder(x, graph_adj, mode=mode),
        }

        return frames

    def forward_with_trace(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, FrameTrace]]:
        """Forward with detailed per-frame tracing."""
        frames = {}
        traces = {}

        # Semantic
        frames["semantic"], traces["semantic"] = self.semantic_encoder.forward_with_trace(x)

        # Structural
        frames["structural"], traces["structural"] = self.structural_encoder.forward_with_trace(x)

        # Causal
        frames["causal"], traces["causal"] = self.causal_encoder.forward_with_trace(x)

        # Relational
        frames["relational"], traces["relational"] = self.relational_encoder.forward_with_trace(x)

        # Graph
        frames["graph"], traces["graph"] = self.graph_encoder.forward_with_trace(x, graph_adj)

        return frames, traces

    def get_frame_encoder(self, frame_name: str) -> BaseFrameEncoder:
        """Get specific frame encoder for inspection."""
        encoders = {
            "semantic": self.semantic_encoder,
            "structural": self.structural_encoder,
            "causal": self.causal_encoder,
            "relational": self.relational_encoder,
            "graph": self.graph_encoder,
        }
        return encoders[frame_name]
