"""
V.V.A.L.T Modular PyTorch Implementation

Pure PyTorch implementation replacing NumPy-core for efficient training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Any, List


class ForwardMode(Enum):
    """Forward pass execution modes."""
    FAST = "fast"  # Minimal computation, no trace
    FINE = "fine"  # Standard computation with basic trace
    DIAGNOSTIC = "diagnostic"  # Full computation with detailed trace


@dataclass
class VVALTDetailedTrace:
    """Detailed trace for interpretability and debugging."""
    frames: Optional[torch.Tensor] = None  # All perspective frames
    weighted_frames: Optional[torch.Tensor] = None  # Task-weighted frames
    attention_weights: Optional[torch.Tensor] = None  # Attention distribution
    task_weights: Optional[torch.Tensor] = None  # Task-conditioned weights
    refined_output: Optional[torch.Tensor] = None  # After refinement
    consistency_score: Optional[torch.Tensor] = None  # Consistency verification
    projection_info: Optional[Dict[str, Any]] = None  # Graph projection details


class VectorFrameEncoderPT(nn.Module):
    """PyTorch implementation of Vector Frame Encoder."""

    def __init__(self, input_dim: int, frame_dim: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.frame_dim = frame_dim

        # Projection matrices for each perspective frame
        self.W_semantic = nn.Parameter(torch.randn(input_dim, frame_dim) * 0.01)
        self.W_structural = nn.Parameter(torch.randn(input_dim, frame_dim) * 0.01)
        self.W_causal = nn.Parameter(torch.randn(input_dim, frame_dim) * 0.01)
        self.W_relational = nn.Parameter(torch.randn(input_dim, frame_dim) * 0.01)
        self.W_graph = nn.Parameter(torch.randn(input_dim, frame_dim) * 0.01)

    def forward(self, x: torch.Tensor, graph_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input into multiple perspective frames.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            graph_adj: Optional adjacency matrix (frame_dim, frame_dim)

        Returns:
            Tensor of shape (batch_size, 5, frame_dim) or (5, frame_dim)
        """
        # Handle single sample vs batch
        was_single = x.dim() == 1
        if was_single:
            x = x.unsqueeze(0)  # (input_dim,) -> (1, input_dim)

        # Project to each perspective frame
        semantic = torch.tanh(x @ self.W_semantic)  # (batch, frame_dim)
        structural = torch.tanh(x @ self.W_structural)
        causal = torch.tanh(x @ self.W_causal)
        relational = torch.tanh(x @ self.W_relational)

        # Graph frame with optional topology
        graph = x @ self.W_graph
        if graph_adj is not None:
            graph = graph @ graph_adj  # Apply graph structure
        graph = torch.tanh(graph)

        # Stack frames: (batch, 5, frame_dim)
        frames = torch.stack([semantic, structural, causal, relational, graph], dim=1)

        if was_single:
            frames = frames.squeeze(0)  # (1, 5, frame_dim) -> (5, frame_dim)

        return frames


class VantageSelectorPT(nn.Module):
    """PyTorch implementation of Vantage Selector."""

    def __init__(self, task_dim: int, num_frames: int = 5, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.task_dim = task_dim
        self.num_frames = num_frames

        # Task-to-frame weight projection
        self.W_task = nn.Parameter(torch.randn(task_dim, num_frames) * 0.01)

    def forward(self, frames: torch.Tensor, task_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply task-conditioned weighting to frames.

        Args:
            frames: (batch, num_frames, frame_dim) or (num_frames, frame_dim)
            task_vector: (batch, task_dim) or (task_dim,)

        Returns:
            Tuple of (weighted_frames, task_weights)
        """
        # Handle single sample
        was_single = frames.dim() == 2
        if was_single:
            frames = frames.unsqueeze(0)
            task_vector = task_vector.unsqueeze(0)

        # Compute task-conditioned weights
        task_weights = F.softmax(task_vector @ self.W_task, dim=-1)  # (batch, num_frames)

        # Apply weights: (batch, num_frames, 1) * (batch, num_frames, frame_dim)
        weighted_frames = task_weights.unsqueeze(-1) * frames

        if was_single:
            weighted_frames = weighted_frames.squeeze(0)
            task_weights = task_weights.squeeze(0)

        return weighted_frames, task_weights


class GraphTopologyProjectorPT(nn.Module):
    """PyTorch implementation of Graph Topology Projector."""

    def __init__(self, frame_dim: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.frame_dim = frame_dim

    def forward(self, frames: torch.Tensor, graph_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply graph topology to frames.

        Args:
            frames: (batch, num_frames, frame_dim) or (num_frames, frame_dim)
            graph_adj: Optional (frame_dim, frame_dim) adjacency matrix

        Returns:
            Projected frames with same shape
        """
        if graph_adj is None:
            return frames

        # Handle single sample
        was_single = frames.dim() == 2
        if was_single:
            frames = frames.unsqueeze(0)

        # Apply graph topology: (batch, num_frames, frame_dim) @ (frame_dim, frame_dim)
        projected = frames @ graph_adj

        if was_single:
            projected = projected.squeeze(0)

        return projected


class MultiPerspectiveAttentionPT(nn.Module):
    """PyTorch implementation of Multi-Perspective Attention."""

    def __init__(self, frame_dim: int, num_frames: int = 5, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.frame_dim = frame_dim
        self.num_frames = num_frames
        self.scale = frame_dim ** -0.5

        # Attention projection matrices
        self.W_query = nn.Parameter(torch.randn(frame_dim, frame_dim) * 0.01)
        self.W_key = nn.Parameter(torch.randn(frame_dim, frame_dim) * 0.01)
        self.W_value = nn.Parameter(torch.randn(frame_dim, frame_dim) * 0.01)

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-perspective attention.

        Args:
            frames: (batch, num_frames, frame_dim) or (num_frames, frame_dim)

        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Handle single sample
        was_single = frames.dim() == 2
        if was_single:
            frames = frames.unsqueeze(0)

        # Project to Q, K, V
        Q = frames @ self.W_query  # (batch, num_frames, frame_dim)
        K = frames @ self.W_key
        V = frames @ self.W_value

        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch, num_frames, num_frames)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.bmm(attn_weights, V)  # (batch, num_frames, frame_dim)

        # Aggregate across frames (mean pooling)
        output = attended.mean(dim=1)  # (batch, frame_dim)

        if was_single:
            output = output.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return output, attn_weights


class LogicRefinementUnitPT(nn.Module):
    """PyTorch implementation of Logic Refinement Unit."""

    def __init__(self, frame_dim: int, hidden_dim: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP for refinement
        self.W1 = nn.Parameter(torch.randn(frame_dim, hidden_dim) * 0.01)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, frame_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine logic representation.

        Args:
            x: (batch, frame_dim) or (frame_dim,)

        Returns:
            Refined tensor with same shape
        """
        # Two-layer MLP with ReLU
        hidden = F.relu(x @ self.W1)
        refined = torch.tanh(hidden @ self.W2)

        # Residual connection
        return x + refined


class ConsistencyVerifierPT(nn.Module):
    """PyTorch implementation of Consistency Verifier."""

    def __init__(self, safe_range: Tuple[float, float] = (-10.0, 10.0)):
        super().__init__()
        self.safe_min = safe_range[0]
        self.safe_max = safe_range[1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify and clip output to safe range.

        Args:
            x: Input tensor

        Returns:
            Tuple of (clipped_output, consistency_score)
        """
        # Check if within bounds
        in_bounds = ((x >= self.safe_min) & (x <= self.safe_max)).float()
        consistency_score = in_bounds.mean()

        # Clip to safe range
        clipped = torch.clamp(x, self.safe_min, self.safe_max)

        return clipped, consistency_score


class VVALTModular(nn.Module):
    """
    Modular PyTorch V.V.A.L.T Implementation

    Complete PyTorch implementation replacing NumPy core for efficient
    training, inference, and gradient-based optimization.
    """

    def __init__(
        self,
        input_dim: int,
        frame_dim: int,
        task_dim: int,
        hidden_dim: Optional[int] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        """
        Initialize modular V.V.A.L.T.

        Args:
            input_dim: Input vector dimension
            frame_dim: Frame representation dimension
            task_dim: Task vector dimension
            hidden_dim: Hidden dimension for refinement (default: 2 * frame_dim)
            runtime_config: Runtime configuration dict
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.frame_dim = frame_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2 * frame_dim
        self.runtime_config = runtime_config or {}
        self.seed = seed

        # Initialize modular components
        self.encoder = VectorFrameEncoderPT(input_dim, frame_dim, seed=seed)
        self.selector = VantageSelectorPT(task_dim, num_frames=5, seed=seed)
        self.projector = GraphTopologyProjectorPT(frame_dim, seed=seed)
        self.attention = MultiPerspectiveAttentionPT(frame_dim, num_frames=5, seed=seed)
        self.refiner = LogicRefinementUnitPT(frame_dim, self.hidden_dim, seed=seed)
        self.verifier = ConsistencyVerifierPT(safe_range=(-10.0, 10.0))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        mode: ForwardMode = ForwardMode.FAST,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass through V.V.A.L.T.

        Args:
            inputs_embeds: Input embeddings (batch, input_dim) or (input_dim,)
            task_vector: Task conditioning vector (batch, task_dim) or (task_dim,)
            graph_adj: Optional graph adjacency matrix (frame_dim, frame_dim)
            mode: Forward execution mode
            return_trace: Whether to return detailed trace

        Returns:
            Tuple of (output, trace_dict)
        """
        trace = {} if (return_trace or mode != ForwardMode.FAST) else None

        # Step 1: Encode input into perspective frames
        frames = self.encoder(inputs_embeds, graph_adj)
        if trace is not None:
            trace['frames'] = frames

        # Step 2: Task-conditioned frame weighting
        weighted_frames, task_weights = self.selector(frames, task_vector)
        if trace is not None:
            trace['weighted_frames'] = weighted_frames
            trace['task_weights'] = task_weights

        # Step 3: Graph topology projection (if adjacency provided)
        if graph_adj is not None:
            projected_frames = self.projector(weighted_frames, graph_adj)
        else:
            projected_frames = weighted_frames

        # Step 4: Multi-perspective attention
        attended, attn_weights = self.attention(projected_frames)
        if trace is not None:
            trace['attention_weights'] = attn_weights
            trace['attended'] = attended

        # Step 5: Logic refinement
        refined = self.refiner(attended)
        if trace is not None:
            trace['refined_output'] = refined

        # Step 6: Consistency verification
        output, consistency = self.verifier(refined)
        if trace is not None:
            trace['consistency_score'] = consistency
            trace['final_output'] = output

        # Mode-specific trace handling
        if mode == ForwardMode.FAST and not return_trace:
            trace = None
        elif mode == ForwardMode.DIAGNOSTIC:
            # Add additional diagnostic information
            if trace is not None:
                trace['mode'] = mode.value
                trace['num_frames'] = frames.shape[-2] if frames.dim() > 1 else frames.shape[0]

        return output, trace
