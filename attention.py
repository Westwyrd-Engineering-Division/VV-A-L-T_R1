"""
V.V.A.L.T PyTorch Native Implementation - Attention Components

Modular attention system with task conditioning and graph awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .frame_encoders import ForwardMode


@dataclass
class AttentionTrace:
    """Trace for attention mechanism."""
    component_name: str
    attention_weights: Optional[torch.Tensor]
    weight_entropy: Optional[float]
    frame_weights: Optional[Dict[str, float]]
    processing_time_ms: Optional[float]

    def to_dict(self) -> Dict:
        result = {
            "component_name": self.component_name,
            "processing_time_ms": self.processing_time_ms,
        }
        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights.tolist()
        if self.weight_entropy is not None:
            result["weight_entropy"] = self.weight_entropy
        if self.frame_weights is not None:
            result["frame_weights"] = self.frame_weights
        return result


class VantageSelector(nn.Module):
    """
    Task-conditioned frame weight selector.

    Computes importance weights for each perspective frame based on task vector.
    Uses softmax normalization to ensure weights sum to 1.0.
    """

    def __init__(self, task_dim: int, num_frames: int = 5):
        super().__init__()

        self.task_dim = task_dim
        self.num_frames = num_frames

        # Task to weights projection
        self.task_projection = nn.Linear(task_dim, num_frames)

        # Initialize with small values for balanced initial weighting
        nn.init.normal_(self.task_projection.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.task_projection.bias)

        self.frame_names = ["semantic", "structural", "causal", "relational", "graph"]

    def forward(self, task_vector: torch.Tensor, mode: ForwardMode = ForwardMode.FAST) -> torch.Tensor:
        """
        Compute frame weights from task vector.

        Args:
            task_vector: Task description (batch_size, task_dim) or (task_dim,)
            mode: Forward execution mode

        Returns:
            Frame weights (batch_size, num_frames) or (num_frames,)
        """
        # Project task to logits
        logits = self.task_projection(task_vector)

        # Softmax for normalized weights
        weights = F.softmax(logits, dim=-1)

        return weights

    def forward_with_trace(self, task_vector: torch.Tensor) -> Tuple[torch.Tensor, AttentionTrace]:
        """Forward with detailed tracing."""
        import time
        start = time.perf_counter()

        weights = self.forward(task_vector, mode=ForwardMode.FAST)

        # Compute weight distribution
        if weights.dim() == 1:
            frame_weights = {
                name: weights[i].item()
                for i, name in enumerate(self.frame_names)
            }
        else:
            # Batched: average across batch
            avg_weights = weights.mean(dim=0)
            frame_weights = {
                name: avg_weights[i].item()
                for i, name in enumerate(self.frame_names)
            }

        # Compute entropy (higher = more uniform distribution)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean().item()

        trace = AttentionTrace(
            component_name="VantageSelector",
            attention_weights=weights.detach().cpu(),
            weight_entropy=entropy,
            frame_weights=frame_weights,
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return weights, trace

    def get_weight_distribution(self, task_vector: torch.Tensor) -> Dict[str, float]:
        """Get human-readable weight distribution."""
        weights = self.forward(task_vector)

        if weights.dim() > 1:
            weights = weights.mean(dim=0)

        return {
            name: weights[i].item()
            for i, name in enumerate(self.frame_names)
        }


class GraphTopologyProjector(nn.Module):
    """
    Graph topology-aware projection.

    Projects frames to respect graph structure through graph convolution.
    """

    def __init__(self, frame_dim: int):
        super().__init__()

        self.frame_dim = frame_dim

        # Topology projection matrix
        self.topology_projection = nn.Linear(frame_dim, frame_dim, bias=False)

        # Initialize close to identity for stability
        nn.init.eye_(self.topology_projection.weight)
        self.topology_projection.weight.data += torch.randn_like(self.topology_projection.weight) * 0.01

    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization of adjacency matrix."""
        adj_with_loops = adj + torch.eye(adj.shape[0], device=adj.device)
        degree = adj_with_loops.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        return D_inv_sqrt @ adj_with_loops @ D_inv_sqrt

    def forward(
        self,
        frames: Dict[str, torch.Tensor],
        graph_adj: Optional[torch.Tensor] = None,
        mode: ForwardMode = ForwardMode.FAST
    ) -> Dict[str, torch.Tensor]:
        """
        Project frames onto graph topology.

        Args:
            frames: Dictionary of frame tensors
            graph_adj: Optional adjacency matrix
            mode: Forward execution mode

        Returns:
            Topology-aligned frames
        """
        if graph_adj is None:
            # No graph structure - simple projection
            return {
                name: self.topology_projection(frame)
                for name, frame in frames.items()
            }

        # Normalize adjacency
        norm_adj = self._normalize_adjacency(graph_adj)

        # Apply graph convolution
        aligned_frames = {}
        for name, frame in frames.items():
            # Project
            projected = self.topology_projection(frame)

            # Graph convolution
            if projected.dim() == 1:
                aligned = norm_adj @ projected
            else:
                # Batched: (batch, dim) @ (dim, dim) -> (batch, dim)
                aligned = projected @ norm_adj.T

            aligned_frames[name] = aligned

        return aligned_frames

    def forward_with_trace(
        self,
        frames: Dict[str, torch.Tensor],
        graph_adj: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], AttentionTrace]:
        """Forward with detailed tracing."""
        import time
        start = time.perf_counter()

        aligned_frames = self.forward(frames, graph_adj, mode=ForwardMode.FAST)

        trace = AttentionTrace(
            component_name="GraphTopologyProjector",
            attention_weights=None,
            weight_entropy=None,
            frame_weights=None,
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return aligned_frames, trace


class MultiPerspectiveAttention(nn.Module):
    """
    Multi-perspective attention mechanism.

    Fuses multiple frame perspectives using scaled dot-product attention.
    """

    def __init__(self, frame_dim: int, num_frames: int = 5):
        super().__init__()

        self.frame_dim = frame_dim
        self.num_frames = num_frames

        # Attention projections (Q, K, V)
        self.query_projection = nn.Linear(frame_dim, frame_dim)
        self.key_projection = nn.Linear(frame_dim, frame_dim)
        self.value_projection = nn.Linear(frame_dim, frame_dim)

        # Output projection
        self.output_projection = nn.Linear(frame_dim, frame_dim)

        # Initialize
        for proj in [self.query_projection, self.key_projection, self.value_projection]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        self.frame_names = ["semantic", "structural", "causal", "relational", "graph"]

    def _stack_frames(self, frames: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack frames into tensor."""
        # Ensure consistent ordering
        stacked = []
        for name in self.frame_names:
            if name in frames:
                stacked.append(frames[name])
            else:
                # Missing frame - use zeros
                stacked.append(torch.zeros_like(stacked[0] if stacked else frames[list(frames.keys())[0]]))

        return torch.stack(stacked, dim=0 if stacked[0].dim() == 1 else 1)

    def forward(
        self,
        frames: Dict[str, torch.Tensor],
        mode: ForwardMode = ForwardMode.FAST
    ) -> torch.Tensor:
        """
        Apply multi-perspective attention.

        Args:
            frames: Dictionary of frame tensors
            mode: Forward execution mode

        Returns:
            Fused representation
        """
        # Stack frames: (num_frames, frame_dim) or (batch, num_frames, frame_dim)
        stacked = self._stack_frames(frames)

        # Project to Q, K, V
        Q = self.query_projection(stacked)  # (num_frames, dim) or (batch, num_frames, dim)
        K = self.key_projection(stacked)
        V = self.value_projection(stacked)

        # Scaled dot-product attention
        # Q @ K^T / sqrt(d_k)
        if Q.dim() == 2:
            scores = Q @ K.T / (self.frame_dim ** 0.5)
        else:
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.frame_dim ** 0.5)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        if V.dim() == 2:
            attended = attention_weights @ V
        else:
            attended = torch.bmm(attention_weights, V)

        # Mean pooling across frames
        if attended.dim() == 2:
            pooled = attended.mean(dim=0)
        else:
            pooled = attended.mean(dim=1)

        # Output projection
        output = self.output_projection(pooled)

        return output

    def forward_with_trace(
        self,
        frames: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, AttentionTrace]:
        """Forward with detailed tracing."""
        import time
        start = time.perf_counter()

        # Stack and compute attention
        stacked = self._stack_frames(frames)
        Q = self.query_projection(stacked)
        K = self.key_projection(stacked)
        V = self.value_projection(stacked)

        if Q.dim() == 2:
            scores = Q @ K.T / (self.frame_dim ** 0.5)
        else:
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.frame_dim ** 0.5)

        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention
        if V.dim() == 2:
            attended = attention_weights @ V
            pooled = attended.mean(dim=0)
        else:
            attended = torch.bmm(attention_weights, V)
            pooled = attended.mean(dim=1)

        output = self.output_projection(pooled)

        # Compute entropy
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean().item()

        trace = AttentionTrace(
            component_name="MultiPerspectiveAttention",
            attention_weights=attention_weights.detach().cpu(),
            weight_entropy=entropy,
            frame_weights=None,
            processing_time_ms=(time.perf_counter() - start) * 1000
        )

        return output, trace

    def get_attention_weights(self, frames: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get attention weight matrix for visualization."""
        stacked = self._stack_frames(frames)
        Q = self.query_projection(stacked)
        K = self.key_projection(stacked)

        if Q.dim() == 2:
            scores = Q @ K.T / (self.frame_dim ** 0.5)
        else:
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.frame_dim ** 0.5)

        return F.softmax(scores, dim=-1)
