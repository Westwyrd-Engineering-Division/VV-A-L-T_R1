"""
GraphTopologyProjector

Aligns vector representations with graph topology constraints.
"""

import numpy as np
from typing import Dict, Optional


class GraphTopologyProjector:
    """
    Projects vector frames onto graph topology structure.

    Ensures that vector representations respect the underlying graph
    structure by injecting topology constraints. Bounded by input graph size.
    """

    def __init__(self, frame_dim: int, seed: int = 42):
        """
        Initialize GraphTopologyProjector.

        Args:
            frame_dim: Dimension of frame representations
            seed: Random seed for deterministic behavior
        """
        self.frame_dim = frame_dim
        np.random.seed(seed)

        # Projection matrix for topology alignment
        self.W_topology = np.eye(frame_dim) + np.random.randn(frame_dim, frame_dim) * 0.01

    def _normalize_adjacency(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix using symmetric normalization.

        Args:
            adj_matrix: Adjacency matrix of shape (n, n)

        Returns:
            Normalized adjacency matrix
        """
        # Add self-loops
        adj_with_self_loops = adj_matrix + np.eye(adj_matrix.shape[0])

        # Compute degree matrix
        degree = np.sum(adj_with_self_loops, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        normalized_adj = D_inv_sqrt @ adj_with_self_loops @ D_inv_sqrt

        return normalized_adj

    def _graph_convolution(
        self,
        features: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply graph convolution operation.

        Args:
            features: Feature matrix of shape (..., frame_dim)
            adj_matrix: Normalized adjacency matrix

        Returns:
            Graph-convolved features
        """
        # If features is 1D, treat as single node
        if len(features.shape) == 1:
            # Create a trivial "graph" for single vector
            return features @ self.W_topology

        # For multi-node features, apply graph convolution
        # This aggregates information from neighboring nodes
        aggregated = adj_matrix @ features
        projected = aggregated @ self.W_topology

        return projected

    def project(
        self,
        frames: Dict[str, np.ndarray],
        graph_adj: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Project frames onto graph topology.

        Args:
            frames: Dictionary of weighted frame representations
            graph_adj: Optional adjacency matrix. If None, uses identity (no graph structure)

        Returns:
            Dictionary of topology-aligned frame representations
        """
        if graph_adj is None:
            # No graph structure - apply simple projection
            return {
                name: frame @ self.W_topology
                for name, frame in frames.items()
            }

        # Normalize adjacency matrix
        norm_adj = self._normalize_adjacency(graph_adj)

        # Apply graph convolution to each frame
        aligned_frames = {}
        for name, frame in frames.items():
            aligned_frames[name] = self._graph_convolution(frame, norm_adj)

        return aligned_frames

    def get_topology_influence(
        self,
        frame: np.ndarray,
        graph_adj: np.ndarray
    ) -> float:
        """
        Measure how much topology influences the frame representation.

        Args:
            frame: Frame representation
            graph_adj: Adjacency matrix

        Returns:
            Influence score (higher = more influence)
        """
        if graph_adj is None:
            return 0.0

        # Compare projected vs unprojected
        unprojected = frame @ self.W_topology
        norm_adj = self._normalize_adjacency(graph_adj)
        projected = self._graph_convolution(frame, norm_adj)

        # Compute difference
        diff = np.linalg.norm(projected - unprojected)
        baseline = np.linalg.norm(unprojected) + 1e-8

        return float(diff / baseline)
