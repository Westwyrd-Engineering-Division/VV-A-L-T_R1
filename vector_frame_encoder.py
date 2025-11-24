"""
VectorFrameEncoder

Encodes input into five perspective frames for multi-vantage analysis.
"""

import numpy as np
from typing import Dict, Tuple


class VectorFrameEncoder:
    """
    Encodes input vectors into multiple perspective frames.

    Frames:
    - Semantic: Meaning-based representation (tanh activation)
    - Structural: Pattern-based representation (sin/cos encoding)
    - Causal: Cause-effect representation (gradient-like operation)
    - Relational: Connection-based representation (normalization)
    - Graph: Topology-aligned representation
    """

    def __init__(self, input_dim: int, frame_dim: int, seed: int = 42):
        """
        Initialize VectorFrameEncoder.

        Args:
            input_dim: Dimension of input vectors
            frame_dim: Dimension of each frame representation
            seed: Random seed for deterministic behavior
        """
        self.input_dim = input_dim
        self.frame_dim = frame_dim
        np.random.seed(seed)

        # Initialize projection matrices for each frame
        # Using orthogonal initialization for stability
        self.W_semantic = self._orthogonal_init((input_dim, frame_dim))
        self.W_structural = self._orthogonal_init((input_dim, frame_dim))
        self.W_causal = self._orthogonal_init((input_dim, frame_dim))
        self.W_relational = self._orthogonal_init((input_dim, frame_dim))
        self.W_graph = self._orthogonal_init((input_dim, frame_dim))

        # Bias terms
        self.b_semantic = np.zeros(frame_dim)
        self.b_structural = np.zeros(frame_dim)
        self.b_causal = np.zeros(frame_dim)
        self.b_relational = np.zeros(frame_dim)
        self.b_graph = np.zeros(frame_dim)

    def _orthogonal_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Initialize matrix with orthogonal properties."""
        matrix = np.random.randn(*shape)
        if shape[0] >= shape[1]:
            q, _ = np.linalg.qr(matrix)
            return q
        else:
            q, _ = np.linalg.qr(matrix.T)
            return q.T

    def encode_semantic(self, x: np.ndarray) -> np.ndarray:
        """
        Semantic frame: Meaning-based representation with tanh activation.

        Args:
            x: Input vector of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            Semantic frame representation
        """
        z = x @ self.W_semantic + self.b_semantic
        return np.tanh(z)

    def encode_structural(self, x: np.ndarray) -> np.ndarray:
        """
        Structural frame: Pattern-based representation with sin/cos encoding.

        Args:
            x: Input vector of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            Structural frame representation
        """
        z = x @ self.W_structural + self.b_structural
        # Split into two halves for sin/cos encoding
        half_dim = self.frame_dim // 2
        sin_part = np.sin(z[..., :half_dim])
        cos_part = np.cos(z[..., half_dim:half_dim*2])

        # Handle odd dimensions
        if self.frame_dim % 2 == 1:
            remainder = z[..., -1:]
            return np.concatenate([sin_part, cos_part, remainder], axis=-1)
        else:
            return np.concatenate([sin_part, cos_part], axis=-1)

    def encode_causal(self, x: np.ndarray) -> np.ndarray:
        """
        Causal frame: Cause-effect representation using gradient-like operation.

        Args:
            x: Input vector of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            Causal frame representation
        """
        z = x @ self.W_causal + self.b_causal

        # Gradient-like operation: difference between adjacent elements
        # This captures directional changes (cause -> effect)
        if len(z.shape) == 1:
            causal_signal = np.concatenate([[z[0]], np.diff(z)])
        else:
            first_elem = z[..., :1]
            diffs = np.diff(z, axis=-1)
            causal_signal = np.concatenate([first_elem, diffs], axis=-1)

        return np.tanh(causal_signal)

    def encode_relational(self, x: np.ndarray) -> np.ndarray:
        """
        Relational frame: Connection-based representation with normalization.

        Args:
            x: Input vector of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            Relational frame representation
        """
        z = x @ self.W_relational + self.b_relational

        # L2 normalization to capture relationships
        epsilon = 1e-8
        norm = np.linalg.norm(z, axis=-1, keepdims=True)
        return z / (norm + epsilon)

    def encode_graph(self, x: np.ndarray, graph_adj: np.ndarray = None) -> np.ndarray:
        """
        Graph frame: Topology-aligned representation.

        Args:
            x: Input vector of shape (input_dim,) or (batch_size, input_dim)
            graph_adj: Optional adjacency matrix for graph structure

        Returns:
            Graph frame representation
        """
        z = x @ self.W_graph + self.b_graph

        # If graph adjacency is provided, apply graph convolution-like operation
        if graph_adj is not None:
            # Normalize adjacency matrix
            degree = np.sum(graph_adj, axis=-1, keepdims=True) + 1e-8
            norm_adj = graph_adj / degree

            # Apply graph structure
            if len(z.shape) == 1:
                z = norm_adj @ z
            else:
                z = z @ norm_adj.T

        return np.tanh(z)

    def encode(self, x: np.ndarray, graph_adj: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Encode input into all five perspective frames.

        Args:
            x: Input vector of shape (input_dim,) or (batch_size, input_dim)
            graph_adj: Optional adjacency matrix for graph frame

        Returns:
            Dictionary containing all five frame representations
        """
        return {
            "semantic": self.encode_semantic(x),
            "structural": self.encode_structural(x),
            "causal": self.encode_causal(x),
            "relational": self.encode_relational(x),
            "graph": self.encode_graph(x, graph_adj)
        }
