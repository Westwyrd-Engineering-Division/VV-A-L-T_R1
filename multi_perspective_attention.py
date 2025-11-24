"""
MultiPerspectiveAttention

Fuses multiple perspective frames using attention mechanism.
"""

import numpy as np
from typing import Dict


class MultiPerspectiveAttention:
    """
    Combines multiple perspective frames into unified representation.

    Stacks weighted frames and applies attention mechanism to produce
    a single coherent vector that captures insights from all perspectives.
    """

    def __init__(self, frame_dim: int, num_frames: int = 5, seed: int = 42):
        """
        Initialize MultiPerspectiveAttention.

        Args:
            frame_dim: Dimension of each frame
            num_frames: Number of perspective frames
            seed: Random seed for deterministic behavior
        """
        self.frame_dim = frame_dim
        self.num_frames = num_frames
        np.random.seed(seed)

        # Attention parameters
        self.W_query = np.random.randn(frame_dim, frame_dim) * 0.1
        self.W_key = np.random.randn(frame_dim, frame_dim) * 0.1
        self.W_value = np.random.randn(frame_dim, frame_dim) * 0.1

        # Output projection
        self.W_output = np.random.randn(frame_dim, frame_dim) * 0.1
        self.b_output = np.zeros(frame_dim)

        self.frame_names = ["semantic", "structural", "causal", "relational", "graph"]

    def _stack_frames(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stack frame representations into a matrix.

        Args:
            frames: Dictionary of frame representations

        Returns:
            Stacked frames of shape (num_frames, frame_dim)
        """
        stacked = []
        for name in self.frame_names:
            if name in frames:
                stacked.append(frames[name])
            else:
                # If frame is missing, use zeros
                stacked.append(np.zeros(self.frame_dim))

        return np.stack(stacked, axis=0)

    def _scaled_dot_product_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query matrix (num_frames, frame_dim)
            key: Key matrix (num_frames, frame_dim)
            value: Value matrix (num_frames, frame_dim)

        Returns:
            Attention output (frame_dim,)
        """
        # Compute attention scores
        scores = query @ key.T  # (num_frames, num_frames)
        scores = scores / np.sqrt(self.frame_dim)  # Scale

        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention to values
        attended = attention_weights @ value  # (num_frames, frame_dim)

        # Average across frames (mean pooling)
        output = np.mean(attended, axis=0)  # (frame_dim,)

        return output

    def attend(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply multi-perspective attention to fuse frames.

        Args:
            frames: Dictionary of topology-aligned frame representations

        Returns:
            Unified representation vector of shape (frame_dim,)
        """
        # Stack frames
        stacked = self._stack_frames(frames)  # (num_frames, frame_dim)

        # Compute Q, K, V
        query = stacked @ self.W_query  # (num_frames, frame_dim)
        key = stacked @ self.W_key  # (num_frames, frame_dim)
        value = stacked @ self.W_value  # (num_frames, frame_dim)

        # Apply attention
        attended = self._scaled_dot_product_attention(query, key, value)

        # Output projection
        output = attended @ self.W_output + self.b_output

        return output

    def get_attention_weights(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get attention weights for interpretability.

        Args:
            frames: Dictionary of frame representations

        Returns:
            Attention weight matrix of shape (num_frames, num_frames)
        """
        stacked = self._stack_frames(frames)
        query = stacked @ self.W_query
        key = stacked @ self.W_key

        scores = query @ key.T / np.sqrt(self.frame_dim)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        return attention_weights
