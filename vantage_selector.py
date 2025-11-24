"""
VantageSelector

Task-conditioned frame weighting for perspective selection.
"""

import numpy as np
from typing import Dict


class VantageSelector:
    """
    Selects and weights perspective frames based on task requirements.

    Computes normalized weights that determine the importance of each
    perspective frame for the current task. Uses L1 normalization to
    ensure weights sum to 1.0.
    """

    def __init__(self, task_dim: int, num_frames: int = 5, seed: int = 42):
        """
        Initialize VantageSelector.

        Args:
            task_dim: Dimension of task vector
            num_frames: Number of perspective frames (default: 5)
            seed: Random seed for deterministic behavior
        """
        self.task_dim = task_dim
        self.num_frames = num_frames
        np.random.seed(seed)

        # Task-to-weight projection matrix
        # Maps task vector to frame weights
        self.W_task = np.random.randn(task_dim, num_frames) * 0.1
        self.b_task = np.zeros(num_frames)

        # Frame names for reference
        self.frame_names = ["semantic", "structural", "causal", "relational", "graph"]

    def compute_weights(self, task_vector: np.ndarray) -> np.ndarray:
        """
        Compute frame weights from task vector.

        Args:
            task_vector: Task description vector of shape (task_dim,)

        Returns:
            Normalized weights of shape (num_frames,) that sum to 1.0
        """
        # Project task vector to frame weights
        logits = task_vector @ self.W_task + self.b_task

        # Apply softmax for positive weights that sum to 1
        # This is equivalent to L1 normalization after exp
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        weights = exp_logits / np.sum(exp_logits)

        return weights

    def select_frames(
        self,
        frames: Dict[str, np.ndarray],
        task_vector: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Weight perspective frames according to task requirements.

        Args:
            frames: Dictionary of frame representations
            task_vector: Task description vector

        Returns:
            Dictionary of weighted frame representations
        """
        weights = self.compute_weights(task_vector)

        weighted_frames = {}
        for i, frame_name in enumerate(self.frame_names):
            if frame_name in frames:
                weighted_frames[frame_name] = frames[frame_name] * weights[i]

        return weighted_frames

    def get_weight_distribution(self, task_vector: np.ndarray) -> Dict[str, float]:
        """
        Get human-readable weight distribution for a task.

        Args:
            task_vector: Task description vector

        Returns:
            Dictionary mapping frame names to their weights
        """
        weights = self.compute_weights(task_vector)

        return {
            frame_name: float(weights[i])
            for i, frame_name in enumerate(self.frame_names)
        }
