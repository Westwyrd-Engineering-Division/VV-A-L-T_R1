"""
LogicRefinementUnit

Performs bounded logic refinement on attention output.
"""

import numpy as np


class LogicRefinementUnit:
    """
    Applies bounded logic refinement to vector representations.

    Uses single-pass nonlinear refinement with tanh activation to ensure
    bounded output in range [-1, 1]. No iteration or recursion - deterministic
    single-pass operation only.
    """

    def __init__(self, frame_dim: int, hidden_dim: int = None, seed: int = 42):
        """
        Initialize LogicRefinementUnit.

        Args:
            frame_dim: Dimension of frame representations
            hidden_dim: Hidden layer dimension (default: 2 * frame_dim)
            seed: Random seed for deterministic behavior
        """
        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim if hidden_dim else 2 * frame_dim
        np.random.seed(seed)

        # Two-layer refinement network
        # Layer 1: frame_dim -> hidden_dim
        self.W1 = np.random.randn(frame_dim, self.hidden_dim) * np.sqrt(2.0 / frame_dim)
        self.b1 = np.zeros(self.hidden_dim)

        # Layer 2: hidden_dim -> frame_dim
        self.W2 = np.random.randn(self.hidden_dim, frame_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(frame_dim)

        # Residual connection weight
        self.alpha = 0.5  # Balance between input and refinement

    def refine(self, x: np.ndarray) -> np.ndarray:
        """
        Apply bounded logic refinement.

        Single-pass operation with residual connection for stability.

        Args:
            x: Input vector of shape (frame_dim,)

        Returns:
            Refined vector bounded in [-1, 1]
        """
        # First layer with tanh activation
        h = np.tanh(x @ self.W1 + self.b1)

        # Second layer with tanh activation
        refined = np.tanh(h @ self.W2 + self.b2)

        # Residual connection for stability
        # Blend original input with refinement
        output = self.alpha * np.tanh(x) + (1 - self.alpha) * refined

        # Ensure output is bounded
        output = np.clip(output, -1.0, 1.0)

        return output

    def get_refinement_magnitude(self, x: np.ndarray) -> float:
        """
        Measure how much refinement changed the input.

        Args:
            x: Input vector

        Returns:
            Magnitude of refinement (0 = no change, higher = more change)
        """
        refined = self.refine(x)
        original = np.tanh(x)

        diff = np.linalg.norm(refined - original)
        baseline = np.linalg.norm(original) + 1e-8

        return float(diff / baseline)

    def get_activation_stats(self, x: np.ndarray) -> dict:
        """
        Get statistics about activations during refinement.

        Args:
            x: Input vector

        Returns:
            Dictionary of activation statistics
        """
        # Compute intermediate activations
        h = np.tanh(x @ self.W1 + self.b1)
        refined = np.tanh(h @ self.W2 + self.b2)

        return {
            "hidden_mean": float(np.mean(h)),
            "hidden_std": float(np.std(h)),
            "hidden_sparsity": float(np.mean(np.abs(h) < 0.1)),
            "output_mean": float(np.mean(refined)),
            "output_std": float(np.std(refined)),
            "output_sparsity": float(np.mean(np.abs(refined) < 0.1)),
        }
