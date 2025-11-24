"""
InterpretabilityProjector

Provides operator visibility into V.V.A.L.T reasoning process.
"""

import numpy as np
from typing import Dict, List


class InterpretabilityProjector:
    """
    Projects internal representations to interpretable format.

    Provides complete reasoning trace visibility through vector analysis,
    activation statistics, sparsity metrics, and sample projections.
    Ensures full operator control and interpretability.
    """

    def __init__(self, frame_dim: int, num_samples: int = 5):
        """
        Initialize InterpretabilityProjector.

        Args:
            frame_dim: Dimension of frame representations
            num_samples: Number of sample projections to show
        """
        self.frame_dim = frame_dim
        self.num_samples = min(num_samples, frame_dim)

    def analyze_vector(self, x: np.ndarray, name: str = "vector") -> Dict:
        """
        Comprehensive analysis of a single vector.

        Args:
            x: Vector to analyze
            name: Name/label for the vector

        Returns:
            Dictionary of analysis metrics
        """
        return {
            "name": name,
            "shape": x.shape,
            "statistics": {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "median": float(np.median(x)),
                "norm_l1": float(np.linalg.norm(x, ord=1)),
                "norm_l2": float(np.linalg.norm(x, ord=2)),
                "norm_inf": float(np.linalg.norm(x, ord=np.inf)),
            },
            "sparsity": {
                "zero_ratio": float(np.mean(x == 0.0)),
                "near_zero_ratio": float(np.mean(np.abs(x) < 0.01)),
                "active_ratio": float(np.mean(np.abs(x) >= 0.1)),
            },
            "distribution": {
                "positive_ratio": float(np.mean(x > 0)),
                "negative_ratio": float(np.mean(x < 0)),
                "skewness": float(self._compute_skewness(x)),
                "kurtosis": float(self._compute_kurtosis(x)),
            },
            "samples": {
                "first_n": x[:self.num_samples].tolist(),
                "last_n": x[-self.num_samples:].tolist(),
                "top_magnitude_indices": np.argsort(np.abs(x))[-self.num_samples:][::-1].tolist(),
                "top_magnitude_values": x[np.argsort(np.abs(x))[-self.num_samples:][::-1]].tolist(),
            }
        }

    def analyze_frames(self, frames: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze all perspective frames.

        Args:
            frames: Dictionary of frame representations

        Returns:
            Dictionary mapping frame names to their analyses
        """
        analyses = {}
        for name, frame in frames.items():
            analyses[name] = self.analyze_vector(frame, name=f"{name}_frame")

        # Add comparative analysis
        analyses["comparison"] = self._compare_frames(frames)

        return analyses

    def _compare_frames(self, frames: Dict[str, np.ndarray]) -> Dict:
        """
        Compare multiple frames to each other.

        Args:
            frames: Dictionary of frame representations

        Returns:
            Comparison metrics
        """
        frame_list = list(frames.values())
        frame_names = list(frames.keys())

        if len(frame_list) < 2:
            return {"note": "Need at least 2 frames for comparison"}

        # Compute pairwise similarities
        similarities = {}
        for i, name1 in enumerate(frame_names):
            for j, name2 in enumerate(frame_names[i+1:], start=i+1):
                similarity = self._cosine_similarity(frame_list[i], frame_list[j])
                similarities[f"{name1}_vs_{name2}"] = float(similarity)

        # Compute diversity (lower = more diverse)
        mean_similarity = np.mean(list(similarities.values()))

        return {
            "pairwise_similarities": similarities,
            "mean_similarity": float(mean_similarity),
            "diversity": float(1.0 - mean_similarity),
        }

    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            return 0.0

        return np.dot(x, y) / (norm_x * norm_y)

    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness of distribution."""
        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0.0

        return np.mean(((x - mean) / std) ** 3)

    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute kurtosis of distribution."""
        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0.0

        return np.mean(((x - mean) / std) ** 4) - 3.0

    def create_reasoning_trace(
        self,
        input_vector: np.ndarray,
        frames: Dict[str, np.ndarray],
        weighted_frames: Dict[str, np.ndarray],
        attention_output: np.ndarray,
        refined_output: np.ndarray,
        final_output: np.ndarray,
        task_weights: Dict[str, float] = None
    ) -> Dict:
        """
        Create complete reasoning trace for operator visibility.

        Args:
            input_vector: Original input
            frames: Encoded frames
            weighted_frames: Task-weighted frames
            attention_output: Multi-perspective attention output
            refined_output: Logic-refined output
            final_output: Final verified output
            task_weights: Optional task-conditioned weights

        Returns:
            Complete reasoning trace dictionary
        """
        trace = {
            "input": self.analyze_vector(input_vector, "input"),
            "encoded_frames": self.analyze_frames(frames),
            "weighted_frames": self.analyze_frames(weighted_frames) if weighted_frames else None,
            "attention_output": self.analyze_vector(attention_output, "attention"),
            "refined_output": self.analyze_vector(refined_output, "refined"),
            "final_output": self.analyze_vector(final_output, "final"),
        }

        if task_weights:
            trace["task_weights"] = task_weights

        # Add transformation analysis
        # Note: Only compare vectors in same space (frame_dim)
        trace["transformations"] = {
            "attention_to_refined_change": float(
                np.linalg.norm(refined_output - attention_output) /
                (np.linalg.norm(attention_output) + 1e-8)
            ),
            "refined_to_final_change": float(
                np.linalg.norm(final_output - refined_output) /
                (np.linalg.norm(refined_output) + 1e-8)
            ),
            "attention_to_final_change": float(
                np.linalg.norm(final_output - attention_output) /
                (np.linalg.norm(attention_output) + 1e-8)
            ),
        }

        return trace

    def format_trace_summary(self, trace: Dict) -> str:
        """
        Format reasoning trace as human-readable summary.

        Args:
            trace: Reasoning trace dictionary

        Returns:
            Formatted string summary
        """
        lines = ["=== V.V.A.L.T Reasoning Trace ===\n"]

        # Input summary
        lines.append(f"Input: norm={trace['input']['statistics']['norm_l2']:.4f}")

        # Task weights if available
        if "task_weights" in trace and trace["task_weights"]:
            lines.append("\nTask-Conditioned Frame Weights:")
            for frame, weight in trace["task_weights"].items():
                lines.append(f"  {frame}: {weight:.4f}")

        # Frame diversity
        if trace["encoded_frames"] and "comparison" in trace["encoded_frames"]:
            diversity = trace["encoded_frames"]["comparison"].get("diversity", 0)
            lines.append(f"\nFrame Diversity: {diversity:.4f}")

        # Transformations
        lines.append("\nTransformation Magnitudes:")
        for name, value in trace["transformations"].items():
            lines.append(f"  {name}: {value:.4f}")

        # Final output
        final_stats = trace["final_output"]["statistics"]
        lines.append(f"\nFinal Output: norm={final_stats['norm_l2']:.4f}, "
                    f"mean={final_stats['mean']:.4f}, std={final_stats['std']:.4f}")

        return "\n".join(lines)
