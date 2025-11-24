"""
V.V.A.L.T Core

Main class integrating all components of the Vantage-Vector Autonomous Logic Transformer.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .components import (
    VectorFrameEncoder,
    VantageSelector,
    GraphTopologyProjector,
    MultiPerspectiveAttention,
    LogicRefinementUnit,
    ConsistencyVerifier,
    InterpretabilityProjector,
)


class VVALT:
    """
    Vantage-Vector Autonomous Logic Transformer

    A deterministic, bounded logic reasoning system that operates through
    multi-perspective vector frame analysis.

    Safety Guarantees:
    - No autonomous loops (single-pass execution only)
    - Deterministic (same input always produces same output)
    - Bounded computation (fixed computational complexity)
    - Operator-controlled (no self-directed behavior)
    - Full interpretability (complete reasoning trace visibility)
    """

    def __init__(
        self,
        input_dim: int,
        frame_dim: int,
        task_dim: int,
        hidden_dim: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize V.V.A.L.T system.

        Args:
            input_dim: Dimension of input vectors
            frame_dim: Dimension of each perspective frame
            task_dim: Dimension of task vectors
            hidden_dim: Hidden dimension for logic refinement (default: 2 * frame_dim)
            seed: Random seed for deterministic behavior
        """
        self.input_dim = input_dim
        self.frame_dim = frame_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim if hidden_dim else 2 * frame_dim
        self.seed = seed

        # Initialize all components
        self.encoder = VectorFrameEncoder(input_dim, frame_dim, seed=seed)
        self.selector = VantageSelector(task_dim, num_frames=5, seed=seed)
        self.projector = GraphTopologyProjector(frame_dim, seed=seed)
        self.attention = MultiPerspectiveAttention(frame_dim, num_frames=5, seed=seed)
        self.refiner = LogicRefinementUnit(frame_dim, self.hidden_dim, seed=seed)
        self.verifier = ConsistencyVerifier(safe_range=(-10.0, 10.0))
        self.interpreter = InterpretabilityProjector(frame_dim, num_samples=5)

    def forward(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        return_trace: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Single-pass forward reasoning through V.V.A.L.T.

        This is the main entry point for the system. Performs deterministic,
        bounded reasoning with full interpretability.

        Args:
            x: Input vector of shape (input_dim,)
            task_vector: Task description vector of shape (task_dim,)
            graph_adj: Optional graph adjacency matrix
            return_trace: If True, return full reasoning trace

        Returns:
            Tuple of (output_vector, reasoning_trace)
            - output_vector: Final safe output of shape (frame_dim,)
            - reasoning_trace: Dictionary of interpretability info (if return_trace=True)
        """
        # Step 1: Encode input into multiple perspective frames
        frames = self.encoder.encode(x, graph_adj)

        # Step 2: Task-conditioned frame weighting
        weighted_frames = self.selector.select_frames(frames, task_vector)
        task_weights = self.selector.get_weight_distribution(task_vector)

        # Step 3: Graph topology alignment
        aligned_frames = self.projector.project(weighted_frames, graph_adj)

        # Step 4: Multi-perspective attention fusion
        attention_output = self.attention.attend(aligned_frames)

        # Step 5: Bounded logic refinement
        refined_output = self.refiner.refine(attention_output)

        # Step 6: Consistency verification and safety validation
        final_output = self.verifier.verify(refined_output, strict=False)

        # Step 7: Generate interpretability trace if requested
        trace = None
        if return_trace:
            trace = self.interpreter.create_reasoning_trace(
                input_vector=x,
                frames=frames,
                weighted_frames=weighted_frames,
                attention_output=attention_output,
                refined_output=refined_output,
                final_output=final_output,
                task_weights=task_weights
            )

        return final_output, trace

    def __call__(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        return_trace: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Callable interface for V.V.A.L.T.

        Allows using instance as a function: output = vvalt(input, task)
        """
        return self.forward(x, task_vector, graph_adj, return_trace)

    def verify_determinism(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        num_trials: int = 5
    ) -> bool:
        """
        Verify that system is deterministic.

        Runs the same input multiple times and checks that outputs are identical.

        Args:
            x: Input vector
            task_vector: Task vector
            graph_adj: Optional graph adjacency
            num_trials: Number of times to run

        Returns:
            True if all outputs are identical
        """
        outputs = []
        for _ in range(num_trials):
            output, _ = self.forward(x, task_vector, graph_adj, return_trace=False)
            outputs.append(output)

        # Check all outputs are identical
        reference = outputs[0]
        for output in outputs[1:]:
            if not self.verifier.verify_deterministic(reference, output):
                return False

        return True

    def get_safety_report(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate comprehensive safety report for input.

        Args:
            x: Input vector
            task_vector: Task vector
            graph_adj: Optional graph adjacency

        Returns:
            Dictionary containing safety analysis
        """
        output, _ = self.forward(x, task_vector, graph_adj, return_trace=False)

        return {
            "deterministic": self.verify_determinism(x, task_vector, graph_adj, num_trials=3),
            "output_safety": self.verifier.get_safety_report(output),
            "bounded": self.verifier.verify_bounds(output),
        }

    def explain(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate human-readable explanation of reasoning process.

        Args:
            x: Input vector
            task_vector: Task vector
            graph_adj: Optional graph adjacency

        Returns:
            Formatted explanation string
        """
        output, trace = self.forward(x, task_vector, graph_adj, return_trace=True)

        return self.interpreter.format_trace_summary(trace)

    def batch_forward(
        self,
        X: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process batch of inputs (single-pass per input).

        Args:
            X: Batch of input vectors of shape (batch_size, input_dim)
            task_vector: Task vector (same for all inputs)
            graph_adj: Optional graph adjacency

        Returns:
            Batch of outputs of shape (batch_size, frame_dim)
        """
        outputs = []
        for x in X:
            output, _ = self.forward(x, task_vector, graph_adj, return_trace=False)
            outputs.append(output)

        return np.stack(outputs, axis=0)
