"""
V.V.A.L.T Input Validation Layer

Comprehensive validation for all inputs following API contract specifications.
"""

import numpy as np
from typing import Optional, Tuple
from .errors import *


class InputValidator:
    """Validates input vectors against V.V.A.L.T requirements."""

    def __init__(self, input_dim: int, dtype_strict: bool = False):
        self.input_dim = input_dim
        self.dtype_strict = dtype_strict

    def validate(self, x: np.ndarray, allow_batch: bool = True) -> np.ndarray:
        """
        Validate input vector.

        Args:
            x: Input array
            allow_batch: Allow batch inputs (2D arrays)

        Returns:
            Validated and possibly reshaped input

        Raises:
            InvalidShapeError: Wrong shape
            InvalidDtypeError: Wrong dtype (if strict)
        """
        # Check dtype
        if x.dtype not in [np.float32, np.float64]:
            if self.dtype_strict:
                raise InvalidDtypeError(str(x.dtype))
            else:
                x = x.astype(np.float32)

        # Check shape
        if x.ndim == 1:
            if x.shape[0] != self.input_dim:
                raise InvalidShapeError(self.input_dim, x.shape)
        elif x.ndim == 2:
            if not allow_batch:
                raise InvalidShapeError(self.input_dim, x.shape)
            if x.shape[1] != self.input_dim:
                raise InvalidShapeError(self.input_dim, x.shape)
        else:
            raise InvalidShapeError(self.input_dim, x.shape)

        return x


class TaskVectorValidator:
    """Validates task vectors."""

    def __init__(self, task_dim: int):
        self.task_dim = task_dim

    def validate(self, task_vector: np.ndarray) -> np.ndarray:
        """
        Validate task vector.

        Args:
            task_vector: Task description vector

        Returns:
            Validated task vector

        Raises:
            InvalidTaskVectorError: Wrong dimension
        """
        if task_vector.ndim != 1:
            raise InvalidTaskVectorError(self.task_dim, task_vector.shape[0] if task_vector.ndim > 0 else 0)

        if task_vector.shape[0] != self.task_dim:
            raise InvalidTaskVectorError(self.task_dim, task_vector.shape[0])

        if task_vector.dtype not in [np.float32, np.float64]:
            task_vector = task_vector.astype(np.float32)

        return task_vector


class GraphValidator:
    """Validates graph adjacency matrices."""

    def __init__(self, frame_dim: int, max_size: int = 10000):
        self.frame_dim = frame_dim
        self.max_size = max_size

    def validate(
        self,
        graph_adj: np.ndarray,
        require_symmetric: bool = True,
        require_binary: bool = True,
        auto_fix: bool = False
    ) -> np.ndarray:
        """
        Validate graph adjacency matrix.

        Args:
            graph_adj: Adjacency matrix
            require_symmetric: Enforce symmetry
            require_binary: Enforce binary values
            auto_fix: Attempt automatic fixes

        Returns:
            Validated (and possibly fixed) adjacency matrix

        Raises:
            InvalidGraphShapeError: Not square
            GraphDimensionMismatchError: Size doesn't match frame_dim
            AsymmetricGraphError: Not symmetric (if required and not auto_fix)
            GraphSizeExceededError: Too large
        """
        # Check shape
        if graph_adj.ndim != 2:
            raise InvalidGraphShapeError(graph_adj.shape)

        if graph_adj.shape[0] != graph_adj.shape[1]:
            raise InvalidGraphShapeError(graph_adj.shape)

        # Check size limits
        n = graph_adj.shape[0]
        if n > self.max_size:
            raise GraphSizeExceededError(n, self.max_size)

        # Check dimension match
        if n != self.frame_dim:
            raise GraphDimensionMismatchError(n, self.frame_dim)

        # Check symmetry
        if require_symmetric:
            if not np.allclose(graph_adj, graph_adj.T, atol=1e-8):
                if auto_fix:
                    graph_adj = (graph_adj + graph_adj.T) / 2
                else:
                    raise AsymmetricGraphError()

        # Check binary
        if require_binary:
            unique_vals = np.unique(graph_adj)
            if not np.all(np.isin(unique_vals, [0, 1])):
                if auto_fix:
                    graph_adj = (graph_adj > 0.5).astype(np.float32)
                # Note: Not raising error, just converting

        # Remove self-loops (set diagonal to 0)
        np.fill_diagonal(graph_adj, 0)

        return graph_adj


class BatchValidator:
    """Validates batch processing requests."""

    def __init__(self, max_batch_size: int = 100):
        self.max_batch_size = max_batch_size

    def validate(self, batch_size: int) -> bool:
        """
        Validate batch size.

        Args:
            batch_size: Requested batch size

        Returns:
            True if valid

        Raises:
            BatchSizeExceededError: Batch too large
        """
        if batch_size > self.max_batch_size:
            raise BatchSizeExceededError(batch_size, self.max_batch_size)

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        return True


class OutputValidator:
    """Validates output vectors for contract compliance."""

    def __init__(self, frame_dim: int, bounds: Tuple[float, float] = (-10.0, 10.0)):
        self.frame_dim = frame_dim
        self.bounds = bounds

    def validate(self, output: np.ndarray, strict: bool = False) -> bool:
        """
        Validate output vector.

        Args:
            output: Output vector to validate
            strict: Raise on violations vs. return False

        Returns:
            True if valid

        Raises:
            OutputBoundViolationError: Output exceeds bounds (if strict)
        """
        # Check shape
        if output.shape != (self.frame_dim,):
            if strict:
                raise ValueError(f"Output shape {output.shape} != expected ({self.frame_dim},)")
            return False

        # Check bounds
        if not np.all((output >= self.bounds[0]) & (output <= self.bounds[1])):
            if strict:
                violating_value = output[np.argmax(np.abs(output))]
                raise OutputBoundViolationError(float(violating_value), self.bounds)
            return False

        # Check for NaN/Inf
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            if strict:
                raise NaNDetectedError("OutputValidator", "final_check")
            return False

        return True


class FrameWeightValidator:
    """Validates frame weight distributions."""

    def __init__(self, num_frames: int = 5, tolerance: float = 1e-6):
        self.num_frames = num_frames
        self.tolerance = tolerance

    def validate(self, weights: np.ndarray, strict: bool = False) -> bool:
        """
        Validate frame weights.

        Args:
            weights: Weight array
            strict: Raise on violations

        Returns:
            True if valid

        Raises:
            WeightSumViolationError: Weights don't sum to 1.0 (if strict)
        """
        # Check shape
        if weights.shape[0] != self.num_frames:
            if strict:
                raise FrameCountMismatchError(weights.shape[0], self.num_frames)
            return False

        # Check sum
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0, atol=self.tolerance):
            if strict:
                raise WeightSumViolationError(float(weight_sum), self.tolerance)
            return False

        # Check non-negativity
        if not np.all(weights >= 0):
            return False

        return True


class ValidationPipeline:
    """Complete validation pipeline for V.V.A.L.T inputs."""

    def __init__(
        self,
        input_dim: int,
        frame_dim: int,
        task_dim: int,
        max_batch_size: int = 100,
        max_graph_size: int = 10000
    ):
        self.input_validator = InputValidator(input_dim)
        self.task_validator = TaskVectorValidator(task_dim)
        self.graph_validator = GraphValidator(frame_dim, max_graph_size)
        self.batch_validator = BatchValidator(max_batch_size)
        self.output_validator = OutputValidator(frame_dim)
        self.weight_validator = FrameWeightValidator()

    def validate_inference_inputs(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Validate all inputs for inference.

        Returns:
            Tuple of (validated_x, validated_task, validated_graph)
        """
        x = self.input_validator.validate(x, allow_batch=True)
        task_vector = self.task_validator.validate(task_vector)

        if graph_adj is not None:
            graph_adj = self.graph_validator.validate(graph_adj, auto_fix=True)

        # Validate batch size if batched
        if x.ndim == 2:
            self.batch_validator.validate(x.shape[0])

        return x, task_vector, graph_adj
