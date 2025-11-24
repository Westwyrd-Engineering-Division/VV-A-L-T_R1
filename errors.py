"""
V.V.A.L.T Error Taxonomy

Comprehensive error classification system following engineering specifications.
"""


# Base Error Classes

class VVALTError(Exception):
    """Base exception for all V.V.A.L.T errors."""
    def __init__(self, message: str, error_code: str = None, recovery_hint: str = None):
        self.error_code = error_code
        self.recovery_hint = recovery_hint
        super().__init__(message)


# Input Validation Errors

class InputValidationError(VVALTError, ValueError):
    """Base class for input validation errors."""
    pass


class InvalidShapeError(InputValidationError):
    """E-VAL-001: Input dimension mismatch."""
    def __init__(self, expected: int, actual: tuple):
        message = f"Input dimension mismatch: expected {expected}, got {actual}"
        recovery_hint = "Verify input preprocessing and reshape to correct dimensions"
        super().__init__(message, error_code="E-VAL-001", recovery_hint=recovery_hint)
        self.expected = expected
        self.actual = actual


class InvalidTaskVectorError(InputValidationError):
    """E-VAL-002: Task vector dimension mismatch."""
    def __init__(self, expected: int, actual: int):
        message = f"Task vector dimension mismatch: expected {expected}, got {actual}"
        recovery_hint = "Verify task encoding matches task_dim configuration"
        super().__init__(message, error_code="E-VAL-002", recovery_hint=recovery_hint)
        self.expected = expected
        self.actual = actual


class InvalidGraphShapeError(InputValidationError):
    """E-VAL-003: Graph adjacency matrix is not square."""
    def __init__(self, shape: tuple):
        message = f"Graph adjacency must be square: got shape {shape}"
        recovery_hint = "Verify graph construction produces square adjacency matrix"
        super().__init__(message, error_code="E-VAL-003", recovery_hint=recovery_hint)
        self.shape = shape


class GraphDimensionMismatchError(InputValidationError):
    """E-VAL-004: Graph size doesn't match frame dimension."""
    def __init__(self, graph_size: int, frame_dim: int):
        message = f"Graph size {graph_size} != frame_dim {frame_dim}"
        recovery_hint = "Adjust frame_dim or graph size to match"
        super().__init__(message, error_code="E-VAL-004", recovery_hint=recovery_hint)
        self.graph_size = graph_size
        self.frame_dim = frame_dim


class AsymmetricGraphError(InputValidationError):
    """E-VAL-005: Graph adjacency matrix is not symmetric."""
    def __init__(self):
        message = "Graph adjacency matrix is not symmetric"
        recovery_hint = "Ensure undirected graph: symmetrize with (A + A^T) / 2"
        super().__init__(message, error_code="E-VAL-005", recovery_hint=recovery_hint)


class InvalidDtypeError(InputValidationError):
    """E-VAL-006: Invalid data type for input."""
    def __init__(self, dtype: str):
        message = f"Input dtype must be float32 or float64, got {dtype}"
        recovery_hint = "Cast input to np.float32 or np.float64"
        super().__init__(message, error_code="E-VAL-006", recovery_hint=recovery_hint)
        self.dtype = dtype


# Numerical Errors

class NumericalError(VVALTError, RuntimeError):
    """Base class for numerical computation errors."""
    pass


class NaNDetectedError(NumericalError):
    """E-NUM-001: NaN values detected in computation."""
    def __init__(self, component: str, stage: str):
        message = f"NaN values detected in {component} at stage {stage}"
        recovery_hint = "Check input validity and range; enable auto-sanitization"
        super().__init__(message, error_code="E-NUM-001", recovery_hint=recovery_hint)
        self.component = component
        self.stage = stage


class InfDetectedError(NumericalError):
    """E-NUM-002: Inf values detected in computation."""
    def __init__(self, component: str, stage: str):
        message = f"Inf values detected in {component} at stage {stage}"
        recovery_hint = "Check input magnitude; reduce extreme values"
        super().__init__(message, error_code="E-NUM-002", recovery_hint=recovery_hint)
        self.component = component
        self.stage = stage


class MatrixSingularError(NumericalError):
    """E-NUM-003: Matrix inversion failed."""
    def __init__(self, component: str):
        message = f"Matrix inversion failed in {component}"
        recovery_hint = "Verify graph connectivity; add regularization"
        super().__init__(message, error_code="E-NUM-003", recovery_hint=recovery_hint)
        self.component = component


# Resource Errors

class ResourceError(VVALTError, RuntimeError):
    """Base class for resource allocation errors."""
    pass


class MemoryAllocationError(ResourceError):
    """E-RES-001: Failed to allocate memory."""
    def __init__(self, size_mb: float, component: str):
        message = f"Failed to allocate {size_mb:.2f} MB for {component}"
        recovery_hint = "Reduce batch size or add system RAM"
        super().__init__(message, error_code="E-RES-001", recovery_hint=recovery_hint)
        self.size_mb = size_mb
        self.component = component


class BatchSizeExceededError(ResourceError):
    """E-RES-002: Batch size exceeds configured maximum."""
    def __init__(self, size: int, max_size: int):
        message = f"Batch size {size} exceeds maximum {max_size}"
        recovery_hint = "Reduce batch size or increase batch_size_limit in config"
        super().__init__(message, error_code="E-RES-002", recovery_hint=recovery_hint)
        self.size = size
        self.max_size = max_size


class GraphSizeExceededError(ResourceError):
    """E-RES-003: Graph size exceeds maximum."""
    def __init__(self, size: int, max_size: int = 10000):
        message = f"Graph size {size} exceeds maximum {max_size}"
        recovery_hint = "Reduce graph size or disable graph projection"
        super().__init__(message, error_code="E-RES-003", recovery_hint=recovery_hint)
        self.size = size
        self.max_size = max_size


# State Errors

class StateError(VVALTError, RuntimeError):
    """Base class for state-related errors."""
    pass


class UninitializedComponentError(StateError):
    """E-STATE-001: Component accessed before initialization."""
    def __init__(self, component: str):
        message = f"{component} not initialized"
        recovery_hint = "Verify VVALT.__init__() completed successfully"
        super().__init__(message, error_code="E-STATE-001", recovery_hint=recovery_hint)
        self.component = component


class CheckpointLoadError(StateError):
    """E-STATE-002: Failed to load checkpoint."""
    def __init__(self, path: str, reason: str):
        message = f"Failed to load checkpoint from {path}: {reason}"
        recovery_hint = "Verify checkpoint file integrity and version compatibility"
        super().__init__(message, error_code="E-STATE-002", recovery_hint=recovery_hint)
        self.path = path
        self.reason = reason


class DeterminismViolationError(StateError):
    """E-STATE-003: Determinism check failed."""
    def __init__(self, max_diff: float, tolerance: float):
        message = f"Determinism check failed: max difference {max_diff} > tolerance {tolerance}"
        recovery_hint = "Report as bug - indicates non-deterministic operations"
        super().__init__(message, error_code="E-STATE-003", recovery_hint=recovery_hint)
        self.max_diff = max_diff
        self.tolerance = tolerance


# Contract Violation Errors

class ContractViolationError(VVALTError, AssertionError):
    """Base class for contract violations (critical bugs)."""
    pass


class OutputBoundViolationError(ContractViolationError):
    """E-CONTRACT-001: Output exceeds safety bounds."""
    def __init__(self, value: float, bounds: tuple):
        message = f"Output exceeds bounds: {value} not in {bounds}"
        recovery_hint = "CRITICAL BUG - Report immediately"
        super().__init__(message, error_code="E-CONTRACT-001", recovery_hint=recovery_hint)
        self.value = value
        self.bounds = bounds


class WeightSumViolationError(ContractViolationError):
    """E-CONTRACT-002: Frame weights don't sum to 1.0."""
    def __init__(self, actual_sum: float, tolerance: float = 1e-6):
        message = f"Frame weights sum to {actual_sum}, expected 1.0 ± {tolerance}"
        recovery_hint = "Check task vector magnitude; renormalize if needed"
        super().__init__(message, error_code="E-CONTRACT-002", recovery_hint=recovery_hint)
        self.actual_sum = actual_sum
        self.tolerance = tolerance


class FrameCountMismatchError(ContractViolationError):
    """E-CONTRACT-003: Wrong number of frames encoded."""
    def __init__(self, count: int, expected: int = 5):
        message = f"Expected {expected} frames, got {count}"
        recovery_hint = "CRITICAL BUG - Report immediately"
        super().__init__(message, error_code="E-CONTRACT-003", recovery_hint=recovery_hint)
        self.count = count
        self.expected = expected


# Error Registry for Monitoring

ERROR_REGISTRY = {
    "E-VAL-001": InvalidShapeError,
    "E-VAL-002": InvalidTaskVectorError,
    "E-VAL-003": InvalidGraphShapeError,
    "E-VAL-004": GraphDimensionMismatchError,
    "E-VAL-005": AsymmetricGraphError,
    "E-VAL-006": InvalidDtypeError,
    "E-NUM-001": NaNDetectedError,
    "E-NUM-002": InfDetectedError,
    "E-NUM-003": MatrixSingularError,
    "E-RES-001": MemoryAllocationError,
    "E-RES-002": BatchSizeExceededError,
    "E-RES-003": GraphSizeExceededError,
    "E-STATE-001": UninitializedComponentError,
    "E-STATE-002": CheckpointLoadError,
    "E-STATE-003": DeterminismViolationError,
    "E-CONTRACT-001": OutputBoundViolationError,
    "E-CONTRACT-002": WeightSumViolationError,
    "E-CONTRACT-003": FrameCountMismatchError,
}
