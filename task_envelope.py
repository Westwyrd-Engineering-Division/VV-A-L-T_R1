"""
V.V.A.L.T Task Envelope DSL

Domain-specific language for task specification and encoding.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from uuid import uuid4, UUID


class TaskType(Enum):
    """Task type classification."""
    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"
    CAUSAL = "CAUSAL"
    RELATIONAL = "RELATIONAL"
    GRAPH = "GRAPH"
    HYBRID = "HYBRID"


class Priority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class InputEncoding(Enum):
    """Input vector encoding types."""
    RAW = "RAW"
    NORMALIZED = "NORMALIZED"
    STANDARDIZED = "STANDARDIZED"
    EMBEDDED = "EMBEDDED"


class SourceType(Enum):
    """Input source types."""
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    GRAPH = "GRAPH"
    TABULAR = "TABULAR"
    HYBRID = "HYBRID"


class GraphTopology(Enum):
    """Graph topology types."""
    RANDOM = "RANDOM"
    SCALE_FREE = "SCALE_FREE"
    SMALL_WORLD = "SMALL_WORLD"
    TREE = "TREE"
    STAR = "STAR"
    LINE = "LINE"
    RING = "RING"
    COMPLETE = "COMPLETE"
    CUSTOM = "CUSTOM"


@dataclass
class TaskMetadata:
    """Metadata for task envelope."""
    task_id: UUID = field(default_factory=uuid4)
    task_type: TaskType = TaskType.HYBRID
    priority: Priority = Priority.MEDIUM
    timeout_ms: int = 1000
    require_trace: bool = False
    graph_required: bool = False
    batch_compatible: bool = True


@dataclass
class TaskConstraints:
    """Constraints for task execution."""
    max_frame_dim: Optional[int] = None
    determinism_required: bool = True
    safety_level: str = "STANDARD"


@dataclass
class InputSource:
    """Input source information."""
    type: SourceType = SourceType.HYBRID
    preprocessor: str = "default"
    version: str = "1.0"


@dataclass
class InputQuality:
    """Input data quality metrics."""
    completeness: float = 1.0  # [0, 1]
    noise_estimate: float = 0.0
    confidence: float = 1.0  # [0, 1]


@dataclass
class InputEnvelope:
    """Complete input specification."""
    vector: np.ndarray
    encoding: InputEncoding = InputEncoding.RAW
    source: InputSource = field(default_factory=InputSource)
    quality: InputQuality = field(default_factory=InputQuality)


@dataclass
class GraphProperties:
    """Graph structural properties."""
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    is_connected: bool = True
    is_directed: bool = False
    has_self_loops: bool = False


@dataclass
class GraphValidation:
    """Graph validation results."""
    symmetry_check: bool = True
    binary_check: bool = True
    dimension_match: bool = True


@dataclass
class GraphEnvelope:
    """Complete graph specification."""
    adjacency: Optional[np.ndarray] = None
    properties: GraphProperties = field(default_factory=GraphProperties)
    topology: GraphTopology = GraphTopology.CUSTOM
    validation: GraphValidation = field(default_factory=GraphValidation)

    def compute_properties(self):
        """Compute graph properties from adjacency matrix."""
        if self.adjacency is None:
            return

        n = self.adjacency.shape[0]
        self.properties.num_nodes = n
        self.properties.num_edges = int(np.sum(self.adjacency)) // 2
        max_edges = n * (n - 1) // 2
        self.properties.density = self.properties.num_edges / max_edges if max_edges > 0 else 0.0
        self.properties.has_self_loops = bool(np.any(np.diag(self.adjacency) != 0))

    def validate(self):
        """Validate graph adjacency matrix."""
        if self.adjacency is None:
            return

        n = self.adjacency.shape[0]

        # Symmetry check
        self.validation.symmetry_check = bool(np.allclose(self.adjacency, self.adjacency.T))

        # Binary check
        unique_vals = np.unique(self.adjacency)
        self.validation.binary_check = bool(np.all(np.isin(unique_vals, [0, 1])))


@dataclass
class TaskEnvelope:
    """Complete task specification envelope."""
    vector: np.ndarray
    metadata: TaskMetadata = field(default_factory=TaskMetadata)
    constraints: TaskConstraints = field(default_factory=TaskConstraints)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vector_shape": self.vector.shape,
            "metadata": {
                "task_id": str(self.metadata.task_id),
                "task_type": self.metadata.task_type.value,
                "priority": self.metadata.priority.value,
                "timeout_ms": self.metadata.timeout_ms,
                "require_trace": self.metadata.require_trace,
                "graph_required": self.metadata.graph_required,
                "batch_compatible": self.metadata.batch_compatible,
            },
            "constraints": {
                "max_frame_dim": self.constraints.max_frame_dim,
                "determinism_required": self.constraints.determinism_required,
                "safety_level": self.constraints.safety_level,
            }
        }


class TaskVectorBuilder:
    """Builder for creating task vectors with semantic patterns."""

    def __init__(self, task_dim: int):
        self.task_dim = task_dim

    def semantic_dominant(self, noise_level: float = 0.1) -> np.ndarray:
        """Create semantic-dominant task vector."""
        base = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        if self.task_dim > 5:
            base = np.concatenate([base, np.zeros(self.task_dim - 5)])
        elif self.task_dim < 5:
            base = base[:self.task_dim]

        noise = np.random.randn(self.task_dim) * noise_level
        return base + noise

    def structural_dominant(self, noise_level: float = 0.1) -> np.ndarray:
        """Create structural-dominant task vector."""
        base = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        if self.task_dim > 5:
            base = np.concatenate([base, np.zeros(self.task_dim - 5)])
        elif self.task_dim < 5:
            base = base[:self.task_dim]

        noise = np.random.randn(self.task_dim) * noise_level
        return base + noise

    def causal_dominant(self, noise_level: float = 0.1) -> np.ndarray:
        """Create causal-dominant task vector."""
        base = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        if self.task_dim > 5:
            base = np.concatenate([base, np.zeros(self.task_dim - 5)])
        elif self.task_dim < 5:
            base = base[:self.task_dim]

        noise = np.random.randn(self.task_dim) * noise_level
        return base + noise

    def relational_dominant(self, noise_level: float = 0.1) -> np.ndarray:
        """Create relational-dominant task vector."""
        base = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        if self.task_dim > 5:
            base = np.concatenate([base, np.zeros(self.task_dim - 5)])
        elif self.task_dim < 5:
            base = base[:self.task_dim]

        noise = np.random.randn(self.task_dim) * noise_level
        return base + noise

    def graph_dominant(self, noise_level: float = 0.1) -> np.ndarray:
        """Create graph-dominant task vector."""
        base = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        if self.task_dim > 5:
            base = np.concatenate([base, np.zeros(self.task_dim - 5)])
        elif self.task_dim < 5:
            base = base[:self.task_dim]

        noise = np.random.randn(self.task_dim) * noise_level
        return base + noise

    def balanced(self) -> np.ndarray:
        """Create balanced task vector (uniform distribution)."""
        if self.task_dim >= 5:
            base = np.ones(5) * 0.5
            if self.task_dim > 5:
                base = np.concatenate([base, np.ones(self.task_dim - 5) * 0.5])
        else:
            base = np.ones(self.task_dim) * 0.5

        return base

    def custom(self, weights: np.ndarray) -> np.ndarray:
        """Create custom task vector from weights."""
        if len(weights) != self.task_dim:
            raise ValueError(f"Weights must have length {self.task_dim}, got {len(weights)}")

        return np.array(weights, dtype=np.float32)

    def from_task_type(self, task_type: TaskType, noise_level: float = 0.1) -> np.ndarray:
        """Create task vector from TaskType enum."""
        mapping = {
            TaskType.SEMANTIC: self.semantic_dominant,
            TaskType.STRUCTURAL: self.structural_dominant,
            TaskType.CAUSAL: self.causal_dominant,
            TaskType.RELATIONAL: self.relational_dominant,
            TaskType.GRAPH: self.graph_dominant,
            TaskType.HYBRID: self.balanced,
        }

        return mapping[task_type](noise_level) if task_type != TaskType.HYBRID else mapping[task_type]()


def create_task_envelope(
    task_vector: np.ndarray,
    task_type: TaskType = TaskType.HYBRID,
    priority: Priority = Priority.MEDIUM,
    require_trace: bool = False,
    timeout_ms: int = 1000
) -> TaskEnvelope:
    """
    Convenient factory for creating task envelopes.

    Args:
        task_vector: Task description vector
        task_type: Type of task
        priority: Execution priority
        require_trace: Force trace generation
        timeout_ms: Maximum execution time

    Returns:
        TaskEnvelope instance
    """
    metadata = TaskMetadata(
        task_type=task_type,
        priority=priority,
        require_trace=require_trace,
        timeout_ms=timeout_ms
    )

    return TaskEnvelope(vector=task_vector, metadata=metadata)
