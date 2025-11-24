"""
V.V.A.L.T Enhanced Core

Production-grade wrapper with validation, monitoring, and configuration.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import time

from .core import VVALT as BaseVVALT
from .config import VVALTConfig
from .validation import ValidationPipeline
from .monitoring import MetricsCollector, StructuredLogger, PerformanceTracer, InferenceMetrics
from .checkpoint import CheckpointManager
from .errors import *


class VVALTEnhanced:
    """
    Production-grade V.V.A.L.T with comprehensive validation and monitoring.

    This class wraps the base VVALT implementation with:
    - Input validation
    - Metrics collection
    - Structured logging
    - Performance tracing
    - Checkpoint management
    - Configuration management
    """

    def __init__(
        self,
        config: Optional[VVALTConfig] = None,
        config_path: Optional[str] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize enhanced V.V.A.L.T.

        Args:
            config: Configuration object
            config_path: Path to configuration YAML file
            enable_monitoring: Enable metrics collection and logging
        """
        # Load configuration
        if config is None:
            self.config = VVALTConfig.load(config_path)
        else:
            self.config = config

        # Validate configuration
        self.config.validate()

        # Initialize base model
        self.model = BaseVVALT(
            input_dim=self.config.model.input_dim,
            frame_dim=self.config.model.frame_dim,
            task_dim=self.config.model.task_dim,
            hidden_dim=self.config.model.hidden_dim,
            seed=self.config.model.seed
        )

        # Initialize validation pipeline
        self.validator = ValidationPipeline(
            input_dim=self.config.model.input_dim,
            frame_dim=self.config.model.frame_dim,
            task_dim=self.config.model.task_dim,
            max_batch_size=self.config.runtime.batch_size_limit
        )

        # Initialize monitoring if enabled
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.metrics = MetricsCollector()
            self.logger = StructuredLogger(
                level=self.config.logging.level.value,
                log_file=self.config.logging.log_file
            )
        else:
            self.metrics = None
            self.logger = None

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.model)

    def forward(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        return_trace: bool = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Enhanced forward pass with validation and monitoring.

        Args:
            x: Input vector
            task_vector: Task description vector
            graph_adj: Optional graph adjacency matrix
            return_trace: Override config trace setting

        Returns:
            Tuple of (output, trace)
        """
        # Performance tracing
        tracer = PerformanceTracer() if self.enable_monitoring else None
        start_time = time.perf_counter()

        try:
            # Determine trace flag
            if return_trace is None:
                return_trace = self.config.runtime.enable_trace

            # Validation
            if tracer:
                tracer.start("validation")

            x, task_vector, graph_adj = self.validator.validate_inference_inputs(
                x, task_vector, graph_adj
            )

            if tracer:
                tracer.end("validation")

            # Core inference
            if tracer:
                tracer.start("inference")

            output, trace = self.model.forward(x, task_vector, graph_adj, return_trace)

            if tracer:
                tracer.end("inference")

            # Output validation
            if tracer:
                tracer.start("output_validation")

            is_safe = self.validator.output_validator.validate(
                output,
                strict=(self.config.runtime.safety_level.value == "STRICT")
            )

            if not is_safe and self.logger:
                self.logger.log_warning(
                    "Output validation failed",
                    component="OutputValidator"
                )

            if tracer:
                tracer.end("output_validation")

            # Record metrics
            if self.enable_monitoring:
                duration_ms = (time.perf_counter() - start_time) * 1000

                metrics = InferenceMetrics(
                    timestamp=time.time(),
                    duration_ms=duration_ms,
                    input_shape=x.shape,
                    output_shape=output.shape,
                    batch_size=1 if x.ndim == 1 else x.shape[0],
                    trace_generated=return_trace,
                    component_timings=tracer.get_timings() if tracer else {}
                )

                self.metrics.record_inference(metrics)

                if self.config.logging.log_traces and return_trace:
                    self.logger.log_component(
                        "VVALTEnhanced",
                        "trace_generated",
                        {"trace_keys": list(trace.keys()) if trace else []}
                    )

            return output, trace

        except VVALTError as e:
            # Record error
            if self.enable_monitoring:
                self.metrics.record_error(e.error_code or "UNKNOWN")
                self.logger.log_error(e.error_code or "UNKNOWN", str(e), e.recovery_hint)
            raise

        except Exception as e:
            # Unexpected error
            if self.enable_monitoring:
                self.metrics.record_error("UNEXPECTED")
                self.logger.log_error("UNEXPECTED", str(e), "Report as bug")
            raise

    def __call__(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        return_trace: bool = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Callable interface."""
        return self.forward(x, task_vector, graph_adj, return_trace)

    def batch_forward(
        self,
        X: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process batch with validation.

        Args:
            X: Batch of input vectors
            task_vector: Task vector (shared)
            graph_adj: Optional graph adjacency

        Returns:
            Batch of outputs
        """
        # Validate batch
        X, task_vector, graph_adj = self.validator.validate_inference_inputs(
            X, task_vector, graph_adj
        )

        return self.model.batch_forward(X, task_vector, graph_adj)

    def verify_determinism(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        num_trials: int = 5
    ) -> bool:
        """
        Verify deterministic behavior.

        Args:
            x: Input vector
            task_vector: Task vector
            graph_adj: Optional graph
            num_trials: Number of trials

        Returns:
            True if deterministic
        """
        is_det = self.model.verify_determinism(x, task_vector, graph_adj, num_trials)

        if not is_det and self.enable_monitoring:
            self.metrics.record_determinism_violation()
            self.logger.log_error(
                "E-STATE-003",
                "Determinism violation detected",
                "Report as critical bug"
            )

        return is_det

    def get_safety_report(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> Dict:
        """Generate safety report."""
        return self.model.get_safety_report(x, task_vector, graph_adj)

    def explain(
        self,
        x: np.ndarray,
        task_vector: np.ndarray,
        graph_adj: Optional[np.ndarray] = None
    ) -> str:
        """Generate human-readable explanation."""
        return self.model.explain(x, task_vector, graph_adj)

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> str:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            metadata: Optional metadata

        Returns:
            Checkpoint checksum
        """
        checksum = self.checkpoint_manager.save(path, metadata)

        if self.logger:
            self.logger.log_component(
                "CheckpointManager",
                "checkpoint_saved",
                {"path": path, "checksum": checksum}
            )

        return checksum

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
            strict: Strict validation

        Returns:
            Checkpoint metadata
        """
        metadata = self.checkpoint_manager.load(path, strict)

        if self.logger:
            self.logger.log_component(
                "CheckpointManager",
                "checkpoint_loaded",
                {"path": path, "metadata": metadata}
            )

        return metadata

    def get_metrics_summary(self) -> Dict:
        """Get monitoring metrics summary."""
        if not self.enable_monitoring:
            return {}

        return self.metrics.get_summary()

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-compatible metrics."""
        if not self.enable_monitoring:
            return ""

        return self.metrics.get_prometheus_metrics()

    def reset_metrics(self):
        """Reset all collected metrics."""
        if self.enable_monitoring:
            self.metrics = MetricsCollector()

    @property
    def input_dim(self) -> int:
        """Get input dimension."""
        return self.config.model.input_dim

    @property
    def frame_dim(self) -> int:
        """Get frame dimension."""
        return self.config.model.frame_dim

    @property
    def task_dim(self) -> int:
        """Get task dimension."""
        return self.config.model.task_dim
