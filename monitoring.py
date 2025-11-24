"""
V.V.A.L.T Monitoring and Observability

Metrics collection, logging, and alerting for production deployments.
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import numpy as np


@dataclass
class InferenceMetrics:
    """Metrics for a single inference."""
    timestamp: float
    duration_ms: float
    input_shape: tuple
    output_shape: tuple
    batch_size: int
    trace_generated: bool
    had_nan_inf: bool = False
    component_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComponentTiming:
    """Timing information for a component."""
    component_name: str
    start_time: float
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.end_time == 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class MetricsCollector:
    """Collects and aggregates metrics for monitoring."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.inference_history = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.component_timings = defaultdict(list)

        # Counters
        self.total_inferences = 0
        self.total_errors = 0
        self.nan_inf_count = 0
        self.determinism_violations = 0

    def record_inference(self, metrics: InferenceMetrics):
        """Record inference metrics."""
        self.inference_history.append(metrics)
        self.total_inferences += 1

        if metrics.had_nan_inf:
            self.nan_inf_count += 1

        # Record component timings
        for component, timing in metrics.component_timings.items():
            self.component_timings[component].append(timing)
            # Keep only recent timings
            if len(self.component_timings[component]) > 1000:
                self.component_timings[component] = self.component_timings[component][-1000:]

    def record_error(self, error_code: str):
        """Record error occurrence."""
        self.error_counts[error_code] += 1
        self.total_errors += 1

    def record_determinism_violation(self):
        """Record determinism violation."""
        self.determinism_violations += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.inference_history:
            return {"total_inferences": 0}

        durations = [m.duration_ms for m in self.inference_history]
        batch_sizes = [m.batch_size for m in self.inference_history]

        summary = {
            "total_inferences": self.total_inferences,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_inferences if self.total_inferences > 0 else 0,
            "nan_inf_count": self.nan_inf_count,
            "determinism_violations": self.determinism_violations,
            "duration_ms": {
                "p50": np.percentile(durations, 50),
                "p95": np.percentile(durations, 95),
                "p99": np.percentile(durations, 99),
                "mean": np.mean(durations),
                "min": np.min(durations),
                "max": np.max(durations),
            },
            "batch_size": {
                "mean": np.mean(batch_sizes),
                "max": np.max(batch_sizes),
            },
            "error_counts": dict(self.error_counts),
        }

        # Component timings
        component_stats = {}
        for component, timings in self.component_timings.items():
            component_stats[component] = {
                "mean_ms": np.mean(timings),
                "p95_ms": np.percentile(timings, 95),
            }
        summary["component_timings"] = component_stats

        return summary

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics."""
        summary = self.get_summary()

        metrics = []
        metrics.append(f"# HELP vvalt_inference_count Total number of inferences")
        metrics.append(f"# TYPE vvalt_inference_count counter")
        metrics.append(f"vvalt_inference_count {self.total_inferences}")

        metrics.append(f"# HELP vvalt_errors_total Total number of errors")
        metrics.append(f"# TYPE vvalt_errors_total counter")
        metrics.append(f"vvalt_errors_total {self.total_errors}")

        metrics.append(f"# HELP vvalt_nan_inf_count NaN/Inf detection count")
        metrics.append(f"# TYPE vvalt_nan_inf_count counter")
        metrics.append(f"vvalt_nan_inf_count {self.nan_inf_count}")

        if "duration_ms" in summary:
            metrics.append(f"# HELP vvalt_inference_duration_seconds Inference duration")
            metrics.append(f"# TYPE vvalt_inference_duration_seconds histogram")
            for percentile in ["p50", "p95", "p99"]:
                value_ms = summary["duration_ms"][percentile]
                metrics.append(f'vvalt_inference_duration_seconds{{quantile="{percentile}"}} {value_ms/1000:.6f}')

        return "\n".join(metrics)


class StructuredLogger:
    """Structured JSON logger for V.V.A.L.T."""

    def __init__(self, name: str = "vvalt", level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

    def log_inference(
        self,
        duration_ms: float,
        input_shape: tuple,
        output_shape: tuple,
        batch_size: int,
        trace_generated: bool
    ):
        """Log inference event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "component": "VVALT",
            "event": "inference_complete",
            "duration_ms": round(duration_ms, 3),
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "batch_size": batch_size,
            "trace_generated": trace_generated,
        }
        self.logger.info(json.dumps(event))

    def log_error(self, error_code: str, message: str, recovery_hint: Optional[str] = None):
        """Log error event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "component": "VVALT",
            "event": "error",
            "error_code": error_code,
            "message": message,
            "recovery_hint": recovery_hint,
        }
        self.logger.error(json.dumps(event))

    def log_warning(self, message: str, component: Optional[str] = None):
        """Log warning event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "WARNING",
            "component": component or "VVALT",
            "event": "warning",
            "message": message,
        }
        self.logger.warning(json.dumps(event))

    def log_component(self, component: str, event: str, data: Dict[str, Any] = None):
        """Log component-specific event."""
        log_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "DEBUG",
            "component": component,
            "event": event,
        }
        if data:
            log_event.update(data)

        self.logger.debug(json.dumps(log_event))


class PerformanceTracer:
    """Traces component-level performance."""

    def __init__(self):
        self.active_timings: Dict[str, ComponentTiming] = {}
        self.completed_timings: List[ComponentTiming] = []

    def start(self, component_name: str):
        """Start timing a component."""
        self.active_timings[component_name] = ComponentTiming(
            component_name=component_name,
            start_time=time.perf_counter()
        )

    def end(self, component_name: str) -> float:
        """End timing a component and return duration."""
        if component_name in self.active_timings:
            timing = self.active_timings[component_name]
            timing.end_time = time.perf_counter()
            self.completed_timings.append(timing)
            del self.active_timings[component_name]
            return timing.duration_ms
        return 0.0

    def get_timings(self) -> Dict[str, float]:
        """Get all completed timings."""
        return {t.component_name: t.duration_ms for t in self.completed_timings}

    def reset(self):
        """Reset all timings."""
        self.active_timings.clear()
        self.completed_timings.clear()


class AlertManager:
    """Manages alerting based on metrics thresholds."""

    def __init__(self):
        self.thresholds = {
            "error_rate_warning": 0.01,  # 1%
            "error_rate_critical": 0.05,  # 5%
            "latency_p99_warning_ms": 1000,  # 1 second
            "memory_warning_pct": 0.8,  # 80%
            "nan_inf_rate_warning": 0.001,  # 0.1%
        }
        self.alerts: List[Dict[str, Any]] = []

    def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts."""
        self.alerts.clear()

        # Error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.thresholds["error_rate_critical"]:
            self.alerts.append({
                "severity": "CRITICAL",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self.thresholds["error_rate_critical"],
                "message": f"Error rate {error_rate:.2%} exceeds critical threshold"
            })
        elif error_rate > self.thresholds["error_rate_warning"]:
            self.alerts.append({
                "severity": "WARNING",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self.thresholds["error_rate_warning"],
                "message": f"Error rate {error_rate:.2%} exceeds warning threshold"
            })

        # Latency
        if "duration_ms" in metrics and "p99" in metrics["duration_ms"]:
            p99_latency = metrics["duration_ms"]["p99"]
            if p99_latency > self.thresholds["latency_p99_warning_ms"]:
                self.alerts.append({
                    "severity": "WARNING",
                    "metric": "latency_p99",
                    "value": p99_latency,
                    "threshold": self.thresholds["latency_p99_warning_ms"],
                    "message": f"P99 latency {p99_latency:.2f}ms exceeds threshold"
                })

        # NaN/Inf rate
        total = metrics.get("total_inferences", 1)
        nan_inf_rate = metrics.get("nan_inf_count", 0) / total
        if nan_inf_rate > self.thresholds["nan_inf_rate_warning"]:
            self.alerts.append({
                "severity": "WARNING",
                "metric": "nan_inf_rate",
                "value": nan_inf_rate,
                "threshold": self.thresholds["nan_inf_rate_warning"],
                "message": f"NaN/Inf rate {nan_inf_rate:.4%} exceeds threshold"
            })

        # Determinism violations
        if metrics.get("determinism_violations", 0) > 0:
            self.alerts.append({
                "severity": "CRITICAL",
                "metric": "determinism_violations",
                "value": metrics["determinism_violations"],
                "threshold": 0,
                "message": "Determinism violations detected"
            })

        return self.alerts

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return self.alerts


class MonitoringContext:
    """Context manager for monitored inference."""

    def __init__(self, collector: MetricsCollector, logger: StructuredLogger):
        self.collector = collector
        self.logger = logger
        self.tracer = PerformanceTracer()
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self.tracer

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            # Error occurred
            error_code = getattr(exc_val, 'error_code', 'UNKNOWN')
            self.collector.record_error(error_code)
            self.logger.log_error(
                error_code,
                str(exc_val),
                getattr(exc_val, 'recovery_hint', None)
            )

        return False  # Don't suppress exceptions
