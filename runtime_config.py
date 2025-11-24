"""
V.V.A.L.T Runtime Configuration System

Comprehensive runtime configuration for production deployment, including:
- Runtime behavior settings
- Performance optimization
- Safety and validation levels
- Resource management
- Monitoring and telemetry
- Execution policies
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path


class SafetyLevel(Enum):
    """Safety verification levels for runtime execution."""
    PERMISSIVE = "PERMISSIVE"  # Minimal validation, maximum performance
    STANDARD = "STANDARD"      # Balanced validation and performance
    STRICT = "STRICT"          # Maximum validation, safety-first
    PARANOID = "PARANOID"      # All checks enabled, development mode


class MemoryTier(Enum):
    """Memory allocation tiers for resource management."""
    TIER_1 = 1  # Low memory: < 512 MB (embedded, mobile)
    TIER_2 = 2  # Medium memory: 512 MB - 2 GB (standard)
    TIER_3 = 3  # High memory: 2 GB - 8 GB (server)
    TIER_4 = 4  # Very high memory: > 8 GB (HPC, batch)


class LogLevel(Enum):
    """Logging verbosity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SILENT = "SILENT"


class ExecutionMode(Enum):
    """Execution mode for runtime behavior."""
    DEVELOPMENT = "DEVELOPMENT"  # Full logging, validation, debugging
    TESTING = "TESTING"          # Testing with traces and validation
    STAGING = "STAGING"          # Pre-production validation
    PRODUCTION = "PRODUCTION"    # Optimized for performance
    BENCHMARK = "BENCHMARK"      # Minimal overhead for profiling


class CacheStrategy(Enum):
    """Caching strategy for graph and computation results."""
    NONE = "NONE"              # No caching
    LRU = "LRU"                # Least Recently Used
    LFU = "LFU"                # Least Frequently Used
    TTL = "TTL"                # Time To Live
    ADAPTIVE = "ADAPTIVE"      # Adaptive based on usage patterns


class TracingLevel(Enum):
    """Interpretability tracing levels."""
    NONE = "NONE"              # No tracing (production)
    MINIMAL = "MINIMAL"        # Key metrics only
    STANDARD = "STANDARD"      # Standard reasoning trace
    DETAILED = "DETAILED"      # Full component traces
    EXHAUSTIVE = "EXHAUSTIVE"  # All intermediate values


@dataclass
class ExecutionConfig:
    """Runtime execution behavior configuration."""

    # Execution mode
    mode: ExecutionMode = ExecutionMode.PRODUCTION

    # Batch processing
    batch_size_limit: int = 100
    auto_batch: bool = False
    batch_timeout_ms: int = 1000

    # Timeout settings
    inference_timeout_ms: int = 5000
    component_timeout_ms: int = 1000
    enable_timeout: bool = False

    # Retry policy
    max_retries: int = 3
    retry_delay_ms: int = 100
    retry_backoff: float = 2.0

    # Threading
    max_workers: int = 1  # Single-threaded by default (deterministic)
    enable_parallel: bool = False

    # Determinism
    enforce_determinism: bool = True
    verify_determinism: bool = False
    determinism_check_trials: int = 3


@dataclass
class SafetyConfig:
    """Safety verification and validation configuration."""

    # Safety level
    level: SafetyLevel = SafetyLevel.STANDARD

    # Bounds checking
    enable_bounds_check: bool = True
    safe_bounds: tuple = (-10.0, 10.0)
    clip_to_bounds: bool = True

    # NaN/Inf detection
    check_nan: bool = True
    check_inf: bool = True
    fail_on_nan: bool = True
    fail_on_inf: bool = True

    # Input validation
    validate_inputs: bool = True
    validate_shapes: bool = True
    validate_dtypes: bool = True

    # Output validation
    validate_outputs: bool = True
    validate_output_range: bool = True

    # Graph validation
    validate_graph_symmetry: bool = True
    validate_graph_connectivity: bool = False
    validate_graph_weights: bool = True

    # Numerical stability
    epsilon: float = 1e-8
    max_gradient_norm: float = 10.0
    check_numerical_stability: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    # Lazy loading
    lazy_loading: bool = True
    lazy_component_init: bool = False

    # Caching
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    cache_graph_normalization: bool = True
    cache_attention_weights: bool = False
    max_cached_graphs: int = 10
    max_cached_computations: int = 50
    cache_ttl_seconds: int = 3600

    # Memory optimization
    use_float64: bool = False
    use_mixed_precision: bool = False
    clear_intermediate: bool = False

    # Computation optimization
    use_vectorization: bool = True
    use_graph_optimization: bool = True
    fuse_operations: bool = False

    # Profiling
    enable_profiling: bool = False
    profile_components: bool = False
    profile_memory: bool = False


@dataclass
class ResourceConfig:
    """Resource management configuration."""

    # Memory limits
    memory_tier: MemoryTier = MemoryTier.TIER_2
    max_memory_mb: Optional[int] = None
    memory_warning_threshold: float = 0.8
    enable_memory_tracking: bool = False

    # Graph limits
    max_graph_nodes: int = 10000
    max_graph_edges: int = 100000

    # Computation limits
    max_frame_dim: int = 2048
    max_input_dim: int = 4096
    max_batch_size: int = 1000

    # I/O limits
    max_checkpoint_size_mb: int = 500
    max_trace_size_mb: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring and telemetry configuration."""

    # Logging
    log_level: LogLevel = LogLevel.WARNING
    log_traces: bool = False
    log_warnings: bool = True
    log_errors: bool = True
    structured_logging: bool = False
    log_file: Optional[str] = None

    # Metrics collection
    enable_metrics: bool = True
    collect_latency: bool = True
    collect_throughput: bool = True
    collect_errors: bool = True
    collect_resource_usage: bool = False

    # Metrics export
    export_prometheus: bool = False
    prometheus_port: int = 9090
    export_interval_seconds: int = 60

    # Tracing
    tracing_level: TracingLevel = TracingLevel.NONE
    trace_sampling_rate: float = 0.1

    # Alerting
    enable_alerts: bool = False
    alert_on_errors: bool = True
    alert_threshold_errors: int = 10
    alert_webhook: Optional[str] = None


@dataclass
class IntegrationConfig:
    """Integration with external systems configuration."""

    # HuggingFace integration
    hf_compatible: bool = False
    save_pretrained_format: bool = False

    # Checkpoint management
    checkpoint_dir: str = "./checkpoints"
    auto_checkpoint: bool = False
    checkpoint_interval: int = 1000
    keep_n_checkpoints: int = 5

    # Export formats
    enable_onnx_export: bool = False
    enable_torchscript_export: bool = False

    # API configuration
    enable_rest_api: bool = False
    api_port: int = 8000
    enable_grpc: bool = False
    grpc_port: int = 50051


@dataclass
class DebugConfig:
    """Debugging and development configuration."""

    # Debug mode
    debug_mode: bool = False
    verbose: bool = False

    # Assertions
    enable_assertions: bool = False
    enable_type_checks: bool = False

    # Debugging features
    break_on_error: bool = False
    save_error_states: bool = False
    error_state_dir: str = "./debug"

    # Visualization
    enable_visualization: bool = False
    save_attention_maps: bool = False
    save_frame_plots: bool = False
    visualization_dir: str = "./visualizations"


@dataclass
class RuntimeConfig:
    """
    Complete V.V.A.L.T Runtime Configuration.

    This is the master runtime configuration class that combines all
    runtime aspects for production deployment.
    """

    # Sub-configurations
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Metadata
    config_version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'RuntimeConfig':
        """
        Load runtime configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RuntimeConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        runtime_data = data.get('runtime', {})

        return cls(
            execution=cls._load_execution_config(runtime_data.get('execution', {})),
            safety=cls._load_safety_config(runtime_data.get('safety', {})),
            performance=cls._load_performance_config(runtime_data.get('performance', {})),
            resources=cls._load_resource_config(runtime_data.get('resources', {})),
            monitoring=cls._load_monitoring_config(runtime_data.get('monitoring', {})),
            integration=cls._load_integration_config(runtime_data.get('integration', {})),
            debug=cls._load_debug_config(runtime_data.get('debug', {})),
            config_version=runtime_data.get('config_version', '1.0.0'),
            description=runtime_data.get('description', ''),
            tags=runtime_data.get('tags', [])
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'RuntimeConfig':
        """
        Load runtime configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            RuntimeConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        runtime_data = data.get('runtime', {})

        return cls(
            execution=cls._load_execution_config(runtime_data.get('execution', {})),
            safety=cls._load_safety_config(runtime_data.get('safety', {})),
            performance=cls._load_performance_config(runtime_data.get('performance', {})),
            resources=cls._load_resource_config(runtime_data.get('resources', {})),
            monitoring=cls._load_monitoring_config(runtime_data.get('monitoring', {})),
            integration=cls._load_integration_config(runtime_data.get('integration', {})),
            debug=cls._load_debug_config(runtime_data.get('debug', {})),
            config_version=runtime_data.get('config_version', '1.0.0'),
            description=runtime_data.get('description', ''),
            tags=runtime_data.get('tags', [])
        )

    @classmethod
    def from_env(cls) -> 'RuntimeConfig':
        """
        Load runtime configuration from environment variables.

        Environment variables follow the pattern:
        VVALT_RUNTIME_{SECTION}_{PARAMETER}

        Examples:
            VVALT_RUNTIME_EXECUTION_MODE=PRODUCTION
            VVALT_RUNTIME_SAFETY_LEVEL=STRICT
            VVALT_RUNTIME_MONITORING_LOG_LEVEL=INFO

        Returns:
            RuntimeConfig instance
        """
        config = cls()

        # Execution config
        if os.getenv('VVALT_RUNTIME_EXECUTION_MODE'):
            config.execution.mode = ExecutionMode(os.getenv('VVALT_RUNTIME_EXECUTION_MODE'))
        if os.getenv('VVALT_RUNTIME_BATCH_SIZE_LIMIT'):
            config.execution.batch_size_limit = int(os.getenv('VVALT_RUNTIME_BATCH_SIZE_LIMIT'))
        if os.getenv('VVALT_RUNTIME_INFERENCE_TIMEOUT_MS'):
            config.execution.inference_timeout_ms = int(os.getenv('VVALT_RUNTIME_INFERENCE_TIMEOUT_MS'))

        # Safety config
        if os.getenv('VVALT_RUNTIME_SAFETY_LEVEL'):
            config.safety.level = SafetyLevel(os.getenv('VVALT_RUNTIME_SAFETY_LEVEL'))
        if os.getenv('VVALT_RUNTIME_SAFE_BOUNDS'):
            bounds = eval(os.getenv('VVALT_RUNTIME_SAFE_BOUNDS'))
            config.safety.safe_bounds = bounds

        # Performance config
        if os.getenv('VVALT_RUNTIME_CACHE_STRATEGY'):
            config.performance.cache_strategy = CacheStrategy(os.getenv('VVALT_RUNTIME_CACHE_STRATEGY'))
        if os.getenv('VVALT_RUNTIME_USE_FLOAT64'):
            config.performance.use_float64 = os.getenv('VVALT_RUNTIME_USE_FLOAT64').lower() == 'true'

        # Monitoring config
        if os.getenv('VVALT_RUNTIME_LOG_LEVEL'):
            config.monitoring.log_level = LogLevel(os.getenv('VVALT_RUNTIME_LOG_LEVEL'))
        if os.getenv('VVALT_RUNTIME_LOG_FILE'):
            config.monitoring.log_file = os.getenv('VVALT_RUNTIME_LOG_FILE')
        if os.getenv('VVALT_RUNTIME_TRACING_LEVEL'):
            config.monitoring.tracing_level = TracingLevel(os.getenv('VVALT_RUNTIME_TRACING_LEVEL'))

        # Resources config
        if os.getenv('VVALT_RUNTIME_MEMORY_TIER'):
            config.resources.memory_tier = MemoryTier(int(os.getenv('VVALT_RUNTIME_MEMORY_TIER')))
        if os.getenv('VVALT_RUNTIME_MAX_MEMORY_MB'):
            config.resources.max_memory_mb = int(os.getenv('VVALT_RUNTIME_MAX_MEMORY_MB'))

        return config

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> 'RuntimeConfig':
        """
        Load runtime configuration with priority:
        1. Environment variables (highest)
        2. Config file (YAML/JSON)
        3. Defaults (lowest)

        Args:
            config_path: Optional path to configuration file

        Returns:
            RuntimeConfig instance
        """
        # Start with defaults
        config = cls()

        # Override with config file if provided
        if config_path:
            path = Path(config_path)
            if path.exists():
                if path.suffix in ['.yaml', '.yml']:
                    config = cls.from_yaml(path)
                elif path.suffix == '.json':
                    config = cls.from_json(path)

        # Override with environment variables
        env_config = cls.from_env()
        config._merge_from_env(env_config)

        return config

    def _merge_from_env(self, env_config: 'RuntimeConfig'):
        """Merge environment variable overrides."""
        # Only merge if environment variable was explicitly set
        if os.getenv('VVALT_RUNTIME_EXECUTION_MODE'):
            self.execution.mode = env_config.execution.mode
        if os.getenv('VVALT_RUNTIME_SAFETY_LEVEL'):
            self.safety.level = env_config.safety.level
        if os.getenv('VVALT_RUNTIME_LOG_LEVEL'):
            self.monitoring.log_level = env_config.monitoring.log_level
        # Add more as needed

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'execution': self._execution_to_dict(),
            'safety': self._safety_to_dict(),
            'performance': self._performance_to_dict(),
            'resources': self._resources_to_dict(),
            'monitoring': self._monitoring_to_dict(),
            'integration': self._integration_to_dict(),
            'debug': self._debug_to_dict(),
            'config_version': self.config_version,
            'description': self.description,
            'tags': self.tags
        }

    def to_yaml(self, path: Union[str, Path]):
        """Save runtime configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump({'runtime': self.to_dict()}, f, default_flow_style=False)

    def to_json(self, path: Union[str, Path], indent: int = 2):
        """Save runtime configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump({'runtime': self.to_dict()}, f, indent=indent)

    def validate(self) -> bool:
        """
        Validate runtime configuration parameters.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Execution validation
        if self.execution.batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")
        if self.execution.batch_size_limit > 100000:
            raise ValueError("batch_size_limit must be <= 100000")
        if self.execution.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        # Safety validation
        if self.safety.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.safety.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")
        if len(self.safety.safe_bounds) != 2:
            raise ValueError("safe_bounds must be a tuple of (min, max)")
        if self.safety.safe_bounds[0] >= self.safety.safe_bounds[1]:
            raise ValueError("safe_bounds min must be < max")

        # Performance validation
        if self.performance.max_cached_graphs < 0:
            raise ValueError("max_cached_graphs must be non-negative")
        if self.performance.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")

        # Resource validation
        if self.resources.max_frame_dim <= 0:
            raise ValueError("max_frame_dim must be positive")
        if self.resources.max_input_dim <= 0:
            raise ValueError("max_input_dim must be positive")
        if self.resources.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.resources.max_memory_mb is not None and self.resources.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

        # Monitoring validation
        if self.monitoring.trace_sampling_rate < 0 or self.monitoring.trace_sampling_rate > 1:
            raise ValueError("trace_sampling_rate must be in [0, 1]")
        if self.monitoring.export_interval_seconds <= 0:
            raise ValueError("export_interval_seconds must be positive")

        return True

    def get_preset(self, preset: str) -> 'RuntimeConfig':
        """
        Get a preset configuration.

        Args:
            preset: Preset name ('development', 'testing', 'staging', 'production', 'benchmark')

        Returns:
            RuntimeConfig instance with preset values
        """
        if preset == 'development':
            return self._development_preset()
        elif preset == 'testing':
            return self._testing_preset()
        elif preset == 'staging':
            return self._staging_preset()
        elif preset == 'production':
            return self._production_preset()
        elif preset == 'benchmark':
            return self._benchmark_preset()
        else:
            raise ValueError(f"Unknown preset: {preset}")

    # Helper methods for loading sub-configurations
    @staticmethod
    def _load_execution_config(data: Dict) -> ExecutionConfig:
        """Load ExecutionConfig from dict."""
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = ExecutionMode(data['mode'])
        return ExecutionConfig(**data)

    @staticmethod
    def _load_safety_config(data: Dict) -> SafetyConfig:
        """Load SafetyConfig from dict."""
        if 'level' in data and isinstance(data['level'], str):
            data['level'] = SafetyLevel(data['level'])
        if 'safe_bounds' in data and isinstance(data['safe_bounds'], list):
            data['safe_bounds'] = tuple(data['safe_bounds'])
        return SafetyConfig(**data)

    @staticmethod
    def _load_performance_config(data: Dict) -> PerformanceConfig:
        """Load PerformanceConfig from dict."""
        if 'cache_strategy' in data and isinstance(data['cache_strategy'], str):
            data['cache_strategy'] = CacheStrategy(data['cache_strategy'])
        return PerformanceConfig(**data)

    @staticmethod
    def _load_resource_config(data: Dict) -> ResourceConfig:
        """Load ResourceConfig from dict."""
        if 'memory_tier' in data and isinstance(data['memory_tier'], int):
            data['memory_tier'] = MemoryTier(data['memory_tier'])
        return ResourceConfig(**data)

    @staticmethod
    def _load_monitoring_config(data: Dict) -> MonitoringConfig:
        """Load MonitoringConfig from dict."""
        if 'log_level' in data and isinstance(data['log_level'], str):
            data['log_level'] = LogLevel(data['log_level'])
        if 'tracing_level' in data and isinstance(data['tracing_level'], str):
            data['tracing_level'] = TracingLevel(data['tracing_level'])
        return MonitoringConfig(**data)

    @staticmethod
    def _load_integration_config(data: Dict) -> IntegrationConfig:
        """Load IntegrationConfig from dict."""
        return IntegrationConfig(**data)

    @staticmethod
    def _load_debug_config(data: Dict) -> DebugConfig:
        """Load DebugConfig from dict."""
        return DebugConfig(**data)

    # Helper methods for dict conversion
    def _execution_to_dict(self) -> Dict:
        """Convert ExecutionConfig to dict."""
        d = asdict(self.execution)
        d['mode'] = self.execution.mode.value
        return d

    def _safety_to_dict(self) -> Dict:
        """Convert SafetyConfig to dict."""
        d = asdict(self.safety)
        d['level'] = self.safety.level.value
        # Convert tuple to list for YAML/JSON compatibility
        if isinstance(d.get('safe_bounds'), tuple):
            d['safe_bounds'] = list(d['safe_bounds'])
        return d

    def _performance_to_dict(self) -> Dict:
        """Convert PerformanceConfig to dict."""
        d = asdict(self.performance)
        d['cache_strategy'] = self.performance.cache_strategy.value
        return d

    def _resources_to_dict(self) -> Dict:
        """Convert ResourceConfig to dict."""
        d = asdict(self.resources)
        d['memory_tier'] = self.resources.memory_tier.value
        return d

    def _monitoring_to_dict(self) -> Dict:
        """Convert MonitoringConfig to dict."""
        d = asdict(self.monitoring)
        d['log_level'] = self.monitoring.log_level.value
        d['tracing_level'] = self.monitoring.tracing_level.value
        return d

    def _integration_to_dict(self) -> Dict:
        """Convert IntegrationConfig to dict."""
        return asdict(self.integration)

    def _debug_to_dict(self) -> Dict:
        """Convert DebugConfig to dict."""
        return asdict(self.debug)

    # Preset configurations
    @classmethod
    def _development_preset(cls) -> 'RuntimeConfig':
        """Development preset with full debugging."""
        config = cls()
        config.execution.mode = ExecutionMode.DEVELOPMENT
        config.safety.level = SafetyLevel.STRICT
        config.monitoring.log_level = LogLevel.DEBUG
        config.monitoring.tracing_level = TracingLevel.DETAILED
        config.debug.debug_mode = True
        config.debug.verbose = True
        config.debug.enable_assertions = True
        config.description = "Development preset with full debugging"
        return config

    @classmethod
    def _testing_preset(cls) -> 'RuntimeConfig':
        """Testing preset for test suites."""
        config = cls()
        config.execution.mode = ExecutionMode.TESTING
        config.safety.level = SafetyLevel.STRICT
        config.execution.verify_determinism = True
        config.monitoring.log_level = LogLevel.INFO
        config.monitoring.tracing_level = TracingLevel.STANDARD
        config.description = "Testing preset for test suites"
        return config

    @classmethod
    def _staging_preset(cls) -> 'RuntimeConfig':
        """Staging preset for pre-production."""
        config = cls()
        config.execution.mode = ExecutionMode.STAGING
        config.safety.level = SafetyLevel.STANDARD
        config.monitoring.log_level = LogLevel.WARNING
        config.monitoring.enable_metrics = True
        config.monitoring.collect_latency = True
        config.description = "Staging preset for pre-production"
        return config

    @classmethod
    def _production_preset(cls) -> 'RuntimeConfig':
        """Production preset optimized for performance."""
        config = cls()
        config.execution.mode = ExecutionMode.PRODUCTION
        config.safety.level = SafetyLevel.STANDARD
        config.performance.lazy_loading = True
        config.performance.cache_graph_normalization = True
        config.monitoring.log_level = LogLevel.ERROR
        config.monitoring.tracing_level = TracingLevel.NONE
        config.monitoring.enable_metrics = True
        config.description = "Production preset optimized for performance"
        return config

    @classmethod
    def _benchmark_preset(cls) -> 'RuntimeConfig':
        """Benchmark preset with minimal overhead."""
        config = cls()
        config.execution.mode = ExecutionMode.BENCHMARK
        config.safety.level = SafetyLevel.PERMISSIVE
        config.safety.validate_inputs = False
        config.safety.validate_outputs = False
        config.monitoring.log_level = LogLevel.SILENT
        config.monitoring.tracing_level = TracingLevel.NONE
        config.monitoring.enable_metrics = False
        config.description = "Benchmark preset with minimal overhead"
        return config


# Convenience function for quick configuration
def create_runtime_config(
    preset: Optional[str] = None,
    **kwargs
) -> RuntimeConfig:
    """
    Create a runtime configuration with optional preset and overrides.

    Args:
        preset: Optional preset name ('development', 'testing', 'staging', 'production', 'benchmark')
        **kwargs: Additional keyword arguments to override defaults

    Returns:
        RuntimeConfig instance

    Example:
        >>> config = create_runtime_config('production', batch_size_limit=200)
        >>> config = create_runtime_config(safety_level=SafetyLevel.STRICT)
    """
    if preset:
        config = RuntimeConfig().get_preset(preset)
    else:
        config = RuntimeConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
