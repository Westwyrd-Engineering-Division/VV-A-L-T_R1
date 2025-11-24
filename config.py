"""
V.V.A.L.T Configuration Management System

Implements hierarchical configuration loading with environment variable overrides.
"""

import os
import yaml
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict


class SafetyLevel(Enum):
    """Safety verification levels."""
    PERMISSIVE = "PERMISSIVE"
    STANDARD = "STANDARD"
    STRICT = "STRICT"


class MemoryTier(Enum):
    """Memory allocation tiers."""
    TIER_1 = 1  # < 512 MB
    TIER_2 = 2  # 512 MB - 2 GB
    TIER_3 = 3  # > 2 GB


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 10
    frame_dim: int = 8
    task_dim: int = 5
    hidden_dim: Optional[int] = None  # Defaults to 2 * frame_dim
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = 2 * self.frame_dim


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration."""
    batch_size_limit: int = 100
    enable_trace: bool = False
    safety_level: SafetyLevel = SafetyLevel.STANDARD
    memory_tier: MemoryTier = MemoryTier.TIER_2


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    lazy_loading: bool = True
    cache_graph_normalization: bool = True
    max_cached_graphs: int = 10
    use_float64: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.WARNING
    log_traces: bool = False
    log_warnings: bool = True
    structured_logging: bool = False
    log_file: Optional[str] = None


@dataclass
class VVALTConfig:
    """Complete V.V.A.L.T configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'VVALTConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        vvalt_data = data.get('vvalt', {})

        return cls(
            model=ModelConfig(**vvalt_data.get('model', {})),
            runtime=RuntimeConfig(
                **{k: SafetyLevel(v) if k == 'safety_level' and isinstance(v, str) else
                   MemoryTier(v) if k == 'memory_tier' else v
                   for k, v in vvalt_data.get('runtime', {}).items()}
            ),
            performance=PerformanceConfig(**vvalt_data.get('performance', {})),
            logging=LoggingConfig(
                **{k: LogLevel(v) if k == 'level' and isinstance(v, str) else v
                   for k, v in vvalt_data.get('logging', {}).items()}
            )
        )

    @classmethod
    def from_env(cls) -> 'VVALTConfig':
        """Load configuration from environment variables."""
        config = cls()

        # Model config
        if os.getenv('VVALT_INPUT_DIM'):
            config.model.input_dim = int(os.getenv('VVALT_INPUT_DIM'))
        if os.getenv('VVALT_FRAME_DIM'):
            config.model.frame_dim = int(os.getenv('VVALT_FRAME_DIM'))
        if os.getenv('VVALT_TASK_DIM'):
            config.model.task_dim = int(os.getenv('VVALT_TASK_DIM'))
        if os.getenv('VVALT_SEED'):
            config.model.seed = int(os.getenv('VVALT_SEED'))

        # Runtime config
        if os.getenv('VVALT_SAFETY_LEVEL'):
            config.runtime.safety_level = SafetyLevel(os.getenv('VVALT_SAFETY_LEVEL'))
        if os.getenv('VVALT_BATCH_SIZE_LIMIT'):
            config.runtime.batch_size_limit = int(os.getenv('VVALT_BATCH_SIZE_LIMIT'))

        # Logging config
        if os.getenv('VVALT_LOG_LEVEL'):
            config.logging.level = LogLevel(os.getenv('VVALT_LOG_LEVEL'))

        return config

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'VVALTConfig':
        """
        Load configuration with priority:
        1. Environment variables (highest)
        2. YAML file
        3. Defaults (lowest)
        """
        # Start with defaults
        config = cls()

        # Override with YAML if provided
        if config_path and os.path.exists(config_path):
            config = cls.from_yaml(config_path)

        # Override with environment variables
        env_config = cls.from_env()
        config.model.input_dim = env_config.model.input_dim or config.model.input_dim
        config.model.frame_dim = env_config.model.frame_dim or config.model.frame_dim
        config.model.task_dim = env_config.model.task_dim or config.model.task_dim
        config.model.seed = env_config.model.seed or config.model.seed

        if os.getenv('VVALT_SAFETY_LEVEL'):
            config.runtime.safety_level = env_config.runtime.safety_level
        if os.getenv('VVALT_LOG_LEVEL'):
            config.logging.level = env_config.logging.level

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'runtime': {
                **asdict(self.runtime),
                'safety_level': self.runtime.safety_level.value,
                'memory_tier': self.runtime.memory_tier.value
            },
            'performance': asdict(self.performance),
            'logging': {
                **asdict(self.logging),
                'level': self.logging.level.value
            }
        }

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump({'vvalt': self.to_dict()}, f, default_flow_style=False)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.model.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.model.frame_dim <= 0:
            raise ValueError("frame_dim must be positive")
        if self.model.task_dim <= 0:
            raise ValueError("task_dim must be positive")
        if self.runtime.batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")
        if self.runtime.batch_size_limit > 10000:
            raise ValueError("batch_size_limit must be <= 10000")
        if self.performance.max_cached_graphs <= 0:
            raise ValueError("max_cached_graphs must be positive")

        return True
