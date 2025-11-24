"""
V.V.A.L.T Runtime Configuration Demo

Demonstrates how to use the comprehensive runtime configuration system.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvalt import (
    RuntimeConfig,
    ExecutionMode,
    SafetyLevel,
    LogLevel,
    TracingLevel,
    CacheStrategy,
    create_runtime_config,
)


def print_separator(title=""):
    """Print formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_default_config():
    """Demonstrate default configuration."""
    print_separator("Default Runtime Configuration")

    config = RuntimeConfig()

    print("Execution mode:", config.execution.mode.value)
    print("Safety level:", config.safety.level.value)
    print("Log level:", config.monitoring.log_level.value)
    print("Cache strategy:", config.performance.cache_strategy.value)
    print("Memory tier:", config.resources.memory_tier.value)

    # Validate
    try:
        config.validate()
        print("\n✓ Configuration is valid")
    except Exception as e:
        print(f"\n✗ Configuration validation failed: {e}")


def demo_preset_configs():
    """Demonstrate preset configurations."""
    print_separator("Preset Configurations")

    presets = ['development', 'testing', 'staging', 'production', 'benchmark']

    for preset in presets:
        config = RuntimeConfig().get_preset(preset)
        print(f"\n{preset.upper()} preset:")
        print(f"  Mode: {config.execution.mode.value}")
        print(f"  Safety: {config.safety.level.value}")
        print(f"  Logging: {config.monitoring.log_level.value}")
        print(f"  Tracing: {config.monitoring.tracing_level.value}")
        print(f"  Description: {config.description}")


def demo_custom_config():
    """Demonstrate custom configuration."""
    print_separator("Custom Runtime Configuration")

    config = RuntimeConfig()

    # Customize execution
    config.execution.mode = ExecutionMode.PRODUCTION
    config.execution.batch_size_limit = 200
    config.execution.enforce_determinism = True

    # Customize safety
    config.safety.level = SafetyLevel.STRICT
    config.safety.safe_bounds = (-5.0, 5.0)

    # Customize monitoring
    config.monitoring.log_level = LogLevel.INFO
    config.monitoring.tracing_level = TracingLevel.STANDARD
    config.monitoring.enable_metrics = True

    # Customize performance
    config.performance.cache_strategy = CacheStrategy.LRU
    config.performance.max_cached_graphs = 20

    print("Custom configuration created:")
    print(f"  Batch size limit: {config.execution.batch_size_limit}")
    print(f"  Safety bounds: {config.safety.safe_bounds}")
    print(f"  Log level: {config.monitoring.log_level.value}")
    print(f"  Cache strategy: {config.performance.cache_strategy.value}")

    # Validate
    config.validate()
    print("\n✓ Custom configuration is valid")


def demo_save_load_yaml():
    """Demonstrate saving and loading YAML configuration."""
    print_separator("Save/Load YAML Configuration")

    # Create configuration
    config = create_runtime_config('production')
    config.description = "Custom production configuration"
    config.tags = ["production", "v1.0"]

    # Save to YAML
    yaml_path = "/tmp/vvalt_runtime.yaml"
    config.to_yaml(yaml_path)
    print(f"✓ Configuration saved to {yaml_path}")

    # Load from YAML
    loaded_config = RuntimeConfig.from_yaml(yaml_path)
    print(f"✓ Configuration loaded from {yaml_path}")
    print(f"  Description: {loaded_config.description}")
    print(f"  Tags: {loaded_config.tags}")
    print(f"  Mode: {loaded_config.execution.mode.value}")


def demo_save_load_json():
    """Demonstrate saving and loading JSON configuration."""
    print_separator("Save/Load JSON Configuration")

    # Create configuration
    config = create_runtime_config('staging')
    config.description = "Staging environment configuration"

    # Save to JSON
    json_path = "/tmp/vvalt_runtime.json"
    config.to_json(json_path, indent=2)
    print(f"✓ Configuration saved to {json_path}")

    # Load from JSON
    loaded_config = RuntimeConfig.from_json(json_path)
    print(f"✓ Configuration loaded from {json_path}")
    print(f"  Description: {loaded_config.description}")
    print(f"  Mode: {loaded_config.execution.mode.value}")
    print(f"  Safety level: {loaded_config.safety.level.value}")


def demo_environment_override():
    """Demonstrate environment variable overrides."""
    print_separator("Environment Variable Overrides")

    # Set environment variables
    os.environ['VVALT_RUNTIME_EXECUTION_MODE'] = 'DEVELOPMENT'
    os.environ['VVALT_RUNTIME_SAFETY_LEVEL'] = 'STRICT'
    os.environ['VVALT_RUNTIME_LOG_LEVEL'] = 'DEBUG'
    os.environ['VVALT_RUNTIME_BATCH_SIZE_LIMIT'] = '50'

    # Load configuration from environment
    config = RuntimeConfig.from_env()

    print("Configuration loaded from environment variables:")
    print(f"  Execution mode: {config.execution.mode.value}")
    print(f"  Safety level: {config.safety.level.value}")
    print(f"  Log level: {config.monitoring.log_level.value}")
    print(f"  Batch size limit: {config.execution.batch_size_limit}")

    # Clean up
    del os.environ['VVALT_RUNTIME_EXECUTION_MODE']
    del os.environ['VVALT_RUNTIME_SAFETY_LEVEL']
    del os.environ['VVALT_RUNTIME_LOG_LEVEL']
    del os.environ['VVALT_RUNTIME_BATCH_SIZE_LIMIT']


def demo_hierarchical_loading():
    """Demonstrate hierarchical configuration loading."""
    print_separator("Hierarchical Configuration Loading")

    # Create base config file
    base_config = create_runtime_config('production')
    base_config.execution.batch_size_limit = 100
    base_config.to_yaml("/tmp/vvalt_base.yaml")

    # Set environment override
    os.environ['VVALT_RUNTIME_BATCH_SIZE_LIMIT'] = '200'
    os.environ['VVALT_RUNTIME_SAFETY_LEVEL'] = 'PARANOID'

    # Load with priority: env > file > defaults
    config = RuntimeConfig.load("/tmp/vvalt_base.yaml")

    print("Configuration loaded with hierarchical priority:")
    print("  Priority: Environment > File > Defaults")
    print(f"  Batch size limit: {config.execution.batch_size_limit} (from env)")
    print(f"  Safety level: {config.safety.level.value} (from env)")
    print(f"  Mode: {config.execution.mode.value} (from file)")

    # Clean up
    del os.environ['VVALT_RUNTIME_BATCH_SIZE_LIMIT']
    del os.environ['VVALT_RUNTIME_SAFETY_LEVEL']


def demo_to_dict():
    """Demonstrate dictionary conversion."""
    print_separator("Dictionary Conversion")

    config = create_runtime_config('production')
    config_dict = config.to_dict()

    print("Configuration converted to dictionary:")
    print(f"  Keys: {list(config_dict.keys())}")
    print(f"  Execution keys: {list(config_dict['execution'].keys())}")
    print(f"  Safety keys: {list(config_dict['safety'].keys())}")
    print("\nSample values:")
    print(f"  execution.mode: {config_dict['execution']['mode']}")
    print(f"  safety.level: {config_dict['safety']['level']}")
    print(f"  monitoring.log_level: {config_dict['monitoring']['log_level']}")


def demo_convenience_function():
    """Demonstrate convenience function for quick config creation."""
    print_separator("Convenience Function")

    # Quick production config with overrides
    config = create_runtime_config('production')

    print("Quick config creation with create_runtime_config():")
    print(f"  Preset: production")
    print(f"  Mode: {config.execution.mode.value}")
    print(f"  Safety: {config.safety.level.value}")
    print(f"  Description: {config.description}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  V.V.A.L.T Runtime Configuration System Demo")
    print("=" * 80)

    try:
        demo_default_config()
        demo_preset_configs()
        demo_custom_config()
        demo_save_load_yaml()
        demo_save_load_json()
        demo_environment_override()
        demo_hierarchical_loading()
        demo_to_dict()
        demo_convenience_function()

        print_separator("All Demonstrations Completed Successfully!")

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
