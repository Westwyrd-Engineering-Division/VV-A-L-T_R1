"""
V.V.A.L.T Production Features Demonstration

Demonstrates enhanced production-grade features:
- Configuration management
- Task envelope DSL
- Monitoring and metrics
- Checkpoint management
- Error handling
- Validation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvalt import (
    VVALTEnhanced,
    VVALTConfig,
    ModelConfig,
    RuntimeConfig,
    SafetyLevel,
    TaskVectorBuilder,
    TaskType,
    create_task_envelope,
    create_checkpoint_metadata,
)


def print_separator(title=""):
    """Print formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_configuration():
    """Demonstrate configuration management."""
    print_separator("Configuration Management Demo")

    # Create custom configuration
    config = VVALTConfig()
    config.model.input_dim = 10
    config.model.frame_dim = 8
    config.model.task_dim = 5
    config.runtime.safety_level = SafetyLevel.STANDARD
    config.runtime.batch_size_limit = 50

    print("Configuration created:")
    print(f"  Model: input_dim={config.model.input_dim}, frame_dim={config.model.frame_dim}")
    print(f"  Runtime: safety_level={config.runtime.safety_level.value}")
    print(f"  Logging: level={config.logging.level.value}")

    # Validate configuration
    try:
        config.validate()
        print("\n✓ Configuration validated successfully")
    except Exception as e:
        print(f"\n✗ Configuration validation failed: {e}")

    return config


def demo_enhanced_core(config):
    """Demonstrate enhanced V.V.A.L.T with monitoring."""
    print_separator("Enhanced V.V.A.L.T Demo")

    # Initialize with monitoring enabled
    vvalt = VVALTEnhanced(config=config, enable_monitoring=True)

    print("Enhanced V.V.A.L.T initialized with:")
    print(f"  Monitoring: ENABLED")
    print(f"  Validation: ENABLED")
    print(f"  Safety level: {config.runtime.safety_level.value}")

    # Run inference
    x = np.random.randn(10)
    task = np.random.randn(5)

    print("\nRunning inference...")
    output, _ = vvalt(x, task)

    print(f"✓ Inference completed")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    return vvalt


def demo_task_envelope():
    """Demonstrate task envelope DSL."""
    print_separator("Task Envelope DSL Demo")

    builder = TaskVectorBuilder(task_dim=5)

    # Create different task types
    print("Creating task vectors for different types:")

    # Semantic-dominant task
    semantic_task = builder.from_task_type(TaskType.SEMANTIC)
    print(f"\n  Semantic-dominant: {semantic_task[:5]}")

    # Structural-dominant task
    structural_task = builder.from_task_type(TaskType.STRUCTURAL)
    print(f"  Structural-dominant: {structural_task[:5]}")

    # Balanced task
    balanced_task = builder.balanced()
    print(f"  Balanced: {balanced_task[:5]}")

    # Create task envelope
    envelope = create_task_envelope(
        task_vector=semantic_task,
        task_type=TaskType.SEMANTIC,
        require_trace=True,
        timeout_ms=500
    )

    print(f"\nTask envelope created:")
    print(f"  Task ID: {envelope.metadata.task_id}")
    print(f"  Task type: {envelope.metadata.task_type.value}")
    print(f"  Require trace: {envelope.metadata.require_trace}")

    return semantic_task


def demo_monitoring(vvalt):
    """Demonstrate monitoring and metrics."""
    print_separator("Monitoring & Metrics Demo")

    # Run multiple inferences
    print("Running 10 inferences to collect metrics...")
    for i in range(10):
        x = np.random.randn(10)
        task = np.random.randn(5)
        vvalt(x, task)

    # Get metrics summary
    summary = vvalt.get_metrics_summary()

    print("\nMetrics Summary:")
    print(f"  Total inferences: {summary['total_inferences']}")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Error rate: {summary['error_rate']:.2%}")

    if 'duration_ms' in summary:
        print(f"\nLatency Statistics:")
        print(f"  Mean: {summary['duration_ms']['mean']:.2f} ms")
        print(f"  P95: {summary['duration_ms']['p95']:.2f} ms")
        print(f"  P99: {summary['duration_ms']['p99']:.2f} ms")

    # Get Prometheus metrics
    print("\nPrometheus Metrics (sample):")
    prom_metrics = vvalt.get_prometheus_metrics()
    print("\n".join(prom_metrics.split("\n")[:6]))


def demo_checkpoint(vvalt):
    """Demonstrate checkpoint save/load."""
    print_separator("Checkpoint Management Demo")

    checkpoint_path = "/tmp/vvalt_checkpoint.npz"

    # Create metadata
    metadata = create_checkpoint_metadata(
        description="Production model checkpoint",
        tags=["production", "v0.1"],
        metrics={"accuracy": 0.95, "latency_ms": 15.3}
    )

    # Save checkpoint
    print(f"Saving checkpoint to {checkpoint_path}...")
    checksum = vvalt.save_checkpoint(checkpoint_path, metadata)
    print(f"✓ Checkpoint saved")
    print(f"  Checksum: {checksum[:16]}...")

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    loaded_metadata = vvalt.load_checkpoint(checkpoint_path, strict=True)
    print(f"✓ Checkpoint loaded")
    print(f"  Version: {loaded_metadata.get('version')}")
    print(f"  Description: {loaded_metadata.get('description')}")
    print(f"  Metrics: {loaded_metadata.get('metrics')}")


def demo_error_handling():
    """Demonstrate error handling."""
    print_separator("Error Handling Demo")

    config = VVALTConfig()
    config.model.input_dim = 10
    config.model.frame_dim = 8
    config.model.task_dim = 5

    vvalt = VVALTEnhanced(config=config, enable_monitoring=True)

    print("Testing error handling...")

    # Test 1: Invalid input shape
    print("\n1. Invalid input shape:")
    try:
        x = np.random.randn(5)  # Wrong dimension
        task = np.random.randn(5)
        vvalt(x, task)
    except Exception as e:
        print(f"  ✓ Caught: {type(e).__name__}")
        print(f"    Message: {str(e)}")
        if hasattr(e, 'recovery_hint'):
            print(f"    Recovery: {e.recovery_hint}")

    # Test 2: Invalid task vector
    print("\n2. Invalid task vector:")
    try:
        x = np.random.randn(10)
        task = np.random.randn(3)  # Wrong dimension
        vvalt(x, task)
    except Exception as e:
        print(f"  ✓ Caught: {type(e).__name__}")
        print(f"    Message: {str(e)}")

    # Test 3: Invalid graph
    print("\n3. Invalid graph (asymmetric):")
    try:
        x = np.random.randn(10)
        task = np.random.randn(5)
        graph = np.random.rand(8, 8)  # Not symmetric
        vvalt(x, task, graph_adj=graph)
    except Exception as e:
        print(f"  ✓ Caught: {type(e).__name__}")
        print(f"    Message: {str(e)}")

    # Check error metrics
    summary = vvalt.get_metrics_summary()
    print(f"\nError Metrics:")
    print(f"  Total errors recorded: {summary['total_errors']}")
    print(f"  Error counts: {summary.get('error_counts', {})}")


def demo_safety_verification(vvalt):
    """Demonstrate safety verification."""
    print_separator("Safety Verification Demo")

    x = np.random.randn(10)
    task = np.random.randn(5)

    # Test determinism
    print("Testing determinism (10 trials)...")
    is_deterministic = vvalt.verify_determinism(x, task, num_trials=10)
    print(f"  ✓ Deterministic: {is_deterministic}")

    # Get safety report
    print("\nGenerating safety report...")
    report = vvalt.get_safety_report(x, task)
    print(f"  Deterministic: {report['deterministic']}")
    print(f"  Output bounded: {report['bounded']}")
    print(f"  Output safe: {report['output_safety']['safety']['is_safe']}")


def demo_batch_processing(vvalt):
    """Demonstrate batch processing."""
    print_separator("Batch Processing Demo")

    batch_size = 25
    X = np.random.randn(batch_size, 10)
    task = np.random.randn(5)

    print(f"Processing batch of {batch_size} inputs...")
    outputs = vvalt.batch_forward(X, task)

    print(f"✓ Batch processed")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Output norms: min={np.linalg.norm(outputs, axis=1).min():.4f}, "
          f"max={np.linalg.norm(outputs, axis=1).max():.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  V.V.A.L.T Production Features Demonstration")
    print("=" * 80)

    try:
        # Configuration
        config = demo_configuration()

        # Enhanced core
        vvalt = demo_enhanced_core(config)

        # Task envelope
        demo_task_envelope()

        # Monitoring
        demo_monitoring(vvalt)

        # Checkpoint
        demo_checkpoint(vvalt)

        # Error handling
        demo_error_handling()

        # Safety verification
        demo_safety_verification(vvalt)

        # Batch processing
        demo_batch_processing(vvalt)

        print_separator("All Demonstrations Completed Successfully!")

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
