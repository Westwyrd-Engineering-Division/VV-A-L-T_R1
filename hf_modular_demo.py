"""
HuggingFace Modular V.V.A.L.T Demo

Demonstrates the HuggingFace-compatible wrapper around modular PyTorch V.V.A.L.T.

This shows:
- VVALTModelHF for base model usage
- VVALTForSequenceClassificationModular for classification tasks
- VVALTForRegressionModular for regression tasks
- Three forward modes (fast/fine/diagnostic)
- Event hooks integration
- Training examples
- Trace visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvalt.configuration_vvalt import VVALTConfig
from vvalt.modeling_vvalt_modular import (
    VVALTModelHF,
    VVALTForSequenceClassification,
    VVALTForRegression,
    VVALTModularOutput,
)
from vvalt.torch_modules.vvalt_modular import EventHookType
from vvalt.torch_modules.visualization import VVALTVisualizer
from vvalt.utils import create_random_graph


def print_separator(title=""):
    """Print formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_base_model():
    """Demonstrate base VVALTModelHF usage."""
    print_separator("1. Base VVALTModelHF")

    # Create configuration
    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    # Initialize model
    model = VVALTModelHF(config)
    print("Model initialized:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Frame dim: {config.frame_dim}")
    print(f"  Task dim: {config.task_dim}")

    # Create inputs
    batch_size = 4
    inputs_embeds = torch.randn(batch_size, config.input_dim)
    task_vector = torch.randn(batch_size, config.task_dim)
    graph_adj = create_random_graph(batch_size, edge_probability=0.3)

    print(f"\nInput shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  task_vector: {task_vector.shape}")
    print(f"  graph_adj: {graph_adj.shape}")

    # Fast mode (no tracing)
    print("\n--- Fast Mode ---")
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            forward_mode="fast"
        )

    print(f"Output shape: {output.last_hidden_state.shape}")
    print(f"Output range: [{output.last_hidden_state.min():.4f}, {output.last_hidden_state.max():.4f}]")

    # Diagnostic mode with tracing
    print("\n--- Diagnostic Mode with Tracing ---")
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            forward_mode="diagnostic",
            return_trace=True,
            output_hidden_states=True,
            output_attentions=True,
        )

    print(f"Output shape: {output.last_hidden_state.shape}")
    print(f"Has trace: {output.trace is not None}")
    if output.trace:
        print(f"  Total time: {output.trace.total_time_ms:.2f}ms")
        print(f"  Frame traces: {len(output.trace.frame_traces)}")
        print(f"  Is safe: {output.trace.is_safe}")
        print(f"  Deterministic: {output.trace.deterministic_check_passed}")

    print(f"Hidden states: {len(output.hidden_states) if output.hidden_states else 0} frames")
    print(f"Attentions: {output.attentions is not None}")


def demo_classification():
    """Demonstrate VVALTForSequenceClassification."""
    print_separator("2. VVALTForSequenceClassification")

    # Create configuration
    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        num_labels=3,  # 3-class classification
        seed=42
    )

    # Initialize model
    model = VVALTForSequenceClassification(config)
    print(f"Classification model initialized with {config.num_labels} classes")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create inputs
    batch_size = 8
    inputs_embeds = torch.randn(batch_size, config.input_dim)
    task_vector = torch.randn(batch_size, config.task_dim)
    labels = torch.randint(0, config.num_labels, (batch_size,))

    print(f"\nInput shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  labels: {labels.shape}")

    # Forward pass without labels
    print("\n--- Inference (no labels) ---")
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            forward_mode="fast"
        )

    print(f"Logits shape: {output.logits.shape}")
    print(f"Predictions: {output.logits.argmax(dim=-1)}")

    # Forward pass with labels (compute loss)
    print("\n--- Training (with labels) ---")
    output = model(
        inputs_embeds=inputs_embeds,
        task_vector=task_vector,
        labels=labels,
        forward_mode="fast"
    )

    print(f"Loss: {output.loss.item():.4f}")
    print(f"Logits shape: {output.logits.shape}")

    # Show predicted vs. true labels
    predictions = output.logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    print(f"Accuracy: {accuracy:.2%}")


def demo_regression():
    """Demonstrate VVALTForRegression."""
    print_separator("3. VVALTForRegression")

    # Create configuration
    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        num_labels=1,  # Single regression target
        seed=42
    )

    # Initialize model
    model = VVALTForRegression(config)
    print(f"Regression model initialized")

    # Create inputs
    batch_size = 8
    inputs_embeds = torch.randn(batch_size, config.input_dim)
    task_vector = torch.randn(batch_size, config.task_dim)
    labels = torch.randn(batch_size, 1)  # Continuous targets

    print(f"\nInput shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  labels: {labels.shape}")

    # Forward pass without labels
    print("\n--- Inference (no labels) ---")
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            forward_mode="fast"
        )

    print(f"Predictions shape: {output.logits.shape}")
    print(f"Predictions: {output.logits.squeeze()}")

    # Forward pass with labels (compute loss)
    print("\n--- Training (with labels) ---")
    output = model(
        inputs_embeds=inputs_embeds,
        task_vector=task_vector,
        labels=labels,
        forward_mode="fast"
    )

    print(f"Loss (MSE): {output.loss.item():.4f}")
    print(f"Predictions shape: {output.logits.shape}")

    # Compute correlation
    predictions = output.logits
    correlation = torch.corrcoef(torch.cat([labels.T, predictions.T], dim=0))[0, 1]
    print(f"Correlation: {correlation:.4f}")


def demo_event_hooks():
    """Demonstrate event hooks integration."""
    print_separator("4. Event Hooks Integration")

    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        num_labels=3,
        seed=42
    )

    model = VVALTForSequenceClassification(config)

    # Hook counters
    hook_calls = {
        'pre_encoding': 0,
        'post_encoding': 0,
        'post_attention': 0,
        'post_verification': 0,
    }

    # Define hooks
    def pre_encoding_hook(data):
        hook_calls['pre_encoding'] += 1
        print(f"  [PRE_ENCODING] Input shape: {data['input'].shape}")

    def post_encoding_hook(data):
        hook_calls['post_encoding'] += 1
        print(f"  [POST_ENCODING] Generated {len(data['frames'])} frames")

    def post_attention_hook(data):
        hook_calls['post_attention'] += 1
        print(f"  [POST_ATTENTION] Output mean: {data['output'].mean().item():.4f}")

    def post_verification_hook(data):
        hook_calls['post_verification'] += 1
        print(f"  [POST_VERIFICATION] Safety verified")

    # Register hooks
    model.register_event_hook(EventHookType.PRE_ENCODING, pre_encoding_hook)
    model.register_event_hook(EventHookType.POST_ENCODING, post_encoding_hook)
    model.register_event_hook(EventHookType.POST_ATTENTION, post_attention_hook)
    model.register_event_hook(EventHookType.POST_VERIFICATION, post_verification_hook)

    print("Registered hooks:")
    for hook_type in hook_calls.keys():
        print(f"  - {hook_type}")

    # Run forward pass
    print("\nRunning forward with hooks:")
    inputs_embeds = torch.randn(2, config.input_dim)
    task_vector = torch.randn(2, config.task_dim)

    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            forward_mode="diagnostic"
        )

    print(f"\nHook execution summary:")
    for hook_name, count in hook_calls.items():
        print(f"  {hook_name}: called {count} time(s)")


def demo_training_loop():
    """Demonstrate training loop."""
    print_separator("5. Training Loop Example")

    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        num_labels=3,
        seed=42
    )

    model = VVALTForSequenceClassification(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training model for 5 steps...")

    # Dummy training data
    batch_size = 8
    losses = []

    for step in range(5):
        # Generate random training batch
        inputs_embeds = torch.randn(batch_size, config.input_dim)
        task_vector = torch.randn(batch_size, config.task_dim)
        labels = torch.randint(0, config.num_labels, (batch_size,))

        # Forward pass
        optimizer.zero_grad()
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            labels=labels,
            forward_mode="fast"
        )

        # Backward pass
        loss = output.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print(f"\n✓ Training complete")
    print(f"  Average loss: {sum(losses) / len(losses):.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")


def demo_trace_visualization():
    """Demonstrate trace visualization."""
    print_separator("6. Trace Visualization")

    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    model = VVALTModelHF(config)

    # Generate trace
    inputs_embeds = torch.randn(4, config.input_dim)
    task_vector = torch.randn(4, config.task_dim)
    graph_adj = create_random_graph(4, edge_probability=0.3)

    print("Generating diagnostic trace...")
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            forward_mode="diagnostic",
            return_trace=True,
            output_hidden_states=True,
            output_attentions=True,
        )

    trace = output.trace

    print(f"\nTrace details:")
    print(f"  Total time: {trace.total_time_ms:.2f}ms")
    print(f"  Frame traces: {len(trace.frame_traces)}")
    print(f"  Component times: {len(trace.component_times_ms)}")

    # Show component timing
    print(f"\nComponent timing breakdown:")
    for component, time_ms in sorted(trace.component_times_ms.items(),
                                     key=lambda x: x[1], reverse=True):
        percentage = (time_ms / trace.total_time_ms) * 100
        print(f"  {component}: {time_ms:.2f}ms ({percentage:.1f}%)")

    # Visualize
    visualizer = VVALTVisualizer()
    output_dir = Path("/tmp/vvalt_hf_modular_viz")
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Frame activations
    visualizer.plot_frame_activations(
        trace.frame_traces,
        save_path=output_dir / "frame_activations.png"
    )
    print(f"  ✓ Frame activations: {output_dir}/frame_activations.png")

    # Component timing
    visualizer.plot_component_timing(
        trace,
        save_path=output_dir / "timing.png"
    )
    print(f"  ✓ Component timing: {output_dir}/timing.png")

    # Export trace to JSON
    visualizer.export_trace_json(
        trace,
        path=output_dir / "trace.json"
    )
    print(f"  ✓ Trace JSON: {output_dir}/trace.json")

    print(f"\n✓ All visualizations saved to: {output_dir}")


def demo_component_access():
    """Demonstrate component access."""
    print_separator("7. Component Access")

    config = VVALTConfig(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        num_labels=3,
        seed=42
    )

    model = VVALTForSequenceClassification(config)

    print("Accessing internal components...")

    # Get all components
    components = model.vvalt.get_all_components()
    print(f"\nAvailable components: {len(components)}")
    for name, component in components.items():
        num_params = sum(p.numel() for p in component.parameters())
        print(f"  - {name}: {component.__class__.__name__} ({num_params} params)")

    # Access specific components
    print("\nAccessing specific components:")

    frame_encoder = model.vvalt.get_component('frame_encoder')
    print(f"  Frame encoder: {frame_encoder.__class__.__name__}")
    print(f"    Submodules: {list(frame_encoder.encoders.keys())}")

    vantage_selector = model.vvalt.get_component('vantage_selector')
    print(f"  Vantage selector: {vantage_selector.__class__.__name__}")

    attention = model.vvalt.get_component('attention')
    print(f"  Attention: {attention.__class__.__name__}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  HuggingFace Modular V.V.A.L.T - Comprehensive Demo")
    print("=" * 80)

    try:
        demo_base_model()
        demo_classification()
        demo_regression()
        demo_event_hooks()
        demo_training_loop()
        demo_trace_visualization()
        demo_component_access()

        print_separator("All Demonstrations Completed Successfully!")
        print("\n✓ HuggingFace modular V.V.A.L.T is fully operational")
        print("✓ Base model, classification, and regression heads")
        print("✓ Three forward modes (fast/fine/diagnostic)")
        print("✓ Event hooks for governance")
        print("✓ Training-ready with gradient support")
        print("✓ Comprehensive tracing and visualization")
        print("✓ Component inspection and access")

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
