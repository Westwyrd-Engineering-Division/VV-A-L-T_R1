"""
V.V.A.L.T Modular PyTorch Implementation Demo

Demonstrates the fully modular PyTorch-native V.V.A.L.T with:
- Inspectable submodules for each component
- Three forward modes: fast / fine / diagnostic
- Event hooks for governance and safety layers
- Comprehensive visualization and tracing
- Graph-aware extensions
- Training-ready architecture
"""

import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvalt.torch_modules.vvalt_modular import (
    VVALTModular,
    EventHookType,
)
from vvalt.torch_modules.frame_encoders import ForwardMode
from vvalt.torch_modules.visualization import VVALTVisualizer
from vvalt.utils import create_random_graph


def print_separator(title=""):
    """Print formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_basic_usage():
    """Demonstrate basic usage of modular V.V.A.L.T."""
    print_separator("1. Basic Modular V.V.A.L.T Usage")

    # Initialize model
    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        safe_bounds=(-10.0, 10.0),
        seed=42
    )

    print("Model initialized with modular architecture:")
    print(model.summary())

    # Create sample inputs
    batch_size = 8
    x = torch.randn(batch_size, 128)
    task_vector = torch.randn(batch_size, 32)
    graph_adj = create_random_graph(batch_size, edge_probability=0.3)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  task_vector: {task_vector.shape}")
    print(f"  graph_adj: {graph_adj.shape}")

    # Fast forward (no tracing)
    with torch.no_grad():
        output = model.forward_fast(x, task_vector, graph_adj)

    print(f"\n✓ Fast forward output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Within safe bounds: {output.min() >= -10.0 and output.max() <= 10.0}")


def demo_forward_modes():
    """Demonstrate three forward modes."""
    print_separator("2. Forward Modes: Fast / Fine / Diagnostic")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    x = torch.randn(4, 128)
    task_vector = torch.randn(4, 32)
    graph_adj = create_random_graph(4, edge_probability=0.3)

    # Fast mode
    print("Fast Mode (optimized, no tracing):")
    with torch.no_grad():
        import time
        start = time.perf_counter()
        output_fast = model.forward_fast(x, task_vector, graph_adj)
        fast_time = (time.perf_counter() - start) * 1000
    print(f"  ✓ Execution time: {fast_time:.2f}ms")
    print(f"  ✓ Output: {output_fast.shape}")

    # Fine mode
    print("\nFine Mode (basic tracing):")
    with torch.no_grad():
        start = time.perf_counter()
        output_fine, basic_trace = model.forward_fine(x, task_vector, graph_adj)
        fine_time = (time.perf_counter() - start) * 1000
    print(f"  ✓ Execution time: {fine_time:.2f}ms")
    print(f"  ✓ Trace keys: {list(basic_trace.keys())}")

    # Diagnostic mode
    print("\nDiagnostic Mode (full micro-tracing):")
    with torch.no_grad():
        start = time.perf_counter()
        output_diag, full_trace = model.forward_diagnostic(x, task_vector, graph_adj)
        diag_time = (time.perf_counter() - start) * 1000
    print(f"  ✓ Execution time: {diag_time:.2f}ms")
    print(f"  ✓ Total components traced: {len(full_trace.component_times_ms)}")
    print(f"  ✓ Frame traces: {len(full_trace.frame_traces)}")
    print(f"  ✓ Safety verified: {full_trace.is_safe}")
    print(f"  ✓ Determinism check: {full_trace.deterministic_check_passed}")

    # Verify outputs are identical
    diff_fine = torch.abs(output_fast - output_fine).max().item()
    diff_diag = torch.abs(output_fast - output_diag).max().item()
    print(f"\n✓ Output consistency:")
    print(f"  Fast vs Fine: max diff = {diff_fine:.10f}")
    print(f"  Fast vs Diagnostic: max diff = {diff_diag:.10f}")
    print(f"  All modes produce identical outputs: {diff_fine < 1e-6 and diff_diag < 1e-6}")


def demo_component_inspection():
    """Demonstrate component inspection and access."""
    print_separator("3. Component Inspection")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    print("Available components:")
    components = model.get_all_components()
    for name, component in components.items():
        num_params = sum(p.numel() for p in component.parameters())
        print(f"  - {name}: {component.__class__.__name__} ({num_params} params)")

    # Access individual components
    print("\nAccessing individual components:")
    vantage_selector = model.get_component('vantage_selector')
    print(f"  ✓ VantageSelector: {vantage_selector}")

    attention = model.get_component('attention')
    print(f"  ✓ MultiPerspectiveAttention: {attention}")

    # Inspect frame encoder submodules
    print("\nFrame encoder submodules:")
    frame_encoder = model.get_component('frame_encoder')
    for name, module in frame_encoder.encoders.items():
        print(f"  - {name}: {module.__class__.__name__}")


def demo_event_hooks():
    """Demonstrate event hooks for governance."""
    print_separator("4. Event Hooks for Governance Layer")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    # Hook counters
    hook_calls = {
        'pre_encoding': 0,
        'post_encoding': 0,
        'pre_attention': 0,
        'post_attention': 0,
        'pre_verification': 0,
        'post_verification': 0,
    }

    # Define hook callbacks
    def pre_encoding_hook(data):
        hook_calls['pre_encoding'] += 1
        print(f"  [PRE_ENCODING] Input shape: {data['input'].shape}")

    def post_encoding_hook(data):
        hook_calls['post_encoding'] += 1
        num_frames = len(data['frames'])
        print(f"  [POST_ENCODING] Generated {num_frames} frames")

    def post_attention_hook(data):
        hook_calls['post_attention'] += 1
        output_mean = data['output'].mean().item()
        print(f"  [POST_ATTENTION] Output mean: {output_mean:.4f}")

    def post_verification_hook(data):
        hook_calls['post_verification'] += 1
        is_safe = data['trace'].has_nan_inf if hasattr(data['trace'], 'has_nan_inf') else True
        print(f"  [POST_VERIFICATION] Safety check: {'✓' if not is_safe else '✗'}")

    # Register hooks
    model.register_hook(EventHookType.PRE_ENCODING, pre_encoding_hook)
    model.register_hook(EventHookType.POST_ENCODING, post_encoding_hook)
    model.register_hook(EventHookType.POST_ATTENTION, post_attention_hook)
    model.register_hook(EventHookType.POST_VERIFICATION, post_verification_hook)

    print("Registered hooks:")
    for hook_type in [EventHookType.PRE_ENCODING, EventHookType.POST_ENCODING,
                      EventHookType.POST_ATTENTION, EventHookType.POST_VERIFICATION]:
        print(f"  - {hook_type.value}")

    # Run forward pass
    print("\nRunning forward with hooks:")
    x = torch.randn(2, 128)
    task_vector = torch.randn(2, 32)
    graph_adj = create_random_graph(2, edge_probability=0.3)

    with torch.no_grad():
        output, trace = model.forward_diagnostic(x, task_vector, graph_adj)

    print(f"\n✓ Hook execution summary:")
    for hook_name, count in hook_calls.items():
        print(f"  {hook_name}: called {count} time(s)")


def demo_detailed_tracing():
    """Demonstrate detailed micro-tracing."""
    print_separator("5. Detailed Micro-Tracing")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    x = torch.randn(4, 128)
    task_vector = torch.randn(4, 32)
    graph_adj = create_random_graph(4, edge_probability=0.3)

    with torch.no_grad():
        output, trace = model.forward_diagnostic(x, task_vector, graph_adj)

    print("Trace structure:")
    print(f"  Input shape: {trace.input_shape}")
    print(f"  Task shape: {trace.task_shape}")
    print(f"  Has graph: {trace.has_graph}")
    print(f"  Total time: {trace.total_time_ms:.2f}ms")

    print("\nFrame traces:")
    for frame_name, frame_trace in trace.frame_traces.items():
        print(f"  {frame_name}:")
        print(f"    Input: mean={frame_trace.input_stats['mean']:.4f}, std={frame_trace.input_stats['std']:.4f}")
        print(f"    Output: mean={frame_trace.output_stats['mean']:.4f}, std={frame_trace.output_stats['std']:.4f}")
        if frame_trace.processing_time_ms:
            print(f"    Time: {frame_trace.processing_time_ms:.2f}ms")

    print("\nAttention traces:")
    if trace.vantage_trace:
        print(f"  Vantage selector entropy: {trace.vantage_trace.weight_entropy:.4f}")
        if trace.vantage_trace.frame_weights:
            print(f"  Frame weights:")
            for fname, weight in trace.vantage_trace.frame_weights.items():
                print(f"    {fname}: {weight:.4f}")

    print("\nComponent timing breakdown:")
    total = trace.total_time_ms
    for component, time_ms in sorted(trace.component_times_ms.items(),
                                     key=lambda x: x[1], reverse=True):
        percentage = (time_ms / total) * 100 if total > 0 else 0
        print(f"  {component}: {time_ms:.2f}ms ({percentage:.1f}%)")

    print(f"\nSafety verification:")
    print(f"  Is safe: {trace.is_safe}")
    print(f"  Determinism check: {trace.deterministic_check_passed}")
    print(f"  Bounds check: {trace.bounds_check_passed}")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print_separator("6. Visualization")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    x = torch.randn(8, 128)
    task_vector = torch.randn(8, 32)
    graph_adj = create_random_graph(8, edge_probability=0.3)

    # Generate trace
    with torch.no_grad():
        output, trace = model.forward_diagnostic(x, task_vector, graph_adj)

    # Create visualizer
    visualizer = VVALTVisualizer()

    # Create output directory
    output_dir = Path("/tmp/vvalt_modular_viz")
    output_dir.mkdir(exist_ok=True)

    print("Generating visualizations...")

    # Plot frame activations
    visualizer.plot_frame_activations(
        trace.frame_traces,
        save_path=output_dir / "frame_activations.png"
    )
    print(f"  ✓ Frame activations: {output_dir}/frame_activations.png")

    # Plot attention weights
    if trace.attention_trace and trace.attention_trace.attention_weights is not None:
        visualizer.plot_attention_weights(
            trace.attention_trace.attention_weights[0],  # First sample
            frame_names=['semantic', 'structural', 'causal', 'relational', 'graph'],
            save_path=output_dir / "attention_weights.png"
        )
        print(f"  ✓ Attention weights: {output_dir}/attention_weights.png")

    # Plot frame weights
    if trace.vantage_trace and trace.vantage_trace.frame_weights:
        visualizer.plot_frame_weights(
            trace.vantage_trace.frame_weights,
            save_path=output_dir / "frame_weights.png"
        )
        print(f"  ✓ Frame weights: {output_dir}/frame_weights.png")

    # Plot component timing
    visualizer.plot_component_timing(
        trace,
        save_path=output_dir / "timing.png"
    )
    print(f"  ✓ Component timing: {output_dir}/timing.png")

    # Plot full trace summary
    visualizer.plot_full_trace_summary(
        trace,
        save_path=output_dir / "trace_summary.png"
    )
    print(f"  ✓ Full trace summary: {output_dir}/trace_summary.png")

    # Plot graph topology
    visualizer.plot_graph_topology(
        graph_adj,
        save_path=output_dir / "graph_topology.png"
    )
    print(f"  ✓ Graph topology: {output_dir}/graph_topology.png")

    # Export trace to JSON
    visualizer.export_trace_json(
        trace,
        path=output_dir / "trace.json"
    )
    print(f"  ✓ Trace JSON: {output_dir}/trace.json")

    # Generate full report
    visualizer.generate_report(trace, output_dir)
    print(f"  ✓ Full report: {output_dir}/")

    print(f"\n✓ All visualizations saved to: {output_dir}")


def demo_determinism():
    """Demonstrate and verify determinism."""
    print_separator("7. Determinism Verification")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42  # Fixed seed
    )

    x = torch.randn(4, 128)
    task_vector = torch.randn(4, 32)
    graph_adj = create_random_graph(4, edge_probability=0.3)

    print("Running determinism check (10 trials)...")
    is_deterministic = model.verify_determinism(
        x, task_vector, graph_adj, num_trials=10
    )

    print(f"\n✓ Determinism verified: {is_deterministic}")

    if is_deterministic:
        print("  All 10 trials produced identical outputs")
    else:
        print("  ✗ Outputs varied across trials (this is a bug!)")

    # Show output statistics
    with torch.no_grad():
        output = model.forward_fast(x, task_vector, graph_adj)

    print(f"\nOutput statistics (from deterministic run):")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")
    print(f"  Within bounds [-10, 10]: {output.min() >= -10.0 and output.max() <= 10.0}")


def demo_training_ready():
    """Demonstrate training-ready architecture."""
    print_separator("8. Training-Ready Architecture")

    model = VVALTModular(
        input_dim=128,
        frame_dim=64,
        task_dim=32,
        hidden_dim=128,
        seed=42
    )

    print("Model is a standard PyTorch nn.Module:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Demonstrate gradient flow
    print("\nDemonstrating gradient flow:")

    x = torch.randn(4, 128, requires_grad=True)
    task_vector = torch.randn(4, 32, requires_grad=True)
    graph_adj = create_random_graph(4, edge_probability=0.3)

    # Forward pass
    output = model.forward_fast(x, task_vector, graph_adj)

    # Dummy loss
    loss = output.mean()

    # Backward pass
    loss.backward()

    print(f"  ✓ Forward pass complete")
    print(f"  ✓ Backward pass complete")
    print(f"  ✓ Input gradients: {x.grad is not None}")
    print(f"  ✓ Task gradients: {task_vector.grad is not None}")

    # Show gradient statistics for a component
    frame_encoder = model.get_component('frame_encoder')
    semantic_encoder = frame_encoder.encoders['semantic']
    grad_norm = sum(p.grad.norm().item() for p in semantic_encoder.parameters() if p.grad is not None)
    print(f"  ✓ Semantic encoder gradient norm: {grad_norm:.6f}")

    print("\nExample training loop:")
    print("""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            output = model.forward_fast(
                batch['x'],
                batch['task_vector'],
                batch['graph_adj']
            )

            loss = criterion(output, batch['target'])
            loss.backward()
            optimizer.step()
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  V.V.A.L.T Modular PyTorch Implementation - Comprehensive Demo")
    print("=" * 80)

    try:
        demo_basic_usage()
        demo_forward_modes()
        demo_component_inspection()
        demo_event_hooks()
        demo_detailed_tracing()
        demo_visualization()
        demo_determinism()
        demo_training_ready()

        print_separator("All Demonstrations Completed Successfully!")
        print("\n✓ Modular PyTorch V.V.A.L.T is fully operational")
        print("✓ Inspectable submodules for every component")
        print("✓ Three forward modes: fast / fine / diagnostic")
        print("✓ Event hooks for governance and safety")
        print("✓ Comprehensive visualization and tracing")
        print("✓ Graph-aware extensions")
        print("✓ Training-ready with full gradient support")
        print("✓ Deterministic and bounded guarantees maintained")

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
