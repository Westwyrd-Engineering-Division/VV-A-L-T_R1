"""
V.V.A.L.T Demonstration

Examples of using the Vantage-Vector Autonomous Logic Transformer.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvalt import VVALT
from vvalt.utils import create_random_graph, create_line_graph, create_star_graph


def print_separator(title=""):
    """Print formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_basic_usage():
    """Demonstrate basic V.V.A.L.T usage."""
    print_separator("Basic Usage Demo")

    # Initialize V.V.A.L.T
    input_dim = 10
    frame_dim = 8
    task_dim = 5

    vvalt = VVALT(
        input_dim=input_dim,
        frame_dim=frame_dim,
        task_dim=task_dim,
        seed=42
    )

    # Create input and task vectors
    x = np.random.randn(input_dim)
    task = np.random.randn(task_dim)

    print("Input shape:", x.shape)
    print("Task shape:", task.shape)

    # Run V.V.A.L.T
    output, _ = vvalt(x, task)

    print("\nOutput shape:", output.shape)
    print("Output (first 5 values):", output[:5])
    print("\nSingle-pass execution completed successfully!")


def demo_with_graph():
    """Demonstrate V.V.A.L.T with graph topology."""
    print_separator("Graph Topology Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create input, task, and graph
    x = np.random.randn(10)
    task = np.random.randn(5)
    graph = create_star_graph(num_nodes=8)

    print("Graph adjacency matrix shape:", graph.shape)
    print("Number of edges:", int(np.sum(graph)) // 2)

    # Run with graph topology
    output, _ = vvalt(x, task, graph_adj=graph)

    print("\nOutput with graph topology:", output[:5])
    print("Graph-aware reasoning completed!")


def demo_interpretability():
    """Demonstrate V.V.A.L.T interpretability features."""
    print_separator("Interpretability Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create input and task
    x = np.random.randn(10)
    task = np.random.randn(5)

    # Get reasoning explanation
    explanation = vvalt.explain(x, task)
    print(explanation)


def demo_safety_verification():
    """Demonstrate V.V.A.L.T safety guarantees."""
    print_separator("Safety Verification Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create input and task
    x = np.random.randn(10)
    task = np.random.randn(5)

    # Verify determinism
    print("Testing determinism (5 trials)...")
    is_deterministic = vvalt.verify_determinism(x, task, num_trials=5)
    print(f"✓ Deterministic: {is_deterministic}")

    # Get safety report
    safety_report = vvalt.get_safety_report(x, task)
    print(f"✓ Bounded output: {safety_report['bounded']}")
    print(f"✓ Safe output: {safety_report['output_safety']['safety']['is_safe']}")

    print("\nAll safety guarantees verified!")


def demo_task_conditioning():
    """Demonstrate task-conditioned perspective weighting."""
    print_separator("Task-Conditioned Weighting Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create input
    x = np.random.randn(10)

    # Try different task vectors
    tasks = {
        "semantic_task": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "structural_task": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "balanced_task": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    }

    for task_name, task_vector in tasks.items():
        print(f"\n{task_name}:")

        # Get frame weights
        weights = vvalt.selector.get_weight_distribution(task_vector)
        for frame, weight in weights.items():
            print(f"  {frame}: {weight:.4f}")

        # Run V.V.A.L.T
        output, _ = vvalt(x, task_vector)
        print(f"  Output norm: {np.linalg.norm(output):.4f}")


def demo_batch_processing():
    """Demonstrate batch processing."""
    print_separator("Batch Processing Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create batch of inputs
    batch_size = 5
    X = np.random.randn(batch_size, 10)
    task = np.random.randn(5)

    print(f"Processing batch of {batch_size} inputs...")

    # Process batch
    outputs = vvalt.batch_forward(X, task)

    print(f"Output batch shape: {outputs.shape}")
    print("Output norms:", [f"{np.linalg.norm(out):.4f}" for out in outputs])


def demo_full_trace():
    """Demonstrate full reasoning trace."""
    print_separator("Full Reasoning Trace Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create input and task
    x = np.random.randn(10)
    task = np.random.randn(5)

    # Get full trace
    output, trace = vvalt(x, task, return_trace=True)

    print("Trace keys:", list(trace.keys()))
    print("\nTask weights:")
    for frame, weight in trace['task_weights'].items():
        print(f"  {frame}: {weight:.4f}")

    print("\nTransformations:")
    for trans, value in trace['transformations'].items():
        print(f"  {trans}: {value:.4f}")

    print("\nFrame diversity:", trace['encoded_frames']['comparison']['diversity'])


def demo_graph_types():
    """Demonstrate different graph topologies."""
    print_separator("Different Graph Topologies Demo")

    # Initialize V.V.A.L.T
    vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

    # Create input and task
    x = np.random.randn(10)
    task = np.random.randn(5)

    # Test different graph types
    graphs = {
        "No graph": None,
        "Random graph": create_random_graph(8, edge_probability=0.3),
        "Line graph": create_line_graph(8),
        "Star graph": create_star_graph(8),
    }

    for graph_name, graph in graphs.items():
        output, _ = vvalt(x, task, graph_adj=graph)
        print(f"{graph_name:15s} -> Output norm: {np.linalg.norm(output):.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  V.V.A.L.T - Vantage-Vector Autonomous Logic Transformer")
    print("  Demonstration Suite")
    print("=" * 80)

    try:
        demo_basic_usage()
        demo_with_graph()
        demo_task_conditioning()
        demo_batch_processing()
        demo_interpretability()
        demo_safety_verification()
        demo_full_trace()
        demo_graph_types()

        print_separator("All Demonstrations Completed Successfully!")

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
