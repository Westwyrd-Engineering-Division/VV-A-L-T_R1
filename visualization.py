"""
V.V.A.L.T Visualization Module

Visualization tools for frame encodings, attention maps, and topology.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import json

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


from .vvalt_modular import VVALTDetailedTrace, VVALTModular
from .frame_encoders import FrameTrace


class VVALTVisualizer:
    """
    Visualization toolkit for V.V.A.L.T internals.

    Generates plots and visualizations for:
    - Frame activation patterns
    - Attention weight matrices
    - Component timing analysis
    - Safety metrics
    """

    def __init__(self, model: Optional[VVALTModular] = None):
        self.model = model

        if not PLOTTING_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

    def plot_frame_activations(
        self,
        frame_traces: Dict[str, FrameTrace],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8)
    ):
        """
        Plot activation patterns for all frames.

        Args:
            frame_traces: Dictionary of frame traces
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        frame_names = ["semantic", "structural", "causal", "relational", "graph"]

        for i, name in enumerate(frame_names):
            ax = axes[i]
            if name in frame_traces:
                trace = frame_traces[name]

                # Get activation pattern
                if isinstance(trace.activation_pattern, torch.Tensor):
                    activations = trace.activation_pattern.numpy()
                else:
                    activations = np.array(trace.activation_pattern)

                # Plot
                ax.bar(range(len(activations)), activations)
                ax.set_title(f"{name.capitalize()} Frame\n"
                           f"Mean: {trace.output_stats['mean']:.3f}, "
                           f"Std: {trace.output_stats['std']:.3f}")
                ax.set_xlabel("Dimension")
                ax.set_ylabel("Activation")
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3)

        # Remove empty subplot
        fig.delaxes(axes[5])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        frame_names: List[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot attention weight heatmap.

        Args:
            attention_weights: Attention weight matrix (num_frames, num_frames)
            frame_names: Names for frame labels
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        if frame_names is None:
            frame_names = ["Semantic", "Structural", "Causal", "Relational", "Graph"]

        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights = attention_weights.detach().cpu().numpy()
        else:
            weights = attention_weights

        # Handle batched weights (average across batch)
        if weights.ndim == 3:
            weights = weights.mean(axis=0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            weights,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=frame_names,
            yticklabels=frame_names,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )

        ax.set_title("Multi-Perspective Attention Weights")
        ax.set_xlabel("Key Frame")
        ax.set_ylabel("Query Frame")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def plot_frame_weights(
        self,
        frame_weights: Dict[str, float],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot task-conditioned frame weights (from VantageSelector).

        Args:
            frame_weights: Dictionary of frame weights
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        fig, ax = plt.subplots(figsize=figsize)

        frames = list(frame_weights.keys())
        weights = list(frame_weights.values())

        # Create bar plot
        bars = ax.bar(frames, weights, color=plt.cm.viridis(np.linspace(0, 1, len(frames))))

        ax.set_ylabel("Weight")
        ax.set_title("Task-Conditioned Frame Weights (VantageSelector)")
        ax.set_ylim(0, max(weights) * 1.2)
        ax.axhline(y=1.0/len(frames), color='r', linestyle='--', label='Uniform')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def plot_component_timing(
        self,
        trace: VVALTDetailedTrace,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot component execution timing breakdown.

        Args:
            trace: Detailed trace from diagnostic forward pass
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar chart of component times
        components = list(trace.component_times_ms.keys())
        times = list(trace.component_times_ms.values())

        ax1.barh(components, times, color=plt.cm.plasma(np.linspace(0, 1, len(components))))
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Component Execution Time")
        ax1.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        total_time = sum(times)
        for i, (comp, time) in enumerate(zip(components, times)):
            pct = (time / total_time) * 100
            ax1.text(time, i, f'  {pct:.1f}%', va='center')

        # Pie chart
        ax2.pie(times, labels=components, autopct='%1.1f%%', startangle=90,
               colors=plt.cm.plasma(np.linspace(0, 1, len(components))))
        ax2.set_title(f"Time Distribution\n(Total: {trace.total_time_ms:.2f} ms)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def plot_full_trace_summary(
        self,
        trace: VVALTDetailedTrace,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Generate comprehensive visualization of entire trace.

        Args:
            trace: Detailed trace from diagnostic forward pass
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Frame activations (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        frame_names = list(trace.frame_traces.keys())
        mean_activations = [trace.frame_traces[name].output_stats['mean'] for name in frame_names]
        ax1.bar(frame_names, mean_activations)
        ax1.set_title("Frame Mean Activations")
        ax1.set_ylabel("Mean Value")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Frame weights (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        if trace.vantage_trace and trace.vantage_trace.frame_weights:
            weights = trace.vantage_trace.frame_weights
            ax2.bar(weights.keys(), weights.values(),
                   color=plt.cm.viridis(np.linspace(0, 1, len(weights))))
            ax2.set_title("Task-Conditioned Frame Weights")
            ax2.set_ylabel("Weight")
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=1.0/len(weights), color='r', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)

        # 3. Component timing (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        components = list(trace.component_times_ms.keys())
        times = list(trace.component_times_ms.values())
        ax3.barh(components, times)
        ax3.set_xlabel("Time (ms)")
        ax3.set_title("Component Timing")
        ax3.grid(True, alpha=0.3)

        # 4. Safety metrics (middle-right)
        ax4 = fig.add_subplot(gs[1, 1])
        safety_metrics = [
            trace.is_safe,
            trace.bounds_check_passed,
            trace.deterministic_check_passed
        ]
        safety_labels = ["No NaN/Inf", "Bounds OK", "Deterministic"]
        colors = ['green' if m else 'red' for m in safety_metrics]
        ax4.barh(safety_labels, [1]*3, color=colors, alpha=0.7)
        ax4.set_xlim(0, 1.2)
        ax4.set_title("Safety Checks")
        ax4.set_xlabel("Status")
        for i, (label, status) in enumerate(zip(safety_labels, safety_metrics)):
            ax4.text(0.5, i, '✓' if status else '✗', ha='center', va='center',
                    fontsize=20, color='white', weight='bold')

        # 5. Attention entropy (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        if trace.attention_trace:
            entropy = trace.attention_trace.weight_entropy
            ax5.bar(['Attention\nEntropy'], [entropy])
            ax5.set_ylabel("Entropy")
            ax5.set_title("Attention Distribution Entropy\n(Higher = More Uniform)")
            ax5.grid(True, alpha=0.3)

        # 6. Refinement magnitude (bottom-right)
        ax6 = fig.add_subplot(gs[2, 1])
        if trace.refinement_trace:
            ref_mag = trace.refinement_trace.refinement_magnitude
            ax6.bar(['Refinement\nMagnitude'], [ref_mag])
            ax6.set_ylabel("Relative Change")
            ax6.set_title("Logic Refinement Impact")
            ax6.grid(True, alpha=0.3)

        # Main title
        fig.suptitle(f"V.V.A.L.T Detailed Trace Summary\n"
                    f"Total Time: {trace.total_time_ms:.2f} ms | "
                    f"Input: {trace.input_shape} | "
                    f"Task: {trace.task_shape}",
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def plot_graph_topology(
        self,
        graph_adj: torch.Tensor,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ):
        """
        Visualize graph topology from adjacency matrix.

        Args:
            graph_adj: Adjacency matrix
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        if isinstance(graph_adj, torch.Tensor):
            adj = graph_adj.detach().cpu().numpy()
        else:
            adj = graph_adj

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Heatmap
        sns.heatmap(adj, cmap='binary', cbar_kws={'label': 'Edge'},
                   square=True, ax=ax1)
        ax1.set_title("Adjacency Matrix")
        ax1.set_xlabel("Node")
        ax1.set_ylabel("Node")

        # Network plot (if networkx available)
        try:
            import networkx as nx

            G = nx.from_numpy_array(adj)
            pos = nx.spring_layout(G)

            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                 node_size=500, ax=ax2)
            nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5, ax=ax2)
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax2)

            ax2.set_title(f"Graph Visualization\n"
                         f"Nodes: {G.number_of_nodes()}, "
                         f"Edges: {G.number_of_edges()}")
            ax2.axis('off')

        except ImportError:
            ax2.text(0.5, 0.5, 'networkx not available\nfor graph visualization',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def export_trace_json(self, trace: VVALTDetailedTrace, path: str):
        """Export trace as JSON for external analysis."""
        with open(path, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2)

    def generate_report(
        self,
        trace: VVALTDetailedTrace,
        output_dir: str = "./vvalt_report"
    ):
        """
        Generate comprehensive visualization report.

        Args:
            trace: Detailed trace
            output_dir: Directory to save report files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Generate all plots
        if PLOTTING_AVAILABLE:
            self.plot_frame_activations(
                trace.frame_traces,
                save_path=f"{output_dir}/frame_activations.png"
            )

            if trace.vantage_trace and trace.vantage_trace.frame_weights:
                self.plot_frame_weights(
                    trace.vantage_trace.frame_weights,
                    save_path=f"{output_dir}/frame_weights.png"
                )

            if trace.attention_trace and trace.attention_trace.attention_weights is not None:
                self.plot_attention_weights(
                    trace.attention_trace.attention_weights,
                    save_path=f"{output_dir}/attention_weights.png"
                )

            self.plot_component_timing(
                trace,
                save_path=f"{output_dir}/component_timing.png"
            )

            self.plot_full_trace_summary(
                trace,
                save_path=f"{output_dir}/full_summary.png"
            )

        # Export JSON
        self.export_trace_json(trace, f"{output_dir}/trace.json")

        # Generate text report
        with open(f"{output_dir}/report.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("V.V.A.L.T Detailed Trace Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Input Shape: {trace.input_shape}\n")
            f.write(f"Task Shape: {trace.task_shape}\n")
            f.write(f"Has Graph: {trace.has_graph}\n\n")

            f.write("Component Timing:\n")
            for comp, time in trace.component_times_ms.items():
                pct = (time / trace.total_time_ms) * 100
                f.write(f"  {comp:.<40} {time:>8.2f} ms ({pct:>5.1f}%)\n")
            f.write(f"  {'TOTAL':.<40} {trace.total_time_ms:>8.2f} ms\n\n")

            f.write("Safety Checks:\n")
            f.write(f"  Is Safe: {trace.is_safe}\n")
            f.write(f"  Bounds Check Passed: {trace.bounds_check_passed}\n")
            f.write(f"  Deterministic: {trace.deterministic_check_passed}\n\n")

            if trace.vantage_trace and trace.vantage_trace.frame_weights:
                f.write("Frame Weights:\n")
                for frame, weight in trace.vantage_trace.frame_weights.items():
                    f.write(f"  {frame:.<20} {weight:>6.4f}\n")

        print(f"Report generated in: {output_dir}/")
