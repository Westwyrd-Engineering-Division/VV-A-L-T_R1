"""
Integration tests for modular PyTorch V.V.A.L.T implementation.

Tests:
- Modular architecture and component access
- Forward modes (fast, fine, diagnostic)
- Event hooks system
- Detailed tracing
- Determinism verification
- Safety guarantees
- Gradient flow and training readiness
- Visualization pipeline
"""

import unittest
import torch
import tempfile
from pathlib import Path
import json

from vvalt.torch_modules.vvalt_modular import (
    VVALTModular,
    VVALTDetailedTrace,
    EventHookType,
)
from vvalt.torch_modules.frame_encoders import (
    MultiFrameEncoder,
    ForwardMode,
)
from vvalt.torch_modules.attention import (
    VantageSelector,
    GraphTopologyProjector,
    MultiPerspectiveAttention,
)
from vvalt.torch_modules.refinement import (
    LogicRefinementUnit,
    ConsistencyVerifier,
)
from vvalt.torch_modules.visualization import VVALTVisualizer
from vvalt.utils import create_random_graph


class TestModularArchitecture(unittest.TestCase):
    """Test modular architecture and component access."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42
        )
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 128)
        self.task_vector = torch.randn(self.batch_size, 32)
        self.graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

    def test_model_initialization(self):
        """Test model initializes correctly with all components."""
        self.assertIsInstance(self.model, VVALTModular)
        self.assertEqual(self.model.input_dim, 128)
        self.assertEqual(self.model.frame_dim, 64)
        self.assertEqual(self.model.task_dim, 32)

    def test_component_access(self):
        """Test accessing individual components."""
        components = self.model.get_all_components()

        # Check all expected components exist
        expected_components = [
            'frame_encoder',
            'vantage_selector',
            'topology_projector',
            'attention',
            'refinement_unit',
            'verifier'
        ]

        for comp_name in expected_components:
            self.assertIn(comp_name, components)
            self.assertIsNotNone(components[comp_name])

        # Test individual component access
        frame_encoder = self.model.get_component('frame_encoder')
        self.assertIsInstance(frame_encoder, MultiFrameEncoder)

        vantage_selector = self.model.get_component('vantage_selector')
        self.assertIsInstance(vantage_selector, VantageSelector)

        attention = self.model.get_component('attention')
        self.assertIsInstance(attention, MultiPerspectiveAttention)

    def test_frame_encoder_submodules(self):
        """Test frame encoder has all 5 submodules."""
        frame_encoder = self.model.get_component('frame_encoder')

        expected_frames = ['semantic', 'structural', 'causal', 'relational', 'graph']
        for frame_name in expected_frames:
            self.assertIn(frame_name, frame_encoder.encoders)

    def test_parameter_count(self):
        """Test model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All params should be trainable


class TestForwardModes(unittest.TestCase):
    """Test three forward modes: fast, fine, diagnostic."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42
        )
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 128)
        self.task_vector = torch.randn(self.batch_size, 32)
        self.graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

    def test_fast_forward(self):
        """Test fast forward mode (no tracing)."""
        with torch.no_grad():
            output = self.model.forward_fast(self.x, self.task_vector, self.graph_adj)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_fine_forward(self):
        """Test fine forward mode (basic tracing)."""
        with torch.no_grad():
            output, trace = self.model.forward_fine(self.x, self.task_vector, self.graph_adj)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertIsInstance(trace, dict)
        self.assertIn('frame_traces', trace)
        self.assertIn('total_time_ms', trace)

    def test_diagnostic_forward(self):
        """Test diagnostic forward mode (full tracing)."""
        with torch.no_grad():
            output, trace = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertIsInstance(trace, VVALTDetailedTrace)

        # Check trace completeness
        self.assertEqual(trace.input_shape, (self.batch_size, 128))
        self.assertEqual(trace.task_shape, (self.batch_size, 32))
        self.assertTrue(trace.has_graph)
        self.assertIsNotNone(trace.frame_traces)
        self.assertIsNotNone(trace.total_time_ms)
        self.assertIsNotNone(trace.component_times_ms)

    def test_output_consistency_across_modes(self):
        """Test all forward modes produce identical outputs."""
        with torch.no_grad():
            output_fast = self.model.forward_fast(self.x, self.task_vector, self.graph_adj)
            output_fine, _ = self.model.forward_fine(self.x, self.task_vector, self.graph_adj)
            output_diag, _ = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        # All outputs should be identical
        self.assertTrue(torch.allclose(output_fast, output_fine, atol=1e-6))
        self.assertTrue(torch.allclose(output_fast, output_diag, atol=1e-6))
        self.assertTrue(torch.allclose(output_fine, output_diag, atol=1e-6))


class TestEventHooks(unittest.TestCase):
    """Test event hooks system."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42
        )
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 128)
        self.task_vector = torch.randn(self.batch_size, 32)
        self.graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

    def test_hook_registration(self):
        """Test hook registration."""
        hook_called = {'count': 0}

        def test_hook(data):
            hook_called['count'] += 1

        self.model.register_hook(EventHookType.PRE_ENCODING, test_hook)

        with torch.no_grad():
            self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        self.assertEqual(hook_called['count'], 1)

    def test_multiple_hooks(self):
        """Test multiple hooks on different events."""
        hook_calls = {
            'pre_encoding': 0,
            'post_encoding': 0,
            'pre_attention': 0,
            'post_attention': 0,
        }

        def make_hook(name):
            def hook(data):
                hook_calls[name] += 1
            return hook

        self.model.register_hook(EventHookType.PRE_ENCODING, make_hook('pre_encoding'))
        self.model.register_hook(EventHookType.POST_ENCODING, make_hook('post_encoding'))
        self.model.register_hook(EventHookType.PRE_ATTENTION, make_hook('pre_attention'))
        self.model.register_hook(EventHookType.POST_ATTENTION, make_hook('post_attention'))

        with torch.no_grad():
            self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        # All hooks should have been called
        for name, count in hook_calls.items():
            self.assertEqual(count, 1, f"Hook {name} not called")

    def test_hook_receives_data(self):
        """Test hooks receive appropriate data."""
        received_data = {}

        def post_encoding_hook(data):
            received_data['frames'] = data.get('frames')
            received_data['trace'] = data.get('trace')

        self.model.register_hook(EventHookType.POST_ENCODING, post_encoding_hook)

        with torch.no_grad():
            self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        self.assertIn('frames', received_data)
        self.assertIsNotNone(received_data['frames'])
        self.assertEqual(len(received_data['frames']), 5)  # 5 frame types


class TestDetailedTracing(unittest.TestCase):
    """Test detailed tracing system."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42
        )
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 128)
        self.task_vector = torch.randn(self.batch_size, 32)
        self.graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

    def test_trace_structure(self):
        """Test trace has expected structure."""
        with torch.no_grad():
            _, trace = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        # Check basic structure
        self.assertIsInstance(trace, VVALTDetailedTrace)
        self.assertIsNotNone(trace.input_shape)
        self.assertIsNotNone(trace.task_shape)
        self.assertIsNotNone(trace.has_graph)
        self.assertIsNotNone(trace.frame_traces)
        self.assertIsNotNone(trace.total_time_ms)
        self.assertIsNotNone(trace.component_times_ms)

    def test_frame_traces(self):
        """Test frame traces are complete."""
        with torch.no_grad():
            _, trace = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        # Should have 5 frame traces
        self.assertEqual(len(trace.frame_traces), 5)

        expected_frames = ['semantic', 'structural', 'causal', 'relational', 'graph']
        for frame_name in expected_frames:
            self.assertIn(frame_name, trace.frame_traces)
            frame_trace = trace.frame_traces[frame_name]

            # Check trace completeness
            self.assertIsNotNone(frame_trace.input_stats)
            self.assertIsNotNone(frame_trace.output_stats)
            self.assertIn('mean', frame_trace.input_stats)
            self.assertIn('std', frame_trace.input_stats)
            self.assertIn('mean', frame_trace.output_stats)
            self.assertIn('std', frame_trace.output_stats)

    def test_component_timing(self):
        """Test component timing is tracked."""
        with torch.no_grad():
            _, trace = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        # Should have timing for all components
        self.assertGreater(len(trace.component_times_ms), 0)
        self.assertGreater(trace.total_time_ms, 0)

        # Sum of component times should be <= total time
        component_sum = sum(trace.component_times_ms.values())
        self.assertLessEqual(component_sum, trace.total_time_ms * 1.1)  # Allow 10% overhead

    def test_safety_checks(self):
        """Test safety verification in trace."""
        with torch.no_grad():
            _, trace = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        self.assertIsNotNone(trace.is_safe)
        self.assertTrue(trace.is_safe)  # Should pass safety checks
        self.assertIsNotNone(trace.verification_trace)


class TestDeterminism(unittest.TestCase):
    """Test determinism guarantees."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42  # Fixed seed
        )
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 128)
        self.task_vector = torch.randn(self.batch_size, 32)
        self.graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

    def test_deterministic_outputs(self):
        """Test outputs are deterministic across runs."""
        outputs = []

        with torch.no_grad():
            for _ in range(5):
                output = self.model.forward_fast(self.x, self.task_vector, self.graph_adj)
                outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            self.assertTrue(torch.allclose(outputs[0], outputs[i], atol=1e-8))

    def test_verify_determinism_method(self):
        """Test verify_determinism method."""
        is_deterministic = self.model.verify_determinism(
            self.x, self.task_vector, self.graph_adj, num_trials=10
        )

        self.assertTrue(is_deterministic)

    def test_determinism_check_in_trace(self):
        """Test determinism check is performed in diagnostic mode."""
        with torch.no_grad():
            _, trace = self.model.forward_diagnostic(self.x, self.task_vector, self.graph_adj)

        self.assertIsNotNone(trace.deterministic_check_passed)
        self.assertTrue(trace.deterministic_check_passed)


class TestSafetyGuarantees(unittest.TestCase):
    """Test safety guarantees are maintained."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            safe_bounds=(-10.0, 10.0),
            seed=42
        )
        self.batch_size = 4

    def test_bounded_outputs(self):
        """Test outputs are bounded."""
        x = torch.randn(self.batch_size, 128)
        task_vector = torch.randn(self.batch_size, 32)
        graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

        with torch.no_grad():
            output = self.model.forward_fast(x, task_vector, graph_adj)

        self.assertGreaterEqual(output.min().item(), -10.0)
        self.assertLessEqual(output.max().item(), 10.0)

    def test_no_nan_inf(self):
        """Test outputs never contain NaN or Inf."""
        x = torch.randn(self.batch_size, 128)
        task_vector = torch.randn(self.batch_size, 32)
        graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

        with torch.no_grad():
            output = self.model.forward_fast(x, task_vector, graph_adj)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_bounds_check_in_trace(self):
        """Test bounds check is recorded in trace."""
        x = torch.randn(self.batch_size, 128)
        task_vector = torch.randn(self.batch_size, 32)
        graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

        with torch.no_grad():
            _, trace = self.model.forward_diagnostic(x, task_vector, graph_adj)

        self.assertIsNotNone(trace.bounds_check_passed)
        self.assertTrue(trace.bounds_check_passed)


class TestGradientFlow(unittest.TestCase):
    """Test gradient flow and training readiness."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42
        )
        self.batch_size = 4

    def test_gradient_computation(self):
        """Test gradients can be computed."""
        x = torch.randn(self.batch_size, 128, requires_grad=True)
        task_vector = torch.randn(self.batch_size, 32, requires_grad=True)
        graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

        output = self.model.forward_fast(x, task_vector, graph_adj)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(task_vector.grad)

        # Check model parameters have gradients
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_gradient_magnitude(self):
        """Test gradients have reasonable magnitude."""
        x = torch.randn(self.batch_size, 128, requires_grad=True)
        task_vector = torch.randn(self.batch_size, 32, requires_grad=True)
        graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

        output = self.model.forward_fast(x, task_vector, graph_adj)
        loss = output.mean()
        loss.backward()

        # Check gradient norms are reasonable (not exploding/vanishing)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertGreater(grad_norm, 1e-8, f"Vanishing gradient in {name}")
                self.assertLess(grad_norm, 1e3, f"Exploding gradient in {name}")


class TestVisualization(unittest.TestCase):
    """Test visualization pipeline."""

    def setUp(self):
        self.model = VVALTModular(
            input_dim=128,
            frame_dim=64,
            task_dim=32,
            hidden_dim=128,
            seed=42
        )
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 128)
        self.task_vector = torch.randn(self.batch_size, 32)
        self.graph_adj = create_random_graph(self.batch_size, edge_probability=0.3)

        # Generate trace
        with torch.no_grad():
            _, self.trace = self.model.forward_diagnostic(
                self.x, self.task_vector, self.graph_adj
            )

        self.visualizer = VVALTVisualizer()
        self.output_dir = Path(tempfile.mkdtemp())

    def test_visualizer_initialization(self):
        """Test visualizer initializes correctly."""
        self.assertIsInstance(self.visualizer, VVALTVisualizer)

    def test_export_trace_json(self):
        """Test trace export to JSON."""
        json_path = self.output_dir / "trace.json"
        self.visualizer.export_trace_json(self.trace, json_path)

        self.assertTrue(json_path.exists())

        # Verify JSON is valid
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.assertIn('input_shape', data)
        self.assertIn('total_time_ms', data)
        self.assertIn('frame_traces', data)

    def test_plot_frame_activations(self):
        """Test frame activations plotting."""
        save_path = self.output_dir / "frame_activations.png"
        self.visualizer.plot_frame_activations(
            self.trace.frame_traces,
            save_path=save_path
        )

        # Matplotlib might not save in headless environments, so just check no errors

    def test_plot_component_timing(self):
        """Test component timing plotting."""
        save_path = self.output_dir / "timing.png"
        self.visualizer.plot_component_timing(
            self.trace,
            save_path=save_path
        )

    def test_generate_report(self):
        """Test full report generation."""
        self.visualizer.generate_report(self.trace, self.output_dir)

        # Check trace JSON was created
        json_path = self.output_dir / "trace.json"
        self.assertTrue(json_path.exists())


if __name__ == '__main__':
    unittest.main()
