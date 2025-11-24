"""
Comprehensive Test Suite for V.V.A.L.T

Tests all components and safety guarantees of the system.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vvalt import VVALT
from vvalt.components import (
    VectorFrameEncoder,
    VantageSelector,
    GraphTopologyProjector,
    MultiPerspectiveAttention,
    LogicRefinementUnit,
    ConsistencyVerifier,
    InterpretabilityProjector,
)
from vvalt.utils import create_random_graph, create_line_graph, create_star_graph


class TestVectorFrameEncoder(unittest.TestCase):
    """Test VectorFrameEncoder component."""

    def setUp(self):
        self.input_dim = 10
        self.frame_dim = 8
        self.encoder = VectorFrameEncoder(self.input_dim, self.frame_dim, seed=42)

    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.input_dim, self.input_dim)
        self.assertEqual(self.encoder.frame_dim, self.frame_dim)

    def test_encode_all_frames(self):
        """Test encoding into all five frames."""
        x = np.random.randn(self.input_dim)
        frames = self.encoder.encode(x)

        self.assertIn("semantic", frames)
        self.assertIn("structural", frames)
        self.assertIn("causal", frames)
        self.assertIn("relational", frames)
        self.assertIn("graph", frames)

        for frame_name, frame in frames.items():
            self.assertEqual(frame.shape, (self.frame_dim,))

    def test_semantic_frame_bounded(self):
        """Test semantic frame uses tanh (bounded to [-1, 1])."""
        x = np.random.randn(self.input_dim) * 10
        semantic = self.encoder.encode_semantic(x)

        self.assertTrue(np.all(semantic >= -1.0))
        self.assertTrue(np.all(semantic <= 1.0))

    def test_relational_frame_normalized(self):
        """Test relational frame is L2 normalized."""
        x = np.random.randn(self.input_dim)
        relational = self.encoder.encode_relational(x)

        norm = np.linalg.norm(relational)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_deterministic(self):
        """Test encoder is deterministic."""
        x = np.random.randn(self.input_dim)

        frames1 = self.encoder.encode(x)
        frames2 = self.encoder.encode(x)

        for frame_name in frames1.keys():
            np.testing.assert_array_almost_equal(frames1[frame_name], frames2[frame_name])


class TestVantageSelector(unittest.TestCase):
    """Test VantageSelector component."""

    def setUp(self):
        self.task_dim = 5
        self.selector = VantageSelector(self.task_dim, num_frames=5, seed=42)

    def test_weights_sum_to_one(self):
        """Test weights sum to 1.0 (L1 normalization)."""
        task = np.random.randn(self.task_dim)
        weights = self.selector.compute_weights(task)

        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)

    def test_weights_positive(self):
        """Test all weights are positive."""
        task = np.random.randn(self.task_dim)
        weights = self.selector.compute_weights(task)

        self.assertTrue(np.all(weights >= 0))

    def test_deterministic(self):
        """Test selector is deterministic."""
        task = np.random.randn(self.task_dim)

        weights1 = self.selector.compute_weights(task)
        weights2 = self.selector.compute_weights(task)

        np.testing.assert_array_almost_equal(weights1, weights2)


class TestGraphTopologyProjector(unittest.TestCase):
    """Test GraphTopologyProjector component."""

    def setUp(self):
        self.frame_dim = 8
        self.projector = GraphTopologyProjector(self.frame_dim, seed=42)

    def test_project_without_graph(self):
        """Test projection without graph structure."""
        frames = {
            "semantic": np.random.randn(self.frame_dim),
            "structural": np.random.randn(self.frame_dim),
        }

        projected = self.projector.project(frames, graph_adj=None)

        self.assertEqual(len(projected), len(frames))
        for name in frames.keys():
            self.assertEqual(projected[name].shape, (self.frame_dim,))

    def test_project_with_graph(self):
        """Test projection with graph structure."""
        frames = {
            "semantic": np.random.randn(self.frame_dim),
        }
        graph = create_star_graph(self.frame_dim)

        projected = self.projector.project(frames, graph_adj=graph)

        self.assertIn("semantic", projected)
        self.assertEqual(projected["semantic"].shape, (self.frame_dim,))


class TestMultiPerspectiveAttention(unittest.TestCase):
    """Test MultiPerspectiveAttention component."""

    def setUp(self):
        self.frame_dim = 8
        self.attention = MultiPerspectiveAttention(self.frame_dim, num_frames=5, seed=42)

    def test_attend_output_shape(self):
        """Test attention output has correct shape."""
        frames = {
            "semantic": np.random.randn(self.frame_dim),
            "structural": np.random.randn(self.frame_dim),
            "causal": np.random.randn(self.frame_dim),
        }

        output = self.attention.attend(frames)

        self.assertEqual(output.shape, (self.frame_dim,))

    def test_attention_weights_valid(self):
        """Test attention weights are valid probabilities."""
        frames = {
            "semantic": np.random.randn(self.frame_dim),
            "structural": np.random.randn(self.frame_dim),
        }

        weights = self.attention.get_attention_weights(frames)

        # Each row should sum to 1
        row_sums = np.sum(weights, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(weights.shape[0]))


class TestLogicRefinementUnit(unittest.TestCase):
    """Test LogicRefinementUnit component."""

    def setUp(self):
        self.frame_dim = 8
        self.refiner = LogicRefinementUnit(self.frame_dim, seed=42)

    def test_output_bounded(self):
        """Test refinement output is bounded to [-1, 1]."""
        x = np.random.randn(self.frame_dim) * 10

        refined = self.refiner.refine(x)

        self.assertTrue(np.all(refined >= -1.0))
        self.assertTrue(np.all(refined <= 1.0))

    def test_deterministic(self):
        """Test refiner is deterministic."""
        x = np.random.randn(self.frame_dim)

        refined1 = self.refiner.refine(x)
        refined2 = self.refiner.refine(x)

        np.testing.assert_array_almost_equal(refined1, refined2)

    def test_single_pass(self):
        """Test refinement is single-pass (no iteration)."""
        x = np.random.randn(self.frame_dim)

        # Should complete instantly (single pass)
        import time
        start = time.time()
        refined = self.refiner.refine(x)
        duration = time.time() - start

        # Should be very fast (< 0.1 seconds)
        self.assertLess(duration, 0.1)


class TestConsistencyVerifier(unittest.TestCase):
    """Test ConsistencyVerifier component."""

    def setUp(self):
        self.verifier = ConsistencyVerifier(safe_range=(-10.0, 10.0))

    def test_detect_nan(self):
        """Test NaN detection."""
        x = np.array([1.0, 2.0, np.nan, 3.0])
        validity = self.verifier.check_validity(x)

        self.assertTrue(validity["has_nan"])
        self.assertTrue(validity["has_invalid"])

    def test_detect_inf(self):
        """Test Inf detection."""
        x = np.array([1.0, 2.0, np.inf, 3.0])
        validity = self.verifier.check_validity(x)

        self.assertTrue(validity["has_inf"])
        self.assertTrue(validity["has_invalid"])

    def test_sanitize_nan(self):
        """Test NaN sanitization."""
        x = np.array([1.0, 2.0, np.nan, 3.0])
        safe = self.verifier.verify(x, strict=False)

        self.assertFalse(np.any(np.isnan(safe)))

    def test_clipping(self):
        """Test output clipping to safe range."""
        x = np.array([-100.0, -5.0, 0.0, 5.0, 100.0])
        safe = self.verifier.verify(x)

        self.assertTrue(np.all(safe >= -10.0))
        self.assertTrue(np.all(safe <= 10.0))

    def test_determinism_check(self):
        """Test determinism verification."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0, 2.0, 3.0])

        self.assertTrue(self.verifier.verify_deterministic(x1, x2))

        x3 = np.array([1.0, 2.0, 3.1])
        self.assertFalse(self.verifier.verify_deterministic(x1, x3))


class TestInterpretabilityProjector(unittest.TestCase):
    """Test InterpretabilityProjector component."""

    def setUp(self):
        self.frame_dim = 8
        self.projector = InterpretabilityProjector(self.frame_dim, num_samples=3)

    def test_analyze_vector(self):
        """Test vector analysis."""
        x = np.random.randn(self.frame_dim)
        analysis = self.projector.analyze_vector(x)

        self.assertIn("statistics", analysis)
        self.assertIn("sparsity", analysis)
        self.assertIn("distribution", analysis)
        self.assertIn("samples", analysis)

    def test_analyze_frames(self):
        """Test frame analysis."""
        frames = {
            "semantic": np.random.randn(self.frame_dim),
            "structural": np.random.randn(self.frame_dim),
        }

        analysis = self.projector.analyze_frames(frames)

        self.assertIn("semantic", analysis)
        self.assertIn("structural", analysis)
        self.assertIn("comparison", analysis)

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([1.0, 0.0, 0.0])

        similarity = self.projector._cosine_similarity(x, y)
        self.assertAlmostEqual(similarity, 1.0, places=6)


class TestVVALT(unittest.TestCase):
    """Test main V.V.A.L.T system."""

    def setUp(self):
        self.input_dim = 10
        self.frame_dim = 8
        self.task_dim = 5

        self.vvalt = VVALT(
            input_dim=self.input_dim,
            frame_dim=self.frame_dim,
            task_dim=self.task_dim,
            seed=42
        )

    def test_initialization(self):
        """Test V.V.A.L.T initialization."""
        self.assertEqual(self.vvalt.input_dim, self.input_dim)
        self.assertEqual(self.vvalt.frame_dim, self.frame_dim)
        self.assertEqual(self.vvalt.task_dim, self.task_dim)

    def test_forward_pass(self):
        """Test single forward pass."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)

        output, _ = self.vvalt(x, task)

        self.assertEqual(output.shape, (self.frame_dim,))

    def test_determinism(self):
        """Test system is deterministic."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)

        is_deterministic = self.vvalt.verify_determinism(x, task, num_trials=5)
        self.assertTrue(is_deterministic)

    def test_bounded_output(self):
        """Test output is bounded."""
        x = np.random.randn(self.input_dim) * 100
        task = np.random.randn(self.task_dim) * 100

        output, _ = self.vvalt(x, task)

        # Output should be clipped to safe range
        self.assertTrue(np.all(output >= -10.0))
        self.assertTrue(np.all(output <= 10.0))

    def test_with_graph(self):
        """Test V.V.A.L.T with graph topology."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)
        graph = create_star_graph(self.frame_dim)

        output, _ = self.vvalt(x, task, graph_adj=graph)

        self.assertEqual(output.shape, (self.frame_dim,))

    def test_return_trace(self):
        """Test reasoning trace generation."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)

        output, trace = self.vvalt(x, task, return_trace=True)

        self.assertIsNotNone(trace)
        self.assertIn("input", trace)
        self.assertIn("encoded_frames", trace)
        self.assertIn("final_output", trace)
        self.assertIn("task_weights", trace)

    def test_batch_processing(self):
        """Test batch processing."""
        batch_size = 5
        X = np.random.randn(batch_size, self.input_dim)
        task = np.random.randn(self.task_dim)

        outputs = self.vvalt.batch_forward(X, task)

        self.assertEqual(outputs.shape, (batch_size, self.frame_dim))

    def test_safety_report(self):
        """Test safety report generation."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)

        report = self.vvalt.get_safety_report(x, task)

        self.assertIn("deterministic", report)
        self.assertIn("output_safety", report)
        self.assertIn("bounded", report)

    def test_explanation(self):
        """Test human-readable explanation."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)

        explanation = self.vvalt.explain(x, task)

        self.assertIsInstance(explanation, str)
        self.assertIn("V.V.A.L.T", explanation)

    def test_no_autonomous_loops(self):
        """Test there are no autonomous loops (safety guarantee)."""
        x = np.random.randn(self.input_dim)
        task = np.random.randn(self.task_dim)

        # Should complete in single pass (very fast)
        import time
        start = time.time()
        output, _ = self.vvalt(x, task)
        duration = time.time() - start

        # Should complete in under 1 second (single pass)
        self.assertLess(duration, 1.0)


class TestGraphUtils(unittest.TestCase):
    """Test graph utility functions."""

    def test_create_random_graph(self):
        """Test random graph creation."""
        num_nodes = 5
        graph = create_random_graph(num_nodes, edge_probability=0.5, seed=42)

        self.assertEqual(graph.shape, (num_nodes, num_nodes))
        # Should be symmetric
        np.testing.assert_array_equal(graph, graph.T)
        # No self-loops
        self.assertEqual(np.trace(graph), 0)

    def test_create_line_graph(self):
        """Test line graph creation."""
        num_nodes = 5
        graph = create_line_graph(num_nodes)

        self.assertEqual(graph.shape, (num_nodes, num_nodes))
        # Should have num_nodes - 1 edges (times 2 for undirected)
        self.assertEqual(np.sum(graph), 2 * (num_nodes - 1))

    def test_create_star_graph(self):
        """Test star graph creation."""
        num_nodes = 5
        graph = create_star_graph(num_nodes)

        self.assertEqual(graph.shape, (num_nodes, num_nodes))
        # Center node (0) should connect to all others
        self.assertEqual(np.sum(graph[0, :]), num_nodes - 1)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
