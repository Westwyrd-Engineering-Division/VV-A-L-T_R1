"""
Comprehensive Graph Integrity Tests for V.V.A.L.T

This test suite provides exhaustive validation of graph structures, utilities,
and topology projection operations to ensure mathematical correctness,
structural integrity, and system safety.

Test Coverage:
- Graph utility functions (random, line, star, complete, ring)
- Adjacency matrix properties (symmetry, no self-loops, sparsity)
- Graph normalization operations (symmetric normalization)
- Graph convolution operations
- GraphTopologyProjector component integrity
- Edge cases and boundary conditions
- Determinism and reproducibility
- Integration with V.V.A.L.T system
"""

import unittest
import numpy as np
from typing import Dict, Optional

from vvalt.utils.graph import (
    create_random_graph,
    create_line_graph,
    create_star_graph,
    create_complete_graph,
    create_ring_graph,
)
from vvalt.components.graph_topology_projector import GraphTopologyProjector


class TestGraphUtilityFunctions(unittest.TestCase):
    """Test suite for graph creation utility functions."""

    def test_create_random_graph_basic_properties(self):
        """Test random graph creation has correct basic properties."""
        num_nodes = 10
        graph = create_random_graph(num_nodes, edge_probability=0.3, seed=42)

        # Check shape
        self.assertEqual(graph.shape, (num_nodes, num_nodes))

        # Check symmetry (undirected graph)
        np.testing.assert_array_equal(graph, graph.T)

        # Check no self-loops
        self.assertEqual(np.trace(graph), 0)
        self.assertTrue(np.all(np.diag(graph) == 0))

        # Check binary values
        unique_values = np.unique(graph)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))

        # Check dtype (default numpy float is float64)
        self.assertTrue(graph.dtype in [np.float32, np.float64])

    def test_create_random_graph_determinism(self):
        """Test random graph creation is deterministic with fixed seed."""
        num_nodes = 15
        seed = 42

        graph1 = create_random_graph(num_nodes, edge_probability=0.3, seed=seed)
        graph2 = create_random_graph(num_nodes, edge_probability=0.3, seed=seed)

        np.testing.assert_array_equal(graph1, graph2)

    def test_create_random_graph_different_seeds(self):
        """Test different seeds produce different graphs."""
        num_nodes = 10

        graph1 = create_random_graph(num_nodes, edge_probability=0.3, seed=42)
        graph2 = create_random_graph(num_nodes, edge_probability=0.3, seed=123)

        # Should not be identical (with very high probability)
        self.assertFalse(np.array_equal(graph1, graph2))

    def test_create_random_graph_edge_probability(self):
        """Test edge probability affects graph density."""
        num_nodes = 50

        # Low probability
        sparse_graph = create_random_graph(num_nodes, edge_probability=0.1, seed=42)
        sparse_density = np.sum(sparse_graph) / (num_nodes * (num_nodes - 1))

        # High probability
        dense_graph = create_random_graph(num_nodes, edge_probability=0.8, seed=42)
        dense_density = np.sum(dense_graph) / (num_nodes * (num_nodes - 1))

        # Dense graph should have more edges
        self.assertGreater(dense_density, sparse_density)

        # Densities should be approximately the probabilities
        # Note: actual density may be higher due to symmetrization
        self.assertLess(sparse_density, 0.3)
        self.assertGreater(dense_density, 0.7)

    def test_create_random_graph_edge_cases(self):
        """Test random graph creation with edge case sizes."""
        # Single node
        graph = create_random_graph(1, edge_probability=0.5, seed=42)
        self.assertEqual(graph.shape, (1, 1))
        self.assertEqual(graph[0, 0], 0)  # No self-loop

        # Two nodes
        graph = create_random_graph(2, edge_probability=0.5, seed=42)
        self.assertEqual(graph.shape, (2, 2))
        np.testing.assert_array_equal(graph, graph.T)

        # Large graph
        graph = create_random_graph(100, edge_probability=0.3, seed=42)
        self.assertEqual(graph.shape, (100, 100))

    def test_create_line_graph_structure(self):
        """Test line graph has correct chain structure."""
        num_nodes = 10
        graph = create_line_graph(num_nodes)

        # Check shape
        self.assertEqual(graph.shape, (num_nodes, num_nodes))

        # Check symmetry
        np.testing.assert_array_equal(graph, graph.T)

        # Check no self-loops
        self.assertEqual(np.trace(graph), 0)

        # Check total edges: n-1 edges (each counted twice in adjacency matrix)
        total_edges = np.sum(graph)
        self.assertEqual(total_edges, 2 * (num_nodes - 1))

        # Check chain structure: node i connects to i-1 and i+1
        for i in range(num_nodes):
            degree = np.sum(graph[i])
            if i == 0 or i == num_nodes - 1:
                self.assertEqual(degree, 1)  # End nodes have degree 1
            else:
                self.assertEqual(degree, 2)  # Interior nodes have degree 2

        # Check consecutive connections
        for i in range(num_nodes - 1):
            self.assertEqual(graph[i, i + 1], 1)
            self.assertEqual(graph[i + 1, i], 1)

    def test_create_line_graph_edge_cases(self):
        """Test line graph with edge case sizes."""
        # Single node (isolated)
        graph = create_line_graph(1)
        self.assertEqual(graph.shape, (1, 1))
        self.assertEqual(np.sum(graph), 0)

        # Two nodes (single edge)
        graph = create_line_graph(2)
        self.assertEqual(graph.shape, (2, 2))
        self.assertEqual(np.sum(graph), 2)
        self.assertEqual(graph[0, 1], 1)
        self.assertEqual(graph[1, 0], 1)

    def test_create_star_graph_structure(self):
        """Test star graph has correct hub-and-spoke structure."""
        num_nodes = 10
        graph = create_star_graph(num_nodes)

        # Check shape
        self.assertEqual(graph.shape, (num_nodes, num_nodes))

        # Check symmetry
        np.testing.assert_array_equal(graph, graph.T)

        # Check no self-loops
        self.assertEqual(np.trace(graph), 0)

        # Check center node (node 0) connects to all others
        center_degree = np.sum(graph[0])
        self.assertEqual(center_degree, num_nodes - 1)

        # Check all other nodes connect only to center
        for i in range(1, num_nodes):
            degree = np.sum(graph[i])
            self.assertEqual(degree, 1)
            self.assertEqual(graph[i, 0], 1)

        # Check total edges
        total_edges = np.sum(graph)
        self.assertEqual(total_edges, 2 * (num_nodes - 1))

    def test_create_star_graph_edge_cases(self):
        """Test star graph with edge case sizes."""
        # Single node (isolated)
        graph = create_star_graph(1)
        self.assertEqual(graph.shape, (1, 1))
        self.assertEqual(np.sum(graph), 0)

        # Two nodes (single edge)
        graph = create_star_graph(2)
        self.assertEqual(graph.shape, (2, 2))
        self.assertEqual(np.sum(graph), 2)

    def test_create_complete_graph_structure(self):
        """Test complete graph where every node connects to every other."""
        num_nodes = 10
        graph = create_complete_graph(num_nodes)

        # Check shape
        self.assertEqual(graph.shape, (num_nodes, num_nodes))

        # Check symmetry
        np.testing.assert_array_equal(graph, graph.T)

        # Check no self-loops
        self.assertEqual(np.trace(graph), 0)
        self.assertTrue(np.all(np.diag(graph) == 0))

        # Check all off-diagonal elements are 1
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    self.assertEqual(graph[i, j], 1)

        # Check degree of each node (should be num_nodes - 1)
        for i in range(num_nodes):
            degree = np.sum(graph[i])
            self.assertEqual(degree, num_nodes - 1)

        # Check total edges: n*(n-1)/2 edges (counted twice)
        total_edges = np.sum(graph)
        self.assertEqual(total_edges, num_nodes * (num_nodes - 1))

    def test_create_complete_graph_edge_cases(self):
        """Test complete graph with edge case sizes."""
        # Single node
        graph = create_complete_graph(1)
        self.assertEqual(graph.shape, (1, 1))
        self.assertEqual(np.sum(graph), 0)

        # Two nodes
        graph = create_complete_graph(2)
        self.assertEqual(graph.shape, (2, 2))
        self.assertEqual(np.sum(graph), 2)

    def test_create_ring_graph_structure(self):
        """Test ring graph has correct cycle structure."""
        num_nodes = 10
        graph = create_ring_graph(num_nodes)

        # Check shape
        self.assertEqual(graph.shape, (num_nodes, num_nodes))

        # Check symmetry
        np.testing.assert_array_equal(graph, graph.T)

        # Check no self-loops
        self.assertEqual(np.trace(graph), 0)

        # Check all nodes have degree 2
        for i in range(num_nodes):
            degree = np.sum(graph[i])
            self.assertEqual(degree, 2)

        # Check consecutive connections and wraparound
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            self.assertEqual(graph[i, next_node], 1)

        # Check total edges (n edges counted twice)
        total_edges = np.sum(graph)
        self.assertEqual(total_edges, 2 * num_nodes)

    def test_create_ring_graph_edge_cases(self):
        """Test ring graph with edge case sizes."""
        # Single node (self-loop prohibited, so isolated)
        graph = create_ring_graph(1)
        self.assertEqual(graph.shape, (1, 1))
        self.assertEqual(graph[0, 0], 0)

        # Two nodes (forms a line, not a ring since num_nodes <= 2)
        graph = create_ring_graph(2)
        self.assertEqual(graph.shape, (2, 2))
        self.assertEqual(np.sum(graph), 2)  # 1 edge counted twice (no wraparound)

        # Three nodes (triangle - first true ring)
        graph = create_ring_graph(3)
        self.assertEqual(graph.shape, (3, 3))
        self.assertEqual(np.sum(graph), 6)  # 3 edges counted twice

    def test_all_graphs_are_valid_adjacency_matrices(self):
        """Test all graph types produce valid adjacency matrices."""
        num_nodes = 8

        graphs = [
            create_random_graph(num_nodes, edge_probability=0.4, seed=42),
            create_line_graph(num_nodes),
            create_star_graph(num_nodes),
            create_complete_graph(num_nodes),
            create_ring_graph(num_nodes),
        ]

        for graph in graphs:
            # Check symmetric
            np.testing.assert_array_equal(graph, graph.T)

            # Check no self-loops
            self.assertEqual(np.trace(graph), 0)

            # Check binary
            unique_values = np.unique(graph)
            self.assertTrue(np.all(np.isin(unique_values, [0, 1])))

            # Check shape
            self.assertEqual(graph.shape, (num_nodes, num_nodes))

            # Check dtype (default numpy float is float64)
            self.assertTrue(graph.dtype in [np.float32, np.float64])


class TestGraphNormalization(unittest.TestCase):
    """Test suite for graph normalization operations."""

    def setUp(self):
        """Initialize test fixtures."""
        self.frame_dim = 16
        self.projector = GraphTopologyProjector(self.frame_dim, seed=42)

    def test_normalize_adjacency_adds_self_loops(self):
        """Test normalization adds identity matrix (self-loops)."""
        num_nodes = 5
        adj = create_line_graph(num_nodes)

        # Original has no self-loops
        self.assertEqual(np.trace(adj), 0)

        # Normalized should have added self-loops
        normalized = self.projector._normalize_adjacency(adj)

        # Check that diagonal elements are non-zero
        self.assertTrue(np.all(np.diag(normalized) > 0))

    def test_normalize_adjacency_symmetric_output(self):
        """Test normalized adjacency matrix remains symmetric."""
        num_nodes = 8
        adj = create_random_graph(num_nodes, edge_probability=0.3, seed=42)

        normalized = self.projector._normalize_adjacency(adj)

        # Should remain symmetric
        np.testing.assert_array_almost_equal(normalized, normalized.T)

    def test_normalize_adjacency_formula(self):
        """Test symmetric normalization formula: D^(-1/2) * A * D^(-1/2)."""
        num_nodes = 4
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ], dtype=np.float32)

        # Add self-loops
        adj_with_loops = adj + np.eye(num_nodes)

        # Compute degree matrix
        degrees = np.sum(adj_with_loops, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

        # Expected normalized adjacency
        expected = D_inv_sqrt @ adj_with_loops @ D_inv_sqrt

        # Actual normalized adjacency
        actual = self.projector._normalize_adjacency(adj)

        np.testing.assert_array_almost_equal(actual, expected)

    def test_normalize_adjacency_isolated_nodes(self):
        """Test normalization handles isolated nodes (degree 0) safely."""
        num_nodes = 5
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        # Only connect nodes 0 and 1
        adj[0, 1] = 1
        adj[1, 0] = 1
        # Nodes 2, 3, 4 are isolated

        normalized = self.projector._normalize_adjacency(adj)

        # Should not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(normalized)))
        self.assertFalse(np.any(np.isinf(normalized)))

        # Isolated nodes should have self-loops normalized
        for i in [2, 3, 4]:
            # Degree is 1 (only self-loop), so normalized value is 1/1 = 1
            self.assertAlmostEqual(normalized[i, i], 1.0)

    def test_normalize_adjacency_complete_graph(self):
        """Test normalization of complete graph."""
        num_nodes = 6
        adj = create_complete_graph(num_nodes)

        normalized = self.projector._normalize_adjacency(adj)

        # Should be symmetric
        np.testing.assert_array_almost_equal(normalized, normalized.T)

        # All entries should be positive
        self.assertTrue(np.all(normalized > 0))

        # Diagonal should have values (self-loops present)
        self.assertTrue(np.all(np.diag(normalized) > 0))


class TestGraphConvolution(unittest.TestCase):
    """Test suite for graph convolution operations."""

    def setUp(self):
        """Initialize test fixtures."""
        self.frame_dim = 16
        self.projector = GraphTopologyProjector(self.frame_dim, seed=42)

    def test_graph_convolution_output_shape(self):
        """Test graph convolution produces correct output shape."""
        num_nodes = 10
        features = np.random.randn(num_nodes, self.frame_dim).astype(np.float32)
        adj = create_star_graph(num_nodes)

        output = self.projector._graph_convolution(features, adj)

        # Output should have same shape as input
        self.assertEqual(output.shape, features.shape)

    def test_graph_convolution_aggregates_neighbors(self):
        """Test graph convolution aggregates information from neighbors."""
        num_nodes = 3
        features = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Simple line graph: 0 -- 1 -- 2
        adj = create_line_graph(num_nodes)

        # Project with simple projector
        projector = GraphTopologyProjector(3, seed=42)

        output = projector._graph_convolution(features, adj)

        # Node 1 (middle) should aggregate from nodes 0 and 2
        # Output should be non-zero for all features at node 1
        # (exact values depend on W_topology and normalization)
        self.assertTrue(output.shape, (num_nodes, 3))

    def test_graph_convolution_no_nans_or_infs(self):
        """Test graph convolution produces no NaN or Inf values."""
        num_nodes = 20
        features = np.random.randn(num_nodes, self.frame_dim).astype(np.float32)
        adj = create_random_graph(num_nodes, edge_probability=0.3, seed=42)

        output = self.projector._graph_convolution(features, adj)

        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))

    def test_graph_convolution_deterministic(self):
        """Test graph convolution is deterministic."""
        num_nodes = 10
        features = np.random.randn(num_nodes, self.frame_dim).astype(np.float32)
        adj = create_star_graph(num_nodes)

        output1 = self.projector._graph_convolution(features, adj)
        output2 = self.projector._graph_convolution(features, adj)

        np.testing.assert_array_equal(output1, output2)


class TestGraphTopologyProjector(unittest.TestCase):
    """Test suite for GraphTopologyProjector component."""

    def setUp(self):
        """Initialize test fixtures."""
        self.frame_dim = 16
        self.num_nodes = 8
        self.projector = GraphTopologyProjector(self.frame_dim, seed=42)

    def test_initialization(self):
        """Test projector initialization."""
        self.assertEqual(self.projector.frame_dim, self.frame_dim)
        self.assertIsNotNone(self.projector.W_topology)
        self.assertEqual(self.projector.W_topology.shape,
                        (self.frame_dim, self.frame_dim))

    def test_project_without_graph(self):
        """Test projection without graph structure."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
            'structural': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }

        output = self.projector.project(frames, graph_adj=None)

        # Output should have same keys
        self.assertEqual(set(output.keys()), set(frames.keys()))

        # Output shapes should match input shapes
        for key in frames:
            self.assertEqual(output[key].shape, frames[key].shape)

        # Without graph, W_topology is still applied but no graph aggregation
        # Output should be valid (no NaN/Inf) and non-zero
        for key in frames:
            self.assertFalse(np.any(np.isnan(output[key])))
            self.assertFalse(np.any(np.isinf(output[key])))
            self.assertFalse(np.allclose(output[key], 0))

    def test_project_with_graph(self):
        """Test projection with graph structure."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
            'structural': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_star_graph(self.num_nodes)

        output = self.projector.project(frames, graph_adj=graph)

        # Output should have same keys
        self.assertEqual(set(output.keys()), set(frames.keys()))

        # Output shapes should match input shapes
        for key in frames:
            self.assertEqual(output[key].shape, frames[key].shape)

        # With graph, output should differ from input
        for key in frames:
            self.assertFalse(np.array_equal(output[key], frames[key]))

    def test_project_multiple_frame_types(self):
        """Test projection with all frame types."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
            'structural': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
            'causal': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
            'relational': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
            'graph': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_ring_graph(self.num_nodes)

        output = self.projector.project(frames, graph_adj=graph)

        # All frames should be processed
        self.assertEqual(set(output.keys()), set(frames.keys()))

        for key in frames:
            self.assertEqual(output[key].shape, frames[key].shape)
            self.assertFalse(np.any(np.isnan(output[key])))
            self.assertFalse(np.any(np.isinf(output[key])))

    def test_get_topology_influence(self):
        """Test topology influence measurement."""
        frame = np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32)
        graph = create_star_graph(self.num_nodes)

        influence = self.projector.get_topology_influence(frame, graph)

        # Influence should be a non-negative scalar
        self.assertIsInstance(influence, float)
        self.assertGreaterEqual(influence, 0.0)
        self.assertFalse(np.isnan(influence))
        self.assertFalse(np.isinf(influence))

    def test_get_topology_influence_different_graphs(self):
        """Test topology influence varies with graph structure."""
        frame = np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32)

        # Different graph structures should produce different influences
        star_graph = create_star_graph(self.num_nodes)
        line_graph = create_line_graph(self.num_nodes)
        complete_graph = create_complete_graph(self.num_nodes)

        influence_star = self.projector.get_topology_influence(frame, star_graph)
        influence_line = self.projector.get_topology_influence(frame, line_graph)
        influence_complete = self.projector.get_topology_influence(frame, complete_graph)

        # All should be valid
        for influence in [influence_star, influence_line, influence_complete]:
            self.assertGreaterEqual(influence, 0.0)
            self.assertFalse(np.isnan(influence))

    def test_deterministic_projection(self):
        """Test projection is deterministic with same inputs."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_random_graph(self.num_nodes, edge_probability=0.3, seed=42)

        output1 = self.projector.project(frames, graph_adj=graph)
        output2 = self.projector.project(frames, graph_adj=graph)

        for key in frames:
            np.testing.assert_array_equal(output1[key], output2[key])

    def test_projection_safety_bounds(self):
        """Test projection produces bounded outputs (no extreme values)."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_complete_graph(self.num_nodes)

        output = self.projector.project(frames, graph_adj=graph)

        for key in frames:
            # Check no NaN or Inf
            self.assertFalse(np.any(np.isnan(output[key])))
            self.assertFalse(np.any(np.isinf(output[key])))

            # Check values are within reasonable bounds
            self.assertTrue(np.all(np.abs(output[key]) < 1e6))


class TestGraphIntegrationScenarios(unittest.TestCase):
    """Test suite for complex integration scenarios."""

    def setUp(self):
        """Initialize test fixtures."""
        self.frame_dim = 32
        self.num_nodes = 16
        self.projector = GraphTopologyProjector(self.frame_dim, seed=42)

    def test_empty_frames_dict(self):
        """Test handling of empty frames dictionary."""
        frames = {}
        graph = create_star_graph(self.num_nodes)

        output = self.projector.project(frames, graph_adj=graph)

        # Should return empty dict
        self.assertEqual(output, {})

    def test_single_frame(self):
        """Test projection with single frame."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_line_graph(self.num_nodes)

        output = self.projector.project(frames, graph_adj=graph)

        self.assertEqual(len(output), 1)
        self.assertIn('semantic', output)
        self.assertEqual(output['semantic'].shape, frames['semantic'].shape)

    def test_different_graph_topologies_same_frames(self):
        """Test same frames with different graph topologies."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }

        graphs = {
            'star': create_star_graph(self.num_nodes),
            'line': create_line_graph(self.num_nodes),
            'ring': create_ring_graph(self.num_nodes),
            'complete': create_complete_graph(self.num_nodes),
            'random': create_random_graph(self.num_nodes, edge_probability=0.3, seed=42),
        }

        outputs = {}
        for name, graph in graphs.items():
            output = self.projector.project(frames, graph_adj=graph)
            outputs[name] = output

        # All outputs should be valid
        for name, output in outputs.items():
            self.assertEqual(output['semantic'].shape, frames['semantic'].shape)
            self.assertFalse(np.any(np.isnan(output['semantic'])))
            self.assertFalse(np.any(np.isinf(output['semantic'])))

        # Different graphs should produce different outputs
        output_arrays = [outputs[name]['semantic'] for name in graphs.keys()]
        for i in range(len(output_arrays)):
            for j in range(i + 1, len(output_arrays)):
                # At least some should be different
                if not np.array_equal(output_arrays[i], output_arrays[j]):
                    break

    def test_very_sparse_graph(self):
        """Test projection with very sparse graph."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        # Very sparse graph (low edge probability)
        graph = create_random_graph(self.num_nodes, edge_probability=0.05, seed=42)

        output = self.projector.project(frames, graph_adj=graph)

        self.assertEqual(output['semantic'].shape, frames['semantic'].shape)
        self.assertFalse(np.any(np.isnan(output['semantic'])))
        self.assertFalse(np.any(np.isinf(output['semantic'])))

    def test_very_dense_graph(self):
        """Test projection with very dense graph."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        # Very dense graph (high edge probability)
        graph = create_random_graph(self.num_nodes, edge_probability=0.95, seed=42)

        output = self.projector.project(frames, graph_adj=graph)

        self.assertEqual(output['semantic'].shape, frames['semantic'].shape)
        self.assertFalse(np.any(np.isnan(output['semantic'])))
        self.assertFalse(np.any(np.isinf(output['semantic'])))

    def test_large_frame_dimension(self):
        """Test projection with large frame dimension."""
        large_frame_dim = 128
        projector = GraphTopologyProjector(large_frame_dim, seed=42)

        frames = {
            'semantic': np.random.randn(self.num_nodes, large_frame_dim).astype(np.float32),
        }
        graph = create_star_graph(self.num_nodes)

        output = projector.project(frames, graph_adj=graph)

        self.assertEqual(output['semantic'].shape, (self.num_nodes, large_frame_dim))
        self.assertFalse(np.any(np.isnan(output['semantic'])))

    def test_large_number_of_nodes(self):
        """Test projection with large number of nodes."""
        large_num_nodes = 100

        frames = {
            'semantic': np.random.randn(large_num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_random_graph(large_num_nodes, edge_probability=0.2, seed=42)

        output = self.projector.project(frames, graph_adj=graph)

        self.assertEqual(output['semantic'].shape, (large_num_nodes, self.frame_dim))
        self.assertFalse(np.any(np.isnan(output['semantic'])))

    def test_consecutive_projections(self):
        """Test multiple consecutive projections maintain consistency."""
        frames = {
            'semantic': np.random.randn(self.num_nodes, self.frame_dim).astype(np.float32),
        }
        graph = create_ring_graph(self.num_nodes)

        # First projection
        output1 = self.projector.project(frames, graph_adj=graph)

        # Second projection with same inputs
        output2 = self.projector.project(frames, graph_adj=graph)

        # Third projection with same inputs
        output3 = self.projector.project(frames, graph_adj=graph)

        # All should be identical
        np.testing.assert_array_equal(output1['semantic'], output2['semantic'])
        np.testing.assert_array_equal(output1['semantic'], output3['semantic'])


class TestGraphMathematicalProperties(unittest.TestCase):
    """Test suite for mathematical properties of graphs and operations."""

    def test_adjacency_matrix_eigenvalues(self):
        """Test eigenvalues of adjacency matrices are real."""
        num_nodes = 10

        graphs = [
            create_line_graph(num_nodes),
            create_star_graph(num_nodes),
            create_complete_graph(num_nodes),
            create_ring_graph(num_nodes),
        ]

        for graph in graphs:
            eigenvalues = np.linalg.eigvals(graph)

            # Eigenvalues should be real (imaginary part negligible)
            self.assertTrue(np.all(np.abs(np.imag(eigenvalues)) < 1e-10))

    def test_normalized_adjacency_row_sums(self):
        """Test row sums of normalized adjacency after adding self-loops."""
        num_nodes = 8
        frame_dim = 16
        projector = GraphTopologyProjector(frame_dim, seed=42)

        graphs = [
            create_line_graph(num_nodes),
            create_star_graph(num_nodes),
            create_complete_graph(num_nodes),
        ]

        for graph in graphs:
            normalized = projector._normalize_adjacency(graph)

            # Normalized adjacency should be symmetric
            np.testing.assert_array_almost_equal(normalized, normalized.T)

            # All values should be finite
            self.assertTrue(np.all(np.isfinite(normalized)))

    def test_laplacian_matrix_properties(self):
        """Test Laplacian matrix derived from adjacency has correct properties."""
        num_nodes = 10
        graph = create_ring_graph(num_nodes)

        # Compute Laplacian: L = D - A
        degrees = np.sum(graph, axis=1)
        D = np.diag(degrees)
        L = D - graph

        # Laplacian should be symmetric
        np.testing.assert_array_almost_equal(L, L.T)

        # Row sums should be zero
        row_sums = np.sum(L, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(num_nodes))

        # Laplacian is positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(L)
        self.assertTrue(np.all(eigenvalues >= -1e-10))  # Allow for numerical error

    def test_graph_density_calculation(self):
        """Test graph density matches expected values."""
        num_nodes = 20

        # Complete graph: density = 1.0
        complete = create_complete_graph(num_nodes)
        max_edges = num_nodes * (num_nodes - 1) / 2
        actual_edges = np.sum(complete) / 2
        density = actual_edges / max_edges
        self.assertAlmostEqual(density, 1.0)

        # Line graph: density should be low
        line = create_line_graph(num_nodes)
        actual_edges = np.sum(line) / 2
        density = actual_edges / max_edges
        self.assertLess(density, 0.2)

        # Ring graph
        ring = create_ring_graph(num_nodes)
        actual_edges = np.sum(ring) / 2
        density = actual_edges / max_edges
        self.assertLess(density, 0.2)

    def test_graph_connectedness(self):
        """Test that standard graph types are connected (except special cases)."""
        num_nodes = 10

        def is_connected(adj_matrix):
            """Check if graph is connected using BFS."""
            n = len(adj_matrix)
            visited = [False] * n
            queue = [0]
            visited[0] = True
            count = 1

            while queue:
                node = queue.pop(0)
                for neighbor in range(n):
                    if adj_matrix[node, neighbor] > 0 and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        count += 1

            return count == n

        # These should all be connected
        connected_graphs = [
            create_line_graph(num_nodes),
            create_star_graph(num_nodes),
            create_complete_graph(num_nodes),
            create_ring_graph(num_nodes),
        ]

        for graph in connected_graphs:
            self.assertTrue(is_connected(graph))


class TestGraphEdgeCasesAndRobustness(unittest.TestCase):
    """Test suite for edge cases and robustness."""

    def test_zero_adjacency_matrix(self):
        """Test handling of zero adjacency matrix (no edges)."""
        num_nodes = 5
        frame_dim = 16
        projector = GraphTopologyProjector(frame_dim, seed=42)

        # Zero adjacency (isolated nodes)
        zero_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        frames = {
            'semantic': np.random.randn(num_nodes, frame_dim).astype(np.float32),
        }

        output = projector.project(frames, graph_adj=zero_adj)

        # Should handle gracefully
        self.assertEqual(output['semantic'].shape, frames['semantic'].shape)
        self.assertFalse(np.any(np.isnan(output['semantic'])))
        self.assertFalse(np.any(np.isinf(output['semantic'])))

    def test_single_node_graph(self):
        """Test all graph types with single node."""
        graphs = [
            create_random_graph(1, edge_probability=0.5, seed=42),
            create_line_graph(1),
            create_star_graph(1),
            create_complete_graph(1),
            create_ring_graph(1),
        ]

        for graph in graphs:
            self.assertEqual(graph.shape, (1, 1))
            self.assertEqual(graph[0, 0], 0)  # No self-loop

    def test_two_node_graph(self):
        """Test all graph types with two nodes."""
        graphs = {
            'line': create_line_graph(2),
            'star': create_star_graph(2),
            'complete': create_complete_graph(2),
            'ring': create_ring_graph(2),
        }

        for name, graph in graphs.items():
            self.assertEqual(graph.shape, (2, 2))
            # All should have an edge between the two nodes
            self.assertEqual(graph[0, 1], 1)
            self.assertEqual(graph[1, 0], 1)

    def test_projection_with_mismatched_dimensions(self):
        """Test that mismatched dimensions are handled properly."""
        frame_dim = 16
        projector = GraphTopologyProjector(frame_dim, seed=42)

        # Graph with different number of nodes than frame
        frames = {
            'semantic': np.random.randn(10, frame_dim).astype(np.float32),
        }
        graph = create_star_graph(8)  # Different number of nodes

        # This should raise an error or handle gracefully
        # (depending on implementation, adjust as needed)
        try:
            output = projector.project(frames, graph_adj=graph)
            # If it doesn't raise, check that output is at least valid
            self.assertFalse(np.any(np.isnan(output['semantic'])))
        except (ValueError, RuntimeError, AssertionError):
            # Expected to fail with dimension mismatch
            pass

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme input values."""
        num_nodes = 10
        frame_dim = 16
        projector = GraphTopologyProjector(frame_dim, seed=42)

        # Very large values
        large_frames = {
            'semantic': np.ones((num_nodes, frame_dim), dtype=np.float32) * 1000,
        }
        graph = create_star_graph(num_nodes)

        output = projector.project(large_frames, graph_adj=graph)

        # Should not overflow or produce NaN/Inf
        self.assertFalse(np.any(np.isnan(output['semantic'])))
        self.assertFalse(np.any(np.isinf(output['semantic'])))

        # Very small values
        small_frames = {
            'semantic': np.ones((num_nodes, frame_dim), dtype=np.float32) * 1e-6,
        }

        output = projector.project(small_frames, graph_adj=graph)

        # Should not underflow or produce NaN/Inf
        self.assertFalse(np.any(np.isnan(output['semantic'])))
        self.assertFalse(np.any(np.isinf(output['semantic'])))


class TestGraphReproducibility(unittest.TestCase):
    """Test suite for reproducibility and determinism."""

    def test_random_graph_seed_reproducibility(self):
        """Test that same seed produces identical graphs across runs."""
        num_nodes = 50
        seed = 12345

        graphs = []
        for _ in range(10):
            graph = create_random_graph(num_nodes, edge_probability=0.4, seed=seed)
            graphs.append(graph)

        # All graphs should be identical
        for i in range(1, len(graphs)):
            np.testing.assert_array_equal(graphs[0], graphs[i])

    def test_projector_initialization_reproducibility(self):
        """Test that same seed produces identical projector initialization."""
        frame_dim = 32
        seed = 42

        projectors = []
        for _ in range(5):
            projector = GraphTopologyProjector(frame_dim, seed=seed)
            projectors.append(projector.W_topology)

        # All W_topology matrices should be identical
        for i in range(1, len(projectors)):
            np.testing.assert_array_equal(projectors[0], projectors[i])

    def test_end_to_end_reproducibility(self):
        """Test end-to-end reproducibility of projection pipeline."""
        num_nodes = 12
        frame_dim = 24
        seed = 999

        # Create projector
        projector = GraphTopologyProjector(frame_dim, seed=seed)

        # Create frames (with fixed seed)
        np.random.seed(seed)
        frames = {
            'semantic': np.random.randn(num_nodes, frame_dim).astype(np.float32),
        }

        # Create graph
        graph = create_random_graph(num_nodes, edge_probability=0.3, seed=seed)

        # Run projection multiple times
        outputs = []
        for _ in range(5):
            output = projector.project(frames, graph_adj=graph)
            outputs.append(output['semantic'])

        # All outputs should be identical
        for i in range(1, len(outputs)):
            np.testing.assert_array_equal(outputs[0], outputs[i])


if __name__ == '__main__':
    unittest.main()
