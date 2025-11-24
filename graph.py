"""
Graph Utilities

Helper functions for creating and manipulating graph structures.
"""

import numpy as np


def create_random_graph(num_nodes: int, edge_probability: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Create random undirected graph using Erdős-Rényi model.

    Args:
        num_nodes: Number of nodes in graph
        edge_probability: Probability of edge between any two nodes
        seed: Random seed for reproducibility

    Returns:
        Adjacency matrix of shape (num_nodes, num_nodes)
    """
    np.random.seed(seed)

    # Create random adjacency matrix
    adj = (np.random.rand(num_nodes, num_nodes) < edge_probability).astype(float)

    # Make symmetric (undirected)
    adj = (adj + adj.T) / 2
    adj = (adj > 0).astype(float)

    # Remove self-loops
    np.fill_diagonal(adj, 0)

    return adj


def create_line_graph(num_nodes: int) -> np.ndarray:
    """
    Create line graph (path graph) where nodes form a chain.

    Args:
        num_nodes: Number of nodes in graph

    Returns:
        Adjacency matrix of shape (num_nodes, num_nodes)
    """
    adj = np.zeros((num_nodes, num_nodes))

    # Connect consecutive nodes
    for i in range(num_nodes - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1

    return adj


def create_star_graph(num_nodes: int) -> np.ndarray:
    """
    Create star graph where one central node connects to all others.

    Args:
        num_nodes: Number of nodes in graph (including center)

    Returns:
        Adjacency matrix of shape (num_nodes, num_nodes)
    """
    adj = np.zeros((num_nodes, num_nodes))

    # Connect center (node 0) to all other nodes
    if num_nodes > 1:
        adj[0, 1:] = 1
        adj[1:, 0] = 1

    return adj


def create_complete_graph(num_nodes: int) -> np.ndarray:
    """
    Create complete graph where every node connects to every other node.

    Args:
        num_nodes: Number of nodes in graph

    Returns:
        Adjacency matrix of shape (num_nodes, num_nodes)
    """
    adj = np.ones((num_nodes, num_nodes))

    # Remove self-loops
    np.fill_diagonal(adj, 0)

    return adj


def create_ring_graph(num_nodes: int) -> np.ndarray:
    """
    Create ring graph (cycle graph) where nodes form a cycle.

    Args:
        num_nodes: Number of nodes in graph

    Returns:
        Adjacency matrix of shape (num_nodes, num_nodes)
    """
    adj = create_line_graph(num_nodes)

    # Connect last node to first node to form cycle
    if num_nodes > 2:
        adj[0, num_nodes - 1] = 1
        adj[num_nodes - 1, 0] = 1

    return adj
