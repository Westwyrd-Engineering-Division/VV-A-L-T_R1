# V.V.A.L.T

**Vantage-Vector Autonomous Logic Transformer**

A deterministic, bounded logic reasoning system that operates through multi-perspective vector frame analysis.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

V.V.A.L.T transforms single-perspective reasoning into multi-vantage analysis, enabling sophisticated logic operations while maintaining strict safety guarantees.

### Core Principle

Traditional reasoning systems operate from a single perspective. V.V.A.L.T analyzes problems through **five distinct perspective frames**, each capturing different aspects of the input:

- **Semantic**: Meaning-based representation
- **Structural**: Pattern-based representation
- **Causal**: Cause-effect relationships
- **Relational**: Connection-based representation
- **Graph**: Topology-aligned representation

### Safety Guarantees

V.V.A.L.T is designed with safety as a first-class concern:

- ✅ **No autonomous loops** — Single-pass execution only
- ✅ **Deterministic** — Same input always produces same output
- ✅ **Bounded computation** — Fixed computational complexity
- ✅ **Operator-controlled** — No self-directed behavior
- ✅ **Full interpretability** — Complete reasoning trace visibility

## Installation

### From Source

```bash
git clone https://github.com/VValtDisney/V.V.A.L.T.git
cd V.V.A.L.T
pip install -e .
```

### Requirements

- Python 3.7+
- NumPy 1.20.0+

## Quick Start

```python
import numpy as np
from vvalt import VVALT

# Initialize V.V.A.L.T
vvalt = VVALT(
    input_dim=10,    # Input vector dimension
    frame_dim=8,     # Frame representation dimension
    task_dim=5,      # Task vector dimension
    seed=42          # Random seed for reproducibility
)

# Create input and task vectors
x = np.random.randn(10)      # Input data
task = np.random.randn(5)    # Task description

# Run reasoning
output, trace = vvalt(x, task, return_trace=True)

print("Output:", output)
print("\nReasoning trace available:", trace is not None)
```

## Architecture

V.V.A.L.T consists of seven core components working in sequence:

```
Input → VectorFrameEncoder → VantageSelector → GraphTopologyProjector
    → MultiPerspectiveAttention → LogicRefinementUnit
    → ConsistencyVerifier → Output
```

### Component Descriptions

#### 1. VectorFrameEncoder
Encodes input into five perspective frames using different mathematical transformations:
- Semantic: tanh activation for meaning capture
- Structural: sin/cos encoding for patterns
- Causal: gradient-like operation for cause-effect
- Relational: L2 normalization for relationships
- Graph: topology-aware encoding

#### 2. VantageSelector
Task-conditioned frame weighting using softmax normalization. Determines which perspectives are most relevant for the current task.

#### 3. GraphTopologyProjector
Aligns vector representations with graph structure using graph convolution operations. Optional component when graph topology is available.

#### 4. MultiPerspectiveAttention
Fuses multiple perspective frames using scaled dot-product attention, creating a unified representation.

#### 5. LogicRefinementUnit
Single-pass bounded refinement using two-layer network with tanh activation. Ensures output stays in [-1, 1] range.

#### 6. ConsistencyVerifier
Safety validation detecting NaN/Inf values and clipping outputs to safe range [-10, 10]. Provides fail-safe behavior.

#### 7. InterpretabilityProjector
Generates complete reasoning traces for operator visibility and control.

## Usage Examples

### Basic Usage

```python
from vvalt import VVALT
import numpy as np

# Initialize
vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5, seed=42)

# Run inference
x = np.random.randn(10)
task = np.random.randn(5)
output, _ = vvalt(x, task)
```

### With Graph Topology

```python
from vvalt import VVALT
from vvalt.utils import create_star_graph
import numpy as np

# Initialize
vvalt = VVALT(input_dim=10, frame_dim=8, task_dim=5)

# Create graph structure
graph = create_star_graph(num_nodes=8)

# Run with graph awareness
x = np.random.randn(10)
task = np.random.randn(5)
output, _ = vvalt(x, task, graph_adj=graph)
```

### Get Reasoning Explanation

```python
# Get human-readable explanation
explanation = vvalt.explain(x, task)
print(explanation)
```

Output:
```
=== V.V.A.L.T Reasoning Trace ===

Input: norm=3.1623

Task-Conditioned Frame Weights:
  semantic: 0.2341
  structural: 0.1876
  causal: 0.2103
  relational: 0.1891
  graph: 0.1789

Frame Diversity: 0.3456

Transformation Magnitudes:
  input_to_attention_change: 0.4521
  attention_to_refined_change: 0.1234
  refined_to_final_change: 0.0123
  total_change: 0.5123

Final Output: norm=2.8456, mean=0.0234, std=0.4567
```

### Safety Verification

```python
# Verify determinism
is_deterministic = vvalt.verify_determinism(x, task, num_trials=10)
print(f"Deterministic: {is_deterministic}")

# Get safety report
safety_report = vvalt.get_safety_report(x, task)
print(f"Output is safe: {safety_report['bounded']}")
```

### Batch Processing

```python
# Process multiple inputs
batch_size = 100
X = np.random.randn(batch_size, 10)
task = np.random.randn(5)

outputs = vvalt.batch_forward(X, task)
print(f"Processed {batch_size} inputs -> shape: {outputs.shape}")
```

### Full Reasoning Trace

```python
# Get complete interpretability trace
output, trace = vvalt(x, task, return_trace=True)

# Inspect trace components
print("Available trace data:")
print("- Input analysis:", trace['input'].keys())
print("- Frame encodings:", trace['encoded_frames'].keys())
print("- Task weights:", trace['task_weights'])
print("- Transformations:", trace['transformations'])
print("- Final output:", trace['final_output'].keys())
```

## Running Examples

The repository includes comprehensive examples:

```bash
# Run demonstration suite
python examples/demo.py
```

This will run demonstrations of:
- Basic usage
- Graph topology integration
- Task-conditioned weighting
- Batch processing
- Interpretability features
- Safety verification
- Full reasoning traces
- Different graph types

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_vvalt.py

# Run with coverage
python -m pytest tests/ --cov=vvalt --cov-report=html
```

The test suite verifies:
- ✅ All component functionality
- ✅ Deterministic behavior
- ✅ Bounded outputs
- ✅ Safety guarantees
- ✅ Graph integration
- ✅ Interpretability features

## API Reference

### Main Class: `VVALT`

```python
VVALT(
    input_dim: int,
    frame_dim: int,
    task_dim: int,
    hidden_dim: Optional[int] = None,
    seed: int = 42
)
```

**Methods:**

- `forward(x, task_vector, graph_adj=None, return_trace=False)` - Main reasoning forward pass
- `__call__(x, task_vector, graph_adj=None, return_trace=False)` - Callable interface
- `verify_determinism(x, task_vector, graph_adj=None, num_trials=5)` - Verify deterministic behavior
- `get_safety_report(x, task_vector, graph_adj=None)` - Generate safety analysis
- `explain(x, task_vector, graph_adj=None)` - Human-readable explanation
- `batch_forward(X, task_vector, graph_adj=None)` - Process batch of inputs

### Utility Functions

```python
from vvalt.utils import (
    create_random_graph,
    create_line_graph,
    create_star_graph,
    create_complete_graph,
    create_ring_graph,
)
```

## Design Philosophy

V.V.A.L.T is built on several key principles:

1. **Safety First**: All operations are bounded, deterministic, and verifiable
2. **Interpretability**: Complete reasoning trace visibility for operator control
3. **Multi-Perspective**: Different viewpoints capture complementary information
4. **Task-Conditioned**: Adapt perspective importance based on task requirements
5. **Graph-Aware**: Optional topology constraints for structured reasoning

## Performance Characteristics

- **Time Complexity**: O(d²) where d is frame_dim (dominated by attention)
- **Space Complexity**: O(d²) for weight matrices
- **Deterministic**: Same input → same output (with same seed)
- **Single-Pass**: No iterations or loops
- **Bounded**: All outputs clipped to safe ranges

## Limitations

- Fixed computational complexity (no adaptive depth)
- Requires task vector specification
- Graph topology must fit in memory if used
- Single-pass only (no iterative refinement)

These limitations are **intentional design choices** for safety and predictability.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass
2. Code follows existing style
3. New features include tests
4. Documentation is updated
5. Safety guarantees are maintained

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use V.V.A.L.T in your research, please cite:

```bibtex
@software{vvalt2025,
  title={V.V.A.L.T: Vantage-Vector Autonomous Logic Transformer},
  author={V.V.A.L.T Contributors},
  year={2025},
  url={https://github.com/VValtDisney/V.V.A.L.T}
}
```

## Acknowledgments

V.V.A.L.T builds on ideas from:
- Multi-head attention mechanisms
- Graph neural networks
- Bounded logic systems
- Interpretable AI research

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review examples and tests

---

**Built with safety, interpretability, and determinism as core principles.**
