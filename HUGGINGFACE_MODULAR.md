# HuggingFace Modular V.V.A.L.T

This document describes the HuggingFace-compatible wrapper around the modular PyTorch implementation of V.V.A.L.T.

## Overview

The `modeling_vvalt_modular` module provides drop-in HuggingFace-style models that use the modular `VVALTModular` core instead of the NumPy-based implementation. This combines the benefits of:

- **Modular PyTorch Architecture**: Inspectable components, gradient support, training-ready
- **HuggingFace Ecosystem**: Compatible with transformers library, standard APIs
- **Three Forward Modes**: fast / fine / diagnostic for performance vs. diagnostics
- **Event Hooks**: Governance and safety monitoring at every stage
- **Full Tracing**: Micro-level execution traces with component timing

## Available Models

### VVALTModelHF

Base V.V.A.L.T model that outputs raw hidden states.

```python
from vvalt import VVALTConfig, VVALTModelHF
import torch

config = VVALTConfig(
    input_dim=128,
    frame_dim=64,
    task_dim=32,
    hidden_dim=128
)

model = VVALTModelHF(config)

inputs_embeds = torch.randn(4, 128)
task_vector = torch.randn(4, 32)

# Fast mode (no tracing)
output = model(inputs_embeds, task_vector, forward_mode="fast")
print(output.last_hidden_state.shape)  # (4, 64)

# Diagnostic mode (full tracing)
output = model(inputs_embeds, task_vector, forward_mode="diagnostic", return_trace=True)
print(f"Trace: {output.trace.total_time_ms:.2f}ms")
```

### VVALTForSequenceClassification

V.V.A.L.T with a classification head for sequence classification tasks.

```python
from vvalt import VVALTConfig, VVALTForSequenceClassificationModular
import torch

config = VVALTConfig(
    input_dim=128,
    frame_dim=64,
    task_dim=32,
    hidden_dim=128,
    num_labels=3  # 3-class classification
)

model = VVALTForSequenceClassificationModular(config)

inputs_embeds = torch.randn(4, 128)
task_vector = torch.randn(4, 32)
labels = torch.randint(0, 3, (4,))

# Training
output = model(inputs_embeds, task_vector, labels=labels)
loss = output.loss
logits = output.logits

# Inference
with torch.no_grad():
    output = model(inputs_embeds, task_vector)
    predictions = output.logits.argmax(dim=-1)
```

### VVALTForRegression

V.V.A.L.T with a regression head for continuous value prediction.

```python
from vvalt import VVALTConfig, VVALTForRegressionModular
import torch

config = VVALTConfig(
    input_dim=128,
    frame_dim=64,
    task_dim=32,
    hidden_dim=128,
    num_labels=1  # Single regression target
)

model = VVALTForRegressionModular(config)

inputs_embeds = torch.randn(4, 128)
task_vector = torch.randn(4, 32)
labels = torch.randn(4, 1)

# Training
output = model(inputs_embeds, task_vector, labels=labels)
loss = output.loss  # MSE loss
predictions = output.logits

# Inference
with torch.no_grad():
    output = model(inputs_embeds, task_vector)
    predictions = output.logits
```

## Configuration

Use `VVALTConfig` to configure models:

```python
from vvalt import VVALTConfig

config = VVALTConfig(
    # Core dimensions
    input_dim=128,        # Input embedding dimension
    frame_dim=64,         # Frame representation dimension
    task_dim=32,          # Task vector dimension
    hidden_dim=128,       # Hidden layer dimension

    # Task-specific
    num_labels=3,         # Number of classes (classification) or targets (regression)

    # Regularization
    classifier_dropout=0.1,  # Dropout for classification head
    regressor_dropout=0.1,   # Dropout for regression head

    # Safety
    safe_bounds=(-10.0, 10.0),  # Output bounds

    # Reproducibility
    seed=42,              # Random seed for determinism
)
```

## Forward Modes

All models support three forward modes:

### Fast Mode (Default)

Optimized execution path with no tracing overhead. Use for production inference and training.

```python
output = model(inputs_embeds, task_vector, forward_mode="fast")
```

### Fine Mode

Basic tracing with frame-level detail. Minimal overhead.

```python
output = model(inputs_embeds, task_vector, forward_mode="fine", return_trace=True)
trace = output.trace
```

### Diagnostic Mode

Full micro-level tracing with component timing and event hooks. Use for analysis, debugging, and safety audits.

```python
output = model(
    inputs_embeds,
    task_vector,
    forward_mode="diagnostic",
    return_trace=True,
    output_hidden_states=True,
    output_attentions=True
)

trace = output.trace
print(f"Total time: {trace.total_time_ms:.2f}ms")
print(f"Component times: {trace.component_times_ms}")
print(f"Is safe: {trace.is_safe}")
```

## Output Format

Models return `VVALTModularOutput` with:

```python
@dataclass
class VVALTModularOutput:
    last_hidden_state: torch.Tensor      # Final output (batch_size, frame_dim)
    hidden_states: Tuple[torch.Tensor]   # Frame representations (if requested)
    attentions: Tuple[torch.Tensor]      # Attention weights (if requested)
    trace: VVALTDetailedTrace            # Execution trace (if requested)
    loss: torch.Tensor                   # Training loss (if labels provided)
    logits: torch.Tensor                 # Task predictions (classification/regression)
```

Access fields directly:

```python
output = model(inputs_embeds, task_vector, labels=labels)

# Direct access
loss = output.loss
logits = output.logits
hidden_state = output.last_hidden_state

# Dict-style access (HuggingFace compatible)
loss = output['loss']
logits = output['logits']
```

## Event Hooks

Register hooks for governance and safety monitoring:

```python
from vvalt import VVALTForSequenceClassificationModular, EventHookType

model = VVALTForSequenceClassificationModular(config)

# Define hook
def safety_monitor(data):
    print(f"[SAFETY] Verification complete")
    print(f"  Has NaN/Inf: {data['trace'].has_nan_inf}")

# Register hook
model.register_event_hook(EventHookType.POST_VERIFICATION, safety_monitor)

# Hook triggers during forward (diagnostic mode only)
output = model(inputs_embeds, task_vector, forward_mode="diagnostic")
```

Available hook types:

- `PRE_ENCODING` - Before frame encoding
- `POST_ENCODING` - After frame encoding
- `PRE_ATTENTION` - Before attention computation
- `POST_ATTENTION` - After attention fusion
- `PRE_REFINEMENT` - Before logic refinement
- `POST_REFINEMENT` - After refinement
- `PRE_VERIFICATION` - Before safety verification
- `POST_VERIFICATION` - After verification

## Component Access

Access internal components for inspection or fine-tuning:

```python
# Get all components
components = model.vvalt.get_all_components()
for name, component in components.items():
    print(f"{name}: {component.__class__.__name__}")

# Get specific component
frame_encoder = model.vvalt.get_component('frame_encoder')
vantage_selector = model.vvalt.get_component('vantage_selector')
attention = model.vvalt.get_component('attention')

# Access frame encoder submodules
semantic_encoder = frame_encoder.encoders['semantic']
structural_encoder = frame_encoder.encoders['structural']
```

## Training

Standard PyTorch training loop:

```python
import torch.optim as optim

model = VVALTForSequenceClassificationModular(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        output = model(
            inputs_embeds=batch['inputs_embeds'],
            task_vector=batch['task_vector'],
            labels=batch['labels'],
            forward_mode="fast"  # Use fast mode for training
        )

        loss = output.loss
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
```

## Visualization

Visualize traces using the integrated visualization toolkit:

```python
from vvalt import VVALTVisualizer

# Get trace in diagnostic mode
output = model(
    inputs_embeds,
    task_vector,
    forward_mode="diagnostic",
    return_trace=True
)

# Visualize
visualizer = VVALTVisualizer()
visualizer.plot_frame_activations(output.trace.frame_traces, "frames.png")
visualizer.plot_component_timing(output.trace, "timing.png")
visualizer.export_trace_json(output.trace, "trace.json")
```

## Comparison: NumPy vs. Modular Backend

| Feature | `modeling_vvalt.py` (NumPy) | `modeling_vvalt_modular.py` (PyTorch) |
|---------|---------------------------|-------------------------------------|
| Backend | NumPy core | Modular PyTorch |
| Training | Manual gradients | Native PyTorch autograd |
| Inspection | Limited | Full component access |
| Forward Modes | Single | Three (fast/fine/diagnostic) |
| Tracing | Basic | Micro-level with timing |
| Event Hooks | None | 8 hook types |
| Visualization | External | Built-in |
| Component Access | None | get_component() API |
| Determinism | Yes | Yes (verified) |
| Safety Guarantees | Yes | Yes (maintained) |

## Integration with Transformers

If you have `transformers` installed, the models inherit from `PreTrainedModel`:

```python
from transformers import PreTrainedModel

# Models are PreTrainedModel subclasses
assert isinstance(model, PreTrainedModel)

# Use standard transformers methods
model.save_pretrained("./my_vvalt_model")
model = VVALTModelHF.from_pretrained("./my_vvalt_model")
```

## Safety Guarantees

All V.V.A.L.T safety guarantees are maintained:

1. **Determinism**: Same input → same output (verified in diagnostic mode)
2. **Bounded Outputs**: All outputs clipped to `safe_bounds` (default: [-10, 10])
3. **Single-Pass Execution**: No loops, fixed complexity
4. **NaN/Inf Detection**: Automatic detection and handling
5. **Operator Control**: No autonomous loops, full human control

Verify in diagnostic mode:

```python
output = model(inputs_embeds, task_vector, forward_mode="diagnostic", return_trace=True)

print(f"Deterministic: {output.trace.deterministic_check_passed}")
print(f"Bounds OK: {output.trace.bounds_check_passed}")
print(f"Is safe: {output.trace.is_safe}")
print(f"Has NaN/Inf: {output.trace.verification_trace.has_nan_inf}")
```

## Examples

See `examples/hf_modular_demo.py` for comprehensive demonstrations of:

1. Base model usage
2. Classification tasks
3. Regression tasks
4. Event hooks
5. Training loops
6. Trace visualization
7. Component access

Run the demo:

```bash
python examples/hf_modular_demo.py
```

## Requirements

The modular HuggingFace implementation requires:

```bash
pip install torch>=1.10.0
pip install transformers>=4.0.0  # Optional, for PreTrainedModel inheritance
```

Or install all requirements:

```bash
pip install -r requirements-transformers.txt
```

## API Reference

### VVALTModelHF

```python
class VVALTModelHF(PreTrainedModel):
    def __init__(self, config: VVALTConfig)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        forward_mode: str = "fast",
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VVALTModularOutput

    def register_event_hook(self, hook_type: EventHookType, callback)
    def get_component(self, name: str) -> nn.Module
    def get_all_components(self) -> Dict[str, nn.Module]
```

### VVALTForSequenceClassification

```python
class VVALTForSequenceClassification(PreTrainedModel):
    def __init__(self, config: VVALTConfig)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        forward_mode: str = "fast",
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VVALTModularOutput

    def register_event_hook(self, hook_type: EventHookType, callback)
```

### VVALTForRegression

```python
class VVALTForRegression(PreTrainedModel):
    def __init__(self, config: VVALTConfig)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        forward_mode: str = "fast",
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VVALTModularOutput

    def register_event_hook(self, hook_type: EventHookType, callback)
```

## Best Practices

1. **Use Fast Mode for Training**: `forward_mode="fast"` for optimal performance
2. **Use Diagnostic Mode for Debugging**: Enable tracing and hooks when investigating issues
3. **Register Hooks Sparingly**: Hooks add overhead, only use in diagnostic mode
4. **Check Traces in Development**: Verify safety and determinism during development
5. **Profile with Component Timing**: Use trace timing to identify bottlenecks

## Troubleshooting

### Import Errors

If you get import errors:

```python
from vvalt import is_transformers_modular_available

if not is_transformers_modular_available():
    print("Install PyTorch: pip install torch>=1.10.0")
```

### Performance Issues

Use fast mode for production:

```python
# Slow (full tracing)
output = model(x, task_vector, forward_mode="diagnostic", return_trace=True)

# Fast (no tracing)
output = model(x, task_vector, forward_mode="fast")
```

### NaN/Inf Errors

Check trace for safety violations:

```python
output = model(x, task_vector, forward_mode="diagnostic", return_trace=True)
if output.trace.verification_trace.has_nan_inf:
    print("Safety violation detected!")
    print(f"Clipped values: {output.trace.verification_trace.clipped_values}")
```

## Migration from NumPy Backend

To migrate from `modeling_vvalt.py` to `modeling_vvalt_modular.py`:

```python
# Old (NumPy backend)
from vvalt import VVALTModel, VVALTForSequenceClassification

# New (Modular PyTorch backend)
from vvalt import VVALTModelHF, VVALTForSequenceClassificationModular

# API is mostly compatible
# Just replace class names and optionally add forward_mode parameter
```

Key differences:

1. Add `forward_mode` parameter for fast/fine/diagnostic modes
2. Use `return_trace=True` to get detailed traces
3. Use `register_event_hook()` for governance monitoring
4. Access components via `model.vvalt.get_component()`

## Conclusion

The HuggingFace modular V.V.A.L.T provides a production-ready, inspectable, and training-ready implementation that maintains all safety guarantees while offering unprecedented visibility into the model's internal operations.
