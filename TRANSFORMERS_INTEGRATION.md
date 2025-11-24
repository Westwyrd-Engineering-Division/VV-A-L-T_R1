# V.V.A.L.T HuggingFace Transformers Integration

V.V.A.L.T is fully compatible with the HuggingFace Transformers ecosystem, allowing you to use it like any standard transformer model.

## Installation

```bash
# Standard installation (NumPy only)
pip install -e .

# With HuggingFace transformers support
pip install -r requirements-transformers.txt
```

## Quick Start

### Basic Model

```python
from vvalt.configuration_vvalt import VVALTConfig
from vvalt.modeling_vvalt import VVALTModel
import torch

# Create configuration
config = VVALTConfig(
    input_dim=768,    # Like BERT's hidden size
    frame_dim=512,    # Output dimension
    task_dim=64,      # Task conditioning size
)

# Initialize model
model = VVALTModel(config)

# Forward pass
batch_size = 32
inputs = torch.randn(batch_size, 768)
task = torch.randn(batch_size, 64)

outputs = model(inputs, task_vector=task)
print(outputs.last_hidden_state.shape)  # torch.Size([32, 512])
```

### Classification

```python
from vvalt.modeling_vvalt import VVALTForSequenceClassification

# Binary classification
config = VVALTConfig(
    input_dim=768,
    frame_dim=512,
    task_dim=64,
    num_labels=2
)

model = VVALTForSequenceClassification(config)

# Training
inputs = torch.randn(32, 768)
task = torch.randn(32, 64)
labels = torch.randint(0, 2, (32,))

outputs = model(inputs, task_vector=task, labels=labels)
loss = outputs.loss
logits = outputs.last_hidden_state

loss.backward()
```

### Regression

```python
from vvalt.modeling_vvalt import VVALTForRegression

config = VVALTConfig(
    input_dim=768,
    frame_dim=512,
    task_dim=64,
    num_labels=1  # Single output
)

model = VVALTForRegression(config)

inputs = torch.randn(32, 768)
task = torch.randn(32, 64)
labels = torch.randn(32, 1)

outputs = model(inputs, task_vector=task, labels=labels)
```

## Integration with Pretrained Models

### Using BERT Embeddings

```python
from transformers import BertModel, BertTokenizer
from vvalt.modeling_vvalt import VVALTModel, VVALTConfig

# Load BERT
bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize V.V.A.L.T
vvalt_config = VVALTConfig(
    input_dim=768,  # BERT hidden size
    frame_dim=512,
    task_dim=64
)
vvalt = VVALTModel(vvalt_config)

# Process text
text = "V.V.A.L.T is a deterministic logic transformer"
inputs = tokenizer(text, return_tensors="pt")

# Get BERT embeddings
with torch.no_grad():
    bert_output = bert(**inputs)
    embeddings = bert_output.last_hidden_state[:, 0, :]  # [CLS]

# Process with V.V.A.L.T
task_vector = torch.randn(1, 64)
vvalt_output = vvalt(embeddings, task_vector=task_vector)
```

## Save & Load (HuggingFace Style)

```python
# Save
save_path = "./my-vvalt-model"
config.save_pretrained(save_path)
torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")

# Load
from vvalt.configuration_vvalt import VVALTConfig
from vvalt.modeling_vvalt import VVALTModel

config = VVALTConfig.from_pretrained(save_path)
model = VVALTModel(config)
model.load_state_dict(torch.load(f"{save_path}/pytorch_model.bin"))
```

## Training Loop

```python
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Initialize
config = VVALTConfig(input_dim=768, frame_dim=512, task_dim=64, num_labels=2)
model = VVALTForSequenceClassification(config)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training
model.train()
for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(
        inputs_embeds=batch['embeddings'],
        task_vector=batch['task'],
        labels=batch['labels']
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
```

## Key Differences from Standard Transformers

### 1. Task Vector Required

V.V.A.L.T requires a task vector for conditioning:

```python
# Standard transformer
outputs = model(input_ids)

# V.V.A.L.T
outputs = model(inputs_embeds, task_vector=task)
```

### 2. No Tokenization

V.V.A.L.T operates on continuous embeddings, not tokens:

```python
# Use pretrained embeddings from BERT, etc.
inputs_embeds = get_embeddings(text)  # Your embedding function
outputs = vvalt(inputs_embeds, task_vector=task)
```

### 3. Deterministic Guarantee

Same inputs always produce identical outputs:

```python
# Run multiple times - outputs are identical
output1 = model(inputs, task_vector=task)
output2 = model(inputs, task_vector=task)
assert torch.allclose(output1.last_hidden_state, output2.last_hidden_state)
```

## Model Outputs

All V.V.A.L.T models return `VVALTOutput`:

```python
@dataclass
class VVALTOutput:
    last_hidden_state: torch.Tensor  # Main output
    hidden_states: Optional[Tuple[torch.Tensor]]  # Frame representations
    attentions: Optional[Tuple[torch.Tensor]]  # Attention weights
    trace: Optional[Dict]  # Interpretability trace
    loss: Optional[torch.Tensor]  # Loss (if labels provided)
```

## Available Models

| Model | Use Case | Output |
|-------|----------|--------|
| `VVALTModel` | Base model | (batch_size, frame_dim) |
| `VVALTForSequenceClassification` | Classification | (batch_size, num_labels) |
| `VVALTForRegression` | Regression | (batch_size, num_labels) |

## Configuration Options

```python
VVALTConfig(
    input_dim=768,              # Input embedding dimension
    frame_dim=512,              # Output dimension
    task_dim=64,                # Task vector dimension
    hidden_dim=1024,            # Hidden dimension (default: 2*frame_dim)
    num_labels=2,               # Number of output labels
    seed=42,                    # Random seed for determinism
    use_return_dict=True,       # Return ModelOutput
    initializer_range=0.02,     # Weight initialization std
    safe_bounds=(-10.0, 10.0),  # Output safety bounds
)
```

## Safety Guarantees

V.V.A.L.T maintains strict safety guarantees even in the transformer interface:

- ✅ **Deterministic**: Same input → same output
- ✅ **Bounded**: Outputs clipped to safe range
- ✅ **Single-pass**: No iterative loops
- ✅ **Verifiable**: Complete interpretability traces

```python
# Verify determinism
config = VVALTConfig(seed=42)
model1 = VVALTModel(config)
model2 = VVALTModel(config)

inputs = torch.randn(1, 768)
task = torch.randn(1, 64)

out1 = model1(inputs, task_vector=task).last_hidden_state
out2 = model2(inputs, task_vector=task).last_hidden_state

assert torch.allclose(out1, out2, atol=1e-8)  # Passes
```

## Model Hub Integration

V.V.A.L.T models can be uploaded to HuggingFace Hub:

```python
# Save with metadata
config = VVALTConfig(input_dim=768, frame_dim=512, task_dim=64)
model = VVALTModel(config)

save_directory = "./vvalt-base"
config.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# Upload to Hub (requires huggingface-hub)
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=save_directory,
    repo_id="username/vvalt-base",
    repo_type="model"
)
```

## Example: Fine-tuning on Custom Task

```python
import torch
from torch.utils.data import Dataset, DataLoader
from vvalt.modeling_vvalt import VVALTForSequenceClassification
from vvalt.configuration_vvalt import VVALTConfig

class CustomDataset(Dataset):
    def __init__(self, embeddings, tasks, labels):
        self.embeddings = embeddings
        self.tasks = tasks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'task': self.tasks[idx],
            'labels': self.labels[idx]
        }

# Create dataset
embeddings = torch.randn(1000, 768)
tasks = torch.randn(1000, 64)
labels = torch.randint(0, 2, (1000,))

dataset = CustomDataset(embeddings, tasks, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
config = VVALTConfig(input_dim=768, frame_dim=512, task_dim=64, num_labels=2)
model = VVALTForSequenceClassification(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        outputs = model(
            inputs_embeds=batch['embeddings'],
            task_vector=batch['task'],
            labels=batch['labels']
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

## Comparison with Standard Transformers

| Feature | V.V.A.L.T | Standard Transformers |
|---------|-----------|----------------------|
| Deterministic | ✅ Always | ❌ Dropout, etc. |
| Bounded outputs | ✅ Yes | ❌ No |
| Interpretability | ✅ Full traces | ⚠️ Limited |
| Task conditioning | ✅ Required | ❌ Not built-in |
| Single-pass | ✅ Yes | ⚠️ Variable |
| Attention | ✅ Multi-perspective | ✅ Multi-head |
| HuggingFace API | ✅ Compatible | ✅ Native |

## See Also

- [Production Demo](examples/production_demo.py)
- [Transformers Demo](examples/transformers_demo.py)
- [Base Demo](examples/demo.py)
- [Configuration Guide](config.yaml)
- [Engineering Specs](README.md)
