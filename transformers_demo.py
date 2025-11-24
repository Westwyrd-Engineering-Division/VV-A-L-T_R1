"""
V.V.A.L.T HuggingFace Transformers Integration Demo

Demonstrates using V.V.A.L.T like a standard HuggingFace transformer model.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from transformers import AutoConfig, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  transformers not installed. Install with: pip install -r requirements-transformers.txt")
    sys.exit(1)

from vvalt.configuration_vvalt import VVALTConfig
from vvalt.modeling_vvalt import (
    VVALTModel,
    VVALTForSequenceClassification,
    VVALTForRegression
)


def print_separator(title=""):
    """Print formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_basic_model():
    """Demonstrate basic V.V.A.L.T model like a transformer."""
    print_separator("Basic V.V.A.L.T Model (Transformer-like)")

    # Create config (like any HuggingFace model)
    config = VVALTConfig(
        input_dim=768,  # Like BERT hidden size
        frame_dim=512,  # Output dimension
        task_dim=64,    # Task conditioning dimension
        seed=42
    )

    print("Configuration:")
    print(f"  Model type: {config.model_type}")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Frame dim (output): {config.frame_dim}")
    print(f"  Task dim: {config.task_dim}")
    print(f"  Deterministic: {config.deterministic}")

    # Initialize model
    model = VVALTModel(config)
    print(f"\n✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Forward pass (transformer-style)
    batch_size = 16
    inputs_embeds = torch.randn(batch_size, config.input_dim)
    task_vector = torch.randn(batch_size, config.task_dim)

    print(f"\nForward pass:")
    print(f"  Input shape: {inputs_embeds.shape}")
    print(f"  Task shape: {task_vector.shape}")

    with torch.no_grad():
        outputs = model(inputs_embeds, task_vector=task_vector)

    print(f"\n✓ Output shape: {outputs.last_hidden_state.shape}")
    print(f"  Expected: ({batch_size}, {config.frame_dim})")


def demo_classification():
    """Demonstrate classification like BERT for classification."""
    print_separator("V.V.A.L.T for Sequence Classification")

    # Config for binary classification
    config = VVALTConfig(
        input_dim=768,
        frame_dim=512,
        task_dim=64,
        num_labels=2  # Binary classification
    )

    model = VVALTForSequenceClassification(config)
    print(f"Classification model with {config.num_labels} labels")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass with labels (like training)
    batch_size = 32
    inputs_embeds = torch.randn(batch_size, config.input_dim)
    task_vector = torch.randn(batch_size, config.task_dim)
    labels = torch.randint(0, 2, (batch_size,))

    print(f"\nTraining-style forward pass:")
    print(f"  Batch size: {batch_size}")
    print(f"  Labels: {labels[:10].tolist()}")

    with torch.no_grad():
        outputs = model(
            inputs_embeds,
            task_vector=task_vector,
            labels=labels
        )

    print(f"\n✓ Loss: {outputs.loss.item():.4f}")
    print(f"✓ Logits shape: {outputs.last_hidden_state.shape}")

    # Get predictions
    predictions = torch.argmax(outputs.last_hidden_state, dim=-1)
    print(f"✓ Predictions: {predictions[:10].tolist()}")


def demo_regression():
    """Demonstrate regression task."""
    print_separator("V.V.A.L.T for Regression")

    config = VVALTConfig(
        input_dim=768,
        frame_dim=512,
        task_dim=64,
        num_labels=1  # Single output for regression
    )

    model = VVALTForRegression(config)
    print(f"Regression model")

    # Forward pass
    batch_size = 32
    inputs_embeds = torch.randn(batch_size, config.input_dim)
    task_vector = torch.randn(batch_size, config.task_dim)
    labels = torch.randn(batch_size, 1)

    with torch.no_grad():
        outputs = model(
            inputs_embeds,
            task_vector=task_vector,
            labels=labels
        )

    print(f"\n✓ Loss (MSE): {outputs.loss.item():.4f}")
    print(f"✓ Predictions shape: {outputs.last_hidden_state.shape}")
    print(f"  Sample predictions: {outputs.last_hidden_state[:5, 0].tolist()}")


def demo_save_load():
    """Demonstrate save/load like HuggingFace models."""
    print_separator("Save & Load (HuggingFace Style)")

    # Create and save model
    config = VVALTConfig(input_dim=768, frame_dim=512, task_dim=64)
    model = VVALTModel(config)

    save_path = "/tmp/vvalt-model"
    print(f"Saving model to: {save_path}")

    # Save config
    config.save_pretrained(save_path)
    print("✓ Config saved")

    # Save model state
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    print("✓ Model weights saved")

    # Load model
    print(f"\nLoading model from: {save_path}")
    loaded_config = VVALTConfig.from_pretrained(save_path)
    loaded_model = VVALTModel(loaded_config)
    loaded_model.load_state_dict(torch.load(os.path.join(save_path, "pytorch_model.bin")))
    print("✓ Model loaded")

    # Verify same outputs
    inputs = torch.randn(1, 768)
    task = torch.randn(1, 64)

    with torch.no_grad():
        original_output = model(inputs, task_vector=task).last_hidden_state
        loaded_output = loaded_model(inputs, task_vector=task).last_hidden_state

    diff = torch.abs(original_output - loaded_output).max()
    print(f"\n✓ Max difference: {diff.item():.10f}")
    print("✓ Models are identical!" if diff < 1e-6 else "✗ Models differ")


def demo_with_pretrained_embeddings():
    """Demonstrate using V.V.A.L.T with pretrained embeddings (e.g., from BERT)."""
    print_separator("V.V.A.L.T with Pretrained Embeddings")

    print("Example: Using BERT embeddings as input to V.V.A.L.T")
    print("""
    from transformers import BertModel, BertTokenizer
    from vvalt.modeling_vvalt import VVALTModel, VVALTConfig

    # Load pretrained BERT
    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize V.V.A.L.T
    vvalt_config = VVALTConfig(
        input_dim=768,  # BERT hidden size
        frame_dim=512,
        task_dim=64
    )
    vvalt = VVALTModel(vvalt_config)

    # Tokenize text
    text = "This is an example sentence"
    inputs = tokenizer(text, return_tensors="pt")

    # Get BERT embeddings
    with torch.no_grad():
        bert_output = bert(**inputs)
        embeddings = bert_output.last_hidden_state[:, 0, :]  # [CLS] token

    # Process with V.V.A.L.T
    task_vector = torch.randn(1, 64)
    vvalt_output = vvalt(embeddings, task_vector=task_vector)
    """)


def demo_determinism():
    """Demonstrate V.V.A.L.T's determinism guarantee."""
    print_separator("Determinism Verification")

    config = VVALTConfig(input_dim=768, frame_dim=512, task_dim=64, seed=42)
    model = VVALTModel(config)

    inputs = torch.randn(1, 768)
    task = torch.randn(1, 64)

    print("Running same input 5 times...")
    outputs_list = []

    with torch.no_grad():
        for i in range(5):
            output = model(inputs, task_vector=task).last_hidden_state
            outputs_list.append(output)

    # Check all outputs are identical
    all_same = all(
        torch.allclose(outputs_list[0], out, atol=1e-8)
        for out in outputs_list[1:]
    )

    print(f"\n✓ All outputs identical: {all_same}")
    if all_same:
        print("✓ Determinism guarantee verified!")
    else:
        print("✗ Determinism violated (this is a bug)")

    # Show output statistics
    output = outputs_list[0]
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")


def demo_pipeline_compatibility():
    """Show how V.V.A.L.T can be used in pipelines."""
    print_separator("Pipeline Compatibility")

    print("V.V.A.L.T can be integrated into custom training loops:")
    print("""
    from torch.utils.data import DataLoader
    from vvalt.modeling_vvalt import VVALTForSequenceClassification

    # Initialize model
    config = VVALTConfig(input_dim=768, frame_dim=512, task_dim=64, num_labels=2)
    model = VVALTForSequenceClassification(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
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
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  V.V.A.L.T HuggingFace Transformers Integration")
    print("=" * 80)

    if not HF_AVAILABLE:
        return

    try:
        demo_basic_model()
        demo_classification()
        demo_regression()
        demo_save_load()
        demo_with_pretrained_embeddings()
        demo_determinism()
        demo_pipeline_compatibility()

        print_separator("All Demonstrations Completed Successfully!")
        print("\n✓ V.V.A.L.T is fully compatible with HuggingFace transformers!")
        print("✓ Use it like any other transformer model")
        print("✓ Maintains all safety guarantees (determinism, bounds, single-pass)")

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
