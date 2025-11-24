"""
V.V.A.L.T Modular HuggingFace Wrapper

HuggingFace-compatible wrapper around the modular PyTorch V.V.A.L.T implementation.
Replaces NumPy-core with pure PyTorch for efficient training and inference.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .configuration_vvalt import VVALTConfig
from .torch_modules.vvalt_modular import (
    VVALTModular,
    ForwardMode,
    VVALTDetailedTrace
)

# Import HuggingFace components
try:
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import ModelOutput
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = nn.Module
    ModelOutput = object


@dataclass
class VVALTOutput(ModelOutput if HF_AVAILABLE else object):
    """
    Output class for V.V.A.L.T modular models.

    Args:
        last_hidden_state: Final output tensor of shape (batch_size, frame_dim)
        hidden_states: Tuple of all frame representations (if output_hidden_states=True)
        attentions: Attention weights (if output_attentions=True)
        trace: Full interpretability trace (if return_trace=True)
        loss: Loss value (if labels provided)
    """
    last_hidden_state: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    trace: Optional[Dict[str, Any]] = None
    loss: Optional[torch.Tensor] = None


class VVALTPreTrainedModelModular(PreTrainedModel if HF_AVAILABLE else nn.Module):
    """
    An abstract class to handle weights initialization and pretrained model downloading.
    Compatible with HuggingFace transformers ecosystem.
    """
    config_class = VVALTConfig
    base_model_prefix = "vvalt_modular"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_load_unexpected = []

    def _init_weights(self, module):
        """Initialize the weights (handled by VVALTModular internally)."""
        pass  # Weights are initialized in VVALTModular


class VVALTModelHF(VVALTPreTrainedModelModular):
    """
    HuggingFace-style wrapper around the modular PyTorch V.V.A.L.T implementation.

    This model replaces the NumPy-based core with pure PyTorch modules for
    efficient training, inference, and gradient-based optimization.

    Example:
        ```python
        from vvalt import VVALTConfig, VVALTModelHF
        import torch

        # Initialize configuration
        config = VVALTConfig(
            input_dim=768,
            frame_dim=512,
            task_dim=64
        )

        # Create model
        model = VVALTModelHF(config)

        # Forward pass
        inputs = torch.randn(32, 768)  # batch_size=32
        task = torch.randn(32, 64)
        outputs = model(
            inputs_embeds=inputs,
            task_vector=task,
            forward_mode="fast"
        )

        print(outputs.last_hidden_state.shape)  # (32, 512)
        ```
    """

    config_class = VVALTConfig

    def __init__(self, config: VVALTConfig):
        """
        Initialize V.V.A.L.T modular model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.config = config

        # Core modular PyTorch model
        self.core = VVALTModular(
            input_dim=config.input_dim,
            frame_dim=config.frame_dim,
            task_dim=config.task_dim,
            hidden_dim=config.hidden_dim,
            runtime_config=config.runtime,
            seed=config.seed
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        forward_mode: str = "fast",
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
    ) -> VVALTOutput:
        """
        Forward pass through V.V.A.L.T modular model.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, input_dim) or (input_dim,)
            task_vector: Task conditioning vector of shape (batch_size, task_dim) or (task_dim,)
            graph_adj: Optional graph adjacency matrix (frame_dim, frame_dim)
            labels: Optional labels for computing loss (batch_size,)
            return_trace: Whether to return detailed interpretability trace
            forward_mode: Execution mode - "fast", "fine", or "diagnostic"
            output_hidden_states: Whether to return all frame representations
            output_attentions: Whether to return attention weights
            return_dict: Whether to return VVALTOutput

        Returns:
            VVALTOutput with last_hidden_state and optional trace/loss
        """
        # Convert forward_mode string to enum
        mode_map = {
            "fast": ForwardMode.FAST,
            "fine": ForwardMode.FINE,
            "diagnostic": ForwardMode.DIAGNOSTIC,
        }
        mode = mode_map.get(forward_mode, ForwardMode.FAST)

        # Enable trace if requested via output flags
        should_trace = return_trace or output_hidden_states or output_attentions

        # Forward through core modular model
        output, trace = self.core(
            inputs_embeds,
            task_vector,
            graph_adj=graph_adj,
            mode=mode,
            return_trace=should_trace
        )

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output, labels)

        # Extract hidden states and attentions from trace
        hidden_states = None
        attentions = None
        if trace:
            if output_hidden_states and 'frames' in trace:
                # Return all perspective frames
                hidden_states = (trace['frames'],)
            if output_attentions and 'attention_weights' in trace:
                # Return attention weights
                attentions = (trace['attention_weights'],)

        # Return output
        if not return_dict:
            return (output, trace, loss)

        return VVALTOutput(
            last_hidden_state=output,
            hidden_states=hidden_states,
            attentions=attentions,
            trace=trace if return_trace else None,
            loss=loss
        )

    def get_input_embeddings(self):
        """Return input embeddings (for HuggingFace compatibility)."""
        return None  # V.V.A.L.T operates on pre-computed embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings (for HuggingFace compatibility)."""
        pass  # V.V.A.L.T doesn't have learnable input embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings (not applicable for V.V.A.L.T)."""
        raise NotImplementedError("V.V.A.L.T doesn't use token embeddings")


class VVALTForSequenceClassificationModular(VVALTPreTrainedModelModular):
    """
    V.V.A.L.T Modular Model with sequence classification head.

    Adds a linear classification head on top of the modular V.V.A.L.T model
    for fine-tuning on classification tasks.

    Example:
        ```python
        config = VVALTConfig(
            input_dim=768,
            frame_dim=512,
            task_dim=64,
            num_labels=3
        )
        model = VVALTForSequenceClassificationModular(config)

        inputs = torch.randn(16, 768)
        task = torch.randn(16, 64)
        labels = torch.randint(0, 3, (16,))

        outputs = model(inputs, task_vector=task, labels=labels)
        loss = outputs.loss
        logits = outputs.last_hidden_state
        ```
    """

    def __init__(self, config: VVALTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Base V.V.A.L.T modular model
        self.vvalt = VVALTModelHF(config)

        # Classification head
        self.classifier = nn.Linear(config.frame_dim, config.num_labels)

        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        forward_mode: str = "fast",
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
    ) -> VVALTOutput:
        """Forward pass with classification head."""
        # Get V.V.A.L.T outputs
        outputs = self.vvalt(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            forward_mode=forward_mode,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Apply classification head
        pooled_output = outputs.last_hidden_state  # (batch_size, frame_dim)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + (outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return VVALTOutput(
            loss=loss,
            last_hidden_state=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            trace=outputs.trace,
        )


class VVALTForRegressionModular(VVALTPreTrainedModelModular):
    """
    V.V.A.L.T Modular Model for regression tasks.

    Adds a linear regression head on top of the modular V.V.A.L.T model
    for fine-tuning on regression tasks.

    Example:
        ```python
        config = VVALTConfig(
            input_dim=768,
            frame_dim=512,
            task_dim=64,
            num_labels=1
        )
        model = VVALTForRegressionModular(config)

        inputs = torch.randn(16, 768)
        task = torch.randn(16, 64)
        labels = torch.randn(16, 1)

        outputs = model(inputs, task_vector=task, labels=labels)
        ```
    """

    def __init__(self, config: VVALTConfig):
        super().__init__(config)
        self.config = config

        # Base V.V.A.L.T modular model
        self.vvalt = VVALTModelHF(config)

        # Regression head
        self.regressor = nn.Linear(config.frame_dim, config.num_labels)

        # Initialize regressor weights
        self.regressor.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.regressor.bias is not None:
            self.regressor.bias.data.zero_()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        forward_mode: str = "fast",
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
    ) -> VVALTOutput:
        """Forward pass for regression."""
        # Get V.V.A.L.T outputs
        outputs = self.vvalt(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            forward_mode=forward_mode,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Apply regression head
        pooled_output = outputs.last_hidden_state
        predictions = self.regressor(pooled_output)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.squeeze(), labels.squeeze())

        if not return_dict:
            output = (predictions,) + (outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return VVALTOutput(
            loss=loss,
            last_hidden_state=predictions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            trace=outputs.trace,
        )
