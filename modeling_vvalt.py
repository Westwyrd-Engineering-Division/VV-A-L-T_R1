"""
V.V.A.L.T HuggingFace Transformers Compatible Model

Makes V.V.A.L.T compatible with HuggingFace transformers ecosystem.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Import transformers components
try:
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import ModelOutput
    from transformers.utils import logging
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = nn.Module  # Fallback to PyTorch Module
    ModelOutput = object

from .core import VVALT as BaseVVALT
from .configuration_vvalt import VVALTConfig as VVALTHFConfig


@dataclass
class VVALTOutput(ModelOutput if HF_AVAILABLE else object):
    """
    Output class for V.V.A.L.T models (HuggingFace compatible).

    Args:
        last_hidden_state: Final output vector of shape (batch_size, frame_dim)
        hidden_states: Tuple of all frame representations (if output_hidden_states=True)
        attentions: Attention weights (if output_attentions=True)
        trace: Interpretability trace (if return_trace=True)
    """
    last_hidden_state: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    trace: Optional[Dict[str, Any]] = None
    loss: Optional[torch.Tensor] = None


class VVALTPreTrainedModel(PreTrainedModel if HF_AVAILABLE else nn.Module):
    """
    An abstract class to handle weights initialization and pretrained model downloading.
    Compatible with HuggingFace transformers.
    """
    config_class = VVALTHFConfig
    base_model_prefix = "vvalt"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_load_unexpected = []

    def _init_weights(self, module):
        """Initialize weights (called by HuggingFace)."""
        if isinstance(module, VVALTModel):
            # Weights are initialized by BaseVVALT
            pass

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing (not supported)."""
        raise NotImplementedError("Gradient checkpointing not supported for V.V.A.L.T")


class VVALTModel(VVALTPreTrainedModel):
    """
    V.V.A.L.T Model compatible with HuggingFace transformers API.

    This model can be used with HuggingFace pipelines, training loops,
    and model hub.

    Example:
        ```python
        from transformers import AutoModel
        import torch

        # Initialize model
        config = VVALTHFConfig(
            input_dim=768,
            frame_dim=512,
            task_dim=64
        )
        model = VVALTModel(config)

        # Forward pass
        inputs = torch.randn(32, 768)  # batch_size=32
        task = torch.randn(32, 64)
        outputs = model(inputs, task_vector=task)

        print(outputs.last_hidden_state.shape)  # (32, 512)
        ```
    """

    def __init__(self, config: VVALTHFConfig):
        """
        Initialize V.V.A.L.T model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.config = config

        # Initialize base V.V.A.L.T (NumPy backend)
        self.vvalt_core = BaseVVALT(
            input_dim=config.input_dim,
            frame_dim=config.frame_dim,
            task_dim=config.task_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed
        )

        # Convert to PyTorch parameters for HuggingFace compatibility
        self._register_numpy_as_parameters()

    def _register_numpy_as_parameters(self):
        """Register NumPy arrays as PyTorch parameters."""
        # VectorFrameEncoder parameters
        self.encoder_W_semantic = nn.Parameter(
            torch.from_numpy(self.vvalt_core.encoder.W_semantic).float(),
            requires_grad=False
        )
        self.encoder_W_structural = nn.Parameter(
            torch.from_numpy(self.vvalt_core.encoder.W_structural).float(),
            requires_grad=False
        )
        self.encoder_W_causal = nn.Parameter(
            torch.from_numpy(self.vvalt_core.encoder.W_causal).float(),
            requires_grad=False
        )
        self.encoder_W_relational = nn.Parameter(
            torch.from_numpy(self.vvalt_core.encoder.W_relational).float(),
            requires_grad=False
        )
        self.encoder_W_graph = nn.Parameter(
            torch.from_numpy(self.vvalt_core.encoder.W_graph).float(),
            requires_grad=False
        )

        # VantageSelector parameters
        self.selector_W_task = nn.Parameter(
            torch.from_numpy(self.vvalt_core.selector.W_task).float(),
            requires_grad=False
        )

        # Attention parameters
        self.attention_W_query = nn.Parameter(
            torch.from_numpy(self.vvalt_core.attention.W_query).float(),
            requires_grad=False
        )
        self.attention_W_key = nn.Parameter(
            torch.from_numpy(self.vvalt_core.attention.W_key).float(),
            requires_grad=False
        )
        self.attention_W_value = nn.Parameter(
            torch.from_numpy(self.vvalt_core.attention.W_value).float(),
            requires_grad=False
        )

        # Refiner parameters
        self.refiner_W1 = nn.Parameter(
            torch.from_numpy(self.vvalt_core.refiner.W1).float(),
            requires_grad=False
        )
        self.refiner_W2 = nn.Parameter(
            torch.from_numpy(self.vvalt_core.refiner.W2).float(),
            requires_grad=False
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_trace: Optional[bool] = False,
    ) -> VVALTOutput:
        """
        Forward pass through V.V.A.L.T model.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, input_dim) or (input_dim,)
            task_vector: Task conditioning vector of shape (batch_size, task_dim) or (task_dim,)
            graph_adj: Optional graph adjacency matrix (frame_dim, frame_dim)
            output_hidden_states: Whether to return all frame representations
            output_attentions: Whether to return attention weights
            return_dict: Whether to return ModelOutput
            return_trace: Whether to return interpretability trace

        Returns:
            VVALTOutput with last_hidden_state and optional hidden_states/attentions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Convert torch tensors to numpy
        x_np = inputs_embeds.detach().cpu().numpy()
        task_np = task_vector.detach().cpu().numpy()
        graph_np = graph_adj.detach().cpu().numpy() if graph_adj is not None else None

        # Handle batched vs single inputs
        is_batched = len(x_np.shape) == 2
        if is_batched:
            batch_size = x_np.shape[0]
            outputs_list = []
            traces_list = [] if return_trace else None

            for i in range(batch_size):
                x_single = x_np[i]
                task_single = task_np[i] if len(task_np.shape) == 2 else task_np

                # Forward through core
                output_np, trace = self.vvalt_core.forward(
                    x_single,
                    task_single,
                    graph_np,
                    return_trace=return_trace
                )
                outputs_list.append(output_np)
                if return_trace:
                    traces_list.append(trace)

            # Stack outputs
            final_output_np = np.stack(outputs_list, axis=0)
            trace_combined = traces_list if return_trace else None
        else:
            # Single sample
            final_output_np, trace_combined = self.vvalt_core.forward(
                x_np,
                task_np,
                graph_np,
                return_trace=return_trace
            )
            final_output_np = np.expand_dims(final_output_np, axis=0)

        # Convert back to torch
        last_hidden_state = torch.from_numpy(final_output_np).to(inputs_embeds.device)

        if not return_dict:
            return (last_hidden_state,)

        return VVALTOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,  # Could return frames if needed
            attentions=None,  # Could return attention weights if needed
            trace=trace_combined,
        )

    def get_input_embeddings(self):
        """Return input embeddings (for HuggingFace compatibility)."""
        return None  # V.V.A.L.T doesn't have input embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings (for HuggingFace compatibility)."""
        pass  # V.V.A.L.T doesn't have input embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings (not applicable for V.V.A.L.T)."""
        raise NotImplementedError("V.V.A.L.T doesn't use token embeddings")


class VVALTForSequenceClassification(VVALTPreTrainedModel):
    """
    V.V.A.L.T Model with sequence classification head.

    Example:
        ```python
        config = VVALTHFConfig(input_dim=768, frame_dim=512, task_dim=64, num_labels=2)
        model = VVALTForSequenceClassification(config)

        inputs = torch.randn(32, 768)
        task = torch.randn(32, 64)
        labels = torch.randint(0, 2, (32,))

        outputs = model(inputs, task_vector=task, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        ```
    """

    def __init__(self, config: VVALTHFConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Base V.V.A.L.T model
        self.vvalt = VVALTModel(config)

        # Classification head
        self.classifier = nn.Linear(config.frame_dim, config.num_labels)

        # Initialize weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> VVALTOutput:
        """Forward pass with classification head."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get V.V.A.L.T outputs
        outputs = self.vvalt(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Apply classification head
        pooled_output = outputs.last_hidden_state  # (batch_size, frame_dim)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return VVALTOutput(
            loss=loss,
            last_hidden_state=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VVALTForRegression(VVALTPreTrainedModel):
    """
    V.V.A.L.T Model for regression tasks.

    Example:
        ```python
        config = VVALTHFConfig(input_dim=768, frame_dim=512, task_dim=64, num_labels=1)
        model = VVALTForRegression(config)

        inputs = torch.randn(32, 768)
        task = torch.randn(32, 64)
        labels = torch.randn(32, 1)

        outputs = model(inputs, task_vector=task, labels=labels)
        ```
    """

    def __init__(self, config: VVALTHFConfig):
        super().__init__(config)
        self.config = config

        self.vvalt = VVALTModel(config)
        self.regressor = nn.Linear(config.frame_dim, config.num_labels)

        # Initialize weights
        self.regressor.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.regressor.bias is not None:
            self.regressor.bias.data.zero_()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        task_vector: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> VVALTOutput:
        """Forward pass for regression."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vvalt(
            inputs_embeds=inputs_embeds,
            task_vector=task_vector,
            graph_adj=graph_adj,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        pooled_output = outputs.last_hidden_state
        predictions = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.squeeze(), labels.squeeze())

        if not return_dict:
            output = (predictions,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return VVALTOutput(
            loss=loss,
            last_hidden_state=predictions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
