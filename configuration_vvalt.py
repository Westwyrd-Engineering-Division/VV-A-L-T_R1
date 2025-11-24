"""
V.V.A.L.T Configuration for HuggingFace Transformers

Configuration class compatible with transformers.PretrainedConfig
"""

from typing import Dict, Any

try:
    from transformers.configuration_utils import PretrainedConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Fallback minimal config
    class PretrainedConfig:
        model_type = "vvalt"
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class VVALTConfig(PretrainedConfig):
    r"""
    Configuration class for V.V.A.L.T models (HuggingFace compatible).

    This is the configuration class to store the configuration of a [`VVALTModel`]. It is used to instantiate
    a V.V.A.L.T model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        input_dim (`int`, *optional*, defaults to 768):
            Dimension of input vectors.
        frame_dim (`int`, *optional*, defaults to 512):
            Dimension of each perspective frame representation.
        task_dim (`int`, *optional*, defaults to 64):
            Dimension of task conditioning vectors.
        hidden_dim (`int`, *optional*, defaults to None):
            Hidden dimension for logic refinement unit. If None, defaults to 2 * frame_dim.
        num_labels (`int`, *optional*, defaults to 2):
            Number of labels for classification tasks.
        seed (`int`, *optional*, defaults to 42):
            Random seed for deterministic initialization.
        use_return_dict (`bool`, *optional*, defaults to True):
            Whether to return ModelOutput objects.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization in classification heads.
        safe_bounds (`tuple`, *optional*, defaults to (-10.0, 10.0)):
            Safe output bounds for consistency verification.

    Example:
        ```python
        >>> from vvalt import VVALTConfig, VVALTModel
        >>> # Initializing a V.V.A.L.T configuration
        >>> configuration = VVALTConfig(
        ...     input_dim=768,
        ...     frame_dim=512,
        ...     task_dim=64,
        ...     num_labels=2
        ... )
        >>> # Initializing a model from the configuration
        >>> model = VVALTModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "vvalt"

    def __init__(
        self,
        input_dim: int = 768,
        frame_dim: int = 512,
        task_dim: int = 64,
        hidden_dim: int = None,
        num_labels: int = 2,
        seed: int = 42,
        use_return_dict: bool = True,
        initializer_range: float = 0.02,
        safe_bounds: tuple = (-10.0, 10.0),
        runtime: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.frame_dim = frame_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2 * frame_dim
        self.num_labels = num_labels
        self.seed = seed
        self.use_return_dict = use_return_dict
        self.initializer_range = initializer_range
        self.safe_bounds = safe_bounds

        # Runtime configuration for modular implementation
        self.runtime = runtime if runtime is not None else {
            'batch_size_limit': 100,
            'enable_trace': False,
            'safety_level': 'STANDARD'
        }

        # V.V.A.L.T specific parameters
        self.num_frames = 5  # semantic, structural, causal, relational, graph
        self.frame_names = ["semantic", "structural", "causal", "relational", "graph"]

        # Safety guarantees flags
        self.deterministic = True
        self.bounded_computation = True
        self.single_pass = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dictionary of all attributes.
        """
        output = super().to_dict() if hasattr(super(), 'to_dict') else {}
        output.update({
            "input_dim": self.input_dim,
            "frame_dim": self.frame_dim,
            "task_dim": self.task_dim,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
            "seed": self.seed,
            "use_return_dict": self.use_return_dict,
            "initializer_range": self.initializer_range,
            "safe_bounds": self.safe_bounds,
            "runtime": self.runtime,
            "num_frames": self.num_frames,
            "frame_names": self.frame_names,
            "deterministic": self.deterministic,
            "bounded_computation": self.bounded_computation,
            "single_pass": self.single_pass,
        })
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "VVALTConfig":
        """
        Instantiates a configuration from a Python dictionary.

        Args:
            config_dict: Dictionary of configuration parameters.

        Returns:
            VVALTConfig instance.
        """
        return cls(**config_dict, **kwargs)
