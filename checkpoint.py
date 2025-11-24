"""
V.V.A.L.T Checkpoint System

Handles saving and loading of model parameters with versioning and validation.
"""

import numpy as np
import os
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
from .errors import CheckpointLoadError


CHECKPOINT_VERSION = "0.1.0"


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    def __init__(self, vvalt_instance):
        """
        Initialize checkpoint manager.

        Args:
            vvalt_instance: VVALT model instance
        """
        self.model = vvalt_instance

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint (.npz file)
            metadata: Optional metadata to include

        Returns:
            Checksum of saved file
        """
        # Collect all parameters
        params = {}

        # VectorFrameEncoder
        params["encoder_W_semantic"] = self.model.encoder.W_semantic
        params["encoder_W_structural"] = self.model.encoder.W_structural
        params["encoder_W_causal"] = self.model.encoder.W_causal
        params["encoder_W_relational"] = self.model.encoder.W_relational
        params["encoder_W_graph"] = self.model.encoder.W_graph
        params["encoder_b_semantic"] = self.model.encoder.b_semantic
        params["encoder_b_structural"] = self.model.encoder.b_structural
        params["encoder_b_causal"] = self.model.encoder.b_causal
        params["encoder_b_relational"] = self.model.encoder.b_relational
        params["encoder_b_graph"] = self.model.encoder.b_graph

        # VantageSelector
        params["selector_W_task"] = self.model.selector.W_task
        params["selector_b_task"] = self.model.selector.b_task

        # GraphTopologyProjector
        params["projector_W_topology"] = self.model.projector.W_topology

        # MultiPerspectiveAttention
        params["attention_W_query"] = self.model.attention.W_query
        params["attention_W_key"] = self.model.attention.W_key
        params["attention_W_value"] = self.model.attention.W_value
        params["attention_W_output"] = self.model.attention.W_output
        params["attention_b_output"] = self.model.attention.b_output

        # LogicRefinementUnit
        params["refiner_W1"] = self.model.refiner.W1
        params["refiner_W2"] = self.model.refiner.W2
        params["refiner_b1"] = self.model.refiner.b1
        params["refiner_b2"] = self.model.refiner.b2
        params["refiner_alpha"] = np.array([self.model.refiner.alpha])

        # Metadata
        checkpoint_metadata = {
            "version": CHECKPOINT_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "input_dim": self.model.input_dim,
            "frame_dim": self.model.frame_dim,
            "task_dim": self.model.task_dim,
            "hidden_dim": self.model.hidden_dim,
            "seed": self.model.seed,
        }

        if metadata:
            checkpoint_metadata.update(metadata)

        # Save metadata as JSON string in array
        params["metadata"] = np.array([json.dumps(checkpoint_metadata)])

        # Save checkpoint
        np.savez_compressed(path, **params)

        # Compute checksum
        checksum = self._compute_checksum(path)

        return checksum

    def load(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            strict: Raise on validation errors

        Returns:
            Metadata dictionary

        Raises:
            CheckpointLoadError: Load failed
        """
        if not os.path.exists(path):
            raise CheckpointLoadError(path, "File not found")

        try:
            checkpoint = np.load(path, allow_pickle=True)
        except Exception as e:
            raise CheckpointLoadError(path, f"Failed to load file: {str(e)}")

        # Load metadata
        try:
            metadata = json.loads(str(checkpoint["metadata"][0]))
        except (KeyError, json.JSONDecodeError) as e:
            if strict:
                raise CheckpointLoadError(path, f"Invalid metadata: {str(e)}")
            metadata = {}

        # Validate version
        if metadata.get("version") != CHECKPOINT_VERSION:
            if strict:
                raise CheckpointLoadError(
                    path,
                    f"Version mismatch: checkpoint is {metadata.get('version')}, expected {CHECKPOINT_VERSION}"
                )

        # Validate dimensions
        if strict:
            if metadata.get("input_dim") != self.model.input_dim:
                raise CheckpointLoadError(path, f"input_dim mismatch: {metadata.get('input_dim')} != {self.model.input_dim}")
            if metadata.get("frame_dim") != self.model.frame_dim:
                raise CheckpointLoadError(path, f"frame_dim mismatch: {metadata.get('frame_dim')} != {self.model.frame_dim}")
            if metadata.get("task_dim") != self.model.task_dim:
                raise CheckpointLoadError(path, f"task_dim mismatch: {metadata.get('task_dim')} != {self.model.task_dim}")

        # Load parameters
        try:
            # VectorFrameEncoder
            self.model.encoder.W_semantic = checkpoint["encoder_W_semantic"]
            self.model.encoder.W_structural = checkpoint["encoder_W_structural"]
            self.model.encoder.W_causal = checkpoint["encoder_W_causal"]
            self.model.encoder.W_relational = checkpoint["encoder_W_relational"]
            self.model.encoder.W_graph = checkpoint["encoder_W_graph"]
            self.model.encoder.b_semantic = checkpoint["encoder_b_semantic"]
            self.model.encoder.b_structural = checkpoint["encoder_b_structural"]
            self.model.encoder.b_causal = checkpoint["encoder_b_causal"]
            self.model.encoder.b_relational = checkpoint["encoder_b_relational"]
            self.model.encoder.b_graph = checkpoint["encoder_b_graph"]

            # VantageSelector
            self.model.selector.W_task = checkpoint["selector_W_task"]
            self.model.selector.b_task = checkpoint["selector_b_task"]

            # GraphTopologyProjector
            self.model.projector.W_topology = checkpoint["projector_W_topology"]

            # MultiPerspectiveAttention
            self.model.attention.W_query = checkpoint["attention_W_query"]
            self.model.attention.W_key = checkpoint["attention_W_key"]
            self.model.attention.W_value = checkpoint["attention_W_value"]
            self.model.attention.W_output = checkpoint["attention_W_output"]
            self.model.attention.b_output = checkpoint["attention_b_output"]

            # LogicRefinementUnit
            self.model.refiner.W1 = checkpoint["refiner_W1"]
            self.model.refiner.W2 = checkpoint["refiner_W2"]
            self.model.refiner.b1 = checkpoint["refiner_b1"]
            self.model.refiner.b2 = checkpoint["refiner_b2"]
            self.model.refiner.alpha = float(checkpoint["refiner_alpha"][0])

        except KeyError as e:
            raise CheckpointLoadError(path, f"Missing parameter: {str(e)}")

        return metadata

    def _compute_checksum(self, path: str) -> str:
        """Compute SHA256 checksum of checkpoint file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_checkpoint(self, path: str) -> bool:
        """
        Validate checkpoint file integrity.

        Args:
            path: Path to checkpoint

        Returns:
            True if valid
        """
        try:
            checkpoint = np.load(path, allow_pickle=True)

            # Check required keys
            required_keys = [
                "encoder_W_semantic", "encoder_W_structural", "encoder_W_causal",
                "encoder_W_relational", "encoder_W_graph",
                "selector_W_task", "projector_W_topology",
                "attention_W_query", "attention_W_key", "attention_W_value",
                "refiner_W1", "refiner_W2", "metadata"
            ]

            for key in required_keys:
                if key not in checkpoint:
                    return False

            # Validate shapes
            metadata = json.loads(str(checkpoint["metadata"][0]))
            input_dim = metadata.get("input_dim")
            frame_dim = metadata.get("frame_dim")
            task_dim = metadata.get("task_dim")

            if checkpoint["encoder_W_semantic"].shape != (input_dim, frame_dim):
                return False
            if checkpoint["selector_W_task"].shape != (task_dim, 5):
                return False

            return True

        except Exception:
            return False


def create_checkpoint_metadata(
    description: str = "",
    tags: list = None,
    metrics: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create checkpoint metadata dictionary.

    Args:
        description: Human-readable description
        tags: List of tags for organization
        metrics: Performance metrics

    Returns:
        Metadata dictionary
    """
    metadata = {
        "description": description,
        "tags": tags or [],
        "metrics": metrics or {},
        "created_at": datetime.utcnow().isoformat()
    }

    return metadata
