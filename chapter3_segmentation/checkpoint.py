"""Normalize original SAM3 and Trainer checkpoints for the noninteractive image model."""
from collections.abc import Mapping
from contextlib import contextmanager

IMAGE_PREFIXES = ("backbone.", "transformer.", "geometry_encoder.",
                  "segmentation_head.", "dot_prod_scoring.")
INTERACTIVE_NECK_PREFIX = "backbone.vision_backbone.sam2_convs."


def normalize_image_state(checkpoint):
    """Select active image weights; leave unexpected image keys for strict loading to reject."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SAM3 checkpoint must contain a state dictionary.")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
        raise ValueError("SAM3 checkpoint model state is empty or invalid.")
    if any(key.startswith("detector.") for key in state):
        state = {key[len("detector."):]: value for key, value in state.items() if key.startswith("detector.")}
        # The official full checkpoint includes the point-prompt SAM2 neck. The
        # official enable_inst_interactivity=False model has sam2_convs=None.
        state = {key: value for key, value in state.items() if not key.startswith(INTERACTIVE_NECK_PREFIX)}
    if not state or not any(key.startswith(IMAGE_PREFIXES) for key in state):
        raise ValueError("Checkpoint contains no recognizable SAM3 image-model weights.")
    return dict(state)


@contextmanager
def numpy_metadata_context():
    """Temporarily allow official Trainer metrics, preserving enclosing safe scopes."""
    import numpy as np
    import torch

    # Official Trainer checkpoints can contain NumPy float64 COCO metrics beside
    # the tensor state. Keep restricted unpickling rather than weights_only=False.
    allowed = [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]
    existing = torch.serialization.get_safe_globals()
    # PyTorch removes each context's entries on exit. Do not re-add an outer
    # context's entries or an inner image load would revoke them before resume.
    additions = [entry for entry in allowed if entry not in existing]
    with torch.serialization.safe_globals(additions):
        yield


def load_image_state(path):
    """Load tensor weights safely, allowing only NumPy scalar metadata used by Trainer."""
    import torch

    with numpy_metadata_context():
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    return normalize_image_state(checkpoint)
