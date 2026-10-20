"""Input validation and normalisation for standalone MTN demonstrations."""

import torch


def prepare_model_inputs(batch, config, device):
    """Move a synthetic or external batch to ``device`` and validate shapes."""
    required = {"boxes", "speed", "ego_flow", "ped_flow"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"Missing MTN inputs: {sorted(missing)}")
    if tuple(batch["boxes"].shape[1:]) != (config.observe_length, config.box_size):
        raise ValueError("boxes must have shape [batch, observe_length, box_size]")
    if batch["ego_flow"].shape[-2:] != (config.ego_patches, 2):
        raise ValueError("ego_flow must end in [ego_patches, 2]")
    if batch["ped_flow"].shape[-2:] != (config.ped_patches, 2):
        raise ValueError("ped_flow must end in [ped_patches, 2]")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
