"""
Generic checkpoint loading and saving utilities.
"""

import os
import pickle
import torch


def _strip_module_prefix(state_dict):
    """Strip 'module.' prefix from DDP-saved state_dicts for compatibility."""
    if any(k.startswith('module.') for k in state_dict):
        return {k.removeprefix('module.'): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(path: str, modules: dict) -> dict:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        modules: {key: module} — each with load_state_dict().

    Returns:
        Metadata dict (everything except module state_dicts),
        or empty dict if no checkpoint / corrupted.
    """
    if not os.path.exists(path):
        return {}

    print(f"Found checkpoint at {path}")
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
        print(f"⚠ Checkpoint corrupted ({e})")
        return {}

    for key, module in modules.items():
        if key in checkpoint:
            sd = checkpoint[key]
            # Strip DDP module. prefix for nn.Module state_dicts
            if isinstance(module, torch.nn.Module):
                sd = _strip_module_prefix(sd)
            module.load_state_dict(sd)

    # Return everything that isn't a module state_dict
    meta = {k: v for k, v in checkpoint.items() if k not in modules}
    print(f"✓ Resumed (epoch {meta.get('epoch', '?')})")
    return meta


def save_checkpoint(path: str, modules: dict, meta: dict):
    """
    Save a training checkpoint.

    Args:
        path: Save path.
        modules: {key: module} — each gets .state_dict() called.
        meta: Arbitrary metadata dict (epoch, losses, etc.).
    """
    save_dict = dict(meta)
    for key, module in modules.items():
        save_dict[key] = module.state_dict()
    torch.save(save_dict, path)
