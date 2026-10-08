"""Seeding and layer weight helpers."""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def fix_seed(seed: int):
    """Seed python, numpy and torch; deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_layer_weights(model: nn.Module, layer_name: str) -> Optional[torch.Tensor]:
    """Weight of the module at a dotted path, or None if missing, weightless, or on meta."""
    try:
        weight = model.get_submodule(layer_name).weight.detach()
    except AttributeError:
        return None
    return None if weight.is_meta else weight


def set_layer_weights(model: nn.Module, layer_name: str, weights: torch.Tensor):
    """Write weights into the module at a dotted path, cast to its device and dtype."""
    weight = model.get_submodule(layer_name).weight
    weight.data = weights.to(device=weight.device, dtype=weight.dtype)


def get_layer_param_counts(model: nn.Module, layer_names: List[str]) -> Dict[str, int]:
    """Parameter count per layer (1 if the weight is unavailable)."""
    counts = {}
    for name in layer_names:
        w = get_layer_weights(model, name)
        counts[name] = 1 if w is None else w.numel()
    return counts


@torch.no_grad()
def compute_layer_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Relative squared Frobenius error ||Q - W||_F^2 / ||W||_F^2, in float32."""
    w = original.cpu().float()
    return ((quantized.cpu().float() - w).square().sum() / w.square().sum()).item()

