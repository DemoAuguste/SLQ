"""Per-bitwidth layer weights kept on each layer's device for fast swapping during Shapley sampling."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.weight_store import WeightStore


class GPUWeightCache:
    """Swaps WeightStore weights into a model; cache keeps them on the layer's device, preload loads all upfront."""

    def __init__(self, model: nn.Module, weight_store: WeightStore, bitwidths: List[int],
                 cache: bool = False, preload: bool = False):
        self.weight_store = weight_store
        self.layer_names = weight_store.get_layer_names()
        self.bitwidths = sorted(bitwidths, reverse=True)
        self.cache = cache or preload
        self.weights: Dict[Tuple[str, int], torch.Tensor] = {}
        if preload:
            for name in tqdm(self.layer_names, desc="Preloading weights to GPU"):
                for bw in self.bitwidths:
                    self._get(model.get_submodule(name).weight, name, bw)

    def _get(self, target: torch.Tensor, layer_name: str, bitwidth: int) -> torch.Tensor:
        weight = self.weights.get((layer_name, bitwidth))
        if weight is None:
            weight = self.weight_store.get_layer_weight(layer_name, bitwidth).to(target.device, target.dtype)
            if self.cache:
                self.weights[layer_name, bitwidth] = weight
        return weight

    def swap_layer(self, model: nn.Module, layer_name: str, bitwidth: int):
        """Point one layer's weight at its bitwidth-`bitwidth` version."""
        target = model.get_submodule(layer_name).weight
        target.data = self._get(target, layer_name, bitwidth)

    def swap_all_layers(self, model: nn.Module, allocation: Dict[str, int]):
        """Swap every cached layer named in allocation."""
        names = set(self.layer_names)
        for layer_name, bitwidth in allocation.items():
            if layer_name in names:
                self.swap_layer(model, layer_name, bitwidth)
