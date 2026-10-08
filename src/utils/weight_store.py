"""Per-layer multi-bitwidth weight store."""

from pathlib import Path
from typing import Dict, List

import torch


class WeightStore:
    """Dequantized weights read from disk on every access, at layer_dir/<layer_name>/<bits>.pth."""

    def __init__(self, layer_dir: str):
        self.layer_dir = Path(layer_dir)
        self._bitwidths: Dict[str, List[int]] = {}

    def load(self) -> "WeightStore":
        """Index the layer directory; returns self."""
        for layer_path in self.layer_dir.iterdir():
            bits = sorted(int(f.stem) for f in layer_path.glob("*.pth") if f.stem.isdigit())
            if layer_path.is_dir() and bits:
                self._bitwidths[layer_path.name] = bits
        return self

    def get_layer_weight(self, layer_name: str, bitwidth: int) -> torch.Tensor:
        """Weight tensor of a layer at a bitwidth; KeyError if it was not indexed."""
        if bitwidth not in self._bitwidths[layer_name]:
            raise KeyError(f"No {bitwidth}-bit weights for {layer_name}")
        return torch.load(self.layer_dir / layer_name / f"{bitwidth}.pth", map_location="cpu")

    def get_layer_names(self) -> List[str]:
        return list(self._bitwidths)

    def get_available_bitwidths(self, layer_name: str) -> List[int]:
        return self._bitwidths[layer_name]
