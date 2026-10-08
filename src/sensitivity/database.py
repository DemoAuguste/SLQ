"""Error and sensitivity database data structures."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LayerGroup:
    """Layers constrained to share a bitwidth (e.g. vLLM fused QKV)."""
    group_id: int
    layer_names: List[str]
    pattern: str


@dataclass
class LinearSensitivityData:
    """Per-layer linear alphas: cost = alpha * relative weight error (Sec. 3.3)."""
    nll_alphas: Dict[str, float] = field(default_factory=dict)
    kl_alphas: Dict[str, float] = field(default_factory=dict)
    ear_alphas: Dict[str, float] = field(default_factory=dict)

    layer_names: List[str] = field(default_factory=list)
    top_k: int = 10
    calibration_dataset: str = ""
    calibration_samples: int = 0
    num_noise_samples: int = 15
    seed: int = 0
    # Per-layer values store the group value divided by group size (Eqs. 5-8).
    per_layer_normalized: bool = True
    layer_groups: Optional[List[Dict]] = None


@dataclass
class ShapleySensitivityData:
    """Multi-bitwidth Shapley values per player and per layer (Algorithm 1)."""
    nll_shapley_values: Optional[np.ndarray] = None
    kl_shapley_values: Optional[np.ndarray] = None
    ear_shapley_values: Optional[np.ndarray] = None

    nll_shapley_per_level: Optional[Dict[str, Dict[int, float]]] = None
    kl_shapley_per_level: Optional[Dict[str, Dict[int, float]]] = None
    ear_shapley_per_level: Optional[Dict[str, Dict[int, float]]] = None

    layer_names: List[str] = field(default_factory=list)
    bitwidths: List[int] = field(default_factory=list)
    top_k: int = 10
    calibration_dataset: str = ""
    calibration_samples: int = 0
    num_permutations: int = 50
    seed: int = 0
    # Per-layer values store the group value divided by group size (Eqs. 5-8).
    per_layer_normalized: bool = True
    layer_groups: Optional[List[Dict]] = None


class ErrorDatabase:
    """Per-layer relative squared Frobenius error for every available bitwidth."""

    def __init__(self):
        self.layers: Dict[str, Dict[int, float]] = {}
        self.layer_names: List[str] = []
        self.param_counts: Dict[str, int] = {}

    def add_layer_option(self, layer_name: str, bitwidth: int, error: float):
        if layer_name not in self.layers:
            self.layers[layer_name] = {}
            self.layer_names.append(layer_name)
        self.layers[layer_name][bitwidth] = error

    def get_common_bitwidths(self) -> List[int]:
        if not self.layer_names:
            return []
        return sorted(set.intersection(*(set(self.layers[n]) for n in self.layer_names)))

    def save(self, filepath: str):
        layers = {n: {str(bw): {'error': e} for bw, e in errs.items()} for n, errs in self.layers.items()}
        with open(filepath, 'w') as f:
            json.dump({'layers': layers, 'param_counts': self.param_counts}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ErrorDatabase":
        with open(filepath) as f:
            data = json.load(f)
        db = cls()
        for name, opts in data['layers'].items():
            for bw, o in opts.items():
                db.add_layer_option(name, int(bw), o['error'])
        db.param_counts = data.get('param_counts', {})
        return db
