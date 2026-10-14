"""Layer grouping for constraining layers to share a bitwidth (e.g. vLLM-fused projections)."""

import re
from typing import Dict, List, Tuple

from src.sensitivity.database import LayerGroup

# Generic block regex: matches both `model.layers.N.` (dense) and the nested
# `model.language_model.layers.N.` used by VLM/hybrid checkpoints (Qwen3.5).
_BLOCK = re.compile(r'\.layers\.(\d+)\.')


def parse_group_spec(group_spec: str) -> List[List[str]]:
    """Space-separated groups of comma-separated patterns, e.g. 'mlp.gate_proj,mlp.up_proj self_attn.q_proj,self_attn.k_proj'."""
    groups = [[p for p in g.split(',') if p] for g in group_spec.split()]
    return [g for g in groups if g]


def build_layer_groups(layer_names: List[str], group_patterns: List[List[str]]) -> List[LayerGroup]:
    """Per transformer block, layers matching the same pattern group form one group; the rest stay singletons."""
    grouped: Dict[Tuple[int, int], List[str]] = {}
    singles = []
    for name in layer_names:
        m = _BLOCK.search(name)
        g = next((g for g, pats in enumerate(group_patterns) if m and any(p in name[m.end():] for p in pats)), None)
        if g is None:
            singles.append(name)
        else:
            grouped.setdefault((int(m.group(1)), g), []).append(name)
    units = [(sorted(names), f"block_{b}:{','.join(group_patterns[g])}") for (b, g), names in sorted(grouped.items())]
    units += [([name], name) for name in singles]
    return [LayerGroup(i, names, pattern) for i, (names, pattern) in enumerate(units)]
