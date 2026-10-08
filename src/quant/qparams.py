"""Per-(layer, bitwidth) quantization bundle saved as <layer_dir>/<layer>/<bits>_qparams.pt.

Keys: qweight (uint8 codes [d_row, d_col]), scales and zeros ([d_row, n_groups]),
plus bits, group_size, sym, perchannel, shape, dtype, schema.
Dequant: W = scales[:, g] * (qweight - zeros[:, g]) with g = col // group_size.
"""
import os
from typing import Any, Dict

import torch

QPARAMS_SCHEMA_VERSION = 1


def qparams_path(layer_dir: str, bits: int) -> str:
    return os.path.join(str(layer_dir), f"{int(bits)}_qparams.pt")


def save_qparams(layer_dir: str, bits: int, qweight, scales, zeros, handle) -> None:
    """Save one (layer, bitwidth) bundle; handle is the FastOBQ instance."""
    torch.save(
        {
            "qweight": qweight.cpu(),
            "scales": scales.cpu(),
            "zeros": zeros.cpu(),
            "bits": int(bits),
            "group_size": handle.group_size,
            "sym": handle.sym,
            "perchannel": handle.perchannel,
            "shape": tuple(handle.layer.weight.shape),
            "dtype": str(handle.layer.weight.dtype).replace("torch.", ""),
            "schema": QPARAMS_SCHEMA_VERSION,
        },
        qparams_path(layer_dir, bits),
    )


def load_qparams(layer_dir: str, bits: int) -> Dict[str, Any]:
    path = qparams_path(layer_dir, bits)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != QPARAMS_SCHEMA_VERSION:
        raise ValueError(f"qparams schema mismatch at {path}: expected {QPARAMS_SCHEMA_VERSION}")
    return bundle
