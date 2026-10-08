"""Pack SLQ qparams bundles into Humming's HummingWeightSchema tensor layout (no re-quantization)."""
from typing import Any, Dict, Tuple

import torch


def pack_qparams_for_humming(
    qparams: Dict[str, Any],
    target_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """(tensors, schema_config) for HummingLayer.load_from_tensors; zero-points stay in target_dtype, exact to GPTQ."""
    from humming import ops  # https://github.com/inclusionAI/humming

    if qparams.get("perm") is not None:
        raise NotImplementedError("act-order qparams (column permutation) are not supported")
    bits, sym = int(qparams["bits"]), bool(qparams["sym"])
    # ops.pack_weight packs the last dim (K) and needs int32 input on CUDA
    qweight = qparams["qweight"].to(torch.int32).cuda().contiguous()
    tensors = {
        "weight": ops.pack_weight(qweight, bits).cpu(),
        "weight_scale": qparams["scales"].to(target_dtype).contiguous(),
    }
    if not sym:
        tensors["zero_point"] = qparams["zeros"].to(target_dtype).contiguous()
    schema_config = {
        "quant_method": "humming",
        "dtype": f"uint{bits}",
        "group_size": int(qparams["group_size"]),
        "has_zero_point": not sym,
        "is_fp_zero_point": not sym,
    }
    return tensors, schema_config
