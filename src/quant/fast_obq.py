"""GPTQ (OBQ-style) quantization of a single linear layer at several bitwidths from one shared Hessian."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from src.quant import dist_utils
from src.quant.quant_utils import dequantize, find_params, quantize
from src.utils.device import safe_set_tf32


safe_set_tf32(False)


class FastOBQ:

    def __init__(
        self,
        layer: nn.Linear,
        bitwidth_options: List[int],
        group_size: int = 128,
        sym: bool = False,
        perchannel: bool = True,
        rel_damp: float = 1e-2,
    ):
        assert isinstance(layer, nn.Linear), "FastOBQ supports only nn.Linear layers."
        self.layer = layer
        self.bitwidth_options = bitwidth_options
        self.d_row, self.d_col = layer.weight.shape
        assert self.d_col % group_size == 0, f"in_features ({self.d_col}) must be divisible by group_size ({group_size})"
        self.group_size = group_size
        self.sym = sym
        self.perchannel = perchannel
        self.rel_damp = rel_damp
        self.H = None
        self.num_samples = 0

    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """Update the running Hessian estimate 2 X^T X / n with a batch of layer inputs."""
        batch_size = input.shape[0]
        input = input.reshape(-1, input.shape[-1]).float()
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        n = self.num_samples + batch_size
        self.H.addmm_(input.T, input, beta=self.num_samples / n, alpha=2.0 / n)
        self.num_samples = n

    @torch.no_grad()
    def _prepare(self) -> Tuple[Tensor, Tensor]:
        """Dampened Hessian -> upper Cholesky factor of its inverse, plus the float32 weight.

        On Ascend NPU ``torch.linalg`` (cholesky / cholesky_inverse) is unsupported, so the
        factorisation runs on CPU (as in AMCT's GPTQ); the factor and weight are then moved
        back to the layer device so the quantization loop's error-feedback GEMMs run on the
        accelerator rather than on CPU.
        """
        device = self.layer.weight.device
        H, W = self.H, self.layer.weight.float()
        if H.device.type == "npu":
            H, W = H.cpu(), W.cpu()
        W = W.clone()
        # input channels that never fire get unit curvature and zero weight
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        H.diagonal().add_(self.rel_damp * torch.diag(H).mean())
        W[:, dead] = 0
        # all-zero weight columns are decoupled from the rest of the Hessian
        zero_cols = (W == 0).all(0)
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        return W.to(device), H_inv_cho.to(device)

    @torch.no_grad()
    def quantize(self) -> Dict[int, Tuple[Tensor, Tensor, Tensor]]:
        """Return {bits: (qweight uint8 [d_row, d_col], scales, zeros [d_row, n_groups])}, identical on all ranks."""
        assert self.H is not None, "No calibration samples were processed."
        if dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        d_row, d_col, gs = self.d_row, self.d_col, self.group_size
        device, dtype = self.layer.weight.device, self.layer.weight.dtype
        out = {
            bits: (
                torch.empty(d_row, d_col, device=device, dtype=torch.uint8),
                torch.empty(d_row, d_col // gs, device=device, dtype=dtype),
                torch.empty(d_row, d_col // gs, device=device, dtype=dtype),
            )
            for bits in self.bitwidth_options
        }
        if dist_utils.is_main():
            W, H_inv_cho = self._prepare()
            for bits, (qweight, scales, zeros) in out.items():
                maxq = 2**bits - 1
                w = W.clone()
                # one GPTQ block per group; group params (scale/zero) are fixed before the group
                # is quantized, so the whole group is quantized in one shot and its per-column
                # error (relative to the Hessian diagonal) is propagated to all later columns via
                # a single matmul. This vectorised form is the NPU/CUDA adaptation of the original
                # column-by-column loop (equivalent up to the intra-group error feedback, a
                # second-order effect at gs=128).
                for c1 in range(0, d_col, gs):
                    c2 = c1 + gs
                    w_blk = w[:, c1:c2]
                    scale, zero = find_params(w_blk, bits, self.sym, self.perchannel)
                    # quantize with the stored-precision scale so saved weights match the error feedback
                    scale = scale.to(dtype).float()
                    scales[:, c1 // gs] = scale.flatten()
                    zeros[:, c1 // gs] = zero.flatten()
                    q = quantize(w_blk, scale, zero, maxq)
                    qweight[:, c1:c2] = q
                    dq = dequantize(q, scale, zero)
                    diag = H_inv_cho.diagonal()[c1:c2].clamp(min=1e-10)
                    errs = (w_blk - dq) / diag.unsqueeze(0)
                    w[:, c1:c2] = dq
                    w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)
        if dist_utils.is_dist_available_and_initialized():
            for tensors in out.values():
                for t in tensors:
                    dist.broadcast(t, src=0)
        self.H = None
        return out
