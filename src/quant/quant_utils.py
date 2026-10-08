"""Min-max uniform quantizer and the QLinear fake-quant layer used during GPTQ calibration."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_params(w, bits, sym=False, perchannel=True):
    """Min-max scale and zero point per row of w (one shared pair if not perchannel), as [d_row, 1]."""
    maxq = 2**bits - 1
    x = w if perchannel else w.flatten().unsqueeze(0)
    xmin, xmax = x.min(1).values, x.max(1).values
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        neg = xmin < 0
        xmin[neg] = -xmax[neg]
    flat = xmin == xmax
    xmin[flat], xmax[flat] = -1, 1
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-xmin / scale)
    if not perchannel:
        scale, zero = scale.repeat(w.shape[0]), zero.repeat(w.shape[0])
    return scale.unsqueeze(1), zero.unsqueeze(1)


def quantize(x, scale, zero, maxq, eps=1e-9):
    return torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)


def dequantize(q, scale, zero):
    return scale * (q - zero)


class QLinear(nn.Module):
    """nn.Linear replacement holding uint8 codes with group-wise scales and zeros."""

    def __init__(self, qweight, scale, zero, bias=None):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.bias = bias

    def get_weight(self):
        d_row, num_groups = self.scale.shape
        q = self.qweight.view(d_row, num_groups, -1)
        return dequantize(q, self.scale.unsqueeze(-1), self.zero.unsqueeze(-1)).view_as(self.qweight)

    def forward(self, input):
        return F.linear(input, self.get_weight(), self.bias)
