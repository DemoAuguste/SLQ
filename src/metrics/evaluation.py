"""Perplexity, top-K KL and EAR metrics."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


def _token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy, shape of labels."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='none').view(labels.shape)


@torch.no_grad()
def compute_perplexity(model, data: List[torch.Tensor], batch_size: int = 1) -> Tuple[float, float, List[float]]:
    """Mean perplexity (exp of mean per-sample NLL), variance of per-sample PPL, and per-sample PPL."""
    device = next(model.parameters()).device
    nll = []
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        inputs = torch.cat(data[i:i + batch_size]).to(device)
        nll.extend(_token_nll(model(inputs).logits[:, :-1], inputs[:, 1:]).mean(dim=1).tolist())
    ppl = np.exp(nll)
    return float(np.exp(np.mean(nll))), float(np.var(ppl)), ppl.tolist()


@torch.no_grad()
def compute_baseline_topk(
    model, input_ids: torch.Tensor, batch_size: int = 4, top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K baseline logits and indices at every position, shape [N, seq_len-1, top_k]."""
    model.eval()
    device = next(model.parameters()).device
    vals, idx = [], []
    for i in range(0, input_ids.shape[0], batch_size):
        logits = model(input_ids[i:i + batch_size].to(device)).logits[:, :-1].float()
        v, k = logits.topk(top_k, dim=-1)
        vals.append(v.cpu())
        idx.append(k.cpu())
        del logits
    return torch.cat(vals), torch.cat(idx)


@torch.no_grad()
def compute_reference_dist(topk_vals: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Baseline (fixed) top-K distribution renormalized over the top-K, precomputed once on device.

    Returns (log_p, p) with shape [N, seq_len-1, top_k].
    """
    log_p = F.log_softmax(topk_vals.to(device), dim=-1)
    return log_p, log_p.exp()


def prepare_reference(topk_vals: torch.Tensor, topk_idx: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the baseline top-K distribution and move the indices to the model device.

    Returns (log_p, p, topk_idx) all on `device`. p/log_p are fixed across every quantized forward
    (they depend only on the BF16 baseline), so they are computed once instead of per forward pass.
    """
    log_p, p = compute_reference_dist(topk_vals, device)
    return log_p, p, topk_idx.to(device)


@torch.no_grad()
def _nll_kl_ear(model, data, log_p, p, topk_idx, batch_size, masks, with_nll):
    """Token-mean NLL, top-K KL(p||q) and EAR = sum_k min(p, q).

    log_p/p/topk_idx are the baseline reference (precomputed on device, fixed). Only q (the
    quantized model's distribution at p's top-K) is recomputed per forward. Sums are accumulated
    on device and pulled to host once, avoiding a per-batch device sync.
    """
    model.eval()
    device = next(model.parameters()).device
    nll_acc = torch.zeros((), device=device, dtype=torch.float64)
    kl_acc = torch.zeros((), device=device, dtype=torch.float64)
    ear_acc = torch.zeros((), device=device, dtype=torch.float64)
    n_acc = torch.zeros((), device=device, dtype=torch.float64)
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size].to(device)
        logits = model(batch).logits[:, :-1]
        w = masks[i:i + batch_size, 1:].to(device).float() if masks is not None else None
        if with_nll:
            nl = _token_nll(logits, batch[:, 1:])
            nll_acc += (nl * w).sum() if w is not None else nl.sum()
        log_q = F.log_softmax(logits.gather(-1, topk_idx[i:i + batch_size]).float(), dim=-1)
        lp = log_p[i:i + batch_size]
        pp = p[i:i + batch_size]
        kl = (pp * (lp - log_q)).sum(-1)
        # EAR = min(exp(log_p), exp(log_q)) = exp(min(log_p, log_q)) since exp is monotonic.
        ear = torch.exp(torch.minimum(lp, log_q)).sum(-1)
        if w is not None:
            kl_acc += (kl * w).sum()
            ear_acc += (ear * w).sum()
            n_acc += w.sum()
        else:
            kl_acc += kl.sum()
            ear_acc += ear.sum()
            n_acc += kl.numel()
        del logits
    d = max(n_acc.item(), 1.0)
    return nll_acc.item() / d, kl_acc.item() / d, ear_acc.item() / d, int(n_acc.item())


@torch.no_grad()
def compute_all_metrics(
    model, calibration_data: torch.Tensor, log_p: torch.Tensor, p: torch.Tensor, topk_idx: torch.Tensor,
    batch_size: int = 4, masks: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Mean NLL, top-K KL and EAR against the precomputed baseline reference."""
    return _nll_kl_ear(model, calibration_data, log_p, p, topk_idx, batch_size, masks, True)[:3]


@torch.no_grad()
def compute_kl_and_ear(
    model, log_p: torch.Tensor, p: torch.Tensor, topk_idx: torch.Tensor, eval_data: torch.Tensor,
    batch_size: int = 4, masks: Optional[torch.Tensor] = None,
) -> Tuple[float, float, int]:
    """(mean top-K KL, mean EAR, number of scored tokens) against the precomputed baseline reference."""
    return _nll_kl_ear(model, eval_data, log_p, p, topk_idx, batch_size, masks, False)[1:]
