"""Perplexity, top-K KL and EAR metrics."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.utils.device import empty_cache


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
        batch = input_ids[i:i + batch_size].to(device)
        # 保持 bf16 做 topk，避免 .float() 生成 [batch, seq, vocab] 的全量 fp32
        # logits（batch=4×seq=2048 时约 +4.9 GiB），直接撑爆 60 GiB NPU。
        logits = model(batch).logits[:, :-1]
        v, k = logits.topk(top_k, dim=-1)
        vals.append(v.float().cpu())
        idx.append(k.cpu())
        del logits, batch
        empty_cache()
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


def _last_hidden_state(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the backbone's final hidden state (the lm_head input), shape [B, seq, hidden].

    This avoids `model(input_ids).logits`, which materializes the full-vocabulary logits
    [B, seq, vocab] (e.g. ~972 MiB per 2048-token sample on a 248k-vocab 27B model) and is the
    direct OOM trigger inside lm_head. We only need the top-K vocabulary positions for KL/EAR.

    Compatible with Qwen3_5ForCausalLM (model.model == Qwen3_5TextModel) and, defensively, with
    the multimodal Qwen3_5ForConditionalGeneration wrapper (model.model.language_model).
    """
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None:
        raise AttributeError("cannot locate the backbone submodule on the model")
    language_model = getattr(base, "language_model", None)
    if language_model is not None:
        base = language_model
    out = base(input_ids=input_ids)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0]
    return hidden


@torch.no_grad()
def _nll_kl_ear(model, data, log_p, p, topk_idx, batch_size, masks, with_nll):
    """Token-mean NLL, top-K KL(p||q) and EAR = sum_k min(p, q).

    log_p/p/topk_idx are the baseline reference (precomputed on device, fixed). Only q (the
    quantized model's distribution at p's top-K) is recomputed per forward. Sums are accumulated
    on device and pulled to host once, avoiding a per-batch device sync.

    Memory optimization: when `with_nll` is False (the KL/EAR path used by Shapley sampling), we do
    NOT call `model(batch).logits`, which allocates the full-vocabulary logits in lm_head. Instead
    we take the backbone's last hidden state and project only the top-K lm_head weight rows, so the
    peak allocation drops from O(seq*vocab) to O(seq*top_k*hidden).
    """
    model.eval()
    device = next(model.parameters()).device
    nll_acc = torch.zeros((), device=device, dtype=torch.float64)
    kl_acc = torch.zeros((), device=device, dtype=torch.float64)
    ear_acc = torch.zeros((), device=device, dtype=torch.float64)
    n_acc = torch.zeros((), device=device, dtype=torch.float64)
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size].to(device)
        idx = topk_idx[i:i + batch_size]
        w = masks[i:i + batch_size, 1:].to(device).float() if masks is not None else None
        if with_nll:
            logits = model(batch).logits[:, :-1]
            nl = _token_nll(logits, batch[:, 1:])
            nll_acc += (nl * w).sum() if w is not None else nl.sum()
            log_q = F.log_softmax(logits.gather(-1, idx).float(), dim=-1)
            del logits
        else:
            hidden = _last_hidden_state(model, batch)[:, :-1]
            lmw = model.lm_head.weight
            logits_topk = (hidden.unsqueeze(2) * lmw[idx]).sum(-1)
            bias = getattr(model.lm_head, "bias", None)
            if bias is not None:
                logits_topk = logits_topk + bias[idx]
            log_q = F.log_softmax(logits_topk.float(), dim=-1)
            del hidden, logits_topk
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
