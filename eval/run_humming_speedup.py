#!/usr/bin/env python3
r"""Whole-model forward speed of a Humming-quantized model vs. BF16 (needs CUDA >= 12.8).

Usage:
  python eval/run_humming_speedup.py --model Qwen/Qwen3-8B --layer-dir <layer_dir> \
      --uniform-bits 4 --seq-length 1 --no-baseline --single-gpu --compile
"""

import argparse
import gc
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slq_build_checkpoint import parse_config_file
from src.models.loading import get_tokenizer, load_model
from src.quant.qparams import load_qparams
from src.quant.humming_pack import pack_qparams_for_humming


def load_backbone_single_gpu(model_path: str, dtype: torch.dtype, skip: set, device: torch.device) -> nn.Module:
    """Build the model on meta and load every checkpoint tensor not in skip directly to device."""
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.torch_dtype = dtype
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    snapshot = Path(model_path).expanduser()
    if not snapshot.is_dir():
        snapshot = Path(snapshot_download(model_path, allow_patterns=["*.json", "*.safetensors"]))
    shards = sorted(snapshot.glob("model*.safetensors"))
    if not shards:
        raise RuntimeError(f"No safetensors shards found under {snapshot}")

    wanted = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    wanted -= skip
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                name = key.replace("model.language_model.", "model.", 1)
                if name not in wanted:
                    continue
                parent_path, _, leaf = name.rpartition(".")
                parent, t = model.get_submodule(parent_path), f.get_tensor(key).to(device)
                if leaf in parent._parameters:
                    parent._parameters[leaf] = nn.Parameter(t.to(dtype), requires_grad=False)
                else:
                    parent._buffers[leaf] = t

    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()

    # Tensors absent from the checkpoint (e.g. q_norm/k_norm in some Qwen3.5 checkpoints) are
    # filled with ones; values only matter for timing. Non-persistent buffers (rotary inv_freq)
    # are created on CPU by init_empty_weights and moved to device.
    for mod_name, module in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        for n, p in module.named_parameters(recurse=False):
            if p.device.type == "meta" and prefix + n not in skip:
                module._parameters[n] = nn.Parameter(torch.ones(p.shape, dtype=dtype, device=device),
                                                     requires_grad=False)
        for n, b in module.named_buffers(recurse=False):
            if b.device.type == "meta":
                module._buffers[n] = torch.zeros(b.shape, dtype=b.dtype, device=device)
            elif b.device != device:
                module._buffers[n] = b.to(device)
    return model


class HummingLinear(nn.Module):
    """nn.Linear-compatible wrapper around a HummingLayer (flattens to 2D, adds bias)."""

    def __init__(self, layer_dir: Path, name: str, bits: int, dtype: torch.dtype, linear: nn.Linear,
                 device: torch.device):
        from humming.layer import HummingLayer

        super().__init__()
        bundle = load_qparams(str(layer_dir / name), bits)
        self.out_features, self.in_features = bundle["shape"]
        tensors, cfg = pack_qparams_for_humming(bundle, target_dtype=dtype)
        self.humming = HummingLayer(shape_n=self.out_features, shape_k=self.in_features,
                                    weight_config=cfg, torch_dtype=dtype).to(device)
        self.humming.load_from_tensors(tensors)
        self.humming.transform()
        bias = linear.bias
        self.bias = None if bias is None else nn.Parameter(bias.detach().to(device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.humming(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


@torch.no_grad()
def time_forward(model: nn.Module, input_ids: torch.Tensor, num_warmup: int, num_iters: int):
    """(mean, std) seconds per forward pass."""
    model.eval()
    for _ in range(num_warmup):
        model(input_ids)
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def run(name: str, model: nn.Module, input_ids: torch.Tensor, args) -> float:
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    mean, std = time_forward(model, input_ids, args.num_warmup, args.num_iters)
    print(f"{name}: {mean * 1000:.2f} +/- {std * 1000:.2f} ms ({input_ids.numel() / mean:.1f} tok/s)")
    return mean


def main():
    p = argparse.ArgumentParser(description="Humming vs. BF16 whole-model forward speed")
    p.add_argument("--model", required=True)
    p.add_argument("--layer-dir", required=True, help="per-layer qparams from slq_quantize.py")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config", help="per-layer bitwidth config txt")
    grp.add_argument("--uniform-bits", type=int, help="bitwidth for every layer in --layer-dir")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--no-baseline", action="store_true", help="skip the BF16 baseline")
    p.add_argument("--baseline-only", action="store_true", help="only time the BF16 baseline")
    p.add_argument("--compile", action="store_true", help="torch.compile(mode='reduce-overhead')")
    p.add_argument("--single-gpu", action="store_true",
                   help="load non-quantized weights directly to cuda:0, avoiding multi-GPU dispatch hooks")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    layer_dir = Path(args.layer_dir)
    if args.config:
        bw_config = parse_config_file(args.config)[0]
    else:
        bw_config = {d.name: args.uniform_bits for d in sorted(layer_dir.iterdir()) if d.is_dir() and any(d.glob('*.pth'))}
    print(f"bitwidth distribution: {dict(sorted(Counter(bw_config.values()).items()))}")

    # Random token ids; only timing matters. cuda:0 is the first device of device_map="auto".
    torch.manual_seed(0)
    vocab = get_tokenizer(args.model).vocab_size
    input_ids = torch.randint(0, vocab, (args.batch_size, args.seq_length), device="cuda:0")

    bf16_mean = None
    if not args.no_baseline:
        bf16_mean = run("bf16", load_model(args.model, dtype=args.dtype, device_map="auto"), input_ids, args)
        gc.collect()
        torch.cuda.empty_cache()
    if args.baseline_only:
        return

    if args.single_gpu:
        device = torch.device("cuda:0")
        # Quantized weights are never loaded; their biases are.
        model = load_backbone_single_gpu(args.model, dtype, {f"{n}.weight" for n in bw_config}, device)
    else:
        device = None
        model = load_model(args.model, dtype=args.dtype, device_map="auto")
    for name, bits in bw_config.items():
        linear = model.get_submodule(name)
        parent_path, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent_path), leaf,
                HummingLinear(layer_dir, name, bits, dtype, linear, device or linear.weight.device))

    hm_mean = run("humming", model, input_ids, args)
    if bf16_mean is not None:
        print(f"Speedup vs BF16: {bf16_mean / hm_mean:.2f}x")


if __name__ == "__main__":
    main()
