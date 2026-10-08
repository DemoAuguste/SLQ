#!/usr/bin/env python3
r"""Build a full HF checkpoint from an SLQ bitwidth config and a quantized layer store.

Usage:
    python slq_build_checkpoint.py --model meta-llama/Llama-3.1-8B-Instruct \
        --layer-dir ./quantized_layers --config <search_output>.txt --output-dir ./models
"""

import argparse
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path

from src.models.loading import load_model, get_tokenizer
from src.utils.common import set_layer_weights
from src.utils.weight_store import WeightStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Header keys of the search outputs copied into the model card.
CARD_KEYS = ("constraint_min_ear", "min_recovery", "final_kl", "final_ear", "calibrated_kl",
             "predicted_kl", "predicted_ear", "predicted_recovery")

VLLM_FUSED = {"q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
              "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}

SERVE_SH = """#!/usr/bin/env bash
# Serve this SLQ checkpoint with the Humming vLLM build (https://github.com/inclusionAI/humming).
# Usage: ./serve.sh [port] [max_model_len]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export VLLM_HUMMING_ONLINE_QUANT_CONFIG="$HERE/humming_online_quant_config.json"
vllm serve "$HERE" --quantization humming --port "${1:-7000}" --dtype bfloat16 \\
    --max-model-len "${2:-8192}" --gpu-memory-utilization 0.8 --max-num-seqs 16 --enforce-eager
"""


def parse_config_file(config_path):
    """Parse '# Key: value' header lines into metadata and 'layer: bits' lines into a config."""
    config, metadata = {}, {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                key, sep, value = line[1:].partition(':')
                if sep:
                    metadata[key.strip().lower().replace(' ', '_')] = value.strip()
            elif ':' in line:
                name, bits = line.rsplit(':', 1)
                config[name.strip()] = int(bits)
    return config, metadata


def humming_quant_config(config, group_size):
    """VLLM_HUMMING_ONLINE_QUANT_CONFIG: asymmetric uint grid, most common bitwidth as default, 'dynamic' overrides."""
    # vLLM matches the overrides against its fused module names.
    fused = {}
    for name, bits in config.items():
        parent, leaf = name.rsplit('.', 1)
        key = f"{parent}.{VLLM_FUSED.get(leaf, leaf)}"
        if fused.setdefault(key, bits) != bits:
            raise ValueError(f"{key} fuses layers of different bitwidths; search with layer groups")
    default_bw = Counter(fused.values()).most_common(1)[0][0]
    qc = {"quant_method": "humming", "dtype": f"uint{default_bw}", "group_size": group_size,
          "has_zero_point": True, "is_fp_zero_point": True}
    dynamic = {f"+:(.*\\.)?{re.escape(k)}$": {"dtype": f"uint{b}"} for k, b in fused.items() if b != default_bw}
    if dynamic:
        qc["dynamic"] = dynamic
    return qc


def model_card(metadata, config, online, group_size, dtype):
    """HF model card (README.md) describing the allocation and how to serve it."""
    base = metadata["model"]
    counts = ", ".join(f"{b}-bit: {n}" for b, n in sorted(Counter(config.values()).items()))
    search = "".join(f"- {k}: {metadata[k]}\n" for k in CARD_KEYS if k in metadata)
    tag = "  - humming\n" if online else ""
    bits = set(config.values())
    desc = (f"Uniform {bits.pop()}-bit weight quantization (SLQ calibration anchor)" if len(bits) == 1
            else "Statistically-lossless mixed-bitwidth weight quantization")
    if online:
        run = ("Needs the Humming vLLM build (https://github.com/inclusionAI/humming), which packs the "
               "weights to their per-layer bitwidths on load. The bundled `serve.sh` sets "
               "`VLLM_HUMMING_ONLINE_QUANT_CONFIG` and serves on port 7000 (`./serve.sh [port] [max_model_len]`).\n")
    else:
        run = f"A standard checkpoint for stock transformers or vLLM:\n\n```bash\nvllm serve <this-dir> --dtype {dtype}\n```\n"
    return (
        f"---\nlibrary_name: transformers\npipeline_tag: text-generation\nbase_model: {base}\n"
        f"tags:\n  - slq\n{tag}  - quantization\n---\n\n"
        f"# {base} quantized with SLQ\n\n"
        f"{desc} of [{base}](https://huggingface.co/{base}); "
        f"weights are stored fake-quantized in {dtype} on the integer grid. See the "
        "[SLQ paper](https://arxiv.org/abs/2605.02404) and the [code](https://github.com/IST-DASLab/SLQ).\n\n"
        "## Quantization summary\n\n"
        f"- Average bitwidth (parameter-weighted): **{metadata.get('average_bitwidth', 'unknown')}** bpp\n"
        f"- Asymmetric INT GPTQ, group size {group_size}\n"
        f"- Layers per bitwidth: {counts}\n{search}\n"
        f"## How to run\n\n{run}\n"
        "`quantization_config.txt` is the SLQ search output (layer: bits).\n\n"
        "## Citation\n\n```bibtex\n@inproceedings{helcig2026slq,\n"
        "  title     = {Statistically-Lossless Quantization of Large Language Models},\n"
        "  author    = {Michael Helcig and Eldar Kurtic and Dan Alistarh},\n"
        "  booktitle = {Conference on Language Modeling (COLM)},\n"
        "  year      = {2026},\n}\n```\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Build a model checkpoint from an SLQ bitwidth config")
    parser.add_argument('--model', required=True, help="base model name or path")
    parser.add_argument('--layer-dir', required=True, help="quantized layer store (slq_quantize.py output)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help="per-layer bitwidth config (layer: bits) from a search")
    source.add_argument('--uniform', type=int, metavar='BITS', help="every layer at BITS (e.g. the TL anchor)")
    parser.add_argument('--output-dir', required=True, help="writes <output-dir>/<model>/<subdir>")
    parser.add_argument('--subdir-name', default=None, help="default: config stem without 'config_', or uniform<BITS>")
    parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--quant-format', default='humming_online', choices=['humming_online', 'fake'],
                        help="humming_online adds the Humming vLLM config and serve.sh")
    parser.add_argument('--group-size', type=int, default=128, help="must match the layer store")
    parser.add_argument('--force', action='store_true', help="overwrite an existing output directory")
    args = parser.parse_args()

    weight_store = WeightStore(args.layer_dir).load()
    if args.uniform is not None:
        config = {n: args.uniform for n in weight_store.get_layer_names()}
        metadata = {}
        subdir = args.subdir_name or f"uniform{args.uniform}"
    else:
        config, metadata = parse_config_file(args.config)
        subdir = args.subdir_name or Path(args.config).stem.removeprefix("config_")
    if not config:
        raise ValueError(f"No layers to quantize (config {args.config}, layer dir {args.layer_dir})")
    metadata.setdefault("model", args.model)

    out = Path(args.output_dir).expanduser() / Path(args.model).name / subdir
    if out.exists():
        if not args.force:
            logger.error(f"{out} exists; pass --force to overwrite")
            return 1
        shutil.rmtree(out)

    model = load_model(args.model, dtype=args.dtype)
    tokenizer = get_tokenizer(args.model)

    if "average_bitwidth" not in metadata:
        modules = dict(model.named_modules())
        numel = {n: modules[n].weight.numel() for n in config}
        metadata["average_bitwidth"] = f"{sum(config[n] * k for n, k in numel.items()) / sum(numel.values()):.3f}"

    for name, bits in config.items():
        set_layer_weights(model, name, weight_store.get_layer_weight(name, bits))

    logger.info(f"Saving to {out}")
    out.mkdir(parents=True)
    # Explicit state_dict avoids transformers' offloaded-parameter resolution, which fails on some
    # architectures (e.g. Qwen3.5) whose parameter names do not match the module hierarchy.
    model.save_pretrained(out, safe_serialization=True, state_dict=model.state_dict())
    tokenizer.save_pretrained(out)
    if args.config:
        shutil.copy2(args.config, out / "quantization_config.txt")
    else:
        (out / "quantization_config.txt").write_text("".join(f"{n}: {b}\n" for n, b in config.items()))

    online = args.quant_format == "humming_online"
    if online:
        with open(out / "humming_online_quant_config.json", "w") as f:
            json.dump(humming_quant_config(config, args.group_size), f, indent=2)
        (out / "serve.sh").write_text(SERVE_SH)
        (out / "serve.sh").chmod(0o755)
    (out / "README.md").write_text(model_card(metadata, config, online, args.group_size, args.dtype))

    logger.info(f"Wrote {out} ({len(config)} layers, avg bitwidth {metadata['average_bitwidth']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
