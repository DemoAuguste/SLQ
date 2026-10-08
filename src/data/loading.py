"""Calibration and evaluation data loaders."""

import os
import random
from typing import Iterable, List, Tuple, Union

import torch
from datasets import load_dataset

MASKED_DATASETS = {"gsm8k", "openthoughts"}

# Eval reads the same seeded FineWeb-Edu stream as calibration but starts this
# many documents in, so with the same seed it never sees a calibration document.
FINEWEB_EVAL_SKIP_DOCS = 100_000

C4_REVISION = "607bd4c8450a42878aa9ddc051a65a055450ef87"


def _ids(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def get_wikitext2(num_samples, seq_length, tokenizer, train=True, seed=42) -> List[torch.Tensor]:
    """Random train windows, or the contiguous test split chunked into seq_length."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train" if train else "test")
    tokens = _ids(tokenizer, "\n\n".join(dataset["text"]))
    if not train:
        return [tokens[:, i:i + seq_length] for i in range(0, tokens.shape[1] - seq_length + 1, seq_length)]
    random.seed(seed)
    starts = [random.randint(0, tokens.shape[1] - seq_length - 1) for _ in range(num_samples)]
    return [tokens[:, i:i + seq_length] for i in starts]


def get_fineweb_edu(num_samples, seq_length, tokenizer, train=True, seed=42) -> List[torch.Tensor]:
    """FineWeb-Edu sample-10BT, streamed with a seeded buffered shuffle and packed into seq_length."""
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    if not train:
        dataset = dataset.skip(FINEWEB_EVAL_SKIP_DOCS)
    data = []
    buffer = torch.empty(1, 0, dtype=torch.int64)
    for sample in dataset:
        if not sample["text"].strip():
            continue
        buffer = torch.cat([buffer, _ids(tokenizer, sample["text"])], dim=1)
        while buffer.shape[1] >= seq_length and len(data) < num_samples:
            data.append(buffer[:, :seq_length])
            buffer = buffer[:, seq_length:]
        if len(data) >= num_samples:
            break
    return data


def get_fineweb_edu_gptq(num_tokens, seq_length, tokenizer, seed=0) -> List[torch.Tensor]:
    """GPTQ calibration: num_tokens from the shuffled first half of FineWeb-Edu sample-10BT, documents split at seq_length."""
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    dataset = dataset.select(range(dataset.num_rows // 2)).shuffle(seed=seed)
    data = []
    for sample in dataset:
        if num_tokens <= 0:
            break
        tokens = _ids(tokenizer, sample["text"])[:, :num_tokens]
        # Long documents are split rather than truncated, which would bias towards document starts.
        while tokens.shape[1] > seq_length:
            data.append(tokens[:, :seq_length])
            tokens = tokens[:, seq_length:]
            num_tokens -= seq_length
        data.append(tokens)
        num_tokens -= tokens.shape[1]
    return data


def get_c4(num_samples, seq_length, tokenizer, train=True, seed=42) -> List[torch.Tensor]:
    """C4 en, first train shard or the first 1100 validation documents, joined by blank lines until seq_length."""
    if train:
        files, split = {"train": "en/c4-train.00000-of-01024.json.gz"}, "train"
    else:
        files, split = {"validation": "en/c4-validation.00000-of-00008.json.gz"}, "validation[:1100]"
    dataset = load_dataset("allenai/c4", "default", data_files=files, split=split, revision=C4_REVISION)
    newlines = _ids(tokenizer, "\n\n")
    data = []
    current = torch.empty(1, 0, dtype=torch.int64)
    for sample in dataset:
        current = torch.cat([current, _ids(tokenizer, sample["text"])], dim=1)
        if current.shape[1] >= seq_length:
            data.append(current[:, :seq_length])
            current = torch.empty(1, 0, dtype=torch.int64)
            if len(data) >= num_samples:
                break
        else:
            current = torch.cat([current, newlines], dim=1)
    return data


def _chat_ids(tokenizer, messages, add_generation_prompt: bool) -> torch.Tensor:
    """apply_chat_template as a (1, L) tensor; recent transformers return a BatchEncoding."""
    out = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=add_generation_prompt)
    return out if isinstance(out, torch.Tensor) else out["input_ids"]


def _tokenize_chat_pairs(
    pairs: Iterable[Tuple[str, str]], num_samples: int, seq_length: int, tokenizer,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Chat-format (user, assistant) pairs into right-padded sequences with answer-only masks."""
    data, masks = [], []
    for user, assistant in pairs:
        if len(data) >= num_samples:
            break
        prompt = [{"role": "user", "content": user}]
        start = _chat_ids(tokenizer, prompt, True).shape[1]
        tokens = _chat_ids(tokenizer, prompt + [{"role": "assistant", "content": assistant}], False)[:, :seq_length]
        end = tokens.shape[1]
        mask = torch.zeros(1, seq_length, dtype=torch.long)
        mask[:, start:end] = 1
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad = torch.full((1, seq_length - end), pad_id, dtype=torch.long)
        data.append(torch.cat([tokens, pad], dim=1))
        masks.append(mask)
    return data, masks


def get_gsm8k(num_samples, seq_length, tokenizer, train=True, seed=42):
    """GSM8K in chat format with answer-only masks."""
    dataset = load_dataset("gsm8k", "main", split="train" if train else "test").shuffle(seed=seed)
    return _tokenize_chat_pairs(((s["question"], s["answer"]) for s in dataset), num_samples, seq_length, tokenizer)


def get_openthoughts(num_samples, seq_length, tokenizer, train=True, seed=42):
    """OpenThoughts-114k (90/10 train/test split) in chat format with answer-only masks; first user and assistant turn."""
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    split_idx = int(0.9 * dataset.num_rows)
    dataset = dataset.select(range(split_idx) if train else range(split_idx, dataset.num_rows)).shuffle(seed=seed)

    def pairs():
        for sample in dataset:
            turns = sample.get("conversations") or []
            user = next((t.get("value", "") for t in turns if t.get("from", "").lower() in ("human", "user")), None)
            reply = next((t.get("value", "") for t in turns if t.get("from", "").lower() in ("gpt", "assistant", "model")), None)
            if user and reply:
                yield user, reply

    return _tokenize_chat_pairs(pairs(), num_samples, seq_length, tokenizer)


LOADERS = {
    "wikitext2": get_wikitext2,
    "fineweb_edu": get_fineweb_edu,
    "c4": get_c4,
    "gsm8k": get_gsm8k,
    "openthoughts": get_openthoughts,
}


def get_data(
    data_name_or_path: str, num_tokens: int, seq_length: int, tokenizer, train: bool = True, seed: int = 42,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """(1, seq_length) token tensors; MASKED_DATASETS return (data, masks) with 1 on answer tokens."""
    num_samples = num_tokens // seq_length
    if os.path.isfile(data_name_or_path):
        return [sample[:, :seq_length] for sample in torch.load(data_name_or_path)[:num_samples]]
    if data_name_or_path not in LOADERS:
        raise ValueError(f"Unknown dataset: {data_name_or_path}")
    return LOADERS[data_name_or_path](num_samples, seq_length, tokenizer, train, seed)
