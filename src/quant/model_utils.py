"""Layer selection and input capture for block-wise GPTQ."""
import re
from typing import Any, Dict

import torch
import torch.nn as nn


def to(data: Any, *args, **kwargs):
    """Recursively move tensors in nested tuples, lists and dicts."""
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    if isinstance(data, (tuple, list)):
        return type(data)(to(x, *args, **kwargs) for x in data)
    if isinstance(data, dict):
        return {k: to(v, *args, **kwargs) for k, v in data.items()}
    return data


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def __getattr__(self, name):
        # Some HF models read attributes such as attention_type off the decoder layer before calling it.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["module"], name)

    def forward(self, *input_args, **input_kwargs):
        """Record the block inputs and stop the model forward."""
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt


def select_layers(model: nn.Module, layer_prefix: str, layer_regex: str) -> Dict[str, nn.Linear]:
    """nn.Linear modules whose name starts with layer_prefix and matches layer_regex."""
    return {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.startswith(layer_prefix) and re.search(layer_regex, name)
    }
