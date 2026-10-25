"""Capture router inputs from a running PyTorch model (requires torch).

    from routerlab.capture import RouterCapture

    with RouterCapture(model, max_tokens=4096) as capture:
        model(**inputs)
    capture.save("router_inputs.safetensors")       # then: routerlab sweep router_inputs.safetensors

Forward pre-hooks record the hidden states that enter every router module
(any module named ``gate``/``router``/``wg`` with a 2-D ``weight`` [E, H] that is
not a dense MLP projection). Weights, correction bias, and linear bias are
read from the modules when saving. Values are stored losslessly (BF16 stays
BF16, everything else FP32).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._arrays import to_numpy
from ._version import __version__
from .checkpoint import SafetensorsFile, save_safetensors
from .routing import RouterSpec

__all__ = ["RouterCapture", "CapturedRouter", "load_capture", "find_router_modules"]

_ROUTER_NAMES = ("gate", "router", "wg")
_BIAS_ATTRS = ("e_score_correction_bias", "expert_bias")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass
class CapturedRouter:
    name: str
    hidden_states: np.ndarray            # [N, H] float32 (exact copy of the captured values)
    weight: np.ndarray                   # [E, H] float32
    correction_bias: np.ndarray | None   # [E]
    linear_bias: np.ndarray | None       # [E]
    hidden_dtype: str
    weight_dtype: str

    @property
    def layer(self) -> int | None:
        match = _LAYER_RE.search(self.name)
        return int(match.group(1)) if match else None


def _weight_of(module):
    weight = getattr(module, "weight", None)
    return weight if weight is not None and getattr(weight, "ndim", 0) == 2 else None


def find_router_modules(model, num_experts: int | None = None) -> dict[str, object]:
    """Router modules by qualified name: named gate/router/wg with a 2-D weight [E, H]."""
    if num_experts is None:
        config = getattr(model, "config", None)
        if config is not None:
            try:
                num_experts = RouterSpec.from_hf_config(config).num_experts
            except (ValueError, TypeError):
                num_experts = None
    found = {}
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in _ROUTER_NAMES:
            continue
        weight = _weight_of(module)
        if weight is None or weight.shape[0] < 2 or (num_experts and weight.shape[0] != num_experts):
            continue
        found[name] = module
    return found


def _find_bias(model, name: str, module, size: int):
    """Correction bias on the router, its parent block, or a sibling statistics module."""
    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
    parent = model.get_submodule(parent_name) if parent_name else model
    holders = [module, parent, getattr(parent, "moe_statics", None)]
    for holder in holders:
        for attr in _BIAS_ATTRS:
            value = getattr(holder, attr, None) if holder is not None else None
            if value is not None and hasattr(value, "numel") and value.numel() == size:
                return value
    return None


class RouterCapture:
    """Context manager recording router inputs. See the module docstring.

    `modules`: explicit {name: module} or list of names (default: auto-detect).
    `max_tokens`: keep at most this many tokens per router (first come).
    """

    def __init__(self, model, modules=None, *, max_tokens: int | None = None,
                 filter: Callable[[str], bool] | None = None):
        self.model = model
        if modules is None:
            modules = find_router_modules(model)
        elif not isinstance(modules, dict):
            modules = {name: model.get_submodule(name) for name in modules}
        if filter is not None:
            modules = {name: m for name, m in modules.items() if filter(name)}
        if not modules:
            raise ValueError("No router modules found; pass modules=[...] explicitly")
        self.modules = modules
        self.max_tokens = max_tokens
        self._chunks: dict[str, list] = {name: [] for name in modules}
        self._counts = {name: 0 for name in modules}
        self._handles = []

    def _hook(self, name, hidden_size):
        def hook(module, args, kwargs=None):
            hidden = args[0] if args else (kwargs or {}).get("hidden_states")
            if hidden is None:
                return
            remaining = None if self.max_tokens is None else self.max_tokens - self._counts[name]
            if remaining is not None and remaining <= 0:
                return
            flat = hidden.detach().reshape(-1, hidden_size)
            if remaining is not None:
                flat = flat[:remaining]
            self._chunks[name].append(flat.to("cpu", copy=True))
            self._counts[name] += flat.shape[0]
        return hook

    def start(self) -> RouterCapture:
        if self._handles:
            return self
        for name, module in self.modules.items():
            hook = self._hook(name, _weight_of(module).shape[1])
            self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def records(self) -> dict[str, CapturedRouter]:
        import torch

        out = {}
        for name, module in self.modules.items():
            if not self._chunks[name]:
                continue
            hidden = torch.cat(self._chunks[name])
            weight = _weight_of(module)
            bias = _find_bias(self.model, name, module, weight.shape[0])
            linear = getattr(module, "bias", None)
            out[name] = CapturedRouter(
                name=name, hidden_states=to_numpy(hidden).astype(np.float32),
                weight=to_numpy(weight).astype(np.float32),
                correction_bias=None if bias is None else to_numpy(bias).astype(np.float32).reshape(-1),
                linear_bias=None if linear is None or not hasattr(linear, "numel")
                else to_numpy(linear).astype(np.float32).reshape(-1),
                hidden_dtype=str(hidden.dtype).replace("torch.", ""),
                weight_dtype=str(weight.dtype).replace("torch.", ""))
        return out

    def save(self, path, *, spec: RouterSpec | None = None) -> None:
        """Write all captured routers to one .safetensors file (load with `load_capture`)."""
        records = self.records()
        if spec is None and getattr(self.model, "config", None) is not None:
            try:
                spec = RouterSpec.from_hf_config(self.model.config)
            except (ValueError, TypeError):
                spec = None
        save_capture(path, records, spec=spec,
                     model_type=getattr(getattr(self.model, "config", None), "model_type", None))


def save_capture(path, records: dict[str, CapturedRouter], *, spec: RouterSpec | None = None,
                 model_type: str | None = None) -> None:
    tensors, bf16, info = {}, [], {}
    for name, rec in records.items():
        tensors[f"{name}/hidden_states"] = rec.hidden_states
        tensors[f"{name}/weight"] = rec.weight
        if rec.hidden_dtype == "bfloat16":
            bf16.append(f"{name}/hidden_states")
        if rec.weight_dtype == "bfloat16":
            bf16.append(f"{name}/weight")
        if rec.correction_bias is not None:
            tensors[f"{name}/correction_bias"] = rec.correction_bias
        if rec.linear_bias is not None:
            tensors[f"{name}/linear_bias"] = rec.linear_bias
        info[name] = {"hidden_dtype": rec.hidden_dtype, "weight_dtype": rec.weight_dtype}
    metadata = {"format": "routerlab-capture/1", "routerlab_version": __version__,
                "routers": json.dumps(info)}
    if spec is not None:
        metadata["router_spec"] = json.dumps(spec.to_dict())
    if model_type:
        metadata["model_type"] = str(model_type)
    save_safetensors(path, tensors, metadata, bf16=bf16)


def load_capture(path) -> tuple[dict[str, CapturedRouter], RouterSpec | None]:
    """Load a capture file. Returns ({router name: CapturedRouter}, RouterSpec or None)."""
    file = SafetensorsFile(path)
    if file.metadata.get("format") != "routerlab-capture/1":
        raise ValueError(f"{path} is not a routerlab capture file")
    info = json.loads(file.metadata.get("routers", "{}"))
    spec = RouterSpec.from_dict(json.loads(file.metadata["router_spec"])) if "router_spec" in file.metadata else None
    def get(name, key):
        full = f"{name}/{key}"
        return file.get(full).astype(np.float32) if full in file else None

    records = {}
    for name, dtypes in info.items():
        records[name] = CapturedRouter(
            name=name, hidden_states=get(name, "hidden_states"), weight=get(name, "weight"),
            correction_bias=get(name, "correction_bias"), linear_bias=get(name, "linear_bias"),
            hidden_dtype=dtypes["hidden_dtype"], weight_dtype=dtypes["weight_dtype"])
    return records, spec
