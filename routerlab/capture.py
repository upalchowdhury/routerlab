"""Capture router inputs from a running PyTorch model (requires torch).

    from routerlab.capture import RouterCapture

    with RouterCapture(model, max_tokens=4096) as capture:
        model(**inputs)
    capture.save("router_inputs.safetensors")       # then: routerlab sweep router_inputs.safetensors

Forward pre-hooks record the hidden states that enter every router module
(any module named ``gate``/``router``/``wg`` with a 2-D ``weight`` [E, H] that is
not a dense MLP projection), and forward hooks record the router logits the
module actually computed (in its own dtype and kernels), so two runs can be
compared with ``routerlab compare run_a.safetensors run_b.safetensors``.
Weights, correction bias, and linear bias are read from the modules when
saving. Values are stored losslessly (BF16 stays BF16, FP64 stays FP64,
everything else FP32).
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

__all__ = ["RouterCapture", "CapturedRouter", "load_capture", "save_capture", "find_router_modules",
           "is_capture_file"]

_ROUTER_NAMES = ("gate", "router", "wg")
_BIAS_ATTRS = ("e_score_correction_bias", "expert_bias")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass
class CapturedRouter:
    name: str
    hidden_states: np.ndarray            # [N, H] float32 (float64 for float64 models): exact values
    weight: np.ndarray                   # [E, H] float32 / float64
    correction_bias: np.ndarray | None   # [E]
    linear_bias: np.ndarray | None       # [E]
    hidden_dtype: str
    weight_dtype: str
    logits: np.ndarray | None = None     # [N, E] as computed by the module (None if not identified)
    logits_dtype: str | None = None

    @property
    def layer(self) -> int | None:
        match = _LAYER_RE.search(self.name)
        return int(match.group(1)) if match else None


def _exact(tensor) -> np.ndarray:
    """float32 for <= 32-bit floats (exact for BF16/FP16), float64 kept as float64."""
    array = to_numpy(tensor)
    return array if array.dtype == np.float64 else array.astype(np.float32)


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
    holders = [module, getattr(module, "moe_statics", None), parent, getattr(parent, "moe_statics", None)]
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
    `inputs` / `outputs`: record hidden states and/or the module's router logits.
    """

    def __init__(self, model, modules=None, *, max_tokens: int | None = None,
                 filter: Callable[[str], bool] | None = None, inputs: bool = True, outputs: bool = True):
        self.model = model
        if modules is None:
            modules = find_router_modules(model)
        elif not isinstance(modules, dict):
            modules = {name: model.get_submodule(name) for name in modules}
        if filter is not None:
            modules = {name: m for name, m in modules.items() if filter(name)}
        if not modules:
            raise ValueError("No router modules found; pass modules=[...] explicitly")
        if not (inputs or outputs):
            raise ValueError("Capture inputs, outputs, or both")
        self.modules = modules
        self.max_tokens = max_tokens
        self.inputs, self.outputs = inputs, outputs
        self._chunks: dict[str, list] = {name: [] for name in modules}
        self._logit_chunks: dict[str, list] = {name: [] for name in modules}
        self._counts = {name: 0 for name in modules}
        self._pending: dict[str, tuple] = {}
        self._output_slot: dict[str, int | None] = {}
        self._handles = []

    def _pre_hook(self, name, hidden_size):
        def hook(module, args, kwargs=None):
            hidden = args[0] if args else (kwargs or {}).get("hidden_states")
            if hidden is None:
                return
            remaining = None if self.max_tokens is None else self.max_tokens - self._counts[name]
            if remaining is not None and remaining <= 0:
                return
            flat = hidden.detach().reshape(-1, hidden_size)
            take = flat.shape[0] if remaining is None else min(remaining, flat.shape[0])
            self._pending[name] = (flat, take)
            if self.inputs:
                self._chunks[name].append(flat[:take].to("cpu", copy=True))
            self._counts[name] += take
        return hook

    def _post_hook(self, name):
        def hook(module, args, kwargs, output):
            pending = self._pending.pop(name, None)
            if pending is None:
                return
            flat, take = pending
            logits = self._find_logits(name, module, flat, output)
            if logits is not None:
                self._logit_chunks[name].append(logits[:take].detach().to("cpu", copy=True))
        return hook

    def _find_logits(self, name, module, flat, output):
        """The output tensor equal to hidden @ W.T (+ b): router modules return it in
        different tuple positions (and Llama 4 returns sigmoid scores first)."""
        import torch

        weight = _weight_of(module)
        rows, experts = flat.shape[0], weight.shape[0]
        items = list(output) if isinstance(output, (tuple, list)) else [output]
        candidates = [(i, t.reshape(rows, experts)) for i, t in enumerate(items)
                      if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel() == rows * experts]
        slot = self._output_slot.get(name, -1)
        if slot != -1:
            return next((t for i, t in candidates if i == slot), None)
        probe = min(rows, 8)

        def cpu64(t):
            # Move first, then widen: MPS has no float64, and on torch 2.14 a combined
            # .to("cpu", torch.float64) of an MPS bfloat16 tensor returns wrong values.
            return t.detach().to("cpu").to(torch.float64)

        with torch.no_grad():
            expected = cpu64(flat[:probe]) @ cpu64(weight).T
            linear = getattr(module, "bias", None)
            if isinstance(linear, torch.Tensor) and linear.numel() == experts:
                expected = expected + cpu64(linear)
            scale = expected.abs().max().clamp_min(1e-30)
            errors = [(float((cpu64(t[:probe]) - expected).abs().max() / scale), i, t) for i, t in candidates]
        best = min(errors, default=None, key=lambda e: e[0])
        # 5% of the largest logit tolerates FP8/BF16 router GEMMs but rejects scores/probabilities.
        self._output_slot[name] = best[1] if best is not None and best[0] < 0.05 else None
        return best[2] if self._output_slot[name] is not None else None

    def start(self) -> RouterCapture:
        if self._handles:
            return self
        for name, module in self.modules.items():
            hook = self._pre_hook(name, _weight_of(module).shape[1])
            self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
            if self.outputs:
                self._handles.append(module.register_forward_hook(self._post_hook(name), with_kwargs=True))
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
            if not self._chunks[name] and not self._logit_chunks[name]:
                continue
            hidden = torch.cat(self._chunks[name]) if self._chunks[name] else None
            logits = torch.cat(self._logit_chunks[name]) if self._logit_chunks[name] else None
            weight = _weight_of(module)
            bias = _find_bias(self.model, name, module, weight.shape[0])
            linear = getattr(module, "bias", None)
            out[name] = CapturedRouter(
                name=name, hidden_states=None if hidden is None else _exact(hidden), weight=_exact(weight),
                correction_bias=None if bias is None else to_numpy(bias).astype(np.float32).reshape(-1),
                linear_bias=None if linear is None or not hasattr(linear, "numel")
                else to_numpy(linear).astype(np.float32).reshape(-1),
                hidden_dtype=None if hidden is None else str(hidden.dtype).replace("torch.", ""),
                weight_dtype=str(weight.dtype).replace("torch.", ""),
                logits=None if logits is None else _exact(logits),
                logits_dtype=None if logits is None else str(logits.dtype).replace("torch.", ""))
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
        if rec.hidden_states is not None:
            tensors[f"{name}/hidden_states"] = rec.hidden_states
        tensors[f"{name}/weight"] = rec.weight
        if rec.logits is not None:
            tensors[f"{name}/logits"] = rec.logits
        for key, dtype in (("hidden_states", rec.hidden_dtype), ("weight", rec.weight_dtype),
                           ("logits", rec.logits_dtype)):
            if dtype == "bfloat16" and f"{name}/{key}" in tensors:
                bf16.append(f"{name}/{key}")
        if rec.correction_bias is not None:
            tensors[f"{name}/correction_bias"] = rec.correction_bias
        if rec.linear_bias is not None:
            tensors[f"{name}/linear_bias"] = rec.linear_bias
        info[name] = {"hidden_dtype": rec.hidden_dtype, "weight_dtype": rec.weight_dtype,
                      "logits_dtype": rec.logits_dtype}
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
        return file.get(full) if full in file else None

    records = {}
    for name, dtypes in info.items():
        records[name] = CapturedRouter(
            name=name, hidden_states=get(name, "hidden_states"), weight=get(name, "weight"),
            correction_bias=get(name, "correction_bias"), linear_bias=get(name, "linear_bias"),
            hidden_dtype=dtypes["hidden_dtype"], weight_dtype=dtypes["weight_dtype"],
            logits=get(name, "logits"), logits_dtype=dtypes.get("logits_dtype"))
    return records, spec


def is_capture_file(path) -> bool:
    try:
        return SafetensorsFile(path).metadata.get("format") == "routerlab-capture/1"
    except (OSError, ValueError):
        return False
