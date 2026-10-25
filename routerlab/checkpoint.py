"""Read router weights straight from HuggingFace checkpoints (no torch needed).

Includes a small, dependency-free safetensors reader/writer (BF16 and FP8
tensors are decoded exactly to float32) and router discovery: finds every MoE
gate/router weight, its expert-score correction bias, and its linear bias,
across sharded checkpoints, reading only the bytes of the tensors requested.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .precision import quantize
from .routing import RouterSpec

__all__ = ["SafetensorsFile", "save_safetensors", "Checkpoint", "RouterTensors"]

_NATIVE = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
           "I16": np.int16, "I8": np.int8, "U8": np.uint8, "U16": np.uint16, "U32": np.uint32,
           "U64": np.uint64, "BOOL": np.bool_}
_STORAGE = {**_NATIVE, "BF16": np.uint16, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8}
_TO_TAG = {np.dtype(v).newbyteorder("<"): k for k, v in _NATIVE.items()}
_MAX_HEADER = 100 * 1024 * 1024


def _fp8_table(exponent_bits: int, mantissa_bits: int, finite_only: bool) -> np.ndarray:
    """float32 value of every 8-bit pattern of an FP8 format."""
    codes = np.arange(256, dtype=np.int64)
    sign = np.where(codes >> 7, -1.0, 1.0)
    exp = (codes >> mantissa_bits) & ((1 << exponent_bits) - 1)
    man = codes & ((1 << mantissa_bits) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    frac = man / float(1 << mantissa_bits)
    value = np.where(exp == 0, frac * 2.0 ** (1 - bias), (1 + frac) * 2.0 ** (exp - bias))
    top = exp == (1 << exponent_bits) - 1
    if finite_only:  # E4M3FN: only S.1111.111 is NaN; there is no infinity
        value = np.where(top & (man == (1 << mantissa_bits) - 1), np.nan, value)
    else:            # IEEE-like E5M2: all-ones exponent is inf/NaN
        value = np.where(top, np.where(man == 0, np.inf, np.nan), value)
    return (sign * value).astype(np.float32)


_FP8_TABLES = {"F8_E4M3": _fp8_table(4, 3, True), "F8_E5M2": _fp8_table(5, 2, False)}


class SafetensorsFile:
    """Lazy reader for one .safetensors file; reads only the tensors you ask for."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{self.path} is not a safetensors file (truncated header)")
            (length,) = struct.unpack("<Q", prefix)
            if length > _MAX_HEADER:
                raise ValueError(f"{self.path}: implausible header length {length}")
            header = json.loads(handle.read(length))
        self.metadata: dict[str, str] = header.pop("__metadata__", None) or {}
        self._entries: dict[str, dict] = header
        self._data_start = 8 + length

    def keys(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def dtype(self, name: str) -> str:
        return self._entries[name]["dtype"]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._entries[name]["shape"])

    def get(self, name: str) -> np.ndarray:
        """Load a tensor. BF16/FP8 are decoded exactly to float32."""
        entry = self._entries[name]
        tag = entry["dtype"]
        if tag not in _STORAGE:
            raise ValueError(f"Unsupported safetensors dtype {tag} for {name}")
        storage = np.dtype(_STORAGE[tag]).newbyteorder("<")
        begin, end = entry["data_offsets"]
        count = math.prod(entry["shape"])
        if end - begin != count * storage.itemsize:
            raise ValueError(f"{name}: byte range does not match shape {entry['shape']} and dtype {tag}")
        raw = np.fromfile(self.path, dtype=storage, count=count, offset=self._data_start + begin)
        if tag == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        elif tag in _FP8_TABLES:
            raw = _FP8_TABLES[tag][raw]
        else:
            raw = raw.astype(storage.newbyteorder("="))
        return raw.reshape(entry["shape"])


def save_safetensors(path, tensors: Mapping[str, np.ndarray], metadata: Mapping[str, str] | None = None,
                     *, bf16: Iterable[str] = ()) -> None:
    """Write NumPy arrays to a .safetensors file. Keys listed in `bf16` are stored as
    BF16 (round-to-nearest-even). Readable by the `safetensors` library."""
    bf16 = set(bf16)
    header: dict[str, object] = {}
    blobs, offset = [], 0
    for name, value in tensors.items():
        array = np.asarray(value)
        if name in bf16:
            bits = (quantize(array, "bf16").view(np.uint32) >> 16).astype("<u2")
            tag, data = "BF16", np.ascontiguousarray(bits)
        else:
            data = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
            tag = _TO_TAG.get(data.dtype)
            if tag is None:
                raise TypeError(f"Cannot store dtype {array.dtype} for {name}")
        payload = data.tobytes()
        header[name] = {"dtype": tag, "shape": list(array.shape), "data_offsets": [offset, offset + len(payload)]}
        blobs.append(payload)
        offset += len(payload)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for payload in blobs:
            handle.write(payload)


@dataclass(frozen=True)
class RouterTensors:
    """Location of one router in a checkpoint."""

    module: str                     # e.g. "model.layers.3.mlp.gate"
    layer: int | None
    weight: str                     # tensor names below
    num_experts: int
    hidden_size: int
    dtype: str
    correction_bias: str | None = None
    linear_bias: str | None = None
    hash_table: str | None = None   # token-id -> expert table (hash routing, e.g. DeepSeek-V4)

    def to_dict(self):
        return asdict(self)


_ROUTER_NAMES = ("gate", "router", "wg")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class Checkpoint:
    """A HuggingFace-style checkpoint: a directory (config.json + safetensors shards,
    optionally with model.safetensors.index.json) or a single .safetensors file."""

    def __init__(self, path):
        path = Path(path)
        if path.is_file():
            root, files = path.parent, [path]
        elif path.is_dir():
            root = path
            files = sorted(path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No .safetensors files in {path}")
        else:
            raise FileNotFoundError(path)
        self.root = root
        config_path = root / "config.json"
        self.config: dict | None = json.loads(config_path.read_text()) if config_path.is_file() else None
        self._files: dict[Path, SafetensorsFile] = {}
        self._where: dict[str, Path] = {}
        index = root / "model.safetensors.index.json"
        if path.is_dir() and index.is_file():
            for name, filename in json.loads(index.read_text())["weight_map"].items():
                self._where[name] = root / filename
        else:
            for file in files:
                for name in self._open(file).keys():
                    self._where[name] = file

    def _open(self, file: Path) -> SafetensorsFile:
        if file not in self._files:
            self._files[file] = SafetensorsFile(file)
        return self._files[file]

    def names(self) -> list[str]:
        return list(self._where)

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def info(self, name: str) -> tuple[str, tuple[int, ...]]:
        file = self._open(self._where[name])
        return file.dtype(name), file.shape(name)

    def load(self, name: str, *, dequantize: bool = True) -> np.ndarray:
        """Load a tensor as a NumPy array. FP8 weights with a DeepSeek-style
        ``<name>_scale_inv`` block scale are dequantized to float32."""
        array = self._open(self._where[name]).get(name)
        scale_name = f"{name}_scale_inv"
        if dequantize and scale_name in self._where and array.ndim == 2:
            scale = self._open(self._where[scale_name]).get(scale_name).astype(np.float32)
            if scale.ndim != 2:
                raise ValueError(f"{scale_name}: expected a 2-D block scale, got shape {scale.shape}")
            rows, cols = array.shape
            block = self._block_size()
            if block is None or [math.ceil(n / b) for n, b in zip((rows, cols), block, strict=True)] != \
                    list(scale.shape):
                # No (consistent) quantization_config: 128x128 blocks if they fit, else infer.
                block = [128 if math.ceil(n / 128) == s else math.ceil(n / s)
                         for n, s in zip((rows, cols), scale.shape, strict=True)]
            expanded = np.repeat(np.repeat(scale, block[0], axis=0), block[1], axis=1)[:rows, :cols]
            array = (array.astype(np.float32) * expanded).astype(np.float32)
        return array

    def _block_size(self) -> list[int] | None:
        quant = (self.config or {}).get("quantization_config") or {}
        block = quant.get("weight_block_size")
        return [int(b) for b in block] if isinstance(block, (list, tuple)) and len(block) == 2 else None

    # ------------------------------------------------------------------ routers
    def _config_value(self, *keys):
        config = self.config or {}
        sources = [config] + ([config["text_config"]] if isinstance(config.get("text_config"), dict) else [])
        for source in sources:
            for key in keys:
                if source.get(key) is not None:
                    return source[key]
        return None

    def router_spec(self) -> RouterSpec:
        if self.config is None:
            raise ValueError(f"No config.json next to {self.root}; build RouterSpec manually")
        return RouterSpec.from_hf_config(self.config)

    def find_routers(self) -> list[RouterTensors]:
        """Every MoE router weight [E, H] in the checkpoint, ordered by layer."""
        experts = self._config_value("n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts")
        hidden = self._config_value("hidden_size")
        try:
            uses_bias = self.router_spec().use_correction_bias if self.config is not None else None
        except ValueError:
            uses_bias = None
        found = []
        for name in self._where:
            if not name.endswith(".weight"):
                continue
            module = name[: -len(".weight")]
            parts = module.split(".")
            # gate / router / wg, or Granite's router.layer
            if parts[-1] not in _ROUTER_NAMES and parts[-2:] != ["router", "layer"]:
                continue
            dtype, shape = self.info(name)
            if len(shape) != 2 or shape[0] < 2:
                continue
            if (experts is not None and shape[0] != int(experts)) or (hidden is not None and shape[1] != int(hidden)):
                continue
            match = _LAYER_RE.search(module)
            correction = self._find_bias(module, shape[0])
            linear = self._sized(f"{module}.bias", shape[0])
            if correction is None and linear is not None and uses_bias:
                # e.g. DeepSeek-V4 checkpoints store e_score_correction_bias as `<gate>.bias`;
                # no supported router with a correction bias also has a linear bias.
                correction, linear = linear, None
            found.append(RouterTensors(
                module=module, layer=int(match.group(1)) if match else None, weight=name,
                num_experts=shape[0], hidden_size=shape[1], dtype=dtype,
                correction_bias=correction, linear_bias=linear,
                hash_table=f"{module}.tid2eid" if f"{module}.tid2eid" in self._where else None))
        return sorted(found, key=lambda r: (r.layer is None, r.layer or 0, r.module))

    def _sized(self, name: str, size: int) -> str | None:
        if name in self._where and math.prod(self.info(name)[1]) == size:
            return name
        return None

    def _find_bias(self, module: str, size: int) -> str | None:
        parent = module.rsplit(".", 1)[0] if "." in module else ""
        candidates = [f"{module}.e_score_correction_bias", f"{parent}.e_score_correction_bias",
                      f"{parent}.moe_statics.e_score_correction_bias", f"{parent}.expert_bias",
                      f"{module}.expert_bias"]
        for name in candidates:
            if self._sized(name, size):
                return name
        return None

    def load_router(self, router: RouterTensors | int | str):
        """Return (weight [E,H], correction_bias [E] or None, linear_bias [E] or None).
        `router` may be a RouterTensors, a layer index, or a module/weight name."""
        if not isinstance(router, RouterTensors):
            matches = [r for r in self.find_routers()
                       if (isinstance(router, int) and r.layer == router) or router in (r.module, r.weight)]
            if len(matches) != 1:
                raise KeyError(f"Expected exactly one router matching {router!r}, found {len(matches)}")
            router = matches[0]
        bias = None if router.correction_bias is None else self.load(router.correction_bias).reshape(-1)
        linear = None if router.linear_bias is None else self.load(router.linear_bias).reshape(-1)
        return self.load(router.weight), bias, linear
