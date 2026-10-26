"""Low-bit and ternary router emulation (BitNet b1.58-style -1/0/+1 weights, INT4/INT8).

Ternary hardware and BitNet-style models store weights as {-1, 0, +1} times a
scale, and compute matrix products with integer additions and subtractions
only. This module emulates that exactly, so you can measure what it does to
MoE routing before committing a model or a chip design to it:

* `ternarize`: absmean ternary quantization (Ma et al., "The Era of 1-bit
  LLMs", arXiv:2402.17764): scale = mean|W| (per tensor, row, or group),
  codes = round(clip(W / scale, -1, 1)).
* `quantize_int`: symmetric absmax INTn weights (INT4/INT8), per tensor/row/group.
* `quantize_activations`: absmax INT8 per token (BitNet's activation format).
* `lowbit_project`: integer-exact product of quantized activations and codes,
  rescaled once at the end, as a ternary/integer accelerator would compute it.
* `pack_ternary` / `unpack_ternary`: 2 bits per weight, or base-3 with 5 trits
  per byte (1.6 bits per weight). `weight_footprint` reports storage size.

Formats are also accepted by `project` and `PrecisionConfig` as
``weight_format="ternary"`` / ``"int4"`` / ``"int8"`` and ``input_format="int8"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._arrays import as_float64_2d, to_numpy

__all__ = ["QuantizedWeight", "ternarize", "quantize_int", "quantize_activations", "lowbit_project",
           "pack_ternary", "unpack_ternary", "weight_footprint", "LOWBIT_FORMATS"]

LOWBIT_FORMATS = ("ternary", "int2", "int3", "int4", "int8")
_EPS = 1e-5  # BitNet's absmean epsilon


@dataclass(frozen=True)
class QuantizedWeight:
    """Integer codes [E, H] with scales broadcastable over groups of `group_size` columns."""

    codes: np.ndarray        # int8
    scales: np.ndarray       # float64 [E or 1, n_groups]
    group_size: int          # columns per scale group
    bits: float              # nominal bits per weight (1.58 for ternary)

    def dequantize(self) -> np.ndarray:
        return self.codes.astype(np.float64) * np.repeat(self.scales, self.group_size, axis=1)[:, :self.codes.shape[1]]

    @property
    def sparsity(self) -> float:
        """Fraction of zero codes (free in ternary hardware: no add, no memory traffic if skipped)."""
        return float((self.codes == 0).mean())


def _groups(w: np.ndarray, granularity) -> tuple[np.ndarray, int]:
    """Reshape [E, H] to [E or 1, n_groups, group_size] views (zero-padded) for scale computation."""
    e, h = w.shape
    if granularity == "tensor":
        size = h
        grouped = w.reshape(1, e * h)[:, None, :]  # one scale for everything
        return grouped, size
    if granularity == "channel":
        return w[:, None, :], h
    size = int(granularity)
    if size < 1:
        raise ValueError("group size must be >= 1")
    pad = -h % size
    padded = np.pad(w, ((0, 0), (0, pad)))
    return padded.reshape(e, -1, size), size


def _scales_shape(w: np.ndarray, granularity, values: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    if granularity == "tensor":
        return np.full((1, 1), float(values.reshape(-1)[0])), w.shape[1]
    return values.reshape(w.shape[0], -1), size


def ternarize(weight, granularity="tensor") -> QuantizedWeight:
    """Absmean ternary quantization (BitNet b1.58). `granularity`: "tensor", "channel"
    (one scale per expert row), or an integer group size along the hidden dimension."""
    w = as_float64_2d(weight, "weight")
    grouped, size = _groups(w, granularity)
    if granularity in ("tensor", "channel"):
        absmean = np.abs(grouped).mean(axis=-1)
    else:  # mean over the real (unpadded) columns of each group
        counts = np.minimum(size, w.shape[1] - np.arange(grouped.shape[1]) * size)
        absmean = np.abs(grouped).sum(axis=-1) / counts
    scales, size = _scales_shape(w, granularity, absmean + _EPS, size)
    full = np.repeat(scales, size, axis=1)[:, :w.shape[1]]
    codes = np.clip(np.rint(w / full), -1, 1).astype(np.int8)
    return QuantizedWeight(codes, scales, size, bits=float(np.log2(3)))


def quantize_int(weight, bits: int = 4, granularity="channel") -> QuantizedWeight:
    """Symmetric absmax INT`bits` weights: codes in [-(2^(b-1)-1), 2^(b-1)-1]."""
    if not 2 <= int(bits) <= 8:
        raise ValueError("bits must be between 2 and 8")
    w = as_float64_2d(weight, "weight")
    qmax = 2 ** (int(bits) - 1) - 1
    grouped, size = _groups(w, granularity)
    absmax = np.abs(grouped).max(axis=-1)
    scales, size = _scales_shape(w, granularity, np.where(absmax > 0, absmax / qmax, 1.0), size)
    full = np.repeat(scales, size, axis=1)[:, :w.shape[1]]
    codes = np.clip(np.rint(w / full), -qmax, qmax).astype(np.int8)
    return QuantizedWeight(codes, scales, size, bits=float(bits))


def quantize_activations(hidden, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Per-token absmax INT`bits` activations, as in the BitNet b1.58 reference code:
    scale = absmax / (2^(b-1) - 1), codes = clip(round(x / scale), -2^(b-1), 2^(b-1) - 1).
    Returns (int codes [M, H], scales [M, 1]) with hidden ~= codes * scales."""
    x = as_float64_2d(hidden, "hidden states")
    q = 2 ** (int(bits) - 1) - 1
    absmax = np.abs(x).max(axis=1, keepdims=True)
    scales = np.where(absmax > 0, absmax / q, 1.0)
    codes = np.clip(np.rint(x / scales), -q - 1, q).astype(np.int32)
    return codes, scales


def parse_granularity(value, default):
    """None -> default; "tensor"/"channel"; "group128", "g128" or "128" -> 128."""
    if value is None:
        return default
    text = str(value).lower()
    if text in ("tensor", "channel"):
        return text
    digits = text.removeprefix("group").removeprefix("g").strip(":")
    if digits.isdigit():
        return int(digits)
    raise ValueError(f"Unknown granularity {value!r}: use 'tensor', 'channel', or 'group<N>'")


def quantize_weight(weight, fmt: str, granularity=None) -> QuantizedWeight:
    """`fmt` in LOWBIT_FORMATS. Default granularity: per tensor for ternary (BitNet), per channel for INT."""
    if fmt == "ternary":
        return ternarize(weight, parse_granularity(granularity, "tensor"))
    if fmt.startswith("int") and fmt[3:].isdigit():
        return quantize_int(weight, int(fmt[3:]), parse_granularity(granularity, "channel"))
    raise ValueError(f"Unknown low-bit format {fmt!r}; choose from {LOWBIT_FORMATS}")


def lowbit_project(hidden, weight: QuantizedWeight, *, activation_bits: int | None = 8) -> np.ndarray:
    """Router logits as integer/ternary hardware computes them.

    With `activation_bits`, activations are quantized per token and the inner
    products are accumulated exactly in integers (for ternary codes: additions
    and subtractions only), then rescaled once. With ``activation_bits=None``
    the float activations multiply the dequantized weight (weight-only
    quantization). Returns float64 [M, E].
    """
    if activation_bits is None:
        return as_float64_2d(hidden, "hidden states") @ weight.dequantize().T
    codes_x, scale_x = quantize_activations(hidden, activation_bits)
    if codes_x.shape[1] != weight.codes.shape[1]:
        raise ValueError(f"hidden size {codes_x.shape[1]} != weight hidden size {weight.codes.shape[1]}")
    out = np.zeros((codes_x.shape[0], weight.codes.shape[0]))
    n_groups = weight.scales.shape[1]
    for g in range(n_groups):  # exact int64 accumulation within each scale group
        cols = slice(g * weight.group_size, min((g + 1) * weight.group_size, codes_x.shape[1]))
        acc = codes_x[:, cols].astype(np.int64) @ weight.codes[:, cols].astype(np.int64).T
        out += acc.astype(np.float64) * weight.scales[:, g][None, :]
    return out * scale_x


def pack_ternary(codes, scheme: str = "base3") -> bytes:
    """Pack {-1, 0, +1} codes. "2bit": 4 codes per byte (00=0, 01=+1, 10=-1).
    "base3": 5 codes per byte as a base-3 number (3^5 = 243 <= 256): 1.6 bits/weight.
    Unpack with the same scheme and the code count."""
    flat = to_numpy(codes).astype(np.int64).reshape(-1)
    if flat.size and (flat.min() < -1 or flat.max() > 1):
        raise ValueError("ternary codes must be -1, 0, or +1")
    if scheme == "2bit":
        trits = np.where(flat < 0, 2, flat).astype(np.uint8)
        trits = np.pad(trits, (0, -trits.size % 4)).reshape(-1, 4)
        return (trits[:, 0] | trits[:, 1] << 2 | trits[:, 2] << 4 | trits[:, 3] << 6).astype(np.uint8).tobytes()
    if scheme == "base3":
        trits = np.pad(flat + 1, (0, -flat.size % 5)).reshape(-1, 5)  # 0, 1, 2
        return (trits @ (3 ** np.arange(5))).astype(np.uint8).tobytes()
    raise ValueError("scheme must be '2bit' or 'base3'")


def unpack_ternary(data: bytes, count: int, scheme: str = "base3") -> np.ndarray:
    """Inverse of `pack_ternary`: int8 codes of length `count`."""
    raw = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    if scheme == "2bit":
        trits = np.stack([(raw >> s) & 3 for s in (0, 2, 4, 6)], axis=1).reshape(-1)
        if (trits == 3).any():
            raise ValueError("invalid 2-bit ternary code 0b11")
        values = np.where(trits == 2, -1, trits)
    elif scheme == "base3":
        if (raw > 242).any():
            raise ValueError("invalid base-3 byte (> 242)")
        values = np.stack([(raw // 3 ** i) % 3 for i in range(5)], axis=1).reshape(-1) - 1
    else:
        raise ValueError("scheme must be '2bit' or 'base3'")
    if values.size < count:
        raise ValueError(f"{len(data)} bytes hold fewer than {count} codes")
    return values[:count].astype(np.int8)


def weight_footprint(shape, fmt: str, *, group_size: int | None = None, scale_bits: int = 16) -> dict:
    """Storage for a weight of `shape` in `fmt` ("bf16", "fp8", "int4", "ternary-2bit",
    "ternary-base3", ...), including one `scale_bits` scale per group when given."""
    rows, cols = (int(s) for s in shape)
    per_weight = {"fp32": 32, "bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "int4": 4, "int3": 3, "int2": 2,
                  "ternary-2bit": 2, "ternary-base3": 1.6}
    if fmt not in per_weight:
        raise ValueError(f"Unknown format {fmt!r}; choose from {sorted(per_weight)}")
    weight_bits = rows * cols * per_weight[fmt]
    groups = 0 if group_size is None else rows * -(-cols // int(group_size))
    total = weight_bits + groups * scale_bits
    return {"format": fmt, "weights": rows * cols, "bytes": int(np.ceil(total / 8)),
            "bits_per_weight": total / (rows * cols), "vs_bf16": total / (rows * cols * 16)}
