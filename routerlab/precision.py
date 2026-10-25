"""Floating-point format emulation.

`quantize` reproduces a round-to-nearest-even cast into a narrower binary
format (BF16, FP16, TF32, FP8 E4M3/E5M2, FP32), including subnormals, signed
zero, NaN propagation, and overflow. It models the *value* of a cast, not a
device's execution speed or flush-to-zero modes.

Results are returned as float32 arrays (every supported format is a subset of
float32), or float64 for ``fp64``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._arrays import to_numpy

__all__ = [
    "FloatFormat", "FORMATS", "resolve_format", "quantize", "quantize_scaled",
    "round_bf16", "is_representable", "unit_roundoff",
]


@dataclass(frozen=True)
class FloatFormat:
    """A binary floating-point format with round-to-nearest-even casts."""

    name: str
    mantissa_bits: int        # explicit fraction bits (excluding the implicit bit)
    min_exponent: int         # exponent of the smallest normal number
    max_value: float          # largest finite value
    has_infinity: bool

    @property
    def unit_roundoff(self) -> float:
        """Relative rounding error bound for normal numbers: 2**-(mantissa_bits + 1)."""
        return 2.0 ** -(self.mantissa_bits + 1)

    @property
    def epsilon(self) -> float:
        """Spacing between 1.0 and the next representable value."""
        return 2.0 ** -self.mantissa_bits

    @property
    def smallest_normal(self) -> float:
        return 2.0 ** self.min_exponent

    @property
    def smallest_subnormal(self) -> float:
        return 2.0 ** (self.min_exponent - self.mantissa_bits)


FORMATS: dict[str, FloatFormat] = {
    "fp64": FloatFormat("fp64", 52, -1022, float(np.finfo(np.float64).max), True),
    "fp32": FloatFormat("fp32", 23, -126, float(np.finfo(np.float32).max), True),
    "tf32": FloatFormat("tf32", 10, -126, (2 - 2.0 ** -10) * 2.0 ** 127, True),
    "bf16": FloatFormat("bf16", 7, -126, (2 - 2.0 ** -7) * 2.0 ** 127, True),
    "fp16": FloatFormat("fp16", 10, -14, 65504.0, True),
    "fp8_e4m3fn": FloatFormat("fp8_e4m3fn", 3, -6, 448.0, False),
    "fp8_e5m2": FloatFormat("fp8_e5m2", 2, -14, 57344.0, True),
}

_ALIASES = {
    "float64": "fp64", "double": "fp64", "f64": "fp64",
    "float32": "fp32", "float": "fp32", "f32": "fp32",
    "tensorfloat32": "tf32",
    "bfloat16": "bf16",
    "float16": "fp16", "half": "fp16", "f16": "fp16",
    "float8_e4m3fn": "fp8_e4m3fn", "f8_e4m3": "fp8_e4m3fn", "e4m3": "fp8_e4m3fn", "e4m3fn": "fp8_e4m3fn",
    "float8_e5m2": "fp8_e5m2", "f8_e5m2": "fp8_e5m2", "e5m2": "fp8_e5m2",
}


def resolve_format(fmt) -> FloatFormat:
    """Resolve a format name, alias, NumPy dtype, or torch dtype to a FloatFormat."""
    if isinstance(fmt, FloatFormat):
        return fmt
    if isinstance(fmt, np.dtype) or (isinstance(fmt, type) and issubclass(fmt, np.generic)):
        key = np.dtype(fmt).name
    else:
        key = str(fmt)
    key = key.lower().replace("torch.", "").replace("-", "_").strip()
    key = _ALIASES.get(key, key)
    if key not in FORMATS:
        raise ValueError(f"Unknown floating-point format {fmt!r}; choose from {sorted(FORMATS)}")
    return FORMATS[key]


def unit_roundoff(fmt) -> float:
    return resolve_format(fmt).unit_roundoff


_OVERFLOW_MODES = ("default", "saturate", "inf", "nan")


def quantize(values, fmt="bf16", *, overflow: str = "default") -> np.ndarray:
    """Round values to `fmt` with round-to-nearest-even and return them widened.

    Overflow handling (`overflow`):

    * ``"default"``: IEEE behaviour (overflow to +/-inf) for formats with
      infinities; for ``fp8_e4m3fn``, saturation to +/-448, which matches
      PyTorch's ``tensor.to(torch.float8_e4m3fn)`` (infinities saturate too).
    * ``"saturate"``: clamp every out-of-range value, including +/-inf, to +/-max.
    * ``"nan"``: out-of-range values become NaN (the OCP non-saturating
      conversion, as implemented by ``ml_dtypes``/JAX for E4M3FN).
    * ``"inf"``: out-of-range values become +/-inf (formats with infinity only).

    NaN inputs stay NaN. Signed zeros are preserved. The input is not modified.
    """
    spec = resolve_format(fmt)
    if overflow not in _OVERFLOW_MODES:
        raise ValueError(f"overflow must be one of {_OVERFLOW_MODES}")
    with np.errstate(invalid="ignore"):  # signalling-NaN payloads raise 'invalid' when widened
        x = np.array(to_numpy(values), dtype=np.float64, copy=True)
    if spec.name == "fp64":
        return x
    finite = np.isfinite(x) & (x != 0)
    v = x[finite]
    _, exponent = np.frexp(v)  # v = m * 2**exponent with 0.5 <= |m| < 1
    exponent = np.maximum(exponent - 1, spec.min_exponent)
    quantum = np.ldexp(1.0, exponent - spec.mantissa_bits)
    # Division and multiplication by a power of two are exact; rint is
    # round-half-to-even, so this is a single correctly rounded cast.
    x[finite] = np.rint(v / quantum) * quantum

    mode = overflow
    if mode == "default":
        mode = "inf" if spec.has_infinity else "saturate"
    if mode == "inf" and not spec.has_infinity:
        raise ValueError(f"{spec.name} has no infinity; use overflow='saturate' or 'nan'")
    with np.errstate(invalid="ignore"):
        over = np.abs(x) > spec.max_value  # NaN compares False and is left alone
    if mode == "saturate":
        x[over] = np.copysign(spec.max_value, x[over])
    elif mode == "nan":
        x[over] = np.nan
    else:
        x[over] = np.copysign(np.inf, x[over])
    with np.errstate(invalid="ignore"):
        return x.astype(np.float32)


def quantize_scaled(values, fmt="fp8_e4m3fn", *, axis: int | None = -1,
                    overflow: str = "saturate") -> np.ndarray:
    """Quantize with an absmax scale, as FP8 weight quantization does.

    `axis=-1` scales each row (per output channel for a weight [E, H]);
    `axis=None` uses a single per-tensor scale. Scales are float32
    ``amax / fmt.max_value``. Returns dequantized values (float32).
    """
    spec = resolve_format(fmt)
    x = to_numpy(values).astype(np.float64)
    amax = np.abs(x).max() if axis is None else np.abs(x).max(axis=axis, keepdims=True)
    scale = np.asarray(amax / spec.max_value, dtype=np.float32).astype(np.float64)
    scale = np.where(scale > 0, scale, 1.0)
    return (quantize(x / scale, spec, overflow=overflow).astype(np.float64) * scale).astype(np.float32)


def round_bf16(values) -> np.ndarray:
    """Round to BF16 (ties-to-even) and return float32. Shorthand for quantize(values, "bf16")."""
    return quantize(values, "bf16")


def is_representable(values, fmt) -> np.ndarray:
    """Elementwise: is the value exactly representable in `fmt`? (NaN counts as representable.)"""
    x = to_numpy(values).astype(np.float64)
    q = quantize(x, fmt, overflow="inf" if resolve_format(fmt).has_infinity else "nan")
    return (q == x) | (np.isnan(x) & np.isnan(q))
