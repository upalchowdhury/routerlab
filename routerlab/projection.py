"""Router projection (logits = hidden @ W.T + b) with precision emulation and error bounds."""

from __future__ import annotations

import math

import numpy as np

from ._arrays import as_float64_2d, as_vector
from .lowbit import LOWBIT_FORMATS, lowbit_project, quantize_activations, quantize_weight
from .precision import quantize, quantize_scaled, resolve_format

__all__ = ["project", "projection_error_bound"]

_ACCUMULATE = ("fp64", "fp32")


def _cast(values: np.ndarray, fmt, scaling: str | None) -> np.ndarray:
    if fmt is None:
        return values
    if fmt == "int8":  # activations: per-token absmax INT8 (fake-quantized)
        codes, scales = quantize_activations(values, 8)
        return codes * scales
    if scaling is None:
        return quantize(values, fmt).astype(np.float64)
    if scaling not in ("tensor", "channel"):
        raise ValueError("scaling must be None, 'tensor', or 'channel' for float formats")
    return quantize_scaled(values, fmt, axis=None if scaling == "tensor" else -1).astype(np.float64)


def project(hidden, weight, linear_bias=None, *, input_format=None, weight_format=None,
            weight_scaling: str | None = None, accumulate: str = "fp64", output_format=None) -> np.ndarray:
    """Compute router logits [M, E] from hidden states [..., H] and weight [E, H].

    * ``input_format`` / ``weight_format``: emulate casting the operands first
      (e.g. ``"bf16"``); ``weight_scaling="channel"|"tensor"`` applies absmax
      scaling as FP8 weight quantization does. Low-bit weights
      (``"ternary"``, ``"int4"``, ...; granularity via ``weight_scaling``,
      e.g. ``"group128"``) and ``input_format="int8"`` (per-token absmax) use
      the integer-exact path of `routerlab.lowbit.lowbit_project`.
    * ``accumulate``: ``"fp64"`` (reference: the float64 product of the given
      values) or ``"fp32"`` (NumPy float32 GEMM: FP32 accumulation with a BLAS
      summation order, representative in magnitude but not bitwise equal to a
      GPU kernel).
    * ``output_format``: emulate the output dtype (e.g. ``"bf16"`` logits).

    Returns float64 (holding exactly the emulated values).
    """
    if accumulate not in _ACCUMULATE:
        raise ValueError(f"accumulate must be one of {_ACCUMULATE}")
    w = as_float64_2d(weight, "weight")
    x = as_float64_2d(hidden, "hidden states", width=w.shape[1])
    if weight_format in LOWBIT_FORMATS:
        # Integer hardware path: exact integer accumulation (ternary: adds/subtracts only).
        if input_format not in (None, "int8"):
            x = _cast(x, input_format, None)
        logits = lowbit_project(x, quantize_weight(w, weight_format, weight_scaling),
                                activation_bits=8 if input_format == "int8" else None)
        if linear_bias is not None:
            logits = logits + as_vector(linear_bias, "linear_bias", w.shape[0])
        return _finish(logits, output_format)
    x = _cast(x, input_format, None)
    w = _cast(w, weight_format, weight_scaling)
    if accumulate == "fp32":
        x32, w32 = x.astype(np.float32), w.astype(np.float32)
        if not (np.array_equal(x32, x) and np.array_equal(w32, w)):
            raise ValueError("FP32 accumulation needs FP32-representable operands; set input_format/"
                             "weight_format (e.g. 'fp32') to emulate the cast explicitly")
        logits = (x32 @ w32.T)
        if linear_bias is not None:
            bias = as_vector(linear_bias, "linear_bias", w.shape[0])
            if not np.array_equal(bias.astype(np.float32), bias):
                raise ValueError("FP32 accumulation needs an FP32-representable linear_bias; "
                                 "round it explicitly (e.g. quantize(bias, 'fp32'))")
            logits = logits + bias.astype(np.float32)
        logits = logits.astype(np.float64)
    else:
        logits = x @ w.T
        if linear_bias is not None:
            logits = logits + as_vector(linear_bias, "linear_bias", w.shape[0])
    return _finish(logits, output_format)


def _finish(logits: np.ndarray, output_format) -> np.ndarray:
    if not np.isfinite(logits).all():
        raise FloatingPointError("Projection overflowed (non-finite logits) under this precision")
    if output_format is not None:
        logits = quantize(logits, output_format).astype(np.float64)
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"Logits overflow {resolve_format(output_format).name}")
    return logits


def projection_error_bound(hidden, weight, linear_bias=None, *, accumulate="fp32",
                           output_format=None, method: str = "deterministic", lam: float = 6.0,
                           unit_roundoff: float | None = None) -> np.ndarray:
    """Per-element bound [M, E] on |computed logit - exact logit| for the given operands.

    Covers rounding during the dot product in `accumulate` precision (any
    summation order, blocking, split-K or FMA usage, gradual underflow) and,
    optionally, the final cast to `output_format`. Operand casts are *not* included: pass the operands
    the kernel actually multiplies (e.g. BF16-rounded weights).

    * ``method="deterministic"``: worst case, ``gamma_n * sum_i |x_i w_i|`` with
      ``gamma_n = n u / (1 - n u)`` (Higham, *Accuracy and Stability of
      Numerical Algorithms*, 2002, sec. 3.1). Always valid, often pessimistic.
    * ``method="probabilistic"``: ``(exp(lam sqrt(n) u + n u^2/(1-u)) - 1) *
      sum_i |x_i w_i|`` (Higham & Mary, SIAM J. Sci. Comput., 2019), which holds
      with high probability when rounding errors behave like independent
      zero-mean random variables. Much tighter for large H; not a guarantee.

    `unit_roundoff` overrides u for the accumulator (e.g. ``2**-23`` for
    accumulators that truncate instead of rounding to nearest).
    """
    if method not in ("deterministic", "probabilistic"):
        raise ValueError("method must be 'deterministic' or 'probabilistic'")
    w = as_float64_2d(weight, "weight")
    x = as_float64_2d(hidden, "hidden states", width=w.shape[1])
    magnitude = np.abs(x) @ np.abs(w).T
    n = w.shape[1]
    exact = x @ w.T
    if linear_bias is not None:
        bias = as_vector(linear_bias, "linear_bias", w.shape[0])
        magnitude = magnitude + np.abs(bias)
        exact = exact + bias
        n += 1
    u = resolve_format(accumulate).unit_roundoff if unit_roundoff is None else float(unit_roundoff)
    if method == "deterministic":
        if n * u >= 1:
            raise ValueError("n * u >= 1: the deterministic bound is undefined")
        gamma = n * u / (1 - n * u)
    else:
        gamma = math.expm1(lam * math.sqrt(n) * u + n * u * u / (1 - u))
    acc = resolve_format(accumulate)
    # Underflow: each of the <= 2n products/additions may add half a subnormal spacing.
    bound = gamma * magnitude + n * acc.smallest_subnormal
    if output_format is not None:
        out = resolve_format(output_format)
        # Relative rounding for normal results plus half the subnormal spacing for tiny ones.
        bound = bound + out.unit_roundoff * (np.abs(exact) + bound) + 0.5 * out.smallest_subnormal
    return bound
