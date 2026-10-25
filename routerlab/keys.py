"""Selection keys in engine precision, and exact ties at the top-k boundary.

The float64 reference (`route`) rarely sees exact ties. Engines compute the
selection keys (``f(logits) + correction_bias``) in FP32 or BF16, where distinct
values can collapse onto the same number: a small sigmoid score added to a
much larger bias loses its low bits, and saturated sigmoids round to 1.0. When
such a tie straddles the top-k (or group) boundary, the chosen experts depend
on the kernel's tie-breaking, which `torch.topk` leaves unspecified and which
some fused kernels mishandle (sglang#30989: GLM-5.2-NVFP4 routing produced NaN
expert weights from fp32 bias ties).

`boundary_ties` finds those tokens so you can test kernels on them, and so a
precision change that creates them is caught before deployment.
"""

from __future__ import annotations

import numpy as np

from ._arrays import as_float64_2d
from .precision import quantize, resolve_format
from .routing import RouterSpec, score_function, topk_indices

__all__ = ["selection_keys", "boundary_ties"]


def _round(values: np.ndarray, precision) -> np.ndarray:
    return quantize(values, precision).astype(np.float64)


def selection_keys(logits, spec: RouterSpec, precision="fp32", *, on_logits: bool = False) -> np.ndarray:
    """Emulate the keys an engine ranks experts by, computed in `precision`.

    The logits are cast to `precision`, the scoring function is evaluated and
    rounded (a correctly rounded kernel; real kernels may differ by an ulp), and
    the correction bias is cast and added with one rounding. ``on_logits=True``
    models engines that rank by the logits themselves (e.g. Llama 4, GPT-OSS
    top-k before softmax); it is not allowed with a correction bias.
    """
    fmt = resolve_format(precision)
    logits = _round(as_float64_2d(logits, "logits"), fmt)
    spec.validate(logits.shape[1])
    if on_logits:
        if spec.use_correction_bias:
            raise ValueError("on_logits=True cannot model a correction bias (it is added to scores)")
        return logits
    keys = _round(score_function(spec.scoring)(logits), fmt)
    if spec.use_correction_bias:
        if spec.correction_bias is None:
            raise ValueError(f"Router '{spec.name}' needs its correction bias: spec.with_bias(bias)")
        keys = _round(keys + _round(spec.correction_bias, fmt), fmt)
    return keys


def _group_keys(keys: np.ndarray, spec: RouterSpec, fmt) -> np.ndarray:
    m, e = keys.shape
    grouped = keys.reshape(m, spec.n_group, e // spec.n_group)
    if spec.group_score == "max":
        return grouped.max(axis=-1)
    top2 = np.partition(grouped, grouped.shape[-1] - 2, axis=-1)[..., -2:]
    return _round(top2[..., 0] + top2[..., 1], fmt)


def _boundary_tied(values: np.ndarray, k: int) -> np.ndarray:
    """Row has values[k-1] == values[k] after sorting descending (a tie the cut goes through)."""
    if k >= values.shape[1]:
        return np.zeros(values.shape[0], dtype=bool)
    ordered = -np.sort(-values, axis=1)
    return (ordered[:, k - 1] == ordered[:, k]) & np.isfinite(ordered[:, k])


def boundary_ties(logits, spec: RouterSpec, precision="fp32", *, on_logits: bool = False) -> np.ndarray:
    """Boolean [M]: tokens whose selection, computed in `precision`, has an exact tie at
    the group boundary or the expert boundary, so the chosen set depends on tie-breaking."""
    fmt = resolve_format(precision)
    keys = selection_keys(logits, spec, fmt, on_logits=on_logits)
    tied = np.zeros(keys.shape[0], dtype=bool)
    if spec.grouped:
        group_keys = _group_keys(keys, spec, fmt)
        tied |= _boundary_tied(group_keys, spec.topk_group)
        groups = topk_indices(group_keys, spec.topk_group)
        mask = np.zeros(group_keys.shape, dtype=bool)
        np.put_along_axis(mask, groups, True, axis=1)
        keys = np.where(np.repeat(mask, keys.shape[1] // spec.n_group, axis=1), keys, -np.inf)
    return tied | _boundary_tied(keys, spec.top_k)
