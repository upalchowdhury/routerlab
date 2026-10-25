"""Margin certificates: when can a logit error change which experts are chosen?

Given reference logits and a bound on each logit's error, `certify` proves
(per token) that *every* perturbation within the bound selects exactly the same
expert set, including group-limited selection and correction bias. It uses
interval arithmetic on the router's selection scores:

* For selection that is monotone in the logits (softmax/sigmoid without bias,
  optional max-grouping), the intervals are the logits themselves +/- error.
  The result is exact: a token fails to certify only if some perturbation
  within the bound really can change (or tie) its selection.
* With a correction bias or top-2-sum grouping, each score is bounded through
  its scoring function: elementwise for sigmoid/sqrtsoftplus (exact), and with
  joint softmax bounds otherwise (sound, slightly conservative).

`robustness_radius` gives each token's margin in logit units: the largest
uniform logit error the token tolerates. Compare it with your kernel's error.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._arrays import as_float64_2d, broadcast_bound
from .routing import RouterSpec, _group_scores, _select, score_function, selection_scores

__all__ = ["certify", "robustness_radius", "selection_bounds"]

_EPS = np.finfo(np.float64).eps


def _pad(lo: np.ndarray, hi: np.ndarray):
    """Widen intervals by a few float64 ulps so our own rounding cannot unsound them."""
    pad = 8 * _EPS * np.maximum(np.abs(lo), np.abs(hi)) + np.finfo(np.float64).tiny
    return lo - pad, hi + pad


def selection_bounds(logits, spec: RouterSpec, error):
    """Lower/upper bounds [M, E] of the selection scores under |logit error| <= `error`."""
    logits = as_float64_2d(logits, "logits")
    spec.validate(logits.shape[1])
    err = broadcast_bound(error, logits.shape, "error")
    return _bounds(logits, spec, err)


def _bounds(logits: np.ndarray, spec: RouterSpec, err: np.ndarray):
    if spec.selects_on_logits:
        return _pad(logits - err, logits + err)
    a, b = logits - err, logits + err
    if spec.scoring == "softmax":
        # p_i is increasing in l_i and decreasing in every other logit, so these are exact
        # per-element bounds (not jointly attainable, hence conservative across experts).
        c = b.max(axis=1, keepdims=True)
        ea, eb = np.exp(a - c), np.exp(b - c)
        sa, sb = ea.sum(axis=1, keepdims=True), eb.sum(axis=1, keepdims=True)
        hi = eb / (eb + np.maximum(sa - ea, 0.0))
        lo = ea / (ea + np.maximum(sb - eb, 0.0))
    else:
        f = score_function(spec.scoring)  # elementwise and increasing
        lo, hi = f(a), f(b)
    if spec.use_correction_bias:
        if spec.correction_bias is None:
            raise ValueError(f"Router '{spec.name}' needs its correction bias: spec.with_bias(bias)")
        lo, hi = lo + spec.correction_bias, hi + spec.correction_bias
    return _pad(lo, hi)


def _certified_from_bounds(lo, hi, ids, group_mask, spec: RouterSpec) -> np.ndarray:
    m, e = lo.shape
    ok = np.ones(m, dtype=bool)
    eligible = np.ones((m, e), dtype=bool)
    if spec.grouped:
        glo, ghi = _group_scores(lo, spec), _group_scores(hi, spec)
        worst_chosen = np.where(group_mask, glo, np.inf).min(axis=1)
        best_other = np.where(group_mask, -np.inf, ghi).max(axis=1)
        ok &= worst_chosen > best_other
        eligible = np.repeat(group_mask, e // spec.n_group, axis=1)
    chosen = np.zeros((m, e), dtype=bool)
    np.put_along_axis(chosen, ids, True, axis=1)
    worst_chosen = np.where(chosen, lo, np.inf).min(axis=1)
    best_other = np.where(eligible & ~chosen, hi, -np.inf).max(axis=1)
    return ok & (worst_chosen > best_other)


def certify(logits, spec: RouterSpec, error) -> np.ndarray:
    """Boolean [M]: True where no logit perturbation with |delta| <= `error` can change
    (or tie) the selected expert set. `error` may be a scalar, per-row [M, 1],
    per-expert [E], or per-element [M, E] bound."""
    logits = as_float64_2d(logits, "logits")
    spec.validate(logits.shape[1])
    err = broadcast_bound(error, logits.shape, "error")
    ids, group_mask = _select(logits, spec)
    lo, hi = _bounds(logits, spec, err)
    return _certified_from_bounds(lo, hi, ids, group_mask, spec)


def _kth_gap(values: np.ndarray, k: int) -> np.ndarray:
    """values[k-1] - values[k] of each row sorted descending (inf when k == width)."""
    if k >= values.shape[1]:
        return np.full(values.shape[0], np.inf)
    ordered = -np.sort(-values, axis=1)
    return ordered[:, k - 1] - ordered[:, k]


# Bound on how far one selection score can move per unit of uniform logit error
# (infinity norm): |sigmoid'| <= 1/4, |d sqrt(softplus)| <= 1/2, and for softmax
# sum_j |dp_i/dl_j| = 2 p_i (1 - p_i) <= 1/2.
_SLOPE = {"sigmoid": 0.25, "sqrtsoftplus": 0.5, "softmax": 0.5}


def _radius_lower_bound(logits, ids, group_mask, spec: RouterSpec) -> np.ndarray:
    """Cheap per-row lower bound on the radius, from score gaps and the slope bound."""
    m, e = logits.shape
    choice = selection_scores(logits, spec)
    slope = _SLOPE[spec.scoring]
    bound = np.full(m, np.inf)
    eligible = np.ones((m, e), dtype=bool)
    if spec.grouped:
        groups = _group_scores(choice, spec)
        gap = np.where(group_mask, groups, np.inf).min(axis=1) - np.where(group_mask, -np.inf, groups).max(axis=1)
        moves = 2 if spec.group_score == "top2_sum" else 1
        bound = gap / (2 * moves * slope)
        eligible = np.repeat(group_mask, e // spec.n_group, axis=1)
    chosen = np.zeros((m, e), dtype=bool)
    np.put_along_axis(chosen, ids, True, axis=1)
    gap = np.where(chosen, choice, np.inf).min(axis=1) - np.where(eligible & ~chosen, choice, -np.inf).max(axis=1)
    return np.maximum(np.minimum(bound, gap / (2 * slope)), 0.0)


def robustness_radius(logits, spec: RouterSpec, *, rtol: float = 1e-4, workers: int | None = None) -> np.ndarray:
    """Per-token margin in logit units: the supremum of uniform errors e such that
    every perturbation with |delta| <= e keeps the selected expert set.

    Closed form when selection is monotone in the logits (half the gap at the
    k-th/(k+1)-th boundary, and at the group boundary); otherwise found by
    bisection on `certify` to relative precision `rtol` (the result never
    exceeds the true radius by more than that). 0 means an exact tie at the
    boundary; inf means the selection can never change (e.g. top_k == E).
    Large inputs are processed in row chunks on `workers` threads (default:
    up to 8; NumPy releases the GIL in the heavy kernels).
    """
    logits = as_float64_2d(logits, "logits")
    spec.validate(logits.shape[1])
    if spec.selects_on_logits or logits.shape[0] <= _CHUNK:
        return _radius(logits, spec, rtol)
    chunks = [logits[i:i + _CHUNK] for i in range(0, logits.shape[0], _CHUNK)]
    workers = min(len(chunks), workers or min(8, os.cpu_count() or 1))
    if workers <= 1:
        return np.concatenate([_radius(chunk, spec, rtol) for chunk in chunks])
    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(lambda chunk: _radius(chunk, spec, rtol), chunks)))


_CHUNK = 4096


def _radius(logits: np.ndarray, spec: RouterSpec, rtol: float) -> np.ndarray:
    m, e = logits.shape
    ids, group_mask = _select(logits, spec)
    if spec.selects_on_logits:
        if not spec.grouped:
            return _kth_gap(logits, spec.top_k) / 2
        group_radius = _kth_gap(_group_scores(logits, spec), spec.topk_group) / 2
        eligible = np.repeat(group_mask, e // spec.n_group, axis=1)
        within = np.where(eligible, logits, -np.inf)
        eligible_count = spec.topk_group * (e // spec.n_group)
        if spec.top_k >= eligible_count:
            expert_radius = np.full(m, np.inf)
        else:
            expert_radius = _kth_gap(within, spec.top_k) / 2
        return np.minimum(group_radius, expert_radius)

    def ok(rows: np.ndarray, err: np.ndarray) -> np.ndarray:
        """Certification of `rows` under uniform per-row errors `err`."""
        sub = logits[rows]
        lo, hi = _bounds(sub, spec, np.broadcast_to(err[:, None], sub.shape))
        groups = None if group_mask is None else group_mask[rows]
        return _certified_from_bounds(lo, hi, ids[rows], groups, spec)

    radius = np.zeros(m)
    rows = np.flatnonzero(ok(np.arange(m), np.zeros(m)))  # exact ties stay at radius 0
    if rows.size == 0:
        return radius
    # Start from a rigorous lower bound, then grow the upper end until certification fails.
    lo = _radius_lower_bound(logits[rows], ids[rows], None if group_mask is None else group_mask[rows], spec)
    lo = np.where(np.isfinite(lo) & ok(rows, np.where(np.isfinite(lo), lo, 0.0)), lo, 0.0)
    hi = np.where(lo > 0, 2 * lo, max(1.0, float(np.ptp(logits))))
    growing = np.ones(rows.size, dtype=bool)
    for _ in range(128):
        idx = np.flatnonzero(growing)
        if idx.size == 0:
            break
        still = ok(rows[idx], hi[idx])
        lo[idx[still]] = hi[idx[still]]
        hi[idx[still]] *= 2
        growing[idx[~still]] = False
    never = growing  # still certified at an enormous error: the selection cannot change
    active = ~never
    for _ in range(200):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        mid = 0.5 * (lo[idx] + hi[idx])
        good = ok(rows[idx], mid)
        lo[idx[good]] = mid[good]
        hi[idx[~good]] = mid[~good]
        active[idx] = (hi[idx] - lo[idx]) > rtol * hi[idx]
    radius[rows] = np.where(never, np.inf, lo)
    return radius
