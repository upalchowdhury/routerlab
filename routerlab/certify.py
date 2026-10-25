"""Margin certificates: when can a logit error change which experts are chosen?

Given reference logits and a bound on each logit's error, `certify` proves
(per token) that *every* perturbation within the bound selects exactly the same
expert set, including group-limited selection and correction bias. It uses
interval arithmetic on the router's selection scores:

* For selection that is monotone in the logits (softmax/sigmoid without bias,
  optional max-grouping), the intervals are the logits themselves +/- error.
  Without grouping the result is exact: a token fails to certify only if some
  perturbation within the bound really can change (or tie) its selection.
* With a correction bias or top-2-sum grouping, each score is bounded through
  its scoring function: elementwise for sigmoid/sqrtsoftplus, and with
  per-element softmax bounds otherwise (sound, slightly conservative).
* Group-limited selection is certified when every group holding a chosen
  expert is guaranteed to stay selected and no expert of a group that could
  become selected can overtake a chosen one. This is sound; it can be
  conservative when group and expert boundaries interact.

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
_CHUNK = 4096  # rows per block: bounds memory and lets robustness_radius use threads


def _pad(lo: np.ndarray, hi: np.ndarray, ulps: float = 8.0):
    """Widen intervals by `ulps` float64 ulps so our own rounding cannot unsound them."""
    pad = ulps * _EPS * np.maximum(np.abs(lo), np.abs(hi)) + np.finfo(np.float64).tiny
    return lo - pad, hi + pad


def _sum_of_others(z: np.ndarray) -> np.ndarray:
    """sum_{j != i} z_ij for every i, without cancellation for the dominant element.

    total - z_i loses all precision when z_i dominates the row (e.g. a logit 40
    above the rest); only the row maximum can exceed half the total, so that
    entry is recomputed directly.
    """
    others = z.sum(axis=1, keepdims=True) - z
    rows, top = np.arange(z.shape[0]), z.argmax(axis=1)
    masked = z.copy()
    masked[rows, top] = 0.0
    others[rows, top] = masked.sum(axis=1)
    return np.maximum(others, 0.0)


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
        with np.errstate(invalid="ignore"):
            hi = eb / (eb + _sum_of_others(ea))
            lo = ea / (ea + _sum_of_others(eb))
        # 0/0 when everything underflows at enormous errors: [0, 1] is always a valid bound.
        hi, lo = np.where(np.isnan(hi), 1.0, hi), np.where(np.isnan(lo), 0.0, lo)
        ulps = 16.0 + logits.shape[1]  # covers the row sums' rounding (worst case grows with E)
    else:
        f = score_function(spec.scoring)  # elementwise and increasing
        lo, hi = f(a), f(b)
        ulps = 8.0
    if spec.use_correction_bias:
        if spec.correction_bias is None:
            raise ValueError(f"Router '{spec.name}' needs its correction bias: spec.with_bias(bias)")
        lo, hi = lo + spec.correction_bias, hi + spec.correction_bias
    return _pad(lo, hi, ulps)


def _certified_from_bounds(lo, hi, ids, group_mask, spec: RouterSpec) -> np.ndarray:
    """Soundness: under any perturbation every score s lies in [lo, hi] and every group
    score in [glo, ghi] (group scores are monotone). A group that fewer than topk_group
    other groups can reach stays selected; a group strictly beaten by topk_group others
    stays excluded. If all groups holding chosen experts stay selected, every chosen
    expert stays eligible; if each beats every possibly eligible other expert, the
    top-k set cannot change. Ties count as changes (strict inequalities)."""
    m, e = lo.shape
    chosen = np.zeros((m, e), dtype=bool)
    np.put_along_axis(chosen, ids, True, axis=1)
    worst_chosen = np.where(chosen, lo, np.inf).min(axis=1)
    if not spec.grouped:
        return worst_chosen > np.where(chosen, -np.inf, hi).max(axis=1)
    size, need = e // spec.n_group, spec.topk_group
    glo, ghi = _group_scores(lo, spec), _group_scores(hi, spec)
    # threats[g] = #{h != g : ghi_h >= glo_g}; the h == g term is always counted (hi >= lo).
    threats = (ghi[:, None, :] >= glo[:, :, None]).sum(axis=2) - 1
    beaten = (glo[:, None, :] > ghi[:, :, None]).sum(axis=2)  # #{g : glo_g > ghi_h}
    stays_selected = threats <= need - 1
    possibly_selected = beaten < need
    has_chosen = chosen.reshape(m, spec.n_group, size).any(axis=2)
    groups_ok = (stays_selected | ~has_chosen).all(axis=1)
    eligible = np.repeat(possibly_selected, size, axis=1)
    return groups_ok & (worst_chosen > np.where(eligible & ~chosen, hi, -np.inf).max(axis=1))


def certify(logits, spec: RouterSpec, error) -> np.ndarray:
    """Boolean [M]: True where no logit perturbation with |delta| <= `error` can change
    (or tie) the selected expert set. `error` may be a scalar, per-row [M, 1],
    per-expert [E], or per-element [M, E] bound."""
    logits = as_float64_2d(logits, "logits")
    spec.validate(logits.shape[1])
    err = broadcast_bound(error, logits.shape, "error")
    out = np.empty(logits.shape[0], dtype=bool)
    for start in range(0, logits.shape[0], _CHUNK):  # bounds memory for the [rows, G, G] group tables
        part = slice(start, start + _CHUNK)
        ids, group_mask = _select(logits[part], spec)
        lo, hi = _bounds(logits[part], spec, err[part])
        out[part] = _certified_from_bounds(lo, hi, ids, group_mask, spec)
    return out


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
    slope = 1.0 if spec.selects_on_logits else _SLOPE[spec.scoring]
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

    Closed form (half the k-th/(k+1)-th logit gap, exact) for ungrouped selection
    that is monotone in the logits; otherwise found by bisection on `certify` to
    relative precision `rtol` (the result never exceeds the largest certified
    error by more than that). 0 means an exact tie at the
    boundary; inf means the selection can never change (e.g. top_k == E).
    Large inputs are processed in row chunks on `workers` threads (default:
    up to 8; NumPy releases the GIL in the heavy kernels).
    """
    logits = as_float64_2d(logits, "logits")
    spec.validate(logits.shape[1])
    if (spec.selects_on_logits and not spec.grouped) or logits.shape[0] <= _CHUNK:
        return _radius(logits, spec, rtol)
    chunks = [logits[i:i + _CHUNK] for i in range(0, logits.shape[0], _CHUNK)]
    workers = min(len(chunks), workers or min(8, os.cpu_count() or 1))
    if workers <= 1:
        return np.concatenate([_radius(chunk, spec, rtol) for chunk in chunks])
    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(lambda chunk: _radius(chunk, spec, rtol), chunks)))



def _radius(logits: np.ndarray, spec: RouterSpec, rtol: float) -> np.ndarray:
    m, e = logits.shape
    ids, group_mask = _select(logits, spec)
    if spec.selects_on_logits and not spec.grouped:
        return _kth_gap(logits, spec.top_k) / 2

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
    # The bound can be astronomically small (softmax probability gaps of e^-90), so the
    # bracket starts at the logit scale rather than at a multiple of it.
    scale = max(1.0, float(np.ptp(logits)))
    lo = _radius_lower_bound(logits[rows], ids[rows], None if group_mask is None else group_mask[rows], spec)
    lo = np.where(np.isfinite(lo) & ok(rows, np.where(np.isfinite(lo), lo, 0.0)), lo, 0.0)
    lo = np.minimum(lo, 1e8 * scale)
    hi = np.maximum(2 * lo, 1e-3 * scale)
    growing = np.ones(rows.size, dtype=bool)
    while True:
        idx = np.flatnonzero(growing & (hi <= 1e8 * scale))
        if idx.size == 0:
            break
        still = ok(rows[idx], hi[idx])
        lo[idx[still]] = hi[idx[still]]
        hi[idx[still]] *= 4
        growing[idx[~still]] = False
    # Certified even at 1e8 x the logit range: only a bias gap larger than the score range
    # can do that, and certification is monotone in the error, so the set never changes.
    never = growing
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
