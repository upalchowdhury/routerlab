"""Original 0.1 numerics API, kept for the learning material in docs/.

New code should use `routerlab.route`, `routerlab.compare_routing`, and
`routerlab.project`, which generalise these functions.
"""

from __future__ import annotations

import numpy as np

from .compare import compare_routing
from .precision import round_bf16
from .projection import project
from .routing import RouterSpec, topk_indices

__all__ = ["round_bf16", "projection_oracle", "topk_indices", "routing_diagnostics"]


def projection_oracle(x, weight):
    """Float64 dot products of the supplied values, with no hidden input casting."""
    x, weight = np.asarray(x), np.asarray(weight)
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Expected X[M,H] and W[E,H] with the same H")
    if min(*x.shape, *weight.shape) <= 0:
        raise ValueError("Empty dimensions are outside this lab's contract")
    if not np.isfinite(x).all() or not np.isfinite(weight).all():
        raise ValueError("The projection oracle accepts finite inputs only")
    return project(x, weight, accumulate="fp64")


def routing_diagnostics(reference, candidate, k=8):
    """Raw-logit top-k set comparison with a per-row margin certificate (0.1 format)."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate shapes must match")
    if reference.ndim != 2 or not 1 <= k < reference.shape[1]:
        raise ValueError("Expected nonempty scores[M,E] and 1 <= k < E")
    report = compare_routing(reference, candidate, RouterSpec(top_k=k, name="raw-logit top-k"))
    row_error = np.abs(candidate - reference).max(axis=1)
    ordered = np.sort(reference, axis=1)[:, ::-1]
    margins = ordered[:, k - 1] - ordered[:, k]
    certified = margins > 2 * row_error
    return {
        "rows": report.num_tokens,
        "max_abs_error": report.logit_max_abs_error,
        "relative_l2_error": report.logit_relative_l2_error,
        "topk_set_match_fraction": 1.0 - report.mismatch_rate,
        "minimum_boundary_margin": float(margins.min()),
        "near_boundary_rows": int((~certified).sum()),
        "certified_rows": int(certified.sum()),
        "unexpected_changed_rows": int(report.unexplained_mismatches or 0),
    }
