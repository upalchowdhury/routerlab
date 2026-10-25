"""Compare a candidate router (kernel, engine, precision) against a reference.

`compare_routing` answers the questions that matter when two implementations
of the same MoE router disagree:

* How far apart are the logits?
* Which tokens picked a different expert set, and how much weight moved?
* Is each disagreement explained by numerical error (the token's routing
  margin is within the error/tolerance), or is it a semantic bug (the
  token had a comfortable margin and still flipped)?
* How did per-expert load change?
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ._arrays import as_float64_2d, broadcast_bound, to_numpy
from .certify import certify, robustness_radius
from .routing import RouterSpec, Routing, route

__all__ = ["RoutingReport", "compare_routing", "as_routing", "format_reports"]


def as_routing(candidate, num_experts: int, top_k: int | None = None) -> Routing:
    """Coerce a candidate selection to `Routing`.

    Accepts a `Routing`, an `(expert_ids, weights)` tuple, a dict with
    ``expert_ids``/``topk_ids`` and optional ``weights``/``topk_weights``, or an
    integer id array [M, k]. With ``top_k=1``, 1-D ids/weights are read as [M, 1].
    """
    if isinstance(candidate, Routing):
        return candidate
    if isinstance(candidate, Mapping):
        ids = candidate.get("expert_ids", candidate.get("topk_ids"))
        weights = candidate.get("weights", candidate.get("topk_weights"))
        if ids is None:
            raise ValueError("Candidate dict needs 'expert_ids' (or 'topk_ids')")
    elif isinstance(candidate, tuple) and len(candidate) == 2:
        ids, weights = candidate
    else:
        ids, weights = candidate, None
        if to_numpy(ids).dtype.kind not in "iu":
            raise TypeError("Candidate must be logits (float array), Routing, (ids, weights), or integer ids")
    if top_k == 1:
        ids = _column(ids)
        weights = None if weights is None else _column(weights)
    return Routing(ids, weights, num_experts)


def _column(values):
    array = to_numpy(values)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _is_logits(value) -> bool:
    """Float arrays are logits; Routing, tuples, dicts, and integer arrays are selections."""
    if isinstance(value, (Routing, Mapping, tuple)):
        return False
    return to_numpy(value).dtype.kind == "f"


@dataclass
class RoutingReport:
    """Result of `compare_routing`. `str(report)` gives a readable summary and
    `report.to_dict()` a JSON-serialisable dict."""

    router: str
    num_tokens: int
    num_experts: int
    top_k: int
    # Logits (None when the candidate supplied only a selection)
    logit_max_abs_error: float | None = None
    logit_mean_abs_error: float | None = None
    logit_relative_l2_error: float | None = None
    logit_tolerance_violations: int | None = None
    # Selection
    mismatched_tokens: int = 0
    mismatch_rate: float = 0.0
    mean_expert_overlap: float = 1.0
    invalid_tokens: int = 0        # duplicate experts or non-finite weights in the candidate
    candidate_weights: bool = False  # the candidate supplied its own weights (not derived from logits)
    # Weights (None when either side has no weights)
    weight_max_abs_error_matched: float | None = None
    weight_tolerance_violations: int | None = None   # tokens with the same experts but wrong weights
    weight_worst: dict[str, Any] | None = None
    moved_weight_mean: float | None = None
    moved_weight_max: float | None = None
    # Margins (None when the reference is a selection, not logits)
    min_radius: float | None = None
    radius_quantiles: dict[str, float] | None = None
    tolerance: str | None = None
    at_risk_tokens: int | None = None
    explained_mismatches: int | None = None
    unexplained_mismatches: int | None = None
    max_mismatch_radius: float | None = None
    # Expert load
    max_load_shift: int = 0
    max_load_shift_expert: int | None = None
    expert_load_reference: list[int] = field(default_factory=list, repr=False)
    expert_load_candidate: list[int] = field(default_factory=list, repr=False)
    # Examples for debugging
    mismatch_examples: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def passed(self) -> bool | None:
        """True when every check passed; False when one failed (invalid selection, unexplained
        flip, logits or weights outside tolerance); None when there are differences that can
        only be judged against a tolerance and none was given (or the reference has no logits).
        `bool(None)` is False, so `if report.passed:` only accepts verified results."""
        if (self.invalid_tokens or self.unexplained_mismatches or self.logit_tolerance_violations
                or self.weight_tolerance_violations):
            return False
        if self.tolerance is None and (self.mismatched_tokens or (
                self.candidate_weights and self.weight_max_abs_error_matched)):
            return None
        return True

    @property
    def status(self) -> str:
        return {True: "PASS", False: "FAIL", None: "UNCHECKED"}[self.passed]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        data["status"] = self.status
        return _jsonable(data)

    def __str__(self) -> str:
        return self.summary()

    def summary(self, examples: int = 5) -> str:
        lines = [f"Router {self.router}: {self.num_tokens} tokens, top-{self.top_k} of {self.num_experts} experts"]
        if self.logit_max_abs_error is not None:
            line = (f"  logits      max |err| {self.logit_max_abs_error:.3e}   mean {self.logit_mean_abs_error:.3e}"
                    f"   rel-L2 {self.logit_relative_l2_error:.3e}")
            if self.logit_tolerance_violations:
                line += f"   ({self.logit_tolerance_violations} elements exceed tolerance)"
            lines.append(line)
        lines.append(f"  selection   {self.mismatched_tokens} tokens changed expert set "
                     f"({100 * self.mismatch_rate:.4g}%), mean overlap {self.mean_expert_overlap:.6f}")
        if self.invalid_tokens:
            lines.append(f"  INVALID     {self.invalid_tokens} candidate tokens repeat an expert or have "
                         "non-finite weights")
        if self.moved_weight_mean is not None:
            lines.append(f"  weights     moved mass mean {self.moved_weight_mean:.3e}  max {self.moved_weight_max:.3e}"
                         + ("" if self.weight_max_abs_error_matched is None else
                            f"   max |err| (same experts) {self.weight_max_abs_error_matched:.3e}"))
        if self.weight_tolerance_violations:
            w = self.weight_worst
            lines.append(f"  WEIGHTS     {self.weight_tolerance_violations} tokens with the same experts have weights "
                         f"outside tolerance (token {w['token']}, expert {w['expert']}: "
                         f"{w['reference']:.6g} vs {w['candidate']:.6g})")
        if self.radius_quantiles is not None:
            q = self.radius_quantiles
            lines.append(f"  margins     radius min {self.min_radius:.3e}  p1 {q['p1']:.3e}  "
                         f"p5 {q['p5']:.3e}  median {q['p50']:.3e} (logit units)")
        if self.tolerance is not None:
            lines.append(f"  tolerance   {self.tolerance}: {self.at_risk_tokens} tokens at risk")
            if self.mismatched_tokens:
                lines.append(f"  verdict     {self.explained_mismatches} explained by tolerance, "
                             f"{self.unexplained_mismatches} UNEXPLAINED (margin larger than tolerance)")
        elif self.passed is None:
            lines.append("  verdict     UNCHECKED: give the expected logit error (tolerance) to judge "
                         + ("the changed tokens" if self.mismatched_tokens else "the candidate's weights")
                         + ("" if self.radius_quantiles is not None else "; needs reference logits"))
        if self.max_mismatch_radius is not None and self.mismatched_tokens:
            lines.append(f"  largest margin among changed tokens: {self.max_mismatch_radius:.3e}")
        if self.max_load_shift:
            lines.append(f"  load        largest per-expert change {self.max_load_shift} tokens "
                         f"(expert {self.max_load_shift_expert})")
        for example in self.mismatch_examples[:examples]:
            radius = example.get("radius")
            lines.append(f"    token {example['token']}: -{example['only_reference']} +{example['only_candidate']}"
                         + ("" if radius is None else f"  margin {radius:.3e}")
                         + ("  UNEXPLAINED" if example.get("unexplained") else ""))
        lines.append(f"  {self.status}")
        return "\n".join(lines)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if math.isnan(value) else ("inf" if math.isinf(value) else value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def compare_routing(reference, candidate, spec: RouterSpec, *, tolerance=None,
                    weight_atol: float = 1e-6, weight_rtol: float = 1e-5,
                    max_examples: int = 20) -> RoutingReport:
    """Compare `candidate` against `reference` under router `spec`.

    Parameters
    ----------
    reference:
        Reference logits [..., E] (preferred: enables margin analysis), or a
        reference selection (see `as_routing`).
    candidate:
        Candidate logits [..., E] (routed with `spec`), or the candidate's own
        selection (e.g. ``(topk_ids, topk_weights)`` dumped from an engine).
    tolerance:
        Expected logit error: scalar, per-row [M, 1], per-expert [1, E], or
        per-element [M, E] (e.g. from `projection_error_bound`). A changed
        token is *explained* if its routing could change under this
        tolerance, and *unexplained* otherwise. Without a tolerance the
        differences are measured but not judged (``report.passed is None``
        when there are any).
    weight_atol, weight_rtol:
        Extra slack for the candidate's own weight arithmetic when checking
        tokens that kept their experts: |dw| may be at most
        ``w * (exp(2 * tol_row) - 1 + weight_rtol) + weight_atol``, where the
        first term bounds how far the logit tolerance can move any supported
        router's weights. Use ``weight_rtol=2**-8`` for BF16 weights.
    """
    ref_logits = cand_logits = None
    if _is_logits(reference):
        ref_logits = as_float64_2d(reference, "reference logits")
        num_experts = ref_logits.shape[1]
        ref = route(ref_logits, spec)
    else:
        if spec.num_experts is None:
            raise ValueError("spec.num_experts is required when the reference is a selection")
        num_experts = spec.num_experts
        ref = as_routing(reference, num_experts, spec.top_k)
    if _is_logits(candidate):
        cand_logits = as_float64_2d(candidate, "candidate logits", width=num_experts)
        if ref_logits is not None and cand_logits.shape != ref_logits.shape:
            raise ValueError(f"Reference {ref_logits.shape} and candidate {cand_logits.shape} shapes differ")
        cand = route(cand_logits, spec)
    else:
        cand = as_routing(candidate, num_experts, ref.top_k)
    if cand.num_tokens != ref.num_tokens:
        raise ValueError(f"Reference has {ref.num_tokens} tokens, candidate {cand.num_tokens}")
    if cand.top_k != ref.top_k:
        raise ValueError(f"Reference selects {ref.top_k} experts per token, candidate {cand.top_k}")

    report = RoutingReport(router=spec.name, num_tokens=ref.num_tokens, num_experts=num_experts,
                           top_k=ref.top_k)
    tol = None
    if ref_logits is not None and cand_logits is not None:
        diff = np.abs(cand_logits - ref_logits)
        report.logit_max_abs_error = float(diff.max())
        report.logit_mean_abs_error = float(diff.mean())
        report.logit_relative_l2_error = float(np.linalg.norm(cand_logits - ref_logits)
                                               / max(np.linalg.norm(ref_logits), np.finfo(float).tiny))
        if tolerance is not None:
            tol = broadcast_bound(tolerance, ref_logits.shape)
            report.logit_tolerance_violations = int((diff > tol).sum())
    elif tolerance is not None and ref_logits is not None:
        tol = broadcast_bound(tolerance, ref_logits.shape)
    if tol is not None:
        report.tolerance = _describe_tolerance(tolerance)
    report.candidate_weights = cand_logits is None and cand.weights is not None

    ref_mask, cand_mask = ref.selection_mask(), cand.selection_mask()
    invalid = cand.duplicate_rows()
    if cand.weights is not None:
        invalid |= ~np.isfinite(cand.weights).all(axis=1)
    same = (ref_mask == cand_mask).all(axis=1) & ~invalid
    changed = ~same
    report.invalid_tokens = int(invalid.sum())
    report.mismatched_tokens = int(changed.sum())
    report.mismatch_rate = float(changed.mean())
    report.mean_expert_overlap = float((ref_mask & cand_mask).sum(axis=1).mean() / max(ref.top_k, 1))

    if ref.weights is not None and cand.weights is not None:
        dense_ref, dense_cand = ref.dense_weights(), cand.dense_weights()
        finite = np.isfinite(dense_cand).all(axis=1)  # non-finite rows are already counted invalid
        if finite.any():
            moved = np.abs(dense_ref - dense_cand)[finite].sum(axis=1)
            mass = np.abs(dense_ref[finite]).sum(axis=1) + np.abs(dense_cand[finite]).sum(axis=1)
            moved = moved / np.where(mass > 0, mass, 1.0)
            report.moved_weight_mean, report.moved_weight_max = float(moved.mean()), float(moved.max())
        if same.any():
            report.weight_max_abs_error_matched = float(np.abs(dense_ref - dense_cand)[same].max())
            if tol is not None:
                _check_weights(report, dense_ref, dense_cand, same, tol, weight_atol, weight_rtol)

    radius = None
    if ref_logits is not None:
        radius = robustness_radius(ref_logits, spec)
        finite = radius[np.isfinite(radius)]
        report.min_radius = float(radius.min())
        report.radius_quantiles = ({f"p{p}": float(np.percentile(finite, p)) for p in (1, 5, 50)}
                                   if finite.size else {"p1": np.inf, "p5": np.inf, "p50": np.inf})
        if changed.any():
            report.max_mismatch_radius = float(radius[changed].max())
    unexplained = np.zeros(ref.num_tokens, dtype=bool)
    if tol is not None:
        safe = certify(ref_logits, spec, tol)
        report.at_risk_tokens = int((~safe).sum())
        unexplained = changed & safe
        report.unexplained_mismatches = int(unexplained.sum())
        report.explained_mismatches = int((changed & ~safe).sum())

    load_ref, load_cand = ref.expert_load(), cand.expert_load()
    shift = np.abs(load_cand - load_ref)
    report.expert_load_reference, report.expert_load_candidate = load_ref.tolist(), load_cand.tolist()
    report.max_load_shift = int(shift.max())
    report.max_load_shift_expert = int(shift.argmax()) if shift.max() else None

    # Unexplained flips first: those are the ones to debug.
    order = np.concatenate([np.flatnonzero(unexplained), np.flatnonzero(changed & ~unexplained)])
    for token in order[:max_examples]:
        r_ids, c_ids = set(ref.expert_ids[token].tolist()), set(cand.expert_ids[token].tolist())
        example = {"token": int(token), "reference_experts": sorted(r_ids), "candidate_experts":
                   sorted(cand.expert_ids[token].tolist()), "only_reference": sorted(r_ids - c_ids),
                   "only_candidate": sorted(c_ids - r_ids), "duplicate": bool(invalid[token])}
        if radius is not None:
            example["radius"] = float(radius[token])
        if cand_logits is not None:
            example["max_logit_error"] = float(np.abs(cand_logits[token] - ref_logits[token]).max())
        if tol is not None:
            example["unexplained"] = bool(unexplained[token])
        report.mismatch_examples.append(example)
    return report


def _check_weights(report, dense_ref, dense_cand, same, tol, weight_atol, weight_rtol) -> None:
    # Every supported score function (softmax, sigmoid, sqrtsoftplus) moves a weight by at most
    # a factor exp(+/-2e) under logit errors <= e, renormalization and scaling included.
    row_tol = tol.max(axis=1, keepdims=True)
    allowed = np.abs(dense_ref) * (np.expm1(2 * row_tol) + weight_rtol) + weight_atol
    excess = np.where(same[:, None], np.abs(dense_cand - dense_ref) - allowed, -np.inf)
    excess = np.where(np.isnan(excess), np.inf, excess)  # NaN must never pass
    bad = (excess > 0).any(axis=1)
    report.weight_tolerance_violations = int(bad.sum())
    if bad.any():
        token, expert = np.unravel_index(int(np.argmax(excess)), excess.shape)
        report.weight_worst = {"token": int(token), "expert": int(expert),
                               "reference": float(dense_ref[token, expert]),
                               "candidate": float(dense_cand[token, expert]),
                               "allowed_abs_error": float(allowed[token, expert])}


def _describe_tolerance(tolerance) -> str:
    array = np.asarray(to_numpy(tolerance), dtype=np.float64)
    if array.size == 1:
        return f"logit tolerance {float(array.reshape(-1)[0]):.3e}"
    return f"per-element logit tolerance (max {float(array.max()):.3e})"


def format_reports(reports: Mapping[str, RoutingReport]) -> str:
    """Render several reports (e.g. one per layer) as a Markdown table."""
    header = ("| layer | tokens | changed | changed % | unexplained | max logit err | min margin | "
              "moved weight (mean) |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = [header]

    def fmt(value, spec=".2e"):
        return "-" if value is None else format(value, spec)

    for name, r in reports.items():
        rows.append(f"| {name} | {r.num_tokens} | {r.mismatched_tokens} | {100 * r.mismatch_rate:.3f} | "
                    f"{fmt(r.unexplained_mismatches, 'd')} | {fmt(r.logit_max_abs_error)} | "
                    f"{fmt(r.min_radius)} | {fmt(r.moved_weight_mean)} |")
    return "\n".join(rows)
