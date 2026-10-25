"""Tolerance-aware assertions for MoE router tests (pytest/unittest).

Comparing ``topk_ids`` exactly makes router-kernel tests flaky: random inputs
regularly produce near-ties, and any legitimate rounding difference flips them.
Loosening the check hides real bugs. `assert_routing_close` resolves this:
a token may select different experts only if its routing margin is within the
logit tolerance, so near-ties pass while semantic errors (wrong bias, wrong
group rule, off-by-one indices, races) fail with a precise message.

    from routerlab import RouterSpec, assert_routing_close, projection_error_bound

    spec = RouterSpec.from_hf_config("path/to/config.json").with_bias(bias)
    tol = projection_error_bound(x, w, accumulate="fp32")      # rigorous FP32 bound
    assert_routing_close(ref_logits, (topk_ids, topk_weights), spec, tolerance=tol)
"""

from __future__ import annotations

import numpy as np

from ._arrays import as_float64_2d, broadcast_bound
from .compare import compare_routing
from .routing import RouterSpec

__all__ = ["assert_routing_close"]


def assert_routing_close(reference_logits, candidate, spec: RouterSpec, *, tolerance=None,
                         atol: float = 0.0, rtol: float = 0.0, weight_atol: float = 1e-6,
                         weight_rtol: float = 1e-5, check_weights: bool = True,
                         max_report: int = 5) -> None:
    """Assert that `candidate` routes like `reference_logits` up to numerical error.

    The logit tolerance is ``tolerance + atol + rtol * |reference|`` elementwise
    (`tolerance` may be a scalar or an array such as `projection_error_bound`).
    Checks, in order:

    1. candidate logits (if given) are within the tolerance;
    2. every candidate selection is valid (no repeated experts);
    3. every token whose expert set changed has a routing margin within the
       tolerance, i.e. the change is explainable by numerical error;
    4. for tokens with the same expert set, weights agree within the error
       implied by the tolerance (|dw| <= w * (exp(2 * tol_row) - 1)) plus
       ``weight_atol + weight_rtol * |w|``. Increase `weight_rtol` for engines
       that return BF16 weights (e.g. ``2**-8``).

    `candidate` may be logits, a `Routing`, ``(topk_ids, topk_weights)``, a dict,
    or integer ids. Raises AssertionError with per-token diagnostics.
    """
    ref_logits = as_float64_2d(reference_logits, "reference logits")
    spec.validate(ref_logits.shape[1])
    base = 0.0 if tolerance is None else broadcast_bound(tolerance, ref_logits.shape)
    tol = broadcast_bound(base + atol + rtol * np.abs(ref_logits), ref_logits.shape)
    report = compare_routing(ref_logits, candidate, spec, tolerance=tol, max_examples=max_report,
                             weight_atol=weight_atol, weight_rtol=weight_rtol)

    problems = []
    if report.logit_tolerance_violations:
        cand_logits = as_float64_2d(candidate, "candidate logits", width=ref_logits.shape[1])
        excess = np.abs(cand_logits - ref_logits) - tol
        token, expert = np.unravel_index(int(np.argmax(excess)), excess.shape)
        problems.append(
            f"{report.logit_tolerance_violations} logits exceed tolerance; worst at token {token}, "
            f"expert {expert}: |delta| = {abs(cand_logits[token, expert] - ref_logits[token, expert]):.3e} "
            f"> tol {tol[token, expert]:.3e}")
    if report.invalid_tokens:
        problems.append(f"{report.invalid_tokens} tokens select the same expert more than once")
    if report.unexplained_mismatches:
        problems.append(f"{report.unexplained_mismatches} tokens changed experts although their routing "
                        "margin exceeds the tolerance (not a rounding effect):")
        for ex in [e for e in report.mismatch_examples if e.get("unexplained")][:max_report]:
            row = tol[ex["token"]]
            problems.append(f"    token {ex['token']}: reference {ex['reference_experts']} vs candidate "
                            f"{ex['candidate_experts']}; provably stable under the tolerance "
                            f"(margin {ex['radius']:.3e}; tolerance in this row "
                            f"{row.min():.3e}..{row.max():.3e})")
    if check_weights and report.weight_tolerance_violations:
        w = report.weight_worst
        problems.append(f"{report.weight_tolerance_violations} tokens kept their experts but have weights outside "
                        f"tolerance; worst at token {w['token']}, expert {w['expert']}: reference "
                        f"{w['reference']:.6g}, candidate {w['candidate']:.6g} (allowed |delta| "
                        f"{w['allowed_abs_error']:.3e})")

    if problems:
        explained = report.explained_mismatches or 0
        footer = (f"({explained} other tokens changed experts within tolerance: expected near-ties)"
                  if explained else "")
        raise AssertionError("\n".join([f"Routing mismatch for '{spec.name}' "
                                        f"({report.num_tokens} tokens, top-{report.top_k}):"]
                                       + ["  " + p for p in problems] + ([f"  {footer}"] if footer else [])))

