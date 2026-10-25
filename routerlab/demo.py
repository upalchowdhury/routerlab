"""Self-contained demonstration: python -m routerlab.demo (or `routerlab demo`).

Builds a DeepSeek-V3-style router on synthetic data and compares two
candidates against the float64 reference:

1. BF16 logits: a few tokens change experts, all explained by their tiny margins.
2. A semantic bug (DeepSeek-V2's max group score instead of V3's top-2 sum,
   and the correction bias leaking into the weights): expert changes at
   tokens with large margins, and wrong weights, both flagged.

Synthetic data illustrates the mechanics only; run `routerlab sweep` on
captured activations for real conclusions.
"""

from __future__ import annotations

import numpy as np

from .compare import compare_routing
from .precision import FORMATS
from .projection import project, projection_error_bound
from .routing import RouterSpec, Routing, _group_scores, score_function, topk_indices


def _buggy_route(logits: np.ndarray, spec: RouterSpec) -> Routing:
    """A plausible porting bug: DeepSeek-V2's group rule (max) used for a V3 router
    (top-2 sum), and the correction bias left in the returned weights."""
    choice = score_function(spec.scoring)(logits) + spec.correction_bias
    wrong = spec.replace(group_score="max")
    groups = topk_indices(_group_scores(choice, wrong), spec.topk_group)
    mask = np.zeros((logits.shape[0], spec.n_group), dtype=bool)
    np.put_along_axis(mask, groups, True, axis=1)
    masked = np.where(np.repeat(mask, logits.shape[1] // spec.n_group, axis=1), choice, -np.inf)
    ids = topk_indices(masked, spec.top_k)
    weights = np.take_along_axis(choice, ids, axis=1)
    return Routing(ids, weights / weights.sum(axis=1, keepdims=True) * spec.scaling_factor, logits.shape[1])


def main() -> None:
    rng = np.random.default_rng(0)
    tokens, hidden_size, experts = 2048, 512, 64
    hidden = rng.standard_normal((tokens, hidden_size)).astype(np.float32)
    weight = (rng.standard_normal((experts, hidden_size)) / np.sqrt(hidden_size)).astype(np.float32)
    bias = rng.normal(-0.3, 0.6, experts)
    spec = RouterSpec(top_k=8, scoring="sigmoid", n_group=8, topk_group=4, group_score="top2_sum",
                      use_correction_bias=True, renormalize=True, renorm_eps=1e-20, scaling_factor=2.5,
                      num_experts=experts, name="deepseek_v3-style (synthetic)").with_bias(bias)

    reference = project(hidden, weight)                       # exact float64 logits
    bf16 = project(hidden, weight, accumulate="fp32", output_format="bf16")
    tolerance = projection_error_bound(hidden, weight, accumulate="fp32", output_format="bf16")

    print("=== 1. BF16 logits vs float64 reference (tolerance = rigorous FP32+BF16 error bound)\n")
    print(compare_routing(reference, bf16, spec, tolerance=tolerance).summary(examples=3))
    print("\n=== 2. Buggy candidate kernel (same BF16 logits, V2 group rule, biased weights)\n")
    print(compare_routing(reference, _buggy_route(bf16, spec), spec, tolerance=tolerance).summary(examples=3))
    print(f"\nBF16 unit roundoff is {FORMATS['bf16'].unit_roundoff:.2e}: a token whose margin is smaller than "
          "its logit error may legitimately change experts; one with a larger margin may not.")


if __name__ == "__main__":
    main()
