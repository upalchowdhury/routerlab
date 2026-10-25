import numpy as np
import pytest

from routerlab import RouterSpec, boundary_ties, compare_routing, selection_keys
from routerlab.precision import FORMATS


def _logit(p):
    return np.log(p) - np.log1p(-p)


def test_fp32_bias_addition_creates_a_boundary_tie():
    # sigmoid scores 1e-9 and 2e-9 differ in float64 but vanish next to a bias of 1.0 in fp32.
    logits = np.array([[_logit(1e-9), _logit(2e-9), -30.0, -30.0]])
    spec = RouterSpec(top_k=1, scoring="sigmoid").with_bias(np.array([1.0, 1.0, 0.0, 0.0]))
    assert not boundary_ties(logits, spec, "fp64")[0]
    assert boundary_ties(logits, spec, "fp32")[0]
    keys = selection_keys(logits, spec, "fp32")
    assert keys[0, 0] == keys[0, 1] == 1.0


def test_ties_inside_the_chosen_set_do_not_count():
    logits = np.array([[2.0, 2.0, -1.0, -3.0]])  # tie between the two chosen experts only
    assert not boundary_ties(logits, RouterSpec(top_k=2), "fp32")[0]
    assert boundary_ties(logits, RouterSpec(top_k=1), "fp32")[0]


def test_group_boundary_tie():
    # Groups 0 and 1 have identical top-2 sums in bf16 but not in float64.
    eps = FORMATS["bf16"].epsilon / 8
    logits = _logit(np.array([[0.5, 0.25, 0.5 + eps, 0.25, 0.01, 0.01]]))
    spec = RouterSpec(top_k=2, scoring="sigmoid", n_group=3, topk_group=1, group_score="top2_sum").with_bias(
        np.zeros(6))
    assert not boundary_ties(logits, spec, "fp64")[0]
    assert boundary_ties(logits, spec, "bf16")[0]


def test_many_ties_with_large_shared_bias():
    rng = np.random.default_rng(0)
    bias = rng.normal(0, 1, 64)
    bias[:8] = 8.0
    logits = rng.normal(-14, 1, size=(512, 64))  # sigmoid ~1e-6, far below the bias's fp32 ulp
    spec = RouterSpec(top_k=4, scoring="sigmoid", renormalize=True).with_bias(bias)
    assert boundary_ties(logits, spec, "fp64").sum() == 0
    assert boundary_ties(logits, spec, "fp32").mean() > 0.5
    report = compare_routing(logits, logits, spec)
    assert report.boundary_ties["fp32"] > 0 and "tie-breaking" in str(report)


def test_on_logits_and_validation():
    logits = np.array([[1.0, 1.0 + 1e-12, 0.0]])
    assert boundary_ties(logits, RouterSpec(top_k=1), "fp32", on_logits=True)[0]
    with pytest.raises(ValueError, match="correction bias"):
        selection_keys(logits, RouterSpec(top_k=1).with_bias(np.zeros(3)), on_logits=True)
