import json

import numpy as np
import pytest

from routerlab import RouterSpec, Routing, assert_routing_close, compare_routing, format_reports, route

SPEC = RouterSpec(top_k=4, scoring="sigmoid", n_group=4, topk_group=2, group_score="top2_sum",
                  renormalize=True, scaling_factor=2.5, num_experts=32,
                  name="dsv3-test").with_bias(np.random.default_rng(9).normal(0, 0.05, 32))


@pytest.fixture
def logits():
    return np.random.default_rng(0).normal(size=(500, 32))


def test_identical_inputs(logits):
    report = compare_routing(logits, logits.copy(), SPEC)
    assert report.mismatched_tokens == 0 and report.passed
    assert report.logit_max_abs_error == 0 and report.moved_weight_max == 0
    json.dumps(report.to_dict())  # JSON-serialisable
    assert "PASS" in str(report)


def test_small_noise_flips_are_explained(logits):
    noisy = logits + np.random.default_rng(1).uniform(-0.02, 0.02, logits.shape)
    report = compare_routing(logits, noisy, SPEC, tolerance=0.0201)
    assert report.mismatched_tokens > 0, "test needs some near-tie flips"
    assert report.unexplained_mismatches == 0 and report.passed
    assert report.max_mismatch_radius <= 0.0201
    assert report.at_risk_tokens >= report.mismatched_tokens


def test_semantic_bug_is_unexplained(logits):
    buggy = route(logits, SPEC.replace(group_score="max"))  # V2 group rule on a V3 router
    report = compare_routing(logits, buggy, SPEC, tolerance=1e-3)
    assert report.unexplained_mismatches > 0 and not report.passed
    first = report.mismatch_examples[0]
    assert first["unexplained"] and first["radius"] > 1e-3


def test_candidate_selection_formats(logits):
    ref = route(logits, SPEC)
    for candidate in (ref, (ref.expert_ids, ref.weights), {"topk_ids": ref.expert_ids,
                                                          "topk_weights": ref.weights}, ref.expert_ids):
        report = compare_routing(logits, candidate, SPEC, tolerance=1e-6)
        assert report.mismatched_tokens == 0 and report.passed


def test_unsorted_candidate_ids_are_fine(logits):
    ref = route(logits, SPEC)
    perm = np.random.default_rng(2).permuted(np.tile(np.arange(4), (500, 1)), axis=1)
    shuffled = (np.take_along_axis(ref.expert_ids, perm, 1), np.take_along_axis(ref.weights, perm, 1))
    assert compare_routing(logits, shuffled, SPEC, tolerance=1e-6).passed


def test_duplicate_experts_are_invalid(logits):
    ref = route(logits, SPEC)
    ids = ref.expert_ids.copy()
    ids[7, 1] = ids[7, 0]
    report = compare_routing(logits, ids, SPEC)
    assert report.invalid_tokens == 1 and not report.passed


def test_wrong_weights_are_caught(logits):
    ref = route(logits, SPEC)
    weights = ref.weights.copy()
    weights[3] *= 1.01
    report = compare_routing(logits, (ref.expert_ids, weights), SPEC, tolerance=1e-4)
    assert report.weight_tolerance_violations == 1 and not report.passed
    assert report.weight_worst["token"] == 3


def test_selection_only_reference():
    ids = np.array([[0, 1], [2, 3]])
    spec = RouterSpec(top_k=2, num_experts=4)
    report = compare_routing(ids, np.array([[1, 0], [2, 1]]), spec)
    assert report.mismatched_tokens == 1 and report.min_radius is None
    with pytest.raises(ValueError, match="num_experts"):
        compare_routing(ids, ids, RouterSpec(top_k=2))


def test_expert_load_shift(logits):
    ref = route(logits, SPEC)
    ids = ref.expert_ids.copy()
    rows = np.flatnonzero(~(ids == 31).any(axis=1))[:10]
    ids[rows, 0] = 31  # move 10 tokens' first expert to expert 31
    report = compare_routing(logits, ids, SPEC)
    assert report.max_load_shift == 10 and report.max_load_shift_expert == 31
    assert sum(report.expert_load_candidate) == sum(report.expert_load_reference)


def test_shape_mismatch_is_rejected(logits):
    with pytest.raises(ValueError):
        compare_routing(logits, logits[:10], SPEC)


def test_format_reports_table(logits):
    table = format_reports({"layer.0": compare_routing(logits, logits, SPEC)})
    assert "| layer.0 | 500 | 0 |" in table


# ----------------------------------------------------------------------------- assertions
def test_assert_passes_with_near_ties_and_fails_on_bugs(logits):
    noisy = logits + np.random.default_rng(3).uniform(-0.02, 0.02, logits.shape)
    assert_routing_close(logits, noisy, SPEC, atol=0.021)
    with pytest.raises(AssertionError, match="exceed tolerance"):
        assert_routing_close(logits, noisy, SPEC, atol=0.001)
    buggy = route(logits, SPEC.replace(group_score="max"))
    with pytest.raises(AssertionError, match="margin exceeds the tolerance"):
        assert_routing_close(logits, buggy, SPEC, atol=0.02)


def test_assert_checks_weights(logits):
    ref = route(logits, SPEC)
    assert_routing_close(logits, (ref.expert_ids, ref.weights * (1 + 1e-6)), SPEC, atol=1e-6)
    with pytest.raises(AssertionError, match="weights outside tolerance"):
        assert_routing_close(logits, (ref.expert_ids, ref.weights * 1.01), SPEC, atol=1e-6)
    assert_routing_close(logits, (ref.expert_ids, ref.weights * 1.01), SPEC, atol=1e-6, check_weights=False)


def test_assert_relative_tolerance_and_torch_inputs(logits):
    torch = pytest.importorskip("torch")
    candidate = torch.from_numpy(logits).float().bfloat16()
    from routerlab.precision import FORMATS
    assert_routing_close(torch.from_numpy(logits), candidate, SPEC, rtol=FORMATS["bf16"].unit_roundoff)


def test_routing_dense_weights_round_trip(logits):
    ref = route(logits, SPEC)
    back = Routing(ref.expert_ids, ref.weights, 32)
    np.testing.assert_array_equal(back.dense_weights(), ref.dense_weights())


def test_clear_errors_for_mismatched_inputs(logits):
    with pytest.raises(ValueError, match="experts per token"):
        compare_routing(logits, route(logits, SPEC).expert_ids[:, :3], SPEC)
    with pytest.raises(ValueError, match="expects 32 experts"):
        compare_routing(logits[:, :16], logits[:, :16], SPEC)
