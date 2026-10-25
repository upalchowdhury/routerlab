"""Property tests: the certificate is sound for every router family and exact where claimed."""

import numpy as np
import pytest

from routerlab import RouterSpec, certify, robustness_radius, route

E = 32


def _specs(rng):
    bias = rng.normal(0, 0.05, E)
    return {
        "softmax": RouterSpec(top_k=4),
        "softmax-grouped-max": RouterSpec(top_k=4, n_group=8, topk_group=3, group_score="max"),
        "sigmoid": RouterSpec(top_k=4, scoring="sigmoid"),
        "sigmoid-bias": RouterSpec(top_k=4, scoring="sigmoid").with_bias(bias),
        "deepseek_v3": RouterSpec(top_k=4, scoring="sigmoid", n_group=8, topk_group=3,
                                  group_score="top2_sum").with_bias(bias),
        "sqrtsoftplus-bias": RouterSpec(top_k=4, scoring="sqrtsoftplus").with_bias(bias),
        "softmax-bias": RouterSpec(top_k=4).with_bias(bias * 0.1),
    }


def _sets(logits, spec):
    return np.sort(route(logits, spec).expert_ids, axis=1)


@pytest.mark.parametrize("name", list(_specs(np.random.default_rng(0))))
def test_radius_is_sound_under_random_and_corner_perturbations(name):
    rng = np.random.default_rng(1)
    spec = _specs(np.random.default_rng(0))[name]
    logits = rng.normal(size=(200, E))
    radius = robustness_radius(logits, spec)
    assert (radius >= 0).all()
    base = _sets(logits, spec)
    selected = route(logits, spec).selection_mask()
    scale = 0.999 * radius[:, None]
    perturbations = [np.where(selected, -1.0, 1.0), np.where(selected, 1.0, -1.0)]
    perturbations += [rng.choice([-1.0, 1.0], size=logits.shape) for _ in range(40)]
    perturbations += [rng.uniform(-1, 1, size=logits.shape) for _ in range(10)]
    for direction in perturbations:
        np.testing.assert_array_equal(_sets(logits + scale * direction, spec), base, err_msg=name)


@pytest.mark.parametrize("name", ["softmax", "sigmoid"])
def test_radius_is_exact_for_plain_topk(name):
    rng = np.random.default_rng(2)
    spec = _specs(rng)[name]
    logits = rng.normal(size=(200, E))
    radius = robustness_radius(logits, spec)
    order = np.argsort(-logits, axis=1)
    rows = np.arange(len(logits))
    pushed = logits.copy()
    pushed[rows, order[:, spec.top_k - 1]] -= 1.001 * radius
    pushed[rows, order[:, spec.top_k]] += 1.001 * radius
    assert (_sets(pushed, spec) != _sets(logits, spec)).any(axis=1).all()


@pytest.mark.parametrize("name", ["softmax-grouped-max", "sigmoid-bias", "deepseek_v3"])
def test_closed_form_and_bisection_agree_with_certify(name):
    rng = np.random.default_rng(3)
    spec = _specs(np.random.default_rng(0))[name]
    logits = rng.normal(size=(100, E))
    radius = robustness_radius(logits, spec)
    assert certify(logits, spec, 0.999 * radius[:, None]).all()
    assert not certify(logits, spec, 1.001 * radius[:, None] + 1e-12).any()


def test_exact_ties_and_degenerate_cases():
    spec = RouterSpec(top_k=1)
    assert robustness_radius(np.array([[1.0, 1.0, 0.0]]), spec)[0] == 0.0
    assert np.isinf(robustness_radius(np.array([[1.0, 0.0]]), RouterSpec(top_k=2))[0])
    assert not certify(np.array([[1.0, 1.0, 0.0]]), spec, 0.0)[0]
    assert certify(np.array([[1.0, 0.0, 0.0]]), spec, 0.49)[0]
    assert not certify(np.array([[1.0, 0.0, 0.0]]), spec, 0.5)[0]


def test_per_element_bounds_are_used():
    logits = np.array([[1.0, 0.0, -5.0]])
    spec = RouterSpec(top_k=1)
    # A large error on an irrelevant expert does not matter; on the runner-up it does.
    assert certify(logits, spec, np.array([[0.1, 0.1, 5.5]]))[0]
    assert not certify(logits, spec, np.array([[0.1, 0.95, 0.1]]))[0]


def test_bias_too_strong_to_overturn_means_infinite_radius():
    spec = RouterSpec(top_k=1, scoring="sigmoid").with_bias(np.array([2.0, 0.0, 0.0]))
    assert np.isinf(robustness_radius(np.zeros((1, 3)), spec)[0])
