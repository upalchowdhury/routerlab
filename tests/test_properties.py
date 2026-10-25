"""Property-based tests (Hypothesis): the guarantees must hold for *any* router and input."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from routerlab import RouterSpec, certify, quantize, robustness_radius, route, save_safetensors  # noqa: E402
from routerlab.checkpoint import SafetensorsFile  # noqa: E402
from routerlab.precision import FORMATS  # noqa: E402

SETTINGS = settings()  # max_examples comes from the profile (tests/conftest.py; HYPOTHESIS_PROFILE=stress)


@st.composite
def routers(draw):
    n_group = draw(st.sampled_from([1, 2, 4]))
    size = draw(st.integers(2, 6))
    experts = n_group * size
    topk_group = draw(st.integers(1, n_group))
    top_k = draw(st.integers(1, min(experts, topk_group * size)))
    scoring = draw(st.sampled_from(["softmax", "sigmoid", "sqrtsoftplus"]))
    spec = RouterSpec(top_k=top_k, scoring=scoring, n_group=n_group, topk_group=topk_group,
                      group_score=draw(st.sampled_from(["max", "top2_sum"])),
                      renormalize=draw(st.booleans()), scaling_factor=draw(st.sampled_from([1.0, 2.5])))
    if draw(st.booleans()):
        spec = spec.with_bias(np.array(draw(st.lists(st.floats(-1, 1), min_size=experts, max_size=experts))))
    scale = draw(st.sampled_from([0.1, 1.0, 5.0, 30.0]))
    tokens = draw(st.integers(1, 6))
    seed = draw(st.integers(0, 2**31 - 1))
    logits = np.random.default_rng(seed).normal(size=(tokens, experts)) * scale
    if draw(st.booleans()):  # inject exact ties
        logits[:, 1] = logits[:, 0]
    return spec, logits, seed


def _sets(logits, spec):
    return np.sort(route(logits, spec).expert_ids, axis=1)


@SETTINGS
@given(routers(), st.floats(1e-6, 2.0))
def test_certified_tokens_never_change(case, error):
    spec, logits, seed = case
    safe = certify(logits, spec, error)
    if not safe.any():
        return
    rng = np.random.default_rng(seed + 1)
    selected = route(logits, spec).selection_mask()
    base = _sets(logits, spec)
    directions = [np.where(selected, -1.0, 1.0), np.where(selected, 1.0, -1.0)]
    directions += [rng.choice([-1.0, 1.0], size=logits.shape) for _ in range(8)]
    directions += [rng.uniform(-1, 1, size=logits.shape) for _ in range(4)]
    for direction in directions:
        moved = _sets(logits + error * direction, spec)
        assert (moved[safe] == base[safe]).all()


@SETTINGS
@given(routers())
def test_radius_is_sound(case):
    spec, logits, seed = case
    radius = robustness_radius(logits, spec)
    assert (radius >= 0).all()
    finite = np.isfinite(radius)
    rng = np.random.default_rng(seed + 2)
    selected = route(logits, spec).selection_mask()
    base = _sets(logits, spec)
    step = np.where(finite, 0.999 * radius, 1e3)[:, None]
    for direction in [np.where(selected, -1.0, 1.0)] + [rng.choice([-1.0, 1.0], size=logits.shape)
                                                        for _ in range(8)]:
        assert (_sets(logits + step * direction, spec) == base).all()


@SETTINGS
@given(routers())
def test_routing_invariants(case):
    spec, logits, seed = case
    r = route(logits, spec)
    assert not r.duplicate_rows().any()
    assert np.isfinite(r.weights).all() and (r.weights >= 0).all()
    if spec.renormalize:
        np.testing.assert_allclose(r.weights.sum(axis=1), spec.scaling_factor, rtol=1e-9)
    # Relabelling experts (whole groups, and experts within a group) relabels the selection.
    rng = np.random.default_rng(seed + 3)
    size = logits.shape[1] // spec.n_group
    perm = np.concatenate([g * size + rng.permutation(size) for g in rng.permutation(spec.n_group)])
    permuted = spec if spec.correction_bias is None else spec.with_bias(spec.correction_bias[perm])
    tied = robustness_radius(logits, spec) == 0  # tie-breaking by index is not permutation-invariant
    moved = np.sort(perm[route(logits[:, perm], permuted).expert_ids], axis=1)
    assert (moved[~tied] == _sets(logits, spec)[~tied]).all()


@settings()
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=1, max_size=64),
       st.sampled_from(["bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2", "tf32"]))
def test_quantize_is_idempotent_monotone_and_within_half_ulp(values, fmt):
    x = np.sort(np.array(values, dtype=np.float32)).astype(np.float64)
    q = quantize(x, fmt, overflow="saturate").astype(np.float64)
    np.testing.assert_array_equal(quantize(q, fmt, overflow="saturate"), q)
    assert (np.diff(q) >= 0).all()
    spec = FORMATS[fmt]
    normal = (np.abs(x) >= spec.smallest_normal) & (np.abs(x) <= spec.max_value)
    assert (np.abs(q - x)[normal] <= spec.unit_roundoff * np.abs(x)[normal]).all()
    tiny = np.abs(x) < spec.smallest_normal
    assert (np.abs(q - x)[tiny] <= spec.smallest_subnormal / 2).all()


@settings()
@given(layout=st.dictionaries(st.text("abcdefgh./_", min_size=1, max_size=12),
                              st.tuples(st.sampled_from(["<f4", "<f8", "<f2", "<i8", "<i4", "|u1", "|b1"]),
                                        st.lists(st.integers(0, 4), min_size=0, max_size=3)),
                              min_size=1, max_size=5), seed=st.integers(0, 2**31 - 1))
def test_safetensors_round_trip(tmp_path_factory, layout, seed):
    rng = np.random.default_rng(seed)
    tensors = {name: (rng.normal(size=shape) * 100).astype(dtype) for name, (dtype, shape) in layout.items()}
    path = tmp_path_factory.mktemp("st") / "x.safetensors"
    save_safetensors(path, tensors)
    file = SafetensorsFile(path)
    for name, value in tensors.items():
        np.testing.assert_array_equal(file.get(name), value)
