import numpy as np
import pytest

from routerlab import RouterSpec, compare_routing, project
from routerlab.lowbit import (
    lowbit_project,
    pack_ternary,
    parse_granularity,
    quantize_activations,
    quantize_int,
    ternarize,
    unpack_ternary,
    weight_footprint,
)
from routerlab.sweep import LOWBIT_CONFIGS, plan_precision, precision_sweep


def _bitnet_reference(w):
    """BitNet b1.58 weight quantization, written independently from the paper's equations."""
    gamma = np.mean(np.abs(w)) + 1e-5
    return np.clip(np.round(w / gamma), -1, 1), gamma


def test_ternarize_matches_bitnet_absmean():
    w = np.random.default_rng(0).normal(size=(16, 64)) * 0.02
    q = ternarize(w)
    codes, gamma = _bitnet_reference(w)
    np.testing.assert_array_equal(q.codes, codes)
    np.testing.assert_allclose(q.scales, [[gamma]])
    np.testing.assert_allclose(q.dequantize(), codes * gamma)
    assert set(np.unique(q.codes)) <= {-1, 0, 1} and 0 < q.sparsity < 1


def test_granularities():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 10)) * np.array([[1], [10], [100], [1000]])
    per_row = ternarize(w, "channel")
    for row in range(4):
        codes, gamma = _bitnet_reference(w[row])
        np.testing.assert_array_equal(per_row.codes[row], codes)
        np.testing.assert_allclose(per_row.scales[row, 0], gamma)
    grouped = ternarize(w, 4)  # groups of 4, 4, 2 columns; the last group's mean uses only its 2 columns
    assert grouped.scales.shape == (4, 3)
    np.testing.assert_allclose(grouped.scales[0, 2], np.abs(w[0, 8:]).mean() + 1e-5)
    assert parse_granularity("group128", "tensor") == 128 and parse_granularity(None, "tensor") == "tensor"
    with pytest.raises(ValueError):
        parse_granularity("rows", "tensor")


def test_int_quantization_and_activation_quantization():
    w = np.random.default_rng(2).normal(size=(8, 32))
    q = quantize_int(w, 4)
    assert q.codes.min() >= -7 and q.codes.max() <= 7
    assert np.abs(q.dequantize() - w).max() <= (np.abs(w).max(axis=1, keepdims=True) / 7 / 2 + 1e-12).max()
    x = np.random.default_rng(3).normal(size=(5, 32)) * 4
    codes, scales = quantize_activations(x)
    assert codes.min() >= -128 and codes.max() <= 127
    assert np.abs(codes * scales - x).max() <= scales.max() / 2 + 1e-12


def test_integer_path_is_exact():
    rng = np.random.default_rng(4)
    x, w = rng.normal(size=(6, 300)), rng.normal(size=(5, 300))
    for q in (ternarize(w), ternarize(w, "channel"), ternarize(w, 128), quantize_int(w, 4, 64)):
        codes_x, scale_x = quantize_activations(x)
        expected = (codes_x * scale_x) @ q.dequantize().T  # same math in float64
        np.testing.assert_allclose(lowbit_project(x, q), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lowbit_project(x, ternarize(w), activation_bits=None), x @ ternarize(w).dequantize().T)
    np.testing.assert_allclose(project(x, w, weight_format="ternary", input_format="int8"),
                               lowbit_project(x, ternarize(w)))


@pytest.mark.parametrize("scheme,bits", [("2bit", 2.0), ("base3", 1.6)])
def test_ternary_packing_round_trip(scheme, bits):
    codes = np.random.default_rng(5).integers(-1, 2, size=1003).astype(np.int8)
    data = pack_ternary(codes, scheme)
    assert len(data) == -(-codes.size // (4 if scheme == "2bit" else 5))
    np.testing.assert_array_equal(unpack_ternary(data, codes.size, scheme), codes)
    assert weight_footprint((1, 1000), f"ternary-{scheme}")["bits_per_weight"] == bits
    with pytest.raises(ValueError):
        pack_ternary(np.array([2]), scheme)


def test_footprint_with_scales():
    fp = weight_footprint((256, 7168), "ternary-2bit", group_size=128)
    assert fp["bits_per_weight"] == pytest.approx(2 + 16 / 128)
    assert weight_footprint((256, 7168), "bf16")["vs_bf16"] == 1.0


def test_lowbit_sweep_and_plan():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(256, 64))
    w = rng.normal(size=(16, 64)) / 8
    spec = RouterSpec(top_k=2, renormalize=True)
    result = precision_sweep(x, w, spec, LOWBIT_CONFIGS)
    assert all("report" in row for row in result.rows)
    plan = plan_precision({"r": result}, max_change_rate=1.0)
    assert plan["r"]["weight_bits"] == pytest.approx(1.58)  # anything is allowed: ternary is cheapest
    strict = plan_precision({"r": result}, max_change_rate=0.0)
    assert strict["r"]["config"] == "bf16 operands, fp32 logits"
    ternary = project(x, w, weight_format="ternary")
    assert compare_routing(project(x, w), ternary, spec).mismatched_tokens > 0
