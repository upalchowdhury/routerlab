import numpy as np
import pytest

from routerlab import project, projection_error_bound, quantize


def _operands(rng, m=64, h=2048, e=64):
    x = quantize(rng.standard_normal((m, h)) * np.exp(rng.normal(0, 1, h)), "bf16")
    w = quantize(rng.standard_normal((e, h)) / np.sqrt(h), "bf16")
    return x, w


def test_reference_is_exact_for_bf16_operands():
    x = np.array([[2.0 ** 24, 1.0, -(2.0 ** 24)]])
    np.testing.assert_array_equal(project(x, np.ones((2, 3))), [[1.0, 1.0]])


def test_fp32_requires_representable_operands():
    with pytest.raises(ValueError, match="FP32"):
        project(np.array([[0.1]]), np.ones((1, 1)), accumulate="fp32")
    project(np.array([[0.1]]), np.ones((1, 1)), accumulate="fp32", input_format="fp32")


def test_deterministic_bound_holds_for_many_summation_orders():
    rng = np.random.default_rng(0)
    x, w = _operands(rng)
    exact = project(x, w)
    bound = projection_error_bound(x, w, accumulate="fp32")
    x32, w32 = x.astype(np.float32), w.astype(np.float32)
    candidates = [x32 @ w32.T]
    for block in (16, 256):  # blocked / split-K style accumulation
        acc = np.zeros(exact.shape, np.float32)
        for k in range(0, x.shape[1], block):
            acc = acc + x32[:, k:k + block] @ w32[:, k:k + block].T
        candidates.append(acc)
    seq = np.zeros(exact.shape, np.float32)  # naive left-to-right accumulation
    for k in range(x.shape[1]):
        seq = seq + np.outer(x32[:, k], w32[:, k])
    candidates.append(seq)
    for computed in candidates:
        assert (np.abs(computed.astype(np.float64) - exact) <= bound).all()


def test_probabilistic_bound_is_tighter_and_still_covers_typical_errors():
    rng = np.random.default_rng(1)
    x, w = _operands(rng)
    det = projection_error_bound(x, w, accumulate="fp32")
    prob = projection_error_bound(x, w, accumulate="fp32", method="probabilistic")
    assert (prob < det).all()
    computed = project(x, w, accumulate="fp32")
    assert (np.abs(computed - project(x, w)) <= prob).all()


def test_output_rounding_is_bounded():
    rng = np.random.default_rng(2)
    x, w = _operands(rng)
    exact = project(x, w)
    for fmt in ("bf16", "fp16"):
        bound = projection_error_bound(x, w, accumulate="fp32", output_format=fmt)
        out = project(x, w, accumulate="fp32", output_format=fmt)
        assert (np.abs(out - exact) <= bound).all()


def test_linear_bias_and_scaled_fp8_weights():
    rng = np.random.default_rng(3)
    x, w = _operands(rng, m=8, h=128, e=16)
    b = rng.normal(size=16)
    np.testing.assert_allclose(project(x, w, b), x.astype(np.float64) @ w.T + b)
    fp8 = project(x, w, weight_format="fp8_e4m3fn", weight_scaling="channel")
    assert np.isfinite(fp8).all() and not np.array_equal(fp8, project(x, w))


def test_overflow_is_reported():
    with pytest.raises(FloatingPointError):
        project(np.full((1, 4), 300.0), np.full((2, 4), 300.0), output_format="fp16")


def test_bound_covers_underflow_and_bias_is_not_rounded_silently():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((32, 256)).astype(np.float32) * np.float32(1e-22)
    w = rng.standard_normal((16, 256)).astype(np.float32)
    bound = projection_error_bound(x, w, accumulate="fp32")
    assert (np.abs(project(x, w, accumulate="fp32") - project(x, w)) <= bound).all()
    with pytest.raises(ValueError, match="linear_bias"):
        project(x, w, np.full(16, 0.1), accumulate="fp32")
    project(x, w, np.full(16, 0.5), accumulate="fp32")
