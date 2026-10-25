import importlib.util

import numpy as np
import pytest

from routerlab import FORMATS, is_representable, quantize, quantize_scaled, resolve_format, round_bf16

HAS_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _bits_equal(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    both_nan = np.isnan(a) & np.isnan(b)
    same = a.view(np.uint32) == b.view(np.uint32)
    return bool(np.all(both_nan | same))


def _probe_values(fmt_name, rng):
    """Every representable value of a 16/8-bit format, their midpoints +/- 1 float32 ulp,
    plus random float32 bit patterns."""
    spec = FORMATS[fmt_name]
    if fmt_name == "bf16":
        grid = (np.arange(0, 0x7F80, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    elif fmt_name == "fp16":
        grid = np.arange(0, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)
    else:
        e_bits = 4 if fmt_name == "fp8_e4m3fn" else 5
        m_bits = spec.mantissa_bits
        codes = np.arange(0, 128)
        exp, man = codes >> m_bits, codes & ((1 << m_bits) - 1)
        bias = (1 << (e_bits - 1)) - 1
        grid = np.where(exp == 0, man / 2**m_bits * 2.0 ** (1 - bias), (1 + man / 2**m_bits) * 2.0 ** (exp - bias))
        grid = np.unique(grid[np.isfinite(grid) & (grid <= spec.max_value)])
    mids = (grid[:-1] + grid[1:]) / 2
    mids32 = mids.astype(np.float32)
    up = np.nextafter(mids32, np.float32(np.inf))
    down = np.nextafter(mids32, np.float32(0))
    with np.errstate(over="ignore"):
        beyond = np.array([spec.max_value * 1.01, spec.max_value * 1.2, spec.max_value * 4], dtype=np.float32)
    randoms = rng.integers(0, 2**32, 200_000, dtype=np.uint64).astype(np.uint32).view(np.float32)
    values = np.concatenate([grid.astype(np.float32), mids32, up, down, beyond, randoms,
                             np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)])
    return np.concatenate([values, -values])


@pytest.mark.skipif(not HAS_ML_DTYPES, reason="ml_dtypes not installed")
@pytest.mark.parametrize("fmt,mode", [("bf16", "default"), ("fp8_e4m3fn", "nan"), ("fp8_e5m2", "default")])
def test_matches_ml_dtypes_bitwise(fmt, mode):
    import ml_dtypes

    dtype = {"bf16": ml_dtypes.bfloat16, "fp8_e4m3fn": ml_dtypes.float8_e4m3fn,
             "fp8_e5m2": ml_dtypes.float8_e5m2}[fmt]
    values = _probe_values(fmt, np.random.default_rng(1))
    with np.errstate(over="ignore", invalid="ignore"):
        expected = values.astype(dtype).astype(np.float32)
    assert _bits_equal(quantize(values, fmt, overflow=mode), expected)


def test_fp16_matches_numpy_from_float32_and_float64():
    rng = np.random.default_rng(2)
    values = _probe_values("fp16", rng)
    with np.errstate(over="ignore", invalid="ignore"):
        assert _bits_equal(quantize(values, "fp16"), values.astype(np.float16).astype(np.float32))
        wide = rng.standard_normal(100_000) * 10.0 ** rng.integers(-9, 6, 100_000)
        assert _bits_equal(quantize(wide, "fp16"), wide.astype(np.float16).astype(np.float32))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("fmt", ["bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2"])
def test_default_overflow_matches_torch_casts(fmt):
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8_e4m3fn": torch.float8_e4m3fn,
             "fp8_e5m2": torch.float8_e5m2}[fmt]
    values = _probe_values(fmt, np.random.default_rng(3))
    expected = torch.from_numpy(values).to(dtype).float().numpy()
    assert _bits_equal(quantize(values, fmt), expected)


def test_ties_to_even_and_special_values():
    x = np.array([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256], dtype=np.float32)
    np.testing.assert_array_equal(round_bf16(x), [1.0, 1 + 2 / 128, -1.0])
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    original = special.copy()
    out = quantize(special, "bf16")
    assert np.signbit(out[1]) and np.isposinf(out[2]) and np.isneginf(out[3]) and np.isnan(out[4])
    np.testing.assert_array_equal(special.view(np.uint32), original.view(np.uint32))  # not mutated


def test_e4m3_overflow_modes():
    x = np.array([448.0, 464.0, 465.0, 1e6, np.inf, -np.inf])
    np.testing.assert_array_equal(quantize(x, "fp8_e4m3fn"), [448, 448, 448, 448, 448, -448])
    assert np.isnan(quantize(x, "fp8_e4m3fn", overflow="nan")[2:]).all()
    with pytest.raises(ValueError):
        quantize(x, "fp8_e4m3fn", overflow="inf")


def test_subnormals_and_double_rounding_free():
    spec = FORMATS["bf16"]
    tiny = spec.smallest_subnormal
    np.testing.assert_array_equal(quantize([tiny, tiny / 2, tiny * 0.51], "bf16"), [tiny, 0.0, tiny])
    # float64 input slightly above a bf16 midpoint must round up, even though rounding it
    # to float32 first would land exactly on the midpoint and round to even (down).
    value = 1 + 2 ** -8 + 2 ** -40
    assert quantize([value], "bf16")[0] == np.float32(1 + 2 ** -7)


def test_resolve_and_representable():
    assert resolve_format("torch.bfloat16") is FORMATS["bf16"]
    assert resolve_format(np.float16) is FORMATS["fp16"]
    assert resolve_format("E4M3") is FORMATS["fp8_e4m3fn"]
    with pytest.raises(ValueError):
        resolve_format("fp7")
    assert is_representable([1.0, 1 + 2 ** -7, 1 + 2 ** -9], "bf16").tolist() == [True, True, False]


def test_quantize_scaled_per_channel():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((16, 64)) * np.logspace(-3, 3, 16)[:, None]
    q = quantize_scaled(w, "fp8_e4m3fn", axis=-1)
    rel = np.abs(q - w).max(axis=1) / np.abs(w).max(axis=1)
    assert rel.max() <= 2 ** -4 + 1e-6  # within half an E4M3 ulp of each row's absmax
    assert np.isfinite(q).all()
