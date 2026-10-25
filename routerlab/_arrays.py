"""Array conversion helpers shared by every module.

RouterLab works on NumPy arrays internally, but accepts anything array-like:
NumPy arrays, nested lists, PyTorch tensors (any device, including BF16/FP8),
and JAX/ml_dtypes arrays. Nothing here imports torch.
"""

from __future__ import annotations

import numpy as np


def is_torch_tensor(value) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach")


def to_numpy(value) -> np.ndarray:
    """Convert an array-like (including torch tensors) to a NumPy array without
    changing any value. Sub-float32 floating types are widened exactly."""
    if is_torch_tensor(value):
        tensor = value.detach().cpu()  # move first, then convert (accelerator casts are not always exact)
        if tensor.is_floating_point() and str(tensor.dtype) not in (
                "torch.float16", "torch.float32", "torch.float64"):
            tensor = tensor.float()
        return tensor.numpy()
    array = np.asarray(value)
    # Note the underscores: "float64".startswith("float6") is True.
    if array.dtype.name == "bfloat16" or array.dtype.name.startswith(("float8_", "float6_", "float4_")):
        # ml_dtypes types (also used by JAX); every value is exact in float32.
        array = array.astype(np.float32)
    return array


def as_float64_2d(value, name: str, width: int | None = None) -> np.ndarray:
    """Return a finite float64 matrix [rows, width], flattening leading dims."""
    array = to_numpy(value).astype(np.float64, copy=False)
    if array.ndim == 1 and width is None:
        array = array[None, :]
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if width is not None and array.shape[-1] != width:
        raise ValueError(f"{name} has last dimension {array.shape[-1]}, expected {width}")
    array = array.reshape(-1, array.shape[-1])
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {bad} non-finite values")
    return array


def as_vector(value, name: str, length: int) -> np.ndarray:
    array = to_numpy(value).astype(np.float64, copy=False).reshape(-1)
    if array.shape[0] != length:
        raise ValueError(f"{name} has {array.shape[0]} elements, expected {length}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def broadcast_bound(bound, shape, name: str = "tolerance") -> np.ndarray:
    """Broadcast a bound to `shape` = (M, E).

    Accepts a scalar, per-row [M, 1], per-expert [1, E] or [E], or per-element
    [M, E]. A 1-D array of length M (when M != E) is also read as per-row; when
    M == E a 1-D bound is ambiguous and rejected.
    """
    array = to_numpy(bound).astype(np.float64, copy=False)
    if array.ndim == 1 and array.shape[0] > 1 and array.shape[0] == shape[0]:
        if shape[0] == shape[1]:
            raise ValueError(f"1-D {name} of length {shape[0]} is ambiguous for {shape[0]} tokens x "
                             f"{shape[1]} experts; pass shape [M, 1] (per token) or [1, E] (per expert)")
        array = array[:, None]
    elif array.ndim >= 2:
        array = array.reshape(-1, array.shape[-1])
    try:
        array = np.broadcast_to(array, shape)
    except ValueError:
        raise ValueError(f"{name} with shape {np.shape(bound)} cannot broadcast to {shape}") from None
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    return array
