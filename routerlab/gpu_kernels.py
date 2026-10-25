"""Educational GPU candidates, NOT GPU-validated or benchmarked in this release.

Contract: BF16 X[M,H], BF16 W[E,H], FP32 output[M,E]. Read-only inputs.
CUDA only. Inputs can have positive noncontiguous strides. Output is contiguous.
Caller supplies a non-aliasing output for allocation-free timed execution.
These deliberately simple candidates are starting points, not claimed speedups.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rowdot(X, W, Y, H: tl.constexpr, E: tl.constexpr,
            XM: tl.constexpr, XH: tl.constexpr,
            WE: tl.constexpr, WH: tl.constexpr,
            BE: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(0)
    experts = tl.program_id(1) * BE + tl.arange(0, BE)
    lane = tl.arange(0, BK)
    accum = tl.zeros((BE, BK), dtype=tl.float32)
    for block in range(tl.cdiv(H, BK)):
        h = block * BK + lane
        x = tl.load(X + token * XM + h * XH, h < H, other=0).to(tl.float32)
        w = tl.load(W + experts[:, None] * WE + h[None, :] * WH,
                    (experts[:, None] < E) & (h[None, :] < H), other=0).to(tl.float32)
        accum += w * x[None, :]
    result = tl.sum(accum, axis=1)
    tl.store(Y + token * E + experts, result, experts < E)


@triton.jit
def _tiled(X, W, Y, M: tl.constexpr, H: tl.constexpr, E: tl.constexpr,
           XM: tl.constexpr, XH: tl.constexpr, WE: tl.constexpr, WH: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(H, BK)):
        h = block * BK + lane
        a = tl.load(X + m[:, None] * XM + h[None, :] * XH,
                    (m[:, None] < M) & (h[None, :] < H), other=0)
        b = tl.load(W + n[None, :] * WE + h[:, None] * WH,
                    (n[None, :] < E) & (h[:, None] < H), other=0)
        accum = tl.dot(a, b, accum)
    tl.store(Y + m[:, None] * E + n[None, :], accum,
             (m[:, None] < M) & (n[None, :] < E))


def _validate(x, weight, out):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Expected X[M,H] and W[E,H]")
    if min(*x.shape, *weight.shape) <= 0:
        raise ValueError("Empty dimensions are unsupported")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("This candidate supports BF16 inputs and BF16 weights only")
    if x.device.type != "cuda" or x.device != weight.device or torch.version.hip:
        raise ValueError("This starter targets an NVIDIA CUDA device")
    if min(*x.stride(), *weight.stride()) <= 0:
        raise ValueError("Only positive input strides are supported")
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32)
    if (out.shape != (x.shape[0], weight.shape[0]) or out.dtype != torch.float32
            or out.device != x.device or not out.is_contiguous()):
        raise ValueError("Output must be contiguous FP32[M,E] on the input device")
    return out


def rowdot(x, weight, out=None):
    """SIMT reductions; investigate this for M=1..8 before tuning."""
    out = _validate(x, weight, out)
    with torch.cuda.device(x.device):
        _rowdot[(x.shape[0], triton.cdiv(weight.shape[0], 4))](
            x, weight, out, x.shape[1], weight.shape[0],
            *x.stride(), *weight.stride(), BE=4, BK=256, num_warps=4)
    return out


def tiled(x, weight, out=None):
    """Tensor-core GEMM; padded M=16 deliberately exposes tiny-batch tradeoffs."""
    out = _validate(x, weight, out)
    with torch.cuda.device(x.device):
        _tiled[(triton.cdiv(x.shape[0], 16), triton.cdiv(weight.shape[0], 32))](
            x, weight, out, x.shape[0], x.shape[1], weight.shape[0],
            *x.stride(), *weight.stride(), BM=16, BN=32, BK=32,
            num_warps=4, num_stages=2)
    return out
