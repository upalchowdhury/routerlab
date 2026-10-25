"""Template: testing a fused MoE router kernel with RouterLab.

Replace `candidate_router` with your kernel (Triton/CUDA/engine op). The
stand-in here is a plain PyTorch implementation of DeepSeek-V3 routing that
computes in FP32 and returns BF16 weights, as many engines do.

    pip install "moe-routerlab[torch] @ git+https://github.com/upalchowdhury/routerlab" pytest
    pytest examples/test_router_kernel.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from routerlab import RouterSpec, assert_routing_close, project, projection_error_bound  # noqa: E402

SPEC = RouterSpec.from_hf_config({
    "model_type": "deepseek_v3", "n_routed_experts": 256, "num_experts_per_tok": 8, "n_group": 8,
    "topk_group": 4, "norm_topk_prob": True, "routed_scaling_factor": 2.5, "topk_method": "noaux_tc"})


def candidate_router(x, weight, bias, spec):
    """Stand-in for the kernel under test: FP32 math, BF16 weights out, unsorted top-k."""
    scores = torch.sigmoid(x.float() @ weight.float().T)
    choice = scores + bias
    groups = choice.view(-1, spec.n_group, 256 // spec.n_group).topk(2, dim=-1).values.sum(-1)
    keep = torch.zeros_like(groups).scatter_(1, groups.topk(spec.topk_group, dim=-1).indices, 1.0)
    mask = keep.repeat_interleave(256 // spec.n_group, dim=1).bool()
    ids = choice.masked_fill(~mask, float("-inf")).topk(spec.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * spec.scaling_factor
    return ids, weights.bfloat16()


@pytest.mark.parametrize("tokens", [1, 7, 128, 4096])
def test_candidate_matches_reference(tokens):
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, 1024, generator=gen).bfloat16()
    weight = (torch.randn(256, 1024, generator=gen) / 32).bfloat16()
    bias = torch.randn(256, generator=gen) * 0.05
    spec = SPEC.with_bias(bias)

    ids, weights = candidate_router(x, weight, bias, spec)

    reference = project(x, weight)                                   # exact float64 logits
    tolerance = projection_error_bound(x, weight, accumulate="fp32")  # holds for any FP32 summation order
    assert_routing_close(reference, (ids, weights), spec, tolerance=tolerance,
                         weight_rtol=2.0 ** -8)                       # BF16 weights


def test_a_real_bug_is_caught():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(512, 1024, generator=gen).bfloat16()
    weight = (torch.randn(256, 1024, generator=gen) / 32).bfloat16()
    bias = torch.randn(256, generator=gen) * 0.05
    spec = SPEC.with_bias(bias)
    ids, weights = candidate_router(x, weight, torch.zeros(256), spec)  # bug: bias dropped
    with pytest.raises(AssertionError, match="margin exceeds the tolerance"):
        assert_routing_close(project(x, weight), (ids, weights), spec,
                             tolerance=projection_error_bound(x, weight), weight_rtol=2.0 ** -8)
    assert np.isfinite(weights.float().numpy()).all()
