"""Side-by-side parity with the router modules in HuggingFace transformers.

For every supported architecture: build a tiny config, instantiate the model's
own router module with random weights (and random correction bias where the
model has one), run it, and check that `RouterSpec.from_hf_config(config)`
reproduces its expert selection and weights from the same logits.

Requires torch and transformers>=5; skipped otherwise.
"""

import importlib
import importlib.util

import numpy as np
import pytest

from routerlab import RouterSpec, robustness_radius, route

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="requires torch and transformers")

E, K, H, M = 16, 4, 32, 300

SOFTMAX = dict(num_experts_per_tok=K)
GROUPED = dict(n_routed_experts=E, num_experts_per_tok=K, n_group=4, topk_group=2,
               routed_scaling_factor=2.5, norm_topk_prob=True)

# model_type: (modeling module, router class, config kwargs, call style)
CASES = {
    "mixtral": ("mixtral", "MixtralTopKRouter", dict(num_local_experts=E, **SOFTMAX), "lsi"),
    "minimax": ("minimax", "MiniMaxTopKRouter", dict(num_local_experts=E, **SOFTMAX), "lsi"),
    "gpt_oss": ("gpt_oss", "GptOssTopKRouter", dict(num_local_experts=E, **SOFTMAX), "lsi"),
    "granitemoe": ("granitemoe", "GraniteMoeTopKRouter", dict(num_local_experts=E, **SOFTMAX), "isl"),
    "granitemoeshared": ("granitemoeshared", "GraniteMoeSharedTopKRouter",
                         dict(num_local_experts=E, **SOFTMAX), "isl"),
    "granitemoehybrid": ("granitemoehybrid", "GraniteMoeHybridTopKRouter",
                         dict(num_local_experts=E, mamba_n_heads=8, mamba_d_head=8, **SOFTMAX), "isl"),
    "hunyuan_v1_moe": ("hunyuan_v1_moe", "HunYuanMoEV1Gate", dict(num_experts=E, moe_topk=K), "lsi"),
    "qwen2_moe": ("qwen2_moe", "Qwen2MoeTopKRouter", dict(num_experts=E, norm_topk_prob=False, **SOFTMAX), "lsi"),
    "qwen2_moe/norm": ("qwen2_moe", "Qwen2MoeTopKRouter", dict(num_experts=E, norm_topk_prob=True, **SOFTMAX),
                       "lsi"),
    "qwen3_moe": ("qwen3_moe", "Qwen3MoeTopKRouter", dict(num_experts=E, norm_topk_prob=True, **SOFTMAX), "lsi"),
    "qwen3_next": ("qwen3_next", "Qwen3NextTopKRouter", dict(num_experts=E, norm_topk_prob=True, **SOFTMAX),
                   "lsi"),
    "qwen3_vl_moe_text": ("qwen3_vl_moe", "Qwen3VLMoeTextTopKRouter",
                          dict(num_experts=E, norm_topk_prob=False, **SOFTMAX), "lsi"),
    "qwen3_5_moe_text": ("qwen3_5_moe", "Qwen3_5MoeTopKRouter", dict(num_experts=E, **SOFTMAX), "lsi"),
    "olmoe": ("olmoe", "OlmoeTopKRouter", dict(num_experts=E, norm_topk_prob=False, **SOFTMAX), "lsi"),
    "flex_olmo": ("flex_olmo", "FlexOlmoTopKRouter", dict(num_experts=E, norm_topk_prob=True, **SOFTMAX), "lsi"),
    "jamba": ("jamba", "JambaSparseMoeBlock", dict(num_experts=E, **SOFTMAX), "jamba"),
    "deepseek_v2": ("deepseek_v2", "DeepseekV2TopkRouter",
                    dict(n_routed_experts=E, num_experts_per_tok=K, topk_method="greedy",
                         routed_scaling_factor=1.5), "lsi"),
    "deepseek_v2/grouped": ("deepseek_v2", "DeepseekV2TopkRouter",
                            dict(n_routed_experts=E, num_experts_per_tok=K, topk_method="group_limited_greedy",
                                 n_group=4, topk_group=2, routed_scaling_factor=16.0), "lsi"),
    "deepseek_v3": ("deepseek_v3", "DeepseekV3TopkRouter", GROUPED, "lsi"),
    "deepseek_v3/no-norm": ("deepseek_v3", "DeepseekV3TopkRouter", {**GROUPED, "norm_topk_prob": False}, "lsi"),
    "deepseek_v32": ("deepseek_v32", "DeepseekV32TopkRouter", GROUPED, "lsi"),
    "glm4_moe": ("glm4_moe", "Glm4MoeTopkRouter", {**GROUPED, "n_group": 1, "topk_group": 1}, "lsi"),
    "dots1": ("dots1", "Dots1TopkRouter", GROUPED, "lsi"),
    "exaone_moe": ("exaone_moe", "ExaoneMoeTopkRouter", {**GROUPED, "num_experts": E}, "lsi"),
    "kimi_linear": ("kimi_linear", "KimiLinearTopkRouter",
                    {**{k: v for k, v in GROUPED.items() if k != "n_routed_experts"}, "num_experts": E}, "lsi"),
    "deepseek_v4": ("deepseek_v4", "DeepseekV4TopKRouter",
                    dict(n_routed_experts=E, num_experts_per_tok=K, routed_scaling_factor=1.5), "lsi"),
    "llama4_text": ("llama4", "Llama4Router", dict(num_local_experts=E, num_experts_per_tok=1), "llama4"),
    "ernie4_5_moe": ("ernie4_5_moe", "Ernie4_5_MoeTopKRouter", dict(moe_num_experts=E, moe_k=K), "lsi"),
    "minimax_m2": ("minimax_m2", "MiniMaxM2TopKRouter", dict(num_local_experts=E, **SOFTMAX), "bias_arg"),
    "lfm2_moe": ("lfm2_moe", "Lfm2MoeTopKRouter",
                 dict(num_experts=E, norm_topk_prob=True, routed_scaling_factor=1.7, **SOFTMAX), "bias_arg"),
}


def _run(case):
    import torch
    from transformers import AutoConfig

    module_name, class_name, kwargs, style = CASES[case]
    model_type = case.split("/")[0]
    config = AutoConfig.for_model(model_type, hidden_size=H, intermediate_size=64, moe_intermediate_size=16,
                                  num_attention_heads=4, num_key_value_heads=4, head_dim=8, **kwargs)
    modeling = importlib.import_module(f"transformers.models.{module_name}.modeling_{module_name}")
    torch.manual_seed(abs(hash(case)) % 2**31)
    module = getattr(modeling, class_name)(config).float()
    bias = None
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor.dim() == 2 and tensor.shape[0] == E:
                tensor.copy_(torch.randn_like(tensor) / H ** 0.5)
            elif "bias" in name and tensor.numel() == E:
                tensor.copy_(torch.randn_like(tensor) * 0.1)
                if "correction" in name or "statics" in name:
                    bias = tensor.detach().reshape(-1).numpy().copy()
        if style == "bias_arg":
            bias = (torch.randn(E) * 0.1).numpy()
    hidden = torch.randn(M, H)
    with torch.no_grad():
        if style == "lsi":
            logits, weights, ids = module(hidden)
        elif style == "isl":
            ids, weights, logits = module(hidden)
        elif style == "bias_arg":
            logits, weights, ids = module(hidden, torch.from_numpy(bias))
        elif style == "jamba":
            logits = module.router(hidden)
            ids, weights = module.route_tokens_to_experts(hidden, logits)
        elif style == "llama4":
            dense, logits = module(hidden)
            weights, ids = torch.topk(dense, config.num_experts_per_tok, dim=-1)
    return config, logits.double().numpy(), ids.numpy(), weights.double().numpy(), bias


@pytest.mark.parametrize("case", sorted(CASES))
def test_reference_router_matches_transformers(case):
    config, logits, hf_ids, hf_weights, bias = _run(case)
    spec = RouterSpec.from_hf_config(config)
    # A serialised config.json must give the same spec as the live config object.
    assert RouterSpec.from_hf_config(config.to_dict()).to_dict() == spec.to_dict()
    if spec.use_correction_bias:
        assert bias is not None, f"{case}: spec expects a bias the module does not have"
        spec = spec.with_bias(bias)
    else:
        assert bias is None, f"{case}: the module has a correction bias the spec ignores"
    ours = route(logits, spec)
    assert ours.top_k == hf_ids.shape[1]

    same = np.sort(ours.expert_ids, axis=1) == np.sort(hf_ids, axis=1)
    same = same.all(axis=1)
    # HF computes in float32 and we in float64: only genuine near-ties may differ.
    radius = robustness_radius(logits, spec)
    assert (radius[~same] < 1e-5).all(), f"{case}: {int((~same).sum())} selections differ beyond near-ties"
    assert same.mean() > 0.98

    dense_ours = ours.dense_weights()[same]
    dense_hf = np.zeros_like(ours.dense_weights())
    np.put_along_axis(dense_hf, hf_ids, hf_weights, axis=1)
    np.testing.assert_allclose(dense_ours, dense_hf[same], rtol=2e-5, atol=2e-6, err_msg=case)


def test_multimodal_wrapper_configs_use_their_text_config():
    from transformers import AutoConfig

    text = dict(num_local_experts=E, num_experts_per_tok=1, hidden_size=H)
    config = AutoConfig.for_model("llama4", text_config=text)
    spec = RouterSpec.from_hf_config(config)
    assert (spec.scoring, spec.top_k, spec.num_experts) == ("sigmoid", 1, E)
    assert RouterSpec.from_hf_config(config.to_dict()).to_dict() == spec.to_dict()
    kimi = AutoConfig.for_model("kimi_k25", text_config={"model_type": "kimi_k2", **GROUPED, "hidden_size": H})
    assert RouterSpec.from_hf_config(kimi).use_correction_bias


@pytest.mark.parametrize("model_type", sorted({c.split("/")[0] for c in CASES} | {"kimi_k2"}))
def test_minimal_config_json_uses_transformers_defaults(model_type):
    """A config.json with only the expert counts must give the same spec as transformers would."""
    from transformers import AutoConfig

    live = AutoConfig.for_model("deepseek_v3" if model_type == "kimi_k2" else model_type)
    if live.num_experts_per_tok is None:  # some classes have no default top-k / expert count
        live.num_experts_per_tok = 4
    if getattr(live, "n_routed_experts", 0) is None:
        live.n_routed_experts = 16
    full = RouterSpec.from_hf_config(live)
    minimal = {"model_type": model_type, "num_experts_per_tok": full.top_k, "n_routed_experts": full.num_experts}

    def routing_fields(spec):
        return {k: v for k, v in spec.to_dict().items() if k not in ("name", "notes")}

    assert routing_fields(RouterSpec.from_hf_config(minimal)) == routing_fields(full)
