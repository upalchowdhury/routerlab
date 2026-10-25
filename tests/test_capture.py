"""RouterCapture on tiny HuggingFace models (requires torch + transformers)."""

import importlib.util

import numpy as np
import pytest

from routerlab import RouterSpec, assert_routing_close, project
from routerlab.capture import RouterCapture, find_router_modules, load_capture

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="requires torch and transformers")

TINY = dict(hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, head_dim=8, vocab_size=97)
MODELS = {
    "deepseek_v3": dict(n_routed_experts=16, num_experts_per_tok=4, n_group=4, topk_group=2,
                        first_k_dense_replace=1, kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8,
                        qk_nope_head_dim=8, v_head_dim=8),
    "mixtral": dict(num_local_experts=8, num_experts_per_tok=2),
    "minimax_m2": dict(num_local_experts=8, num_experts_per_tok=2),
    "gpt_oss": dict(num_local_experts=8, num_experts_per_tok=2, layer_types=["full_attention"] * 2),
}


def _model(model_type, dtype=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(model_type, **TINY, **MODELS[model_type])
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith(("e_score_correction_bias", "expert_bias")):
                buffer.copy_(torch.randn_like(buffer) * 0.1)
        for name, param in model.named_parameters():
            if name.endswith(("gate.weight", "router.weight")):
                param.copy_(torch.randn_like(param))
    return model.to(dtype) if dtype is not None else model


@pytest.mark.parametrize("model_type", sorted(MODELS))
def test_capture_records_router_inputs(model_type, tmp_path):
    import torch

    model = _model(model_type)
    routers = find_router_modules(model)
    assert routers, f"no routers found in {model_type}"
    ids = torch.randint(0, 97, (2, 11))
    with RouterCapture(model) as capture, torch.no_grad():
        model(input_ids=ids)
    records = capture.records()
    assert set(records) == set(routers)
    spec = RouterSpec.from_hf_config(model.config)
    for name, rec in records.items():
        assert rec.hidden_states.shape == (22, model.config.hidden_size)
        assert (rec.correction_bias is not None) == spec.use_correction_bias, name
        assert (rec.linear_bias is not None) == (model_type == "gpt_oss"), name
        # The captured activations reproduce the module's own logits.
        module = routers[name]
        with torch.no_grad():
            out = module(torch.from_numpy(rec.hidden_states), *(
                [torch.from_numpy(rec.correction_bias)] if model_type == "minimax_m2" else []))
        logits = (out[2] if model_type.startswith("granite") else out[0]).double().numpy()
        np.testing.assert_allclose(project(rec.hidden_states, rec.weight, rec.linear_bias), logits,
                                   rtol=1e-5, atol=1e-5)

    capture.save(tmp_path / "cap.safetensors")
    loaded, saved_spec = load_capture(tmp_path / "cap.safetensors")
    assert saved_spec.to_dict() == spec.to_dict()
    for name, rec in records.items():
        np.testing.assert_array_equal(loaded[name].hidden_states, rec.hidden_states)
        np.testing.assert_array_equal(loaded[name].weight, rec.weight)


def test_bf16_capture_is_lossless_and_compact(tmp_path):
    import torch

    model = _model("deepseek_v3", torch.bfloat16)
    with RouterCapture(model, max_tokens=5) as capture, torch.no_grad():
        model(input_ids=torch.randint(0, 97, (1, 9)))
        model(input_ids=torch.randint(0, 97, (1, 9)))
    (record,) = capture.records().values()
    assert record.hidden_states.shape[0] == 5 and record.hidden_dtype == "bfloat16"
    capture.save(tmp_path / "cap.safetensors")
    loaded, spec = load_capture(tmp_path / "cap.safetensors")
    np.testing.assert_array_equal(next(iter(loaded.values())).hidden_states, record.hidden_states)


def test_capture_then_assert_against_live_router():
    """End to end: the model's own top-k matches the reference router within FP32 error."""
    import torch

    model = _model("deepseek_v3")
    gate_name, gate = next(iter(find_router_modules(model).items()))
    outputs = {}
    gate.register_forward_hook(lambda m, a, out: outputs.setdefault("out", out))
    with RouterCapture(model) as capture, torch.no_grad():
        model(input_ids=torch.randint(0, 97, (4, 16)))
    rec = capture.records()[gate_name]
    spec = RouterSpec.from_hf_config(model.config).with_bias(rec.correction_bias)
    logits, weights, ids = outputs["out"]
    reference = project(rec.hidden_states, rec.weight)
    assert_routing_close(reference, (ids, weights), spec, rtol=1e-5, atol=1e-5)


def test_ernie_bias_under_the_router_and_float64_models(tmp_path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model("ernie4_5_moe", **TINY, moe_num_experts=8, moe_k=2)
    model = AutoModelForCausalLM.from_config(config).eval()
    with RouterCapture(model) as capture, torch.no_grad():
        model(input_ids=torch.randint(0, 97, (1, 7)))
    assert all(rec.correction_bias is not None for rec in capture.records().values())

    class Block(torch.nn.Module):  # toy float64 model with an MoE-style router
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(32, 8, bias=False, dtype=torch.float64)

        def forward(self, x):
            return self.gate(x)

    model64 = torch.nn.Sequential(Block())
    with RouterCapture(model64) as capture, torch.no_grad():
        model64(torch.randn(5, 32, dtype=torch.float64))
    capture.save(tmp_path / "f64.safetensors")
    loaded, _ = load_capture(tmp_path / "f64.safetensors")
    for name, rec in capture.records().items():
        assert rec.hidden_states.dtype == np.float64
        np.testing.assert_array_equal(loaded[name].hidden_states, rec.hidden_states)
