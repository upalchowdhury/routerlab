import json

import numpy as np
import pytest

from routerlab import RouterSpec, Routing, route, topk_indices


def test_softmax_topk_renormalized_by_hand():
    logits = np.array([[2.0, 1.0, 0.0, -1.0]])
    routing = route(logits, RouterSpec(top_k=2, renormalize=True))
    assert routing.expert_ids.tolist() == [[0, 1]]
    np.testing.assert_allclose(routing.weights, [[np.e / (np.e + 1), 1 / (np.e + 1)]])


def test_softmax_without_renormalization_uses_full_softmax():
    logits = np.log(np.array([[0.5, 0.3, 0.2]]))
    routing = route(logits, RouterSpec(top_k=2))
    np.testing.assert_allclose(routing.weights, [[0.5, 0.3]])


def test_deepseek_v3_semantics_by_hand():
    # 8 experts, 4 groups of 2, keep 2 groups, top-2 experts. Bias steers selection only.
    logits = np.array([[3.0, 2.9, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0]])
    bias = np.array([0, 0, 0.9, 0.9, 0, 0, 0, 0.0])
    spec = RouterSpec(top_k=2, scoring="sigmoid", n_group=4, topk_group=2, group_score="top2_sum",
                      renormalize=True, scaling_factor=2.0).with_bias(bias)
    sig = 1 / (1 + np.exp(-logits[0]))
    # choice = sig + bias = [.953, .948, 1.4, 1.4, .731, .731, .269, .269]
    # group top-2 sums = [1.90, 2.80, 1.46, 0.54] -> groups 1 and 0; best experts there: 2 and 3.
    routing = route(logits, spec)
    assert sorted(routing.expert_ids[0].tolist()) == [2, 3]
    # Weights come from the *unbiased* sigmoid (0.5 each), renormalized, then scaled by 2.
    np.testing.assert_allclose(routing.weights[0], [1.0, 1.0])
    assert sig[2] < sig[0]  # without the bias these experts would have lost


def test_group_limit_excludes_experts_outside_selected_groups():
    # Expert 7 has the single highest logit but lives in a weak group.
    logits = np.array([[1.0, 1.0, 1.0, 1.0, -5.0, -5.0, -5.0, 2.0]])
    spec = RouterSpec(top_k=2, scoring="sigmoid", n_group=2, topk_group=1, group_score="top2_sum",
                      use_correction_bias=True).with_bias(np.zeros(8))
    assert 7 not in route(logits, spec).expert_ids[0]
    spec_max = RouterSpec(top_k=2, n_group=2, topk_group=1, group_score="max")
    assert 7 in route(logits, spec_max).expert_ids[0]


def test_ties_break_to_lower_index():
    assert topk_indices(np.array([[1.0, 2.0, 2.0, 2.0]]), 2).tolist() == [[1, 2]]


def test_selection_on_logits_avoids_sigmoid_saturation():
    # sigmoid(40) == sigmoid(50) == 1.0 in float64, but the logits still order them.
    logits = np.array([[40.0, 50.0, 0.0]])
    assert route(logits, RouterSpec(top_k=1, scoring="sigmoid")).expert_ids.tolist() == [[1]]


def test_bias_required_when_spec_uses_it():
    spec = RouterSpec(top_k=2, scoring="sigmoid", use_correction_bias=True)
    with pytest.raises(ValueError, match="with_bias"):
        route(np.zeros((1, 4)), spec)


def test_spec_validation():
    with pytest.raises(ValueError):
        RouterSpec(top_k=2, scoring="relu")
    with pytest.raises(ValueError):
        route(np.zeros((1, 4)), RouterSpec(top_k=5))
    with pytest.raises(ValueError):
        route(np.zeros((1, 6)), RouterSpec(top_k=2, n_group=4, topk_group=2))
    with pytest.raises(ValueError, match="non-finite"):
        route(np.array([[0.0, np.nan]]), RouterSpec(top_k=1))


def test_spec_round_trips_through_json():
    spec = RouterSpec(top_k=8, scoring="sigmoid", n_group=8, topk_group=4, group_score="top2_sum",
                      use_correction_bias=True, renormalize=True, renorm_eps=1e-20, scaling_factor=2.5,
                      num_experts=256, name="deepseek_v3")
    again = RouterSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again.to_dict() == spec.to_dict()


@pytest.mark.parametrize("config,expected", [
    ({"model_type": "deepseek_v3", "n_routed_experts": 256, "num_experts_per_tok": 8, "n_group": 8,
      "topk_group": 4, "norm_topk_prob": True, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
      "topk_method": "noaux_tc"},
     dict(scoring="sigmoid", top_k=8, num_experts=256, n_group=8, topk_group=4, group_score="top2_sum",
          use_correction_bias=True, renormalize=True, scaling_factor=2.5)),
    ({"model_type": "kimi_k2", "n_routed_experts": 384, "num_experts_per_tok": 8, "n_group": 1,
      "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 2.827, "topk_method": "noaux_tc"},
     dict(scoring="sigmoid", top_k=8, num_experts=384, use_correction_bias=True, scaling_factor=2.827)),
    ({"model_type": "mixtral", "num_local_experts": 8, "num_experts_per_tok": 2},
     dict(scoring="softmax", top_k=2, num_experts=8, renormalize=True)),
    ({"model_type": "qwen3_moe", "num_experts": 128, "num_experts_per_tok": 8, "norm_topk_prob": True},
     dict(scoring="softmax", top_k=8, renormalize=True)),
    ({"model_type": "qwen2_moe", "num_experts": 60, "num_experts_per_tok": 4, "norm_topk_prob": False},
     dict(renormalize=False)),
    ({"model_type": "deepseek_v2", "n_routed_experts": 160, "num_experts_per_tok": 6, "n_group": 8,
      "topk_group": 3, "topk_method": "group_limited_greedy", "routed_scaling_factor": 16.0},
     dict(scoring="softmax", n_group=8, topk_group=3, group_score="max", scaling_factor=16.0)),
    ({"model_type": "llama4", "text_config": {"model_type": "llama4_text", "num_local_experts": 16,
                                              "num_experts_per_tok": 1}},
     dict(scoring="sigmoid", top_k=1, num_experts=16, renormalize=False)),
    ({"model_type": "gpt_oss", "num_local_experts": 32, "num_experts_per_tok": 4},
     dict(scoring="softmax", renormalize=True)),
    ({"model_type": "kimi_linear", "num_experts": 256, "num_experts_per_token": 8, "num_expert_group": 1,
      "moe_renormalize": True, "routed_scaling_factor": 2.446},
     dict(top_k=8, num_experts=256, renormalize=True, use_correction_bias=True)),
    ({"model_type": "some_new_deepseek", "n_routed_experts": 64, "num_experts_per_tok": 6,
      "topk_method": "noaux_tc", "norm_topk_prob": True},
     dict(scoring="sigmoid", use_correction_bias=True)),
])
def test_from_hf_config_dicts(config, expected):
    spec = RouterSpec.from_hf_config(config)
    for key, value in expected.items():
        assert getattr(spec, key) == value, key


def test_from_hf_config_reads_paths(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mixtral", "num_local_experts": 8,
                                                      "num_experts_per_tok": 2}))
    assert RouterSpec.from_hf_config(tmp_path).top_k == 2
    assert RouterSpec.from_hf_config(tmp_path / "config.json").num_experts == 8


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="not modelled"):
        RouterSpec.from_hf_config({"model_type": "dbrx", "num_experts": 16, "num_experts_per_tok": 4})
    with pytest.raises(ValueError, match="no MoE"):
        RouterSpec.from_hf_config({"model_type": "llama"})


def test_routing_container_validation_and_helpers():
    r = Routing(np.array([[0, 2], [1, 1]]), np.array([[0.6, 0.4], [0.5, 0.5]]), num_experts=3)
    assert r.duplicate_rows().tolist() == [False, True]
    assert r.expert_load().tolist() == [1, 2, 1]
    np.testing.assert_allclose(r.dense_weights(), [[0.6, 0, 0.4], [0, 1.0, 0]])
    with pytest.raises(ValueError, match="outside"):
        Routing(np.array([[0, 3]]), None, num_experts=3)


def test_accepts_torch_bf16_tensors():
    torch = pytest.importorskip("torch")
    logits = torch.randn(5, 8).bfloat16()
    expected = route(logits.float().numpy(), RouterSpec(top_k=2))
    np.testing.assert_array_equal(route(logits, RouterSpec(top_k=2)).expert_ids, expected.expert_ids)
