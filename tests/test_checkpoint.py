import importlib.util
import json

import numpy as np
import pytest

from routerlab import Checkpoint, SafetensorsFile, quantize, save_safetensors

HAS_ST_TORCH = all(importlib.util.find_spec(m) is not None for m in ("safetensors", "torch"))


def test_safetensors_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {"f32": rng.normal(size=(3, 5)).astype(np.float32), "f16": rng.normal(size=7).astype(np.float16),
               "f64": rng.normal(size=(2, 2)), "i64": np.arange(6).reshape(2, 3), "flag": np.array([True, False]),
               "bf": rng.normal(size=(4, 4)).astype(np.float32), "empty": np.zeros((0, 3), np.float32)}
    save_safetensors(tmp_path / "t.safetensors", tensors, {"note": "hi"}, bf16=["bf"])
    file = SafetensorsFile(tmp_path / "t.safetensors")
    assert file.metadata == {"note": "hi"} and file.dtype("bf") == "BF16"
    for name, value in tensors.items():
        expected = quantize(value, "bf16") if name == "bf" else value
        np.testing.assert_array_equal(file.get(name), expected)
        assert file.get(name).shape == value.shape


@pytest.mark.skipif(not HAS_ST_TORCH, reason="safetensors+torch not installed")
def test_interoperates_with_the_safetensors_library(tmp_path):
    import torch
    from safetensors.numpy import load_file
    from safetensors.torch import save_file

    reference = {"bf16": torch.randn(8, 16).bfloat16(), "e4m3": (torch.randn(8, 16) * 50).to(torch.float8_e4m3fn),
                 "e5m2": (torch.randn(8, 16) * 1e3).to(torch.float8_e5m2), "f32": torch.randn(3)}
    save_file(reference, tmp_path / "lib.safetensors")
    ours = SafetensorsFile(tmp_path / "lib.safetensors")
    for name, tensor in reference.items():
        np.testing.assert_array_equal(ours.get(name), tensor.float().numpy())

    save_safetensors(tmp_path / "ours.safetensors", {"a": np.arange(10, dtype=np.int32),
                                                     "b": np.ones((2, 3), np.float32)})
    loaded = load_file(tmp_path / "ours.safetensors")
    np.testing.assert_array_equal(loaded["a"], np.arange(10))


def _write_checkpoint(root, rng, e=16, h=24):
    """A DeepSeek-V3-shaped sharded checkpoint with distractor tensors."""
    root.mkdir()
    config = {"model_type": "deepseek_v3", "n_routed_experts": e, "num_experts_per_tok": 4, "n_group": 4,
              "topk_group": 2, "norm_topk_prob": True, "routed_scaling_factor": 2.5, "hidden_size": h,
              "scoring_func": "sigmoid", "topk_method": "noaux_tc"}
    (root / "config.json").write_text(json.dumps(config))
    w1 = rng.normal(size=(e, h)).astype(np.float32)
    shard1 = {"model.layers.0.mlp.gate_proj.weight": rng.normal(size=(48, h)).astype(np.float32),
              "model.layers.1.mlp.gate.weight": w1,
              "model.layers.1.mlp.gate.e_score_correction_bias": rng.normal(size=e).astype(np.float32),
              "model.layers.1.mlp.shared_expert_gate.weight": rng.normal(size=(1, h)).astype(np.float32),
              "model.layers.1.post_attention_layernorm.weight": np.ones(h, np.float32)}
    q = quantize(rng.normal(size=(e, h)) * 10, "fp8_e4m3fn")
    shard2 = {"model.layers.2.mlp.gate.weight": q, "model.layers.2.mlp.gate.weight_scale_inv":
              np.full((1, 1), 0.5, np.float32),
              "model.layers.2.mlp.gate.e_score_correction_bias": np.zeros(e, np.float32)}
    save_safetensors(root / "model-00001-of-00002.safetensors", shard1, bf16=["model.layers.1.mlp.gate.weight"])
    _save_fp8(root / "model-00002-of-00002.safetensors", shard2)
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return w1, q


def _save_fp8(path, tensors):
    """Write the FP8 tensor with a hand-built header (save_safetensors does not emit FP8)."""
    import struct
    table = {float(v): i for i, v in enumerate(_e4m3_values())}
    header, blobs, offset = {}, [], 0
    for name, value in tensors.items():
        if "scale" in name or "bias" in name:
            data, tag = np.ascontiguousarray(value, "<f4").tobytes(), "F32"
        else:
            data, tag = bytes(table[float(v)] for v in value.ravel()), "F8_E4M3"
        header[name] = {"dtype": tag, "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _e4m3_values():
    from routerlab.checkpoint import _FP8_TABLES
    values = _FP8_TABLES["F8_E4M3"].astype(np.float64)
    values[np.isnan(values)] = 1e9  # never matched
    values[128] = 1e9  # keep +0.0 mapped to code 0, not -0.0
    return values


def test_checkpoint_router_discovery(tmp_path):
    rng = np.random.default_rng(1)
    w1, q = _write_checkpoint(tmp_path / "ckpt", rng)
    ckpt = Checkpoint(tmp_path / "ckpt")
    routers = ckpt.find_routers()
    assert [r.layer for r in routers] == [1, 2]
    assert routers[0].dtype == "BF16" and routers[0].correction_bias.endswith("e_score_correction_bias")
    weight, bias, linear = ckpt.load_router(1)
    np.testing.assert_array_equal(weight, quantize(w1, "bf16"))
    assert bias.shape == (16,) and linear is None
    weight2, _, _ = ckpt.load_router("model.layers.2.mlp.gate")
    np.testing.assert_array_equal(weight2, q * 0.5)  # FP8 block scale applied
    spec = ckpt.router_spec()
    assert spec.name == "deepseek_v3" and spec.use_correction_bias


def test_router_naming_conventions(tmp_path):
    rng = np.random.default_rng(2)
    e, h = 8, 12
    tensors = {
        "model.layers.0.block_sparse_moe.gate.weight": rng.normal(size=(e, h)).astype(np.float32),  # Mixtral
        "model.layers.0.block_sparse_moe.e_score_correction_bias": np.ones(e, np.float32),          # MiniMax-M2
        "model.layers.1.mlp.router.weight": rng.normal(size=(e, h)).astype(np.float32),             # GPT-OSS
        "model.layers.1.mlp.router.bias": np.ones(e, np.float32),
        "model.layers.2.mlp.gate.weight": rng.normal(size=(e, h)).astype(np.float32),               # Ernie 4.5
        "model.layers.2.mlp.moe_statics.e_score_correction_bias": np.ones((1, e), np.float32),
        "model.layers.3.mlp.gate.wg.weight": rng.normal(size=(e, h)).astype(np.float32),            # HunYuan
        "vision.blocks.0.attn.gate.weight": rng.normal(size=(e, 7)).astype(np.float32),             # wrong H
    }
    save_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mixtral", "num_local_experts": e,
                                                      "num_experts_per_tok": 2, "hidden_size": h}))
    routers = {r.layer: r for r in Checkpoint(tmp_path).find_routers()}
    assert sorted(routers) == [0, 1, 2, 3]
    assert routers[0].correction_bias == "model.layers.0.block_sparse_moe.e_score_correction_bias"
    assert routers[1].linear_bias == "model.layers.1.mlp.router.bias" and routers[1].correction_bias is None
    assert routers[2].correction_bias == "model.layers.2.mlp.moe_statics.e_score_correction_bias"
    assert routers[3].weight == "model.layers.3.mlp.gate.wg.weight"


def test_single_file_and_errors(tmp_path):
    save_safetensors(tmp_path / "x.safetensors", {"a": np.ones(2, np.float32)})
    assert Checkpoint(tmp_path / "x.safetensors").names() == ["a"]
    with pytest.raises(FileNotFoundError):
        Checkpoint(tmp_path / "missing")
    (tmp_path / "bad.safetensors").write_bytes(b"\x01")
    with pytest.raises(ValueError):
        SafetensorsFile(tmp_path / "bad.safetensors")


def test_deepseek_v4_and_granite_naming(tmp_path):
    rng = np.random.default_rng(3)
    e, h = 8, 12
    v4 = tmp_path / "v4"
    v4.mkdir()
    (v4 / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "n_routed_experts": e,
                                                "num_experts_per_tok": 2, "hidden_size": h}))
    save_safetensors(v4 / "model.safetensors", {
        "model.layers.0.ffn.gate.weight": rng.normal(size=(e, h)).astype(np.float32),
        "model.layers.0.ffn.gate.tid2eid": np.zeros((97, 2), np.int64),
        "model.layers.3.ffn.gate.weight": rng.normal(size=(e, h)).astype(np.float32),
        "model.layers.3.ffn.gate.bias": rng.normal(size=e).astype(np.float32)})
    routers = {r.layer: r for r in Checkpoint(v4).find_routers()}
    assert routers[3].correction_bias == "model.layers.3.ffn.gate.bias" and routers[3].linear_bias is None
    assert routers[0].hash_table == "model.layers.0.ffn.gate.tid2eid" and routers[3].hash_table is None

    granite = tmp_path / "granite"
    granite.mkdir()
    (granite / "config.json").write_text(json.dumps({"model_type": "granitemoe", "num_local_experts": e,
                                                     "num_experts_per_tok": 2, "hidden_size": h}))
    save_safetensors(granite / "model.safetensors", {
        "model.layers.0.block_sparse_moe.router.layer.weight": rng.normal(size=(e, h)).astype(np.float32)})
    (router,) = Checkpoint(granite).find_routers()
    assert router.layer == 0 and router.weight.endswith("router.layer.weight")


def test_fp8_block_size_from_quantization_config(tmp_path):
    root = tmp_path / "q"
    root.mkdir()
    weight = quantize(np.random.default_rng(4).normal(size=(100, 64)) * 10, "fp8_e4m3fn")
    scale = np.array([[0.5], [2.0]], np.float32)  # 64x64 blocks -> 2 x 1 scales
    (root / "config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [64, 64]}}))
    _save_fp8(root / "model.safetensors", {"w.weight": weight, "w.weight_scale_inv": scale})
    loaded = Checkpoint(root).load("w.weight")
    np.testing.assert_array_equal(loaded[:64], weight[:64] * 0.5)
    np.testing.assert_array_equal(loaded[64:], weight[64:] * 2.0)
