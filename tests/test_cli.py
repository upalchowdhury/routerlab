import json
import subprocess
import sys

import numpy as np
import pytest

from routerlab import RouterSpec, route, save_safetensors
from routerlab.capture import CapturedRouter, save_capture
from routerlab.cli import main

E, H = 16, 24
CONFIG = {"model_type": "deepseek_v3", "n_routed_experts": E, "num_experts_per_tok": 4, "n_group": 4,
          "topk_group": 2, "norm_topk_prob": True, "routed_scaling_factor": 2.5, "hidden_size": H,
          "scoring_func": "sigmoid", "topk_method": "noaux_tc"}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CONFIG))
    return path


def test_spec_command(config_path, capsys):
    assert main(["spec", str(config_path)]) == 0
    assert "top2_sum" in capsys.readouterr().out
    assert main(["spec", str(config_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_group"] == 4


def test_compare_command_multi_layer(tmp_path, config_path, capsys):
    rng = np.random.default_rng(0)
    ref, ok, bug = {}, {}, {}
    spec = RouterSpec.from_hf_config(CONFIG)
    for layer in range(3):
        logits = rng.normal(size=(200, E))
        bias = rng.normal(0, 0.05, E)
        ref[f"layer{layer}/logits"] = logits
        ref[f"layer{layer}/correction_bias"] = bias
        ok[f"layer{layer}/logits"] = logits + rng.uniform(-1e-3, 1e-3, logits.shape)
        wrong = route(logits, spec.with_bias(bias).replace(group_score="max"))
        bug[f"layer{layer}/topk_ids"] = wrong.expert_ids
        bug[f"layer{layer}/topk_weights"] = wrong.weights
    np.savez(tmp_path / "ref.npz", **ref)
    np.savez(tmp_path / "ok.npz", **ok)
    save_safetensors(tmp_path / "bug.safetensors", bug)

    assert main(["compare", str(tmp_path / "ref.npz"), str(tmp_path / "ok.npz"), "--config", str(config_path),
                 "--tolerance", "1.1e-3", "--json", str(tmp_path / "out.json")]) == 0
    out = capsys.readouterr().out
    assert "| layer0 |" in out and "3 routers" in out
    assert set(json.loads((tmp_path / "out.json").read_text())) == {"layer0", "layer1", "layer2"}

    assert main(["compare", str(tmp_path / "ref.npz"), str(tmp_path / "bug.safetensors"),
                 "--config", str(config_path), "--tolerance", "1e-3"]) == 1
    assert main(["compare", str(tmp_path / "ref.npz"), str(tmp_path / "bug.safetensors"),
                 "--config", str(config_path), "--tolerance", "1e-3", "--fail-on", "never"]) == 0


def test_compare_single_arrays_with_manual_spec(tmp_path, capsys):
    logits = np.random.default_rng(1).normal(size=(64, 8))
    np.save(tmp_path / "a.npy", logits)
    np.save(tmp_path / "b.npy", logits)
    assert main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--top-k", "2",
                 "--renormalize"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_inspect_and_sweep_from_checkpoint(tmp_path, capsys):
    rng = np.random.default_rng(2)
    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(CONFIG))
    save_safetensors(ckpt / "model.safetensors", {
        "model.layers.1.mlp.gate.weight": rng.normal(size=(E, H)).astype(np.float32),
        "model.layers.1.mlp.gate.e_score_correction_bias": rng.normal(0, 0.05, E).astype(np.float32),
        "model.layers.1.mlp.gate_proj.weight": rng.normal(size=(32, H)).astype(np.float32)},
        bf16=["model.layers.1.mlp.gate.weight"])
    assert main(["inspect", str(ckpt), "--stats"]) == 0
    out = capsys.readouterr().out
    assert "1 routers" in out and "BF16" in out and "stored as BF16" in out
    assert main(["inspect", str(ckpt), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["routers"][0]["layer"] == 1

    np.save(tmp_path / "hidden.npy", rng.normal(size=(128, H)).astype(np.float32))
    assert main(["sweep", "--checkpoint", str(ckpt), "--layer", "1", "--hidden", str(tmp_path / "hidden.npy"),
                 "--json", str(tmp_path / "sweep.json")]) == 0
    assert "bf16 operands, bf16 logits" in capsys.readouterr().out
    result = json.loads((tmp_path / "sweep.json").read_text())
    assert len(result["layer 1"]["configs"]) == 6
    assert main(["sweep", "--checkpoint", str(ckpt), "--layer", "1", "--synthetic", "32",
                 "--configs", "name=fp16 logits,input_format=fp32,weight_format=fp32,output_format=fp16"]) == 0
    assert "fp16 logits" in capsys.readouterr().out


def test_sweep_from_capture_file(tmp_path, capsys):
    rng = np.random.default_rng(3)
    records = {f"model.layers.{i}.mlp.gate": CapturedRouter(
        name=f"model.layers.{i}.mlp.gate", hidden_states=rng.normal(size=(64, H)).astype(np.float32),
        weight=rng.normal(size=(E, H)).astype(np.float32), correction_bias=rng.normal(0, .05, E).astype(np.float32),
        linear_bias=None, hidden_dtype="float32", weight_dtype="float32") for i in (1, 2)}
    save_capture(tmp_path / "cap.safetensors", records, spec=RouterSpec.from_hf_config(CONFIG))
    assert main(["sweep", str(tmp_path / "cap.safetensors"), "--layers", "2"]) == 0
    out = capsys.readouterr().out
    assert "model.layers.2.mlp.gate" in out and "model.layers.1.mlp.gate" not in out


def test_errors_are_reported_cleanly(tmp_path, capsys):
    assert main(["spec", str(tmp_path / "nope.json")]) == 2
    assert "error" in capsys.readouterr().err


def test_console_entry_point_and_demo():
    out = subprocess.run([sys.executable, "-m", "routerlab.cli", "demo"], capture_output=True, text=True, check=True)
    assert "UNEXPLAINED" in out.stdout and "PASS" in out.stdout
    version = subprocess.run([sys.executable, "-m", "routerlab.cli", "--version"], capture_output=True, text=True)
    assert version.stdout.startswith("routerlab ")


def test_compare_bare_keys_unchecked_and_fail_on(tmp_path, capsys):
    rng = np.random.default_rng(5)
    spec = RouterSpec.from_hf_config(CONFIG)
    logits, bias = rng.normal(size=(300, E)), rng.normal(0, 0.05, E)
    np.savez(tmp_path / "ref.npz", logits=logits, correction_bias=bias)
    good = route(logits, spec.with_bias(bias))
    swapped = good.weights[:, ::-1].copy()
    np.savez(tmp_path / "swapped.npz", topk_ids=good.expert_ids, topk_weights=swapped)
    np.savez(tmp_path / "garbage.npz", logits=rng.normal(size=(300, E)))
    (tmp_path / "c.json").write_text(json.dumps(CONFIG))
    ref, cfg = str(tmp_path / "ref.npz"), ["--config", str(tmp_path / "c.json")]
    # Bare keys form one router with its bias attached (not a second one-token "layer").
    assert main(["compare", ref, ref, *cfg]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Router deepseek_v3: 300 tokens") and "PASS" in out
    # garbage without tolerance: not a PASS
    assert main(["compare", ref, str(tmp_path / "garbage.npz"), *cfg]) == 1
    assert "UNCHECKED" in capsys.readouterr().out
    # same experts, swapped weights: every mode except 'never' fails once a tolerance is given
    for mode, code in (("unexplained", 1), ("any", 1), ("never", 0)):
        assert main(["compare", ref, str(tmp_path / "swapped.npz"), *cfg, "--tolerance", "1e-3",
                     "--fail-on", mode]) == code


def test_selection_reference_with_num_experts(tmp_path):
    ids = np.array([[0, 1], [2, 3]])
    np.save(tmp_path / "ids.npy", ids)
    assert main(["compare", str(tmp_path / "ids.npy"), str(tmp_path / "ids.npy"), "--top-k", "2",
                 "--num-experts", "4"]) == 0


def test_sweep_bias_flag_and_config_parsing(tmp_path, capsys):
    rng = np.random.default_rng(6)
    records = {"model.layers.1.mlp.gate": CapturedRouter(
        name="model.layers.1.mlp.gate", hidden_states=rng.normal(size=(32, H)).astype(np.float32),
        weight=rng.normal(size=(E, H)).astype(np.float32), correction_bias=None, linear_bias=None,
        hidden_dtype="float32", weight_dtype="float32")}
    save_capture(tmp_path / "cap.safetensors", records, spec=RouterSpec.from_hf_config(CONFIG))
    np.save(tmp_path / "bias.npy", rng.normal(0, 0.05, E))
    cap = str(tmp_path / "cap.safetensors")
    with pytest.raises(SystemExit, match="needs a correction bias"):
        main(["sweep", cap])
    capsys.readouterr()
    assert main(["sweep", cap, "--bias", str(tmp_path / "bias.npy"),
                 "--configs", "name=a, weight_format=bf16 ; name=b,output_format=fp16"]) == 0
    out = capsys.readouterr().out
    assert "| a |" in out and "| b |" in out
    with pytest.raises(SystemExit, match="cannot parse"):
        main(["sweep", cap, "--bias", str(tmp_path / "bias.npy"), "--configs", "name=a,weight=bf16"])


def test_compare_vllm_routed_experts_and_capture_files(tmp_path, capsys):
    rng = np.random.default_rng(7)
    spec = RouterSpec(top_k=2, renormalize=True, num_experts=8, name="mixtral-like")
    ref, ids = {}, []
    for layer in (1, 2, 3):
        logits = rng.normal(size=(50, 8))
        ref[f"model.layers.{layer}.mlp.gate/logits"] = logits
        ids.append(route(logits, spec).expert_ids)
    routed = np.stack(ids, axis=1).astype(np.uint8)  # [tokens, layers, k], as vLLM returns it
    np.savez(tmp_path / "trainer.npz", **ref)
    np.save(tmp_path / "routed_experts.npy", routed)
    args = ["compare", str(tmp_path / "trainer.npz"), str(tmp_path / "routed_experts.npy"), "--top-k", "2",
            "--renormalize", "--tolerance", "1e-6"]
    assert main(args) == 0
    assert "| model.layers.3.mlp.gate | 50 | 0 |" in capsys.readouterr().out
    routed[:, 2, :] = route(rng.normal(size=(50, 8)), spec).expert_ids  # last layer routed from wrong logits
    np.save(tmp_path / "routed_experts.npy", routed)
    assert main(args) == 1
    capsys.readouterr()
    assert main([*args, "--layers", "1,2,3"]) == 1
    with pytest.raises(SystemExit, match="3 MoE layers but the reference has 2"):
        main([*args, "--layers", "1,2"])

    # Capture files compare directly; the spec comes from the capture metadata.
    records = {f"model.layers.{i}.mlp.gate": CapturedRouter(
        name=f"model.layers.{i}.mlp.gate", hidden_states=rng.normal(size=(40, H)).astype(np.float32),
        weight=rng.normal(size=(8, H)).astype(np.float32), correction_bias=None, linear_bias=None,
        hidden_dtype="float32", weight_dtype="float32") for i in (1, 2)}
    save_capture(tmp_path / "a.safetensors", records, spec=spec)
    save_capture(tmp_path / "b.safetensors", records, spec=spec)
    capsys.readouterr()
    assert main(["compare", str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")]) == 0
    assert "| model.layers.2.mlp.gate | 40 | 0 |" in capsys.readouterr().out


def test_vllm_layers_pair_in_natural_order(tmp_path):
    rng = np.random.default_rng(8)
    spec = RouterSpec(top_k=2, num_experts=8)
    ref, ids = {}, []
    for layer in range(12):  # names without 'layers.N': 'layer10' must not sort before 'layer2'
        logits = rng.normal(size=(30, 8))
        ref[f"layer{layer}/logits"] = logits
        ids.append(route(logits, spec).expert_ids)
    np.savez(tmp_path / "ref.npz", **ref)
    np.save(tmp_path / "routed.npy", np.stack(ids, axis=1).astype(np.uint8))
    assert main(["compare", str(tmp_path / "ref.npz"), str(tmp_path / "routed.npy"), "--top-k", "2",
                 "--tolerance", "1e-9"]) == 0
