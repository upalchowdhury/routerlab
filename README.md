# RouterLab

[![CI](https://github.com/upalchowdhury/routerlab/actions/workflows/ci.yml/badge.svg)](https://github.com/upalchowdhury/routerlab/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Find out whether your MoE router is wrong or just imprecise.** RouterLab checks whether two implementations of a mixture-of-experts router pick the same experts for every token and layer. It compares a fused kernel against the reference, an inference engine against the trainer, BF16 against FP32, or batch 1 against batch 64. When they disagree, it tells you whether each change is ordinary rounding at a near-tie or a real bug.

## Why this matters

A router makes a hard top-k decision on noisy numbers, so any numerical difference moves tokens near the boundary to other experts, and that compounds across layers. This shows up in real systems:

- **Silent corruption.** In GLM-5.2-NVFP4, FP32 selection keys tied exactly at the top-8 boundary, and the routing kernel emitted NaN expert weights ([sglang#30989](https://github.com/sgl-project/sglang/issues/30989)).
- **Bugs found late.** New routing methods in an FP8 MoE backend dropped DeepSeek-V2-Lite GSM8K to 1.6%. Only a nightly end-to-end eval caught it ([vllm#33532](https://github.com/vllm-project/vllm/issues/33532)).
- **Precision choices argued on intuition.** Correction biases cast to BF16 and BF16 router outputs ([sglang#38695](https://github.com/sgl-project/sglang/issues/38695)). On DeepSeek-V3's real checkpoint, a BF16 cast of the bias merges 62% of expert pairs, leaving a median of 3 distinct values per layer.
- **RL instability.** Routing mismatch between rollout engine and trainer destabilises MoE RL ([R3](https://arxiv.org/abs/2510.11370), [GSPO](https://arxiv.org/abs/2507.18071)).

Measured with RouterLab on real models and real text ([case studies](docs/case-studies/README.md)):

| | OLMoE-1B-7B | LFM2-8B-A1B |
| --- | ---: | ---: |
| BF16 router output vs exact, same inputs: token-layer routings that change experts | 0.71% | 2.43% |
| FP8 router weights | 9.6% | 11.0% |
| Whole model in BF16 vs FP32, worst layer | 6.8% | 27.8% |
| Same tokens, batch 4 vs batch 1 (Apple GPU), worst layer | 5.2% | 15.6% |

The standard checks miss this. Exact `topk_ids` comparisons are flaky, so they get loosened. "Logits match to 1e-3" says nothing about how many tokens that re-routes, and end-to-end evals find problems late without pointing at a cause. **[Full rationale, incidents and use cases →](docs/WHY.md)**

## What it does

- **Reference routers** for 29 MoE architectures (DeepSeek V2/V3/V4, Kimi, GLM, Qwen, Mixtral, Llama 4, GPT-OSS, …), built from `config.json` and tested side by side against the `transformers` implementations.
- **A margin per token:** the largest logit error that cannot change its experts, rigorous even with group-limited routing and correction bias. A changed token is *explained* if its margin is within your error bound and *unexplained* (a bug) otherwise.
- **Drop-in test assertion:** `assert_routing_close` for kernel tests. It's strict but not flaky: near-ties pass, bugs fail with token-level detail, and the tolerance comes from a rigorous FP32 error bound.
- **Comparisons for anything you can dump:** logits or top-k ids from any engine, vLLM's `routed_experts`, or two runs captured from PyTorch.
- **Precision tools:** bit-exact BF16/FP16/FP8 emulation, precision sweeps on real activations, and detection of exact top-k ties in engine precision.
- **Checkpoint inspection without downloading:** `routerlab inspect hf://deepseek-ai/DeepSeek-V3 --stats` reads router weights and biases by HTTP range requests in seconds.

From `routerlab demo`: a candidate kernel with a porting bug (DeepSeek-V2's group rule on a V3-style router, and the correction bias leaking into the weights) checked against the reference:

```text
Router deepseek_v3-style (synthetic): 2048 tokens, top-8 of 64 experts
  selection   891 tokens changed expert set (43.51%), mean overlap 0.918274
  weights     moved mass mean 2.000e-01  max 6.314e-01   max |err| (same experts) 3.458e-01
  WEIGHTS     1157 tokens with the same experts have weights outside tolerance (token 839, expert 39: 0.574148 vs 0.229486)
  margins     radius min 3.828e-05  p1 7.131e-04  p5 5.516e-03  median 7.365e-02 (logit units)
  tolerance   per-element logit tolerance (max 2.216e-02): 72 tokens at risk
  verdict     51 explained by tolerance, 840 UNEXPLAINED (margin larger than tolerance)
  largest margin among changed tokens: 4.976e-01
  load        largest per-expert change 284 tokens (expert 11)
    token 2: -[19] +[45]  margin 1.910e-02  UNEXPLAINED
    token 7: -[2] +[27]  margin 3.192e-01  UNEXPLAINED
  FAIL
```

The same comparison against correctly computed BF16 logits shows 12 changed tokens, all explained, and passes. RouterLab is a NumPy library plus a CLI; PyTorch and GPUs are optional.

## Install

```bash
pip install "moe-routerlab @ git+https://github.com/upalchowdhury/routerlab"
# with extras: "moe-routerlab[torch] @ git+https://github.com/upalchowdhury/routerlab"
#   [torch] capture from PyTorch models, [gpu] CUDA benchmark (Linux), [dev] test tools
```

The package isn't on PyPI yet; install from GitHub as shown.

This needs Python ≥ 3.10 and NumPy ≥ 1.24. It's tested on Linux and macOS, Python 3.10–3.13, and NumPy 1.24–2.x.

## Quick start

### 1. Route like the model does

```python
import numpy as np
from routerlab import RouterSpec, route, robustness_radius

spec = RouterSpec.from_hf_config("/models/DeepSeek-V3/config.json")  # dict, path, model dir, or PretrainedConfig
spec = spec.with_bias(e_score_correction_bias)   # per-layer bias for DeepSeek-V3-style routers

routing = route(logits, spec)            # float64 reference: routing.expert_ids, routing.weights
radius = robustness_radius(logits, spec) # per-token margin, in logit units
```

`logits` (and every other array argument) can be a NumPy array or a PyTorch tensor on any device, including BF16 tensors.

### 2. Make router-kernel tests strict *and* stable

```python
from routerlab import assert_routing_close, projection_error_bound, project

def test_fused_router_kernel():
    x, w, bias = make_inputs()                          # BF16 hidden states and gate weight
    topk_ids, topk_weights = my_fused_router(x, w, bias)   # the kernel under test

    reference = project(x, w)                           # exact float64 logits
    tol = projection_error_bound(x, w, accumulate="fp32")  # rigorous bound for any FP32 summation order
    assert_routing_close(reference, (topk_ids, topk_weights), spec.with_bias(bias), tolerance=tol)
```

The assertion lets a token change experts only if the tolerance can explain it. On failure it reports the offending tokens:

```text
AssertionError: Routing mismatch for 'deepseek_v3-style (synthetic)' (2048 tokens, top-8):
  840 tokens changed experts although their routing margin exceeds the tolerance (not a rounding effect):
      token 2: reference [19, 27, 33, 38, 48, 50, 52, 53] vs candidate [27, 33, 38, 45, 48, 50, 52, 53]; provably stable under the tolerance (margin 1.910e-02; tolerance in this row 5.930e-04..9.919e-03)
      token 7: reference [2, 19, 22, 33, 38, 48, 51, 53] vs candidate [19, 22, 27, 33, 38, 48, 51, 53]; provably stable under the tolerance (margin 3.192e-01; tolerance in this row 4.847e-04..1.113e-02)
  1157 tokens kept their experts but have weights outside tolerance; worst at token 839, expert 39: reference 0.574148, candidate 0.229486 (allowed |delta| 9.993e-03)
  (51 other tokens changed experts within tolerance: expected near-ties)
```

It also checks logits against the tolerance (when the candidate provides logits), rejects duplicate experts and non-finite weights, and checks weights of unchanged tokens against a bound derived from the tolerance. Use `weight_rtol=2**-8` if the engine returns BF16 weights.

### 3. Compare two systems

**Any engine:** save router logits (or `topk_ids`/`topk_weights`) from each system, one key per layer, as `.npz`, `.safetensors`, `.npy` or `.pt`:

```bash
routerlab compare trainer.npz engine.npz --config config.json --tolerance 2e-2 --json report.json
```

```text
| layer | tokens | changed | changed % | unexplained | max logit err | min margin | moved weight (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| model.layers.3.mlp.gate | 4096 | 41 | 1.001 | 0 | 1.28e-02 | 2.12e-05 | 1.62e-03 |
| model.layers.4.mlp.gate | 4096 | 44 | 1.074 | 0 | 1.50e-02 | 1.28e-06 | 1.79e-03 |
| model.layers.5.mlp.gate | 4096 | 36 | 0.879 | 0 | 1.47e-02 | 1.12e-07 | 1.54e-03 |

3 routers, 121 of 12288 token-routings changed (0.9847%)
```

*(Above: synthetic DeepSeek-V3-style layers, float64 reference vs. BF16 logits. About 1% of tokens change experts, and every change is explained by a margin below the BF16 error.)*

Key conventions: use `NAME` or `NAME/logits` for float logits, `NAME/topk_ids` with an optional `NAME/topk_weights` for selections, and `NAME/correction_bias` to attach a bias. Bare `logits`/`topk_ids`/... keys describe a single router.

Exit status 1 means some layer FAILED (unexplained flips, invalid selections, or out-of-tolerance logits or weights) or is UNCHECKED: it differs, but no `--tolerance` was given to judge it. `--fail-on any` also fails on explained flips; `--fail-on never` always exits 0.

Per token, the report gives the logit error, set changes, moved routing weight, the radius distribution, how many tokens are at risk under the tolerance, explained vs. unexplained flips with examples, exact top-k ties in FP32/BF16 selection keys, and per-expert load shift.

**vLLM rollouts vs your trainer (RL):** with `enable_return_routed_experts=True`, vLLM returns `routed_experts` with shape `[tokens, moe_layers, top_k]`. Save it as `.npy` and compare it against the trainer's router logits for the same token positions. Layers are matched in order:

```bash
routerlab compare trainer_logits.npz routed_experts.npy --config config.json --tolerance 1e-2
```

**Two PyTorch runs** (FP32 vs BF16, batch 1 vs batch 64, CPU vs GPU, before vs after a kernel change): capture each run, then compare the captures directly. Each run's own router logits are used, and the router spec comes from the capture:

```python
with RouterCapture(model) as capture:   # records router inputs and the logits each router computed
    model(**batch)
capture.save("run_a.safetensors")
```

```bash
routerlab compare run_a.safetensors run_b.safetensors --tolerance 2e-2
```

### 4. Decide a precision with real activations

Capture the hidden states that enter each router while running your model:

```python
from routerlab.capture import RouterCapture

with RouterCapture(model, max_tokens=8192) as capture:   # auto-detects gate/router modules
    model(**batch)
capture.save("router_inputs.safetensors")                # lossless; BF16 stays BF16
```

Then sweep precision configurations:

```bash
routerlab sweep router_inputs.safetensors
```

```text
## model.layers.7.mlp.gate
Router olmoe, 4096 tokens. Reference margin (logit units): p1 3.093e-04, p5 1.970e-03, p50 2.783e-02

| precision | max logit err | tokens changed | changed % | moved weight (mean) | max load shift | fp32-key ties |
|---|---:|---:|---:|---:|---:|---:|
| fp32 math on stored values | 1.24e-05 | 0 | 0.000 | 1.06e-07 | 0 | 0 |
| fp32 math, bf16 logits | 1.55e-02 | 24 | 0.586 | 7.93e-04 | 4 | 51 |
| bf16 operands, fp32 logits | 1.24e-05 | 0 | 0.000 | 1.06e-07 | 0 | 0 |
| bf16 operands, bf16 logits | 1.55e-02 | 24 | 0.586 | 7.93e-04 | 4 | 51 |
| fp16 operands, fp16 logits | 1.96e-03 | 2 | 0.049 | 8.47e-05 | 1 | 4 |
| fp8-e4m3 weights (per-channel scale) | 8.48e-02 | 388 | 9.473 | 1.21e-02 | 38 | 0 |
```

*(Above: real output, OLMoE-1B-7B run in BF16 on 4,096 WikiText-2 tokens, layer 7. The activations are already BF16, so the "bf16 operands" rows equal the FP32 rows. "fp32-key ties" counts tokens whose top-k boundary is an exact tie once the router logits are stored in that precision.)* Custom configurations look like `--configs "name=bf16 logits,input_format=fp32,weight_format=fp32,output_format=bf16;..."`.

Without a capture, `routerlab sweep --checkpoint DIR --layer N --hidden hidden.npy` reads the router straight from a checkpoint. `--synthetic N` uses Gaussian inputs, scaled by the layer's `post_attention_layernorm` weight when there is one; these results are labelled as indicative only.

### 5. Inspect a checkpoint without downloading it

```bash
routerlab inspect hf://deepseek-ai/DeepSeek-V3 --stats     # or a local directory / .safetensors file
```

```text
hf://deepseek-ai/DeepSeek-V3: 91991 tensors, 59 routers
spec: deepseek_v3: sigmoid scoring, top-8 of 256 experts, group-limited (4 of 8 groups by top2_sum), correction bias (not attached), renormalized, scaled x2.5

layer          shape  dtype  corr. bias  lin. bias  weight
    3    [256, 7168]  BF16   yes         no         model.layers.3.mlp.gate.weight  |w|max 0.248  bias [4.95, 4.96]
  ...
   61    [256, 7168]  BF16   yes         no         model.layers.61.mlp.gate.weight  |w|max 0.324  bias [3.7, 3.88]
note: Casting the correction bias to BF16 merges up to 94% of expert pairs (layer 51: 38 distinct values -> 2); keep it in FP32
note: With FP32 selection keys, score differences below ~4.8e-07 vanish when the bias is added; ...
```

This lists every router weight with its correction bias and linear bias, shapes, dtypes and magnitude stats, plus warnings: bias values that BF16 cannot hold apart, the FP32 tie threshold, hash-routed layers, and FP32-stored weights that are exactly BF16-representable. It reads only the needed bytes over HTTP range requests (about 10 s for DeepSeek-V3's 163 shards), decodes BF16/FP8 exactly, and doesn't need torch. Gated models use `HF_TOKEN`, which is sent only to the Hub host and never to its CDN redirects.

## Supported routers

`RouterSpec.from_hf_config` maps `model_type` to one general pipeline:

```text
scores  = softmax | sigmoid | sqrt(softplus) of logits
choice  = scores + correction_bias                      (selection only)
groups  = top-`topk_group` of `n_group` by max or top-2 sum  (optional)
experts = top-k of choice within the selected groups
weights = scores[experts] -> optional renormalization -> x routed_scaling_factor
```

| Family | `model_type` |
| --- | --- |
| softmax top-k, renormalized | `mixtral`, `minimax`, `gpt_oss`, `granitemoe`, `granitemoeshared`, `granitemoehybrid`, `hunyuan_v1_moe`, `qwen3_vl_moe_text`, `qwen3_5_moe_text` |
| softmax top-k, `norm_topk_prob` | `qwen2_moe`, `qwen3_moe`, `qwen3_next`, `olmoe`, `flex_olmo` |
| softmax top-k, raw | `jamba` |
| softmax + group-limited (max) | `deepseek_v2` (`greedy` and `group_limited_greedy`) |
| sigmoid + correction bias + group-limited (top-2 sum) | `deepseek_v3`, `kimi_k2`, `deepseek_v32`, `glm4_moe`, `dots1`, `exaone_moe`, `kimi_linear`, and any config with `topk_method: noaux_tc` |
| other | `llama4` (sigmoid top-1), `ernie4_5_moe` (softmax + bias), `minimax_m2`, `lfm2_moe` (sigmoid + bias), `deepseek_v4` (sqrt-softplus + bias) |

Multimodal wrappers (`llama4`, `kimi_k25`, and others with a `text_config`) use their text config. Every model type in the table, except the `kimi_k2` alias that transformers itself maps to `deepseek_v3`, is tested side by side against the router module in `transformers` 5.x (`tests/test_hf_parity.py`): same configuration, same weights, same expert sets and weights.

**Not modelled:** DeepSeek-V4 hash-routed layers (the spec covers its learned layers and says so), PhiMoE's sparsemixer, DBRX's p-norm weights, capacity-based token dropping, and router jitter or other training-time noise. For anything else, construct `RouterSpec(...)` directly.

## How the certificate works

Let `S` be the reference expert set of a token, and suppose every logit may be off by at most `e`. Interval arithmetic bounds each selection score `[lo, hi]`. Group scores (max and top-2 sum) are monotone in each element, so they're bounded the same way. The set is **certified** if every selected group's lower bound beats every unselected group's upper bound, and every selected expert's lower bound beats every eligible unselected expert's upper bound. Certified means no perturbation within `e` can change or tie the selection.

- For softmax and sigmoid routers without a bias or grouping, selection is a monotone function of the logits, so the certificate reduces to *gap > 2e* at the k-th/(k+1)-th boundary. It is exact, and so is the radius (half the gap).
- With a correction bias, scores are bounded through the scoring function: elementwise for sigmoid/sqrt-softplus, and per-element (sound but conservative) for softmax.
- Group-limited selection is certified when every group holding a chosen expert is guaranteed to stay selected, and no expert of a group that could become selected can overtake a chosen one. This is sound, but it can be conservative where group and expert boundaries interact.
- The radius is otherwise found by bisection on the certificate, starting from a slope-based lower bound.

The property tests in `tests/test_certify.py` check soundness for every family under random and worst-case ("corner") perturbations, including logits 40 apart (where a naive softmax bound cancels), and check tightness where it's claimed.

**Error bounds.** `projection_error_bound(x, w, accumulate="fp32")` returns per-logit bounds `γ_n · Σ|x_i w_i|` (Higham, 2002). These hold for any summation order, blocking, split-K or FMA use, plus an optional output cast. `method="probabilistic"` gives the much tighter Higham & Mary (2019) bound, which holds with high probability under its rounding model. Pass the operands the kernel actually multiplies, for example BF16-rounded weights.

## API overview

| | |
| --- | --- |
| `RouterSpec`, `RouterSpec.from_hf_config`, `route`, `Routing` | reference routing (float64, ties go to the lower index) |
| `robustness_radius`, `certify`, `selection_bounds` | margin certificates |
| `compare_routing` → `RoutingReport`, `format_reports` | comparison report (`.summary()`, `.to_dict()`; `.passed` is True / False / None = unchecked without a tolerance) |
| `assert_routing_close` | test assertion |
| `project`, `projection_error_bound` | projection with precision emulation; rigorous/probabilistic error bounds |
| `quantize`, `quantize_scaled`, `FORMATS`, `is_representable` | bit-exact BF16/FP16/TF32/FP8-E4M3/FP8-E5M2 casts (RNE, subnormals, overflow modes) |
| `selection_keys`, `boundary_ties` | selection keys in engine precision (FP32/BF16); tokens with exact ties at the top-k or group boundary |
| `precision_sweep`, `PrecisionConfig` | what-if over precision configurations |
| `Checkpoint`, `SafetensorsFile`, `save_safetensors` | router discovery and dependency-free safetensors I/O, local or `hf://org/model` (HTTP range reads) |
| `routerlab.capture.RouterCapture`, `load_capture` | capture router inputs and the logits each router computed, from any PyTorch model (CPU, CUDA, MPS) |

## Precision and scope notes

- The reference computes in float64 from the values you give it. It never casts inputs silently; FP32 emulation refuses operands that aren't FP32-representable unless you ask for the cast.
- For float32 (and narrower) inputs, `quantize` matches PyTorch's casts bit-for-bit by default (E4M3 saturates at ±448, as `tensor.to(torch.float8_e4m3fn)` does). `overflow="nan"` gives the OCP/ml_dtypes behaviour. The tests check this against PyTorch, ml_dtypes and NumPy over every representable value and every rounding midpoint. Float64 inputs are rounded once, so they can differ from PyTorch's float64 → float32 → BF16 double rounding.
- `project(..., accumulate="fp32")` uses NumPy's float32 GEMM. Its errors are representative in size but not bitwise equal to any GPU kernel. Use `projection_error_bound` when you need guarantees.
- Exact ties are broken towards the lower index. Engines' `torch.topk` tie order is unspecified, so tied tokens have radius 0 and any difference at them counts as explained.
- `routerlab bench` and `routerlab.gpu_kernels` (Triton candidates for BF16 → FP32 router projection, CUDA-graph timing) are **experimental** and **have not been validated on GPU hardware by the maintainers**. Every row is checked against the float64 reference before it's timed. See [VALIDATION.md](VALIDATION.md).

## Development

```bash
git clone https://github.com/upalchowdhury/routerlab && cd routerlab
python -m pip install -e ".[dev,torch]" transformers
python -m pytest            # torch/transformers/GPU tests skip automatically when unavailable
ruff check routerlab tests
```

Issues and PRs are welcome, especially new router families, engine dump adapters, and GPU validation runs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Background

RouterLab started as a learning lab on router-projection precision and small-batch kernel performance. That material (project brief, learning plan, contribution path, experiment log) is in [docs/lab/](docs/lab/).

## License

[Apache-2.0](LICENSE)
