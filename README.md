# RouterLab

[![CI](https://github.com/upalchowdhury/routerlab/actions/workflows/ci.yml/badge.svg)](https://github.com/upalchowdhury/routerlab/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Numerical auditing for mixture-of-experts routers.** RouterLab tells you whether two implementations of an MoE router (a fused kernel and a reference, an inference engine and a trainer, BF16 and FP32) select the same experts. When they don't, it tells you whether the difference is ordinary rounding at a near-tie or a real bug.

From `routerlab demo`: a DeepSeek-V3-style router (synthetic data), checked against a candidate kernel with a porting bug. The bug is DeepSeek-V2's group rule and the correction bias leaking into the weights:

```text
Router deepseek_v3-style (synthetic): 2048 tokens, top-8 of 64 experts
  selection   891 tokens changed expert set (43.51%), mean overlap 0.918274
  weights     moved mass mean 2.000e-01  max 6.314e-01   max |err| (same experts) 3.458e-01
  WEIGHTS     1157 tokens with the same experts have weights outside tolerance (token 839, expert 39: 0.574148 vs 0.229486)
  margins     radius min 3.828e-05  p1 6.826e-04  p5 5.292e-03  median 7.218e-02 (logit units)
  tolerance   per-element logit tolerance (max 2.216e-02): 74 tokens at risk
  verdict     52 explained by tolerance, 839 UNEXPLAINED (margin larger than tolerance)
  largest margin among changed tokens: 4.976e-01
  load        largest per-expert change 284 tokens (expert 11)
    token 2: -[19] +[45]  margin 1.910e-02  UNEXPLAINED
    token 7: -[2] +[27]  margin 3.192e-01  UNEXPLAINED
  FAIL
```

The same comparison against correctly computed BF16 logits shows 12 changed tokens, all of them explained, and passes.

It is a NumPy library plus a CLI. PyTorch is optional, and so are GPUs; everything except the CUDA benchmark runs on a laptop.

## Why you might need it

MoE routers choose experts with a hard top-k. A logit error of `1e-3` changes nothing for most tokens, but it flips the selection for tokens whose k-th and (k+1)-th scores are closer than that. That's routine, and it causes several recurring problems:

- **Flaky kernel tests.** `assert torch.equal(topk_ids, ref_ids)` fails on random near-ties. Loosen it and real bugs slip through.
- **"Is this a bug or precision?"** Your fused router disagrees with the reference on 0.4% of tokens. Is that BF16 rounding, or a wrong group rule or a dropped correction bias?
- **Precision decisions.** Can this router's weights or logits be BF16 or FP8? How many tokens would change experts?
- **Train/inference mismatch.** RL on MoE models is sensitive to the rollout engine and the trainer routing tokens differently. You need to measure how often that happens, per layer, and why.

RouterLab answers these with three pieces:

1. **Reference routers** for every major MoE family, built straight from a HuggingFace `config.json`. They are checked side by side against the router modules in `transformers` for 27 architectures.
2. **Margin certificates.** For each token, RouterLab computes the largest logit error that cannot change its expert set (its *robustness radius*), including group-limited selection and correction bias. A token that flips with a radius below your error bound is explained. One that flips with a radius above it is a bug.
3. **Tolerance-aware tools** built on those certificates: an assertion for test suites, a comparison report, rigorous floating-point error bounds for the projection, bit-exact BF16/FP16/FP8 emulation, a precision sweep, checkpoint inspection, and activation capture from PyTorch models.

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
  839 tokens changed experts although their routing margin exceeds the tolerance (not a rounding effect):
      token 2: reference [19, 27, 33, 38, 48, 50, 52, 53] vs candidate [27, 33, 38, 45, 48, 50, 52, 53]; margin 1.910e-02 > tolerance 9.919e-03
      token 7: reference [2, 19, 22, 33, 38, 48, 51, 53] vs candidate [19, 22, 27, 33, 38, 48, 51, 53]; margin 3.192e-01 > tolerance 1.113e-02
  1157 tokens kept their experts but have weights outside tolerance; worst at token 839, expert 39: reference 0.574148, candidate 0.229486 (allowed |delta| 9.993e-03)
  (52 other tokens changed experts within tolerance: expected near-ties)
```

It also checks logits against the tolerance (when the candidate provides logits), rejects duplicate experts, and checks weights of unchanged tokens against a bound derived from the tolerance. Use `weight_rtol=2**-8` if the engine returns BF16 weights.

### 3. Compare dumps from two systems

Save router logits (or `topk_ids`/`topk_weights`) from each system, one key per layer, as `.npz`, `.safetensors`, `.npy` or `.pt`:

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

Key conventions: use `NAME` or `NAME/logits` for float logits, `NAME/topk_ids` with an optional `NAME/topk_weights` for selections, and `NAME/correction_bias` to attach a bias. The exit status is 1 when any layer has unexplained flips, invalid selections, or out-of-tolerance logits or weights. Change that with `--fail-on any|never`.

Per token, the report gives the logit error, set changes, moved routing weight, the radius distribution, how many tokens are at risk under the tolerance, explained vs. unexplained flips with examples, and per-expert load shift.

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
| precision | max logit err | tokens changed | changed % | moved weight (mean) | max load shift |
|---|---:|---:|---:|---:|---:|
| fp32 math on stored values | 9.85e-07 | 0 | 0.000 | 4.86e-08 | 0 |
| fp32 math, bf16 logits | 7.80e-03 | 42 | 2.051 | 2.62e-03 | 3 |
| fp16 operands, fp16 logits | 9.75e-04 | 5 | 0.244 | 3.44e-04 | 1 |
| fp8-e4m3 weights (per-channel scale) | 9.12e-02 | 355 | 17.334 | 2.23e-02 | 13 |
```

*(Above: a randomly initialised tiny Qwen3-MoE, illustrative only.)* Custom configurations look like `--configs "name=bf16 logits,input_format=fp32,weight_format=fp32,output_format=bf16;..."`.

Without a capture, `routerlab sweep --checkpoint DIR --layer N --hidden hidden.npy` reads the router straight from a checkpoint. `--synthetic N` uses Gaussian inputs, scaled by the layer's `post_attention_layernorm` weight when there is one; these results are labelled as indicative only.

### 5. Inspect a checkpoint without loading the model

```bash
routerlab inspect /models/DeepSeek-V3 --stats
```

This lists every router weight, its correction bias and linear bias, shapes and dtypes, magnitude stats, and whether FP32-stored weights are exactly BF16-representable. It reads only the bytes it needs from sharded safetensors, decodes BF16/FP8 exactly, and doesn't need torch.

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

- For softmax and sigmoid routers without a bias, selection is a monotone function of the logits, so the certificate reduces to *gap > 2e* at the expert boundary and at the group boundary. It is exact, and so is the radius (half the gap).
- With a correction bias, scores are bounded through the scoring function. This is exact for elementwise sigmoid/sqrt-softplus and sound but conservative for softmax. The radius is then found by bisection on the certificate, starting from a slope-based lower bound.

The property tests in `tests/test_certify.py` check soundness for every family under random and worst-case ("corner") perturbations, and check tightness where it's claimed.

**Error bounds.** `projection_error_bound(x, w, accumulate="fp32")` returns per-logit bounds `γ_n · Σ|x_i w_i|` (Higham, 2002). These hold for any summation order, blocking, split-K or FMA use, plus an optional output cast. `method="probabilistic"` gives the much tighter Higham & Mary (2019) bound, which holds with high probability under its rounding model. Pass the operands the kernel actually multiplies, for example BF16-rounded weights.

## API overview

| | |
| --- | --- |
| `RouterSpec`, `RouterSpec.from_hf_config`, `route`, `Routing` | reference routing (float64, ties go to the lower index) |
| `robustness_radius`, `certify`, `selection_bounds` | margin certificates |
| `compare_routing` → `RoutingReport`, `format_reports` | comparison report (`.summary()`, `.to_dict()`, `.passed`) |
| `assert_routing_close` | test assertion |
| `project`, `projection_error_bound` | projection with precision emulation; rigorous/probabilistic error bounds |
| `quantize`, `quantize_scaled`, `FORMATS`, `is_representable` | bit-exact BF16/FP16/TF32/FP8-E4M3/FP8-E5M2 casts (RNE, subnormals, overflow modes) |
| `precision_sweep`, `PrecisionConfig` | what-if over precision configurations |
| `Checkpoint`, `SafetensorsFile`, `save_safetensors` | router discovery and dependency-free safetensors I/O |
| `routerlab.capture.RouterCapture`, `load_capture` | activation capture from PyTorch models |

## Precision and scope notes

- The reference computes in float64 from the values you give it. It never casts inputs silently; FP32 emulation refuses operands that aren't FP32-representable unless you ask for the cast.
- `quantize` matches PyTorch's casts bit-for-bit by default (E4M3 saturates at ±448, as `tensor.to(torch.float8_e4m3fn)` does). `overflow="nan"` gives the OCP/ml_dtypes behaviour. The tests check this against PyTorch, ml_dtypes and NumPy over every representable value and every rounding midpoint.
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
