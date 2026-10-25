# Why RouterLab

**An MoE router makes a hard, discrete decision on noisy numbers.** It computes logits, takes
the top k, and sends each token to those experts. Any numerical difference moves tokens near the
top-k boundary to different experts, and the effect compounds over dozens of layers. Sources of
difference include router GEMM precision, a fused kernel, a quantized checkpoint, an engine port,
batch composition, and the difference between the trainer and the rollout engine.

Usually these flips are harmless rounding. Sometimes they are a bug that only surfaces later as a
benchmark drop or corrupted output. Today most teams can't tell which, because the standard checks
don't measure the right thing:

- `assert torch.equal(topk_ids, ref_ids)` is flaky on near-ties, so it gets loosened, and then it
  misses real bugs.
- "Logits match to 1e-3" says nothing about how many tokens that error re-routes.
- End-to-end evals catch routing bugs late, expensively, and without pointing at the cause.

RouterLab answers the question directly. Per token and per layer, it reports whether the experts
changed and how much logit error that token could absorb before it should have. A change at a
small margin is explained by precision. A change at a large margin is a bug.

## This happens in real systems

Each row links the primary source. The last column is what RouterLab measures for that failure mode.

| What happened | Source | What RouterLab shows |
| --- | --- | --- |
| A fused routing kernel produced **NaN expert weights**. FP32 selection keys (sigmoid + a large correction bias) became bitwise equal at the top-8 boundary, and long-context requests came back as `!!!!!!`. Model: GLM-5.2-NVFP4. | [sglang#30989](https://github.com/sgl-project/sglang/issues/30989) | `boundary_ties(logits, spec, "fp32")` counts tokens tied at the boundary in engine precision. `inspect --stats` reports the FP32 spacing at the bias magnitude. Non-finite candidate weights fail as INVALID. |
| New routing methods in an FP8 MoE backend dropped DeepSeek-V2-Lite-FP8 GSM8K to **1.6%**. A nightly end-to-end eval caught it, and the fix disabled those methods. | [vllm#33532](https://github.com/vllm-project/vllm/issues/33532), [vllm#33620](https://github.com/vllm-project/vllm/pull/33620) | A routing-level `assert_routing_close` against the model's reference router fails at unit-test time on any such divergence, as UNEXPLAINED flips (large-margin tokens that changed experts) or weights outside tolerance, and names the tokens. |
| Several models cast the FP32 **correction bias to BF16** and/or downcast router logits, and the precision fixes were debated with contested benchmark effects. | [sglang#38695](https://github.com/sgl-project/sglang/issues/38695), [sglang#29783](https://github.com/sgl-project/sglang/pull/29783) | `inspect hf://deepseek-ai/DeepSeek-V3 --stats`: a BF16 cast collapses each layer's bias to a median of **3** distinct values (from 50), merging 62% of expert pairs. For GLM-4.5-Air the figure is up to 98%. `sweep` measures the routing change on real activations. |
| A reviewer found single-layer top-k "bit-identical, 400/400" for a router precision change, yet the end-to-end benchmarks moved. | [sglang#29892](https://github.com/sgl-project/sglang/pull/29892) | An exact-match sample of 400 tokens on one layer can't detect rare flips that compound over 90+ layers and thousands of generated tokens. RouterLab reports the at-risk fraction for a given error over all tokens and layers (whether routing caused the benchmark change was not established in that thread). |
| The same token can **route differently depending on batch size**, because kernels switch at small token counts. | [sglang#38695](https://github.com/sgl-project/sglang/issues/38695); vLLM's [batch-invariant mode](https://docs.vllm.ai/en/latest/features/batch_invariance/) | Capture at batch 1 and batch N, then `routerlab compare a.safetensors b.safetensors`. |
| RL on MoE models is destabilised by **routing mismatch between rollout and training**, and fixed by replaying the rollout engine's routing. For Qwen3-30B-A3B, about 10% of activated experts change for the same rollout after an update. | [R3, arXiv:2510.11370](https://arxiv.org/abs/2510.11370); [GSPO, arXiv:2507.18071](https://arxiv.org/abs/2507.18071); vLLM [`routed_experts`](https://docs.vllm.ai/en/latest/training/routed_experts/) | `routerlab compare trainer.npz routed_experts.npy` takes vLLM's `[tokens, layers, k]` output and gives per-layer mismatch rates, split into precision-explained and unexplained flips. |

DeepSeek-V3's technical report keeps "MoE gating modules" at their original precision rather than
FP8 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437), §3.3). The model authors treat routing
as precision-sensitive; RouterLab puts a number on how much.

## What we measured on real models

Measured on [OLMoE-1B-7B](case-studies/olmoe-1b-7b/summary.md) and
[LFM2-8B-A1B](case-studies/lfm2-8b-a1b/summary.md) with 16k tokens of WikiText-2. Rates are per
token-layer routing. Setup and caveats are in [case-studies/](case-studies/README.md).

| | OLMoE-1B-7B | LFM2-8B-A1B |
| --- | ---: | ---: |
| BF16 router output, same inputs: tokens that change experts | 0.71% | 2.43% |
| BF16 router output: exact top-k ties (routing by kernel tie-break) | 1.39% | 0.37% |
| FP8-E4M3 router weights: tokens that change experts | 9.6% | 11.0% |
| Whole model BF16 vs FP32: first MoE layer → worst layer | 3.9% → 6.8% | 2.8% → 27.8% |
| Batch of 4 vs batch of 1 (BF16, Apple GPU): worst layer | 5.2% | 15.6% |
| Tokens within BF16 logit rounding of the boundary | 5–9% | 12–28% |

## What it does

| You want to know | Use |
| --- | --- |
| How does this model actually route? | `RouterSpec.from_hf_config(config)` + `route()`: float64 reference for 29 architectures, checked against `transformers` |
| Is my kernel/engine correct, or just imprecise? | `assert_routing_close` (tests), `compare_routing` / `routerlab compare` (reports, CI exit codes) |
| How much error can each token absorb? | `robustness_radius`, `certify` (rigorous margins, including group-limited routing and correction bias) |
| What tolerance should my test use? | `projection_error_bound` (rigorous FP32 bound for any summation order; optional output cast) |
| Can this router run in BF16/FP8? | `routerlab sweep` on activations captured with `RouterCapture` |
| Will engine precision create exact ties? | `boundary_ties(logits, spec, "fp32" or "bf16")` |
| What precision is this checkpoint's router stored in, and is its bias BF16-safe? | `routerlab inspect hf://org/model --stats` (no download) |
| Do two runs route the same? | `RouterCapture` in both runs, then `routerlab compare run_a.safetensors run_b.safetensors` |
| Does the rollout engine route like the trainer? | `routerlab compare trainer.npz vllm_routed_experts.npy` |

## Who uses it, and where

**Kernel and engine developers** writing fused router, top-k or grouped-top-k kernels (Triton,
CUDA, CUTLASS, FlashInfer): replace exact `topk_ids` comparisons with
`assert_routing_close(reference, (ids, weights), spec, tolerance=projection_error_bound(x, w))`.
Near-ties pass, semantic bugs fail with token-level detail, and the tolerance is principled rather
than guessed. See [`examples/test_router_kernel.py`](../examples/test_router_kernel.py).

**Model porters** adding an architecture to vLLM, SGLang, TensorRT-LLM, llama.cpp or MLX: build
the reference from the model's `config.json` and compare the port's router dumps layer by layer.
Wrong group rules, bias placement (selection vs weights), renormalization, scaling factors and
tie-breaking all show up as unexplained flips or weight violations.

**Quantization and precision decisions:** run `routerlab sweep` on captured activations before
choosing BF16/FP16/FP8 for the router, and `inspect --stats` to see whether a checkpoint's bias
survives a cast.

**RL and post-training teams:** measure the routing mismatch between rollout and training per
layer, and tell precision-explained flips from implementation differences, before reaching for
routing replay.

**Serving reliability and determinism:** check batch invariance and exact-tie exposure when
changing kernels, batch-size thresholds or hardware.

## What it does not do

- It measures routing. It does not measure task quality; pair it with your evals.
- It does not hook engines for you, except vLLM's `routed_experts` format and PyTorch models via
  `RouterCapture`. For other engines, dump router logits or top-k ids to `.npz`/`.safetensors`.
- The included Triton kernels and CUDA benchmark have not been validated on GPU hardware (see
  [VALIDATION.md](../VALIDATION.md)).
