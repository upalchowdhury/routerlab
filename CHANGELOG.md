# Changelog

## 0.4.0 — 2026-10-26

### Added
- `routerlab.lowbit`: BitNet b1.58 absmean ternary quantization (per tensor/channel/group),
  symmetric INT2–INT8 weights, per-token INT8 activations, integer-exact ternary/INT projection
  (add/subtract accumulation, one rescale), ternary packing (2-bit, base-3 at 1.6 bits/weight), and
  storage footprints.
- `project`, `PrecisionConfig` and sweeps accept `weight_format="ternary" | "int4" | "int8"`,
  `weight_scaling="group<N>"` and `input_format="int8"`.
- `LOWBIT_CONFIGS` (`routerlab sweep --configs lowbit|all`) and `plan_precision`
  (`--budget PCT`): the cheapest per-router configuration within a routing-change budget, with
  memory totals.
- `routerlab inspect`: the router share of all parameters (from headers, works remotely);
  `--lowbit` adds per-router ternary/INT4/INT8 weight error and ternary sparsity.
- Case study `docs/case-studies/lowbit.md`: post-hoc ternary routers re-route 93–97% of tokens on
  OLMoE and LFM2. INT8 re-routes 5–6%.

## 0.3.0 — 2026-10-25

Real-world validation and the features it called for.

### Added
- `selection_keys` / `boundary_ties`: emulate the engine's selection keys in FP32/BF16 and find tokens
  with exact ties at the top-k or group boundary. These are the tie mechanism behind sglang#30989, where
  GLM-5.2-NVFP4 produced NaN expert weights. `compare_routing` reports them, and `sweep` adds an
  `fp32-key ties` column.
- Remote checkpoints: `Checkpoint("hf://org/name[@rev]")` and `routerlab inspect hf://...` read
  safetensors headers and router tensors by HTTP range requests, with no download. The auth token is
  sent only to the Hub host, transient failures are retried, and shard headers are fetched in parallel.
- `inspect --stats` bias diagnostics: how many expert biases a BF16 cast would merge, and the FP32
  spacing at the bias magnitude. On DeepSeek-V3 a BF16 cast merges 62% of expert pairs; on
  GLM-4.5-Air up to 98% in one layer.
- `RouterCapture` records the logits each router computed (in its own dtype and kernels) as well as
  its inputs. `routerlab compare run_a.safetensors run_b.safetensors` compares two runs directly.
- vLLM `routed_experts` support: `[tokens, layers, k]` arrays are split per layer and matched to the
  reference routers in layer order.
- Case studies on real models with real text (OLMoE-1B-7B, LFM2-8B-A1B; `docs/case-studies/`),
  produced by `examples/real_model_study.py`. `docs/WHY.md` covers incidents and use cases.
- Property-based tests (Hypothesis), with a weekly stress profile in CI.

### Fixed
- Capture on Apple MPS: on torch 2.14, `tensor.to("cpu", torch.float64)` of an MPS bfloat16 tensor
  returns wrong values. RouterLab now moves to the CPU first, then widens, everywhere.
- Renormalised weights were NaN when every selected score underflowed (e.g. sigmoid(-800)). They
  are now computed exactly in log space.
- `robustness_radius` could report `inf` for softmax routers with a correction bias when the
  slope-based lower bound was tiny. Bracketing now starts at the logit scale. Found by Hypothesis.

## 0.2.0 — 2026-10-25

RouterLab is now a general-purpose library and CLI for auditing MoE routers.

### Added
- `RouterSpec` / `route`: float64 reference routers for softmax, sigmoid, and sqrt-softplus scoring,
  correction bias, group-limited selection (max / top-2 sum), renormalization, and scaling.
  `RouterSpec.from_hf_config` covers 29 HuggingFace `model_type`s. 27 of them are tested side by side
  against the `transformers` router modules.
- Margin certificates: `certify`, `robustness_radius`, `selection_bounds`. They are sound for every
  family, and exact for ungrouped selection that is monotone in the logits.
- `compare_routing` / `RoutingReport`: logit error, expert-set changes, moved weight, margins,
  explained vs. unexplained flips, weight-tolerance checks, invalid selections (repeated experts,
  non-finite weights), and expert-load shift. `passed` is True / False / None: None means the
  candidate differs but no tolerance was given to judge it.
- `assert_routing_close`: tolerance-aware assertion for router kernel tests.
- `project` with operand/output precision emulation; `projection_error_bound`, with deterministic
  (Higham) and probabilistic (Higham & Mary) bounds.
- `quantize` / `quantize_scaled`: bit-exact BF16, FP16, TF32, FP8 E4M3FN and FP8 E5M2 casts, with
  PyTorch-compatible and OCP overflow modes.
- `precision_sweep`: what-if analysis across precision configurations.
- `Checkpoint` / `SafetensorsFile` / `save_safetensors`: dependency-free safetensors I/O (BF16/FP8
  decode, DeepSeek FP8 block-scale dequantization) and router discovery across sharded checkpoints.
- `routerlab.capture.RouterCapture`: capture router inputs from PyTorch models.
- CLI: `routerlab spec | inspect | compare | sweep | demo | bench | report`.
- CI on Linux/macOS, Python 3.10–3.13, NumPy 1.24–2.x, plus torch/transformers integration.

### Changed
- `routerlab.numerics` keeps the 0.1 API as thin wrappers over the new modules.
  `topk_indices` now accepts `k == E`.
- The GPU benchmark uses the new comparison code. It now reports bitwise batch invariance and
  `routerlab bench` passes arguments through. The output schema is version 2.
- The learning material moved to `docs/lab/`.
- The license is now Apache-2.0.

## 0.1.0 — 2026-10-24

The initial learning lab: BF16 rounding demo, float64 projection oracle, raw-logit top-k diagnostics,
Triton candidate kernels, and a CUDA benchmark harness.
