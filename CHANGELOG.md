# Changelog

## 0.2.0 — 2026-10-25

RouterLab is now a general-purpose library and CLI for auditing MoE routers.

### Added
- `RouterSpec` / `route`: float64 reference routers for softmax, sigmoid, and sqrt-softplus scoring,
  correction bias, group-limited selection (max / top-2 sum), renormalization, and scaling.
  `RouterSpec.from_hf_config` covers 29 HuggingFace `model_type`s. 27 of them are tested side by side
  against the `transformers` router modules.
- Margin certificates: `certify`, `robustness_radius`, `selection_bounds`. They are sound for every
  family, and exact where selection is monotone in the logits.
- `compare_routing` / `RoutingReport`: logit error, expert-set changes, moved weight, margins,
  explained vs. unexplained flips, weight-tolerance checks, invalid selections, expert-load shift.
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
