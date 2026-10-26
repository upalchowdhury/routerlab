# Validation status

This records what has been checked, how, and what has not. Updated October 26, 2026 (v0.4.0).

## Verified (CPU)

Environment: macOS (Apple silicon), Python 3.12.12, NumPy 2.5.4, PyTorch 2.14.1 (CPU),
transformers 5.20.2, ml_dtypes 0.5.4, safetensors 0.6.0. CI repeats the checks on Linux and macOS
across Python 3.10–3.13 and NumPy 1.24–2.x.

- **Test suite:** 203 passed, 4 skipped (the GPU tests), plus 5 tests in `examples/`.
- **Low-bit emulation** (`tests/test_lowbit.py`): ternary codes and scales match an independent
  implementation of the BitNet b1.58 absmean equations. INT8 activations follow the BitNet
  reference code (scale 127/absmax). The integer path equals the float64 product of the
  dequantized operands to 1e-12. Ternary packing round-trips in both schemes. The low-bit sweeps
  in `docs/case-studies/lowbit.md` are post-hoc quantization on real captured activations. Property-based
  tests (Hypothesis) were also run at 3,000 examples per property with three seeds. One soundness bug
  (an infinite radius) was found and fixed.
- **Real models, real text:** OLMoE-1B-7B and LFM2-8B-A1B on 16,384 WikiText-2 tokens. The workflow
  covers FP32 and BF16 runs on CPU and BF16 runs on Apple MPS, capture of router inputs and logits,
  sweeps, end-to-end and batch-invariance comparisons, and tie detection. Results are in
  `docs/case-studies/`. These runs found and fixed the MPS float64 conversion bug.
- **Real checkpoints, remote:** `inspect hf://...` on DeepSeek-V3 (163 shards, 59 routers, ~10 s),
  Kimi-K2-Instruct, GLM-4.5-Air, Qwen3-30B-A3B and gpt-oss-20b, all without downloading the weights.
- **Independent review:** a separate review pass, with no access to the author's reasoning,
  reported 21 defects, each reproduced. All are fixed and covered by regression tests, and each
  test was confirmed to fail on the pre-fix code. They included NaN weights passing, a vacuous
  verdict without a tolerance, an ambiguous 1-D tolerance shape, softmax bound cancellation at
  logit gaps of about 40, a conservative group certificate, checkpoint naming for DeepSeek-V4 and
  Granite, Ernie bias capture, and config defaults.
  A second independent review of the 0.3.0 additions stopped early on an API rate limit. Instead, the
  author's own adversarial pass found and fixed four more issues: lexicographic layer pairing for
  vLLM arrays, a confusing error for output-only captures, range handling for servers that ignore
  Range, and CLI tracebacks on HTTP errors. Each has a regression test where applicable.
- **Reference routers vs. `transformers`** (`tests/test_hf_parity.py`): 30 configurations across
  27 `model_type`s, plus a check that minimal `config.json`s resolve to the same defaults as the
  transformers config classes. Each one instantiates the model's own router module with random weights and
  biases, then requires identical expert sets (except at genuine float32 near-ties, radius < 1e-5),
  weights within rtol 2e-5, and an identical spec from the live config object and its serialised
  dict. A mutation check (wrong group rule, dropped bias, wrong renormalization or scaling) makes
  the comparison fail.
- **Capture:** works on tiny DeepSeek-V3, Mixtral, MiniMax-M2 and GPT-OSS models. Captured
  activations reproduce each module's logits. BF16 capture is lossless. A live DeepSeek-V3 router's
  top-k passes `assert_routing_close` against the reference.
- **Format emulation** (`tests/test_precision.py`): bit-exact against ml_dtypes (BF16, E4M3FN
  non-saturating, E5M2), NumPy float16 (from float32 and float64), and PyTorch casts (BF16, FP16,
  E4M3FN saturating, E5M2). Each probe covers every finite representable value, every rounding
  midpoint ±1 float32 ulp, 200k random float32 bit patterns, and the special values.
- **Certificates** (`tests/test_certify.py`): radius soundness under random, uniform and corner
  perturbations for 7 router families and at large logit scales; exactness of the closed form for
  plain top-k; agreement between bisection and `certify` within 0.1%.
- **Error bounds** (`tests/test_projection.py`): the deterministic FP32 bound holds for BLAS,
  blocked (split-K-style) and sequential summation orders; the BF16/FP16 output-cast bounds hold.
- **Packaging:** the built wheel installs and passes the suite on Python 3.10 with NumPy 1.24.4,
  and on Python 3.13 with NumPy 2.5.4, with optional dependencies absent.
- **Performance** (one machine): 100k tokens × 256 experts with a DeepSeek-V3-style spec takes
  about 0.8 s to route, 2.2 s for radii (8 threads; bisection) and 5.2 s for a full
  `compare_routing`. Plain softmax radii are closed form (about 0.9 s).

## Not verified

- **GPU kernels and benchmark:** `routerlab/gpu_kernels.py` (Triton) and `routerlab bench` haven't
  been compiled or run on an NVIDIA GPU by the maintainers. `tests/test_gpu.py` is written but
  skipped. The harness checks every candidate against the float64 reference before timing, but
  there are no performance or correctness claims for these kernels.
- **Engines:** no vLLM, SGLang or TensorRT-LLM router has been compared yet. Their outputs can be
  compared through `routerlab compare` or `compare_routing`, but no engine-specific adapters ship.
- **Real-model statistics:** the sweep and compare examples in the README use synthetic or
  randomly initialised models. Routing-margin distributions of production checkpoints on real
  data haven't been measured here.
- **Probabilistic error bound:** this is a heuristic under the Higham–Mary rounding model, not a
  guarantee. Use `method="deterministic"` when you need one.
