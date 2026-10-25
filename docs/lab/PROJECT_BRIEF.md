# RouterLab project brief

Prepared October 24, 2026. Suggested duration: eight weeks at 12–15 focused hours per week. This is a learning and contribution plan, not a promise of performance or upstream acceptance.

## The research question

**For small token counts, which router projection implementation gives the best latency while respecting the model's numerical contract, and how does its behavior change with batch composition?**

The small output makes the problem manageable, but selection makes precision worth investigating. A tiny score difference matters most when experts sit close to the selection boundary.

## The computation, in plain language

For each token:

1. Take its H-dimensional hidden vector.
2. Compute a dot product with each of E expert weight vectors.
3. Produce E logits.
4. Let the model's actual routing function transform scores and select experts.

The first kernel implements steps 1–3. Step 4 is initially a diagnostic and becomes model-specific during integration.

| Symbol | Meaning | Initial values |
| --- | --- | --- |
| M | Token rows in the current operation | 1, 2, 4, 8, 16, 32, 128 |
| H | Hidden-vector width | 1024, 4096, 8192 |
| E | Number of scored experts | 64, 128, 256 |
| k | Experts selected in diagnostic top-k | 8 |

These are proposed experiment shapes. Once you pin a target model, derive its exact shapes, layouts, dtypes, and routing semantics from source and traces. These are not advertised as the exact shapes of every model.

## Precision contract

Initial experiment:

```text
X: BF16 [M,H]
W: BF16 [E,H]
accumulation: FP32
output: FP32 [M,E]
```

The float64 CPU oracle computes dot products of the same BF16-representable values. It measures finite-precision accumulation error without changing the inputs. Different FP32 reduction orders can still differ; FP32 accumulation does not promise a bitwise match with the float64 oracle or another backend.

An FP32 output cast cannot recover information already lost by rounding FP32 weights or inputs to BF16. Therefore:

- Do not silently convert a model's FP32 router weights to BF16.
- Preserve FP32 correction bias when the model requires it.
- Add a separate FP32-weight contract before experimenting on such a model.
- Do not claim that a BF16-weight experiment reproduces every DeepSeek or challenge router.

The Open Kernel Challenge's router has its own contract. This repository is not a drop-in challenge submission. Any challenge adapter must separately match its precision rules, signatures, scored shapes, and harness.

## What to build

### 1. Numerical audit

Extend the existing suite with cancellation, large dynamic range, near-tie boundaries, bias handling, and repeated-token cases. Quantify output error and actual expert selection. For full routing, include score transforms, grouping, renormalization, and selected-expert weights exactly as implemented by the target model.

### 2. Candidate kernels

The starter supplies two readable candidates:

- `rowdot`: one token and a small group of experts per program; vector loads followed by FP32 reductions.
- `tiled`: a conventional BF16 tiled GEMM with FP32 accumulation/output.

Predict where each should work. Then measure. The initial configurations are educational choices and have not been optimized or GPU-validated here.

Try one change at a time: expert tile width, reduction tile width, warps, matrix tiles, and launch partitioning. Record register use and memory behavior when a profiler is available.

### 3. Backend comparison

Compare against the library baseline and a real pinned engine path. Retain the strongest applicable stock implementation as the performance baseline. Repacking, conversion, dispatch, and extra launches must appear in the correct timing scope. Never obtain a win by moving token-dependent work into untimed setup.

### 4. Conservative dispatch

Only after repeated measurements, choose different implementations by shape and hardware. Keep the stock fallback for unsupported or losing cases. Separate tuning cases from held-out shapes/seeds and rerun in a fresh process. Avoid universal thresholds inferred from one GPU.

### 5. Engine integration

Pin SGLang to a concrete commit. Integrate one supported path through its current code organization. Preserve parameter names, checkpoint values, loading behavior, graph compatibility, precision settings, and deterministic-mode semantics. Verify that the intended kernel actually runs with a profiler.

## Numerical interpretation

The diagnostic examines top-k **sets**, not only their ordering. For each row, let e be the largest absolute score error. If the reference gap between the kth and (k+1)th score is greater than 2e, set membership cannot change from perturbations of that size.

This certificate is applied to the actual compared scores. A nonlinear score transform or correction bias changes those scores; recalculate diagnostics after that operation. Exact ties require the target engine's tie semantics. An ordering change among already-selected experts can also matter downstream if weights or grouping use the order, so the engine stage must validate its complete output contract.

The starter's `atol=1e-3`, `rtol=1e-3` are exploratory smoke-test thresholds on normalized synthetic data. They are not production acceptance criteria. Establish target-specific tolerances and evaluation gates before integration.

## Measurement contract

The included runner:

- Warms kernels before timing so compilation is excluded.
- Reuses an FP32 output buffer for every timed backend.
- Measures eager launches and CUDA graph replay separately.
- Uses CUDA events around repeated calls; reports window averages and IQR.
- Labels repeated-input/weight cache behavior explicitly.
- Randomizes backend order per shape and records environment information.
- Checks output before and after timing and records failures.
- Compares one token alone and as part of a batch as a diagnostic.

The runner does not prove absence of races, represent every memory-cache regime, or estimate request-level p99 latency. Extend GPU testing with compute-sanitizer and realistic model traces where applicable. Measure a rotating pool of weight matrices larger than cache as a separate workload; include that regime in the report rather than relabeling warm-cache results.

For a performance report, repeat independent sessions, alternate baseline and candidate, disclose clocks/load, and attach the raw data. A proposed dispatch win should exceed measurement noise and justify its complexity. A 5% repeatable operation gain is a useful project target, not an expected outcome.

## End-to-end validation

Use a compatible small MoE model if one fits your available GPU; otherwise stop at operation-level contributions and arrange a collaborator's model validation. Do not buy large hardware or pretend a dense model validates routing behavior.

Hold model revision, tokenizer, precision, backend settings, prompt set, generated token counts, concurrency, and sampling settings fixed. Start with fixed-token/logit comparisons; then test the complete routing outputs and relevant task quality. Greedy generation may diverge after small numerical changes, so investigate causes and document the acceptance policy rather than asserting that all differences are harmless.

Report operation latency, time per decode step, time to first token, inter-token latency, throughput, and output-quality checks separately. Overall improvement is limited by the fraction of time spent in the optimized operation. A valid outcome can be that an operation improves but the service-level effect is too small to measure.

## Contribution outcomes

1. A reusable numerical regression or missing coverage case.
2. A reproducible backend performance and precision report that helps an active refactor.
3. A narrowly scoped kernel or dispatch patch when measured evidence supports it.

Novel algorithmic research is a later possibility. This starter does not establish novelty. A useful engineering contribution can be small and still improve reliability or remove uncertainty for maintainers.

## Definition of done

- Reproducible setup with a pinned tested environment.
- Correctness evidence for supported shapes, dtypes, and layouts.
- Results from real GPU runs, with failed and losing cases retained.
- Clear distinction between a learning kernel, an optimized kernel, and an integrated contribution.
- One reviewable upstream artifact: test, report, benchmark patch, or optimization patch.
- Honest documentation of limitations and independent reproduction instructions.
