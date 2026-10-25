# From learning project to field contribution

## Current opportunity and overlap

Checked October 24, 2026: SGLang issue [#38695](https://github.com/sgl-project/sglang/issues/38695) is open and discusses router precision, shared gate dispatch, and deterministic behavior. Its discussion includes contributors pursuing model fixes and shared-layer work. [PR #38719](https://github.com/sgl-project/sglang/pull/38719) covers several model-level precision fixes. Do not duplicate these changes or treat the issue as unclaimed. Recheck the current branch, tests, linked PRs, and maintainer guidance before selecting a patch.

RouterLab proposes a complementary numerical and performance audit. Whether a particular benchmark or regression test is useful upstream must be established by inspecting existing coverage and getting feedback when you choose to contact the project. Some simple model-file fixes may not warrant additional tests; follow the actual contribution scope and repository guidance.

## Three contribution sizes

| Size | Concrete contribution | Evidence required |
| --- | --- | --- |
| First | Reproducible numerical/performance report | Pinned code, scripts, data, environment, a question the report resolves |
| Small PR | Missing regression case or benchmark coverage | Demonstrated gap, correct expected behavior, repository checks |
| Optimization PR | One validated kernel/dispatch improvement | Strong stock baseline, reproducible gain, no accuracy regression, graph compatibility |

Accepted PRs are not guaranteed. A report is useful if another engineer can reproduce it and use it to make a decision. Reimplementing an existing kernel without new evidence is primarily a learning exercise.

## Adapter contract

The benchmark accepts an optional `module:function` callable:

```python
def engine_router(x, weight, out):
    # Call the actual implementation from a pinned SGLang revision.
    # Match BF16 X/W -> FP32 output for this initial lab.
    # Fill and return out, with no host synchronization or input-dependent setup.
    return out
```

This is an interface illustration, not an implemented adapter. It must be replaced by a real invocation. Verify that all output entries are written and that a profiler shows the intended engine path. Preserve and time any conversions or copies the real invocation needs. Do not compare an artificially expensive wrapper against a preprocessed candidate.

```bash
python -m routerlab.bench --adapter my_adapter:engine_router \
  --engine-commit ACTUAL_CHECKED_OUT_SHA --mode graph --output results/engine.json
```

The starter deliberately does not hardcode SGLang's changing internal paths or patch a live service. Pin a revision and read its AGENTS/contributor guidance. Use its existing test and benchmark structure when preparing an upstream patch.

## Stage the audit

1. Match one exact model gate, operation shape, device, and dtype contract.
2. Inspect existing tests and avoid collecting redundant evidence.
3. Reproduce the stock behavior first.
4. Compare raw projection logits and the full routing output separately.
5. Test close scores, correction bias, output dtype, masks/strides, token-count thresholds, and same-token batch composition where relevant.
6. Measure actual stock kernel latency and end-to-end impact separately.
7. Keep the contribution small enough to review and explain unsupported cases.

## Proposed first artifact

Title template: **Router projection precision and small-batch latency audit on [GPU]**.

Include:

- What decision or regression the report addresses.
- Exact engine and model revisions.
- Numerical input/output contracts.
- Reproduction commands and raw result files.
- Error, routing behavior, and latency across shapes.
- Known limitations, including synthetic-data and warm-cache scope.
- A narrowly scoped suggested next step justified by the measurements.

Do not claim preserved model accuracy from a random-tensor test, general speedups from one shape, or algorithmic novelty without a literature/source comparison.

## Connection to the original competition

The primary path remains useful for inference-kernel competitions: extracting a computation, matching precision, testing correctness, profiling, and tuning. The initial lab's BF16-weight router differs from the Open Kernel Challenge's FP32-weight router. A later competition adapter must preserve the challenge's exact contract and permitted arithmetic; it cannot submit these kernels unchanged.

## Sources

- SGLang router issue: https://github.com/sgl-project/sglang/issues/38695
- Existing precision-fix PR: https://github.com/sgl-project/sglang/pull/38719
- Current routing implementation to inspect: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py
- Challenge rules: https://openkernelchallenge.dev/rules
