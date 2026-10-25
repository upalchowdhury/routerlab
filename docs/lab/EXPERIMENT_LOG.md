# Experiment log template

Copy this entry for each experiment. No GPU measurements are included with the starter.

## Experiment [number]: [one hypothesis]

- Date:
- Commit and uncommitted changes:
- GPU / driver / CUDA / PyTorch / Triton:
- Engine revision, if applicable:
- Shape, strides, and precision contract:
- Cache regime: warm repeated weights / rotating weights / full model:
- Baseline and why it is the strongest relevant comparison:
- Hypothesis and predicted tradeoff:
- One implementation change:
- Correctness: oracle error, output dtype, finite values:
- Routing: actual selection/weights, top-k margin, tie policy:
- Same-token behavior across batch composition:
- Eager timing and variability:
- CUDA graph timing and variability:
- Profiler finding:
- End-to-end result, or explicitly unmeasured:
- Conclusion: keep / revert / more evidence needed:
- Raw data and reproduction command:

An unsuccessful optimization is informative if you can explain the result and others can reproduce it.
