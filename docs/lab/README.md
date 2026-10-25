# RouterLab learning lab (background material)

RouterLab began as an eight-week learning project: measure and improve one real inference operation,
the MoE router projection (`logits = hidden_states @ router_weight.T`, BF16 inputs → FP32 logits),
with numerical evidence before any performance claim. The library grew out of that work.

| File | Contents |
| --- | --- |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | The research question, precision contract, measurement contract |
| [LEARNING_PLAN.md](LEARNING_PLAN.md) | A week-by-week plan, from CPU numerics to GPU kernels and engine integration |
| [CONTRIBUTING_PATH.md](CONTRIBUTING_PATH.md) | Turning lab results into an upstream contribution (dated October 24, 2026) |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Template for recording experiments |

The commands these documents mention still work: `python -m routerlab.demo`,
`python -m routerlab.bench`, `python -m routerlab.report`, and the 0.1 functions in
`routerlab.numerics`. The demo's output changed in 0.2: it now demonstrates margin-aware comparison.
