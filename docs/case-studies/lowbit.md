# Ternary and low-bit routers: what post-hoc quantization does to MoE routing

For teams targeting ternary or integer hardware (BitNet b1.58-style {-1, 0, +1} weights,
INT4/INT8 accelerators), the question is: **can the MoE router be quantized with the rest of the
model, or must it stay in higher precision?**

Setup: OLMoE-1B-7B and LFM2-8B-A1B run in BF16 on 4,096 WikiText-2 tokens. Router inputs were
captured with `RouterCapture` and swept with `routerlab sweep --configs lowbit`. Quantization is
post-training (no quantization-aware training) and computed on the integer-exact path
(`routerlab.lowbit.lowbit_project`). Raw results: `olmoe-1b-7b/lowbit.json` and
`lfm2-8b-a1b/lowbit.json`. Rates are over all token-layer routings (65,536 for OLMoE, 90,112 for
LFM2).

| Router weights (activations BF16 unless noted) | OLMoE: tokens re-routed | experts kept | LFM2: tokens re-routed | experts kept |
| --- | ---: | ---: | ---: | ---: |
| BF16 (stored) | 0.00% | 100% | 0.00% | 100% |
| INT8 per-channel | 5.96% | 99.3% | 5.22% | 98.7% |
| INT8 + INT8 activations | 7.56% | 99.1% | 5.92% | 98.5% |
| INT4 group-128 | 44.5% | 93.9% | 44.6% | 87.9% |
| INT4 per-channel | 69.3% | 88.8% | 64.3% | 80.4% |
| Ternary, per-tensor absmean (BitNet b1.58) | 97.2% | 68.2% | 93.6% | 55.2% |
| Ternary per-channel / group-128 | 97.0–97.2% | 68–69% | 93.2–93.4% | 56% |
| Ternary + INT8 activations | 97.2% | 68.2% | 93.6% | 55.2% |

"Tokens re-routed" counts tokens where at least one of the top-k experts changed. "Experts kept"
is the mean fraction of each token's top-k that survives.

**Takeaways**

- Post-hoc ternary routers are not usable: about half the routing weight moves and a third to half
  of the selected experts change. Finer scale granularity (per channel, group-128) barely helps.
  The bottleneck is the three-level codebook, not the scale.
- INT4 is also far outside any reasonable budget. Even INT8 routers re-route about 5–6% of tokens,
  which is more than a BF16 router output (0.7–2.4%, see the main case studies).
- Routers are tiny: 0.030% of OLMoE's parameters (4.2 MB in BF16) and 0.016% of DeepSeek-V3's
  (216 MB). A ternary/integer design can keep routers in BF16 (or FP32 output) at negligible
  memory and bandwidth cost. `routerlab inspect <model> --lowbit` prints this share plus per-router
  quantization error and ternary sparsity, and `routerlab sweep --configs lowbit --budget 1` prints
  a per-layer plan.
- Natively ternary MoE models trained with quantization-aware training are a different case. Use
  the same tools to measure them against their own reference.

Caveats: two models, prose text, post-hoc quantization only. The task-quality impact is not
measured.
