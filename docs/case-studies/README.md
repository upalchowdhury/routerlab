# Case studies: router numerics on real MoE models

These are measured with [`examples/real_model_study.py`](../../examples/real_model_study.py) on real
checkpoints and real text: 16,384 tokens of the WikiText-2 test split in 512-token sequences. Raw
per-layer reports are in each `summary.json`.

| Model | Router | Report |
| --- | --- | --- |
| allenai/OLMoE-1B-7B-0924 | softmax, top-8 of 64 experts, 16 MoE layers | [olmoe-1b-7b](olmoe-1b-7b/summary.md) |
| LiquidAI/LFM2-8B-A1B | sigmoid + expert bias, top-4 of 32, 22 MoE layers | [lfm2-8b-a1b](lfm2-8b-a1b/summary.md) |

## Headline numbers

All rates are fractions of token-layer routings, i.e. tokens × MoE layers.

| | OLMoE-1B-7B | LFM2-8B-A1B |
| --- | ---: | ---: |
| A. Router GEMM with BF16 output, same inputs | 0.71% change experts | 2.43% |
| A. … and exact top-k ties from BF16 logits | 1.39% of routings | 0.37% |
| A. FP8-E4M3 router weights (per-channel scale) | 9.6% change experts | 11.0% |
| B. Whole model in BF16 vs FP32, per MoE layer | 3.9% (first) → 6.8% (last) | 2.8% (first), peak 27.8% (layer 15), 16.9% (last) |
| B. … with the router's own BF16 arithmetic alone | 0.7–1.2% per layer | 1.3–3.6% per layer |
| C. Same tokens, batch of 4 vs one at a time (BF16, Apple GPU) | up to 5.2% per layer | up to 15.6% per layer |
| D. Tokens whose margin is within BF16 logit rounding | 5.3–9.1% per layer | 11.5–27.8% per layer |

What this shows:

- Router precision changes routing at rates far above "rare". A BF16 router output alone re-routes
  0.7–2.4% of token-layer routings, and FP8 router weights about 10%. This is consistent with
  DeepSeek-V3 keeping its MoE gating modules out of FP8, and with SGLang switching DeepSeek-family
  router outputs to FP32 ([sglang#29783](https://github.com/sgl-project/sglang/pull/29783)).
- End to end, most divergence is upstream drift that compounds with depth. A per-layer check on
  identical inputs badly underestimates what a full BF16 run does.
- Batch composition alone changes expert selection on this accelerator. Layer 0 is bitwise
  identical; later layers are not.
- A large share of real tokens sit within rounding distance of the top-k boundary. A test that
  compares `topk_ids` exactly on random inputs says little about this.

## Environment and caveats

- Apple M3 Ultra (96 GB), macOS. PyTorch 2.14.1 (CPU for FP32/BF16; MPS for the batch runs),
  transformers 5.20.2, RouterLab 0.2.0. HF eager implementations, not a production serving engine.
- Reference = exact float64 product of the FP32 run's router inputs and the stored router weights.
  BF16-run logits are routed with the FP32 correction bias, so an engine that also casts the bias
  would diverge further.
- WikiText-2 prose only. Code, chat, and long-context workloads can have different margin
  distributions; capture your own traffic with `RouterCapture`.
- Rates are per token-layer routing, not per request. Downstream quality impact is not measured.
- Reproduce: `python examples/real_model_study.py allenai/OLMoE-1B-7B-0924 --tokens 16384 --out <dir>`.
