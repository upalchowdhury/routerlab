# LiquidAI/LFM2-8B-A1B: router numerics on WikiText-2 (16384 tokens)

Router: `lfm2_moe: sigmoid scoring, top-4 of 32 experts, correction bias (not attached), renormalized`

## A. Router-local precision (all layers)

| precision | changed | routings | pct | fp32_key_ties |
|---|---:|---:|---:|---:|
| fp32 math on stored values | 3 | 360448 | 0.0008 | 0 |
| fp32 math, bf16 logits | 8776 | 360448 | 2.4347 | 1330 |
| bf16 operands, fp32 logits | 1477 | 360448 | 0.4098 | 1 |
| bf16 operands, bf16 logits | 8921 | 360448 | 2.475 | 1326 |
| fp16 operands, fp16 logits | 1175 | 360448 | 0.326 | 160 |
| fp8-e4m3 weights (per-channel scale) | 39615 | 360448 | 10.9905 | 0 |

## B. End to end: BF16 model vs FP32 model (CPU)

| router | e2e_changed_pct | router_local_changed_pct | e2e_max_logit_err | moved_weight_mean |
|---|---:|---:|---:|---:|
| model.layers.2.feed_forward.gate | 2.826 | 2.081 | 3.16e-02 | 5.64e-03 |
| model.layers.3.feed_forward.gate | 3.052 | 2.045 | 6.94e-01 | 6.65e-03 |
| model.layers.4.feed_forward.gate | 4.297 | 2.46 | 5.39e-01 | 9.49e-03 |
| model.layers.5.feed_forward.gate | 5.463 | 2.521 | 4.81e-01 | 1.27e-02 |
| model.layers.6.feed_forward.gate | 7.336 | 3.015 | 5.52e-01 | 1.75e-02 |
| model.layers.7.feed_forward.gate | 8.148 | 2.405 | 5.56e-01 | 1.88e-02 |
| model.layers.8.feed_forward.gate | 10.004 | 2.283 | 1.29e+00 | 2.38e-02 |
| model.layers.9.feed_forward.gate | 10.693 | 3.009 | 1.59e+00 | 2.76e-02 |
| model.layers.10.feed_forward.gate | 15.985 | 3.577 | 1.16e+00 | 4.23e-02 |
| model.layers.11.feed_forward.gate | 18.585 | 3.18 | 1.08e+00 | 4.60e-02 |
| model.layers.12.feed_forward.gate | 19.666 | 3.259 | 8.46e-01 | 5.12e-02 |
| model.layers.13.feed_forward.gate | 16.162 | 2.545 | 1.24e+00 | 4.34e-02 |
| model.layers.14.feed_forward.gate | 26.538 | 3.43 | 8.58e-01 | 7.23e-02 |
| model.layers.15.feed_forward.gate | 27.765 | 2.716 | 9.30e-01 | 7.12e-02 |
| model.layers.16.feed_forward.gate | 17.694 | 1.758 | 9.34e-01 | 4.52e-02 |
| model.layers.17.feed_forward.gate | 15.308 | 1.624 | 1.04e+00 | 3.93e-02 |
| model.layers.18.feed_forward.gate | 21.204 | 2.411 | 9.23e-01 | 5.48e-02 |
| model.layers.19.feed_forward.gate | 21.344 | 1.947 | 1.05e+00 | 5.14e-02 |
| model.layers.20.feed_forward.gate | 20.575 | 1.782 | 1.12e+00 | 4.76e-02 |
| model.layers.21.feed_forward.gate | 23.334 | 2.637 | 9.44e-01 | 5.96e-02 |
| model.layers.22.feed_forward.gate | 24.207 | 2.423 | 1.16e+00 | 6.03e-02 |
| model.layers.23.feed_forward.gate | 16.895 | 1.343 | 1.00e+00 | 4.43e-02 |

## C. Batch invariance on mps (BF16, batch 4 vs 1)

| router | tokens_with_different_logits | expert_set_changed | max_logit_diff |
|---|---:|---:|---:|
| model.layers.2.feed_forward.gate | 0 | 0 | 0.00e+00 |
| model.layers.3.feed_forward.gate | 98 | 1 | 1.56e-02 |
| model.layers.4.feed_forward.gate | 434 | 4 | 1.33e-01 |
| model.layers.5.feed_forward.gate | 875 | 21 | 1.88e-01 |
| model.layers.6.feed_forward.gate | 11531 | 181 | 1.56e-01 |
| model.layers.7.feed_forward.gate | 14403 | 342 | 3.59e-01 |
| model.layers.8.feed_forward.gate | 14836 | 460 | 3.75e-01 |
| model.layers.9.feed_forward.gate | 15062 | 614 | 4.53e-01 |
| model.layers.10.feed_forward.gate | 16078 | 1060 | 5.47e-01 |
| model.layers.11.feed_forward.gate | 16166 | 1366 | 3.75e-01 |
| model.layers.12.feed_forward.gate | 16195 | 1620 | 3.59e-01 |
| model.layers.13.feed_forward.gate | 16228 | 1304 | 3.91e-01 |
| model.layers.14.feed_forward.gate | 16310 | 2367 | 5.78e-01 |
| model.layers.15.feed_forward.gate | 16318 | 2551 | 4.38e-01 |
| model.layers.16.feed_forward.gate | 16332 | 1790 | 4.84e-01 |
| model.layers.17.feed_forward.gate | 16335 | 1475 | 5.47e-01 |
| model.layers.18.feed_forward.gate | 16339 | 2120 | 7.03e-01 |
| model.layers.19.feed_forward.gate | 16340 | 2178 | 5.78e-01 |
| model.layers.20.feed_forward.gate | 16343 | 2190 | 6.88e-01 |
| model.layers.21.feed_forward.gate | 16346 | 2458 | 1.05e+00 |
| model.layers.22.feed_forward.gate | 16348 | 2560 | 5.47e-01 |
| model.layers.23.feed_forward.gate | 16348 | 1780 | 8.83e-01 |

## D. Boundary ties and near-ties (reference logits)

| router | fp32_key_ties | bf16_key_ties | at_risk_bf16_logits |
|---|---:|---:|---:|
| model.layers.2.feed_forward.gate | 0 | 471 | 3011 |
| model.layers.3.feed_forward.gate | 0 | 367 | 2866 |
| model.layers.4.feed_forward.gate | 0 | 451 | 3049 |
| model.layers.5.feed_forward.gate | 0 | 511 | 3424 |
| model.layers.6.feed_forward.gate | 0 | 629 | 3841 |
| model.layers.7.feed_forward.gate | 0 | 427 | 3353 |
| model.layers.8.feed_forward.gate | 0 | 459 | 3399 |
| model.layers.9.feed_forward.gate | 0 | 569 | 3828 |
| model.layers.10.feed_forward.gate | 0 | 593 | 4451 |
| model.layers.11.feed_forward.gate | 1 | 594 | 4273 |
| model.layers.12.feed_forward.gate | 0 | 570 | 4547 |
| model.layers.13.feed_forward.gate | 0 | 523 | 3054 |
| model.layers.14.feed_forward.gate | 0 | 654 | 4112 |
| model.layers.15.feed_forward.gate | 0 | 610 | 3938 |
| model.layers.16.feed_forward.gate | 0 | 374 | 2402 |
| model.layers.17.feed_forward.gate | 0 | 319 | 1978 |
| model.layers.18.feed_forward.gate | 0 | 472 | 3084 |
| model.layers.19.feed_forward.gate | 0 | 362 | 2742 |
| model.layers.20.feed_forward.gate | 0 | 345 | 2586 |
| model.layers.21.feed_forward.gate | 0 | 527 | 3245 |
| model.layers.22.feed_forward.gate | 0 | 441 | 3257 |
| model.layers.23.feed_forward.gate | 0 | 266 | 1882 |
