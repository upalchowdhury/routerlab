# allenai/OLMoE-1B-7B-0924: router numerics on WikiText-2 (16384 tokens)

Router: `olmoe: softmax scoring, top-8 of 64 experts`

## A. Router-local precision (all layers)

| precision | changed | routings | pct | fp32_key_ties |
|---|---:|---:|---:|---:|
| fp32 math on stored values | 0 | 262144 | 0.0 | 0 |
| fp32 math, bf16 logits | 1855 | 262144 | 0.7076 | 3646 |
| bf16 operands, fp32 logits | 1445 | 262144 | 0.5512 | 0 |
| bf16 operands, bf16 logits | 2479 | 262144 | 0.9457 | 3679 |
| fp16 operands, fp16 logits | 314 | 262144 | 0.1198 | 498 |
| fp8-e4m3 weights (per-channel scale) | 25085 | 262144 | 9.5692 | 0 |

## B. End to end: BF16 model vs FP32 model (CPU)

| router | e2e_changed_pct | router_local_changed_pct | e2e_max_logit_err | moved_weight_mean |
|---|---:|---:|---:|---:|
| model.layers.0.mlp.gate | 3.888 | 0.726 | 7.17e-02 | 5.08e-03 |
| model.layers.1.mlp.gate | 4.797 | 1.239 | 2.16e-01 | 5.53e-03 |
| model.layers.2.mlp.gate | 4.779 | 0.885 | 1.80e-01 | 5.36e-03 |
| model.layers.3.mlp.gate | 5.261 | 0.83 | 2.55e-01 | 6.09e-03 |
| model.layers.4.mlp.gate | 5.469 | 0.964 | 2.53e-01 | 6.66e-03 |
| model.layers.5.mlp.gate | 6.03 | 0.897 | 2.76e-01 | 7.36e-03 |
| model.layers.6.mlp.gate | 5.914 | 0.867 | 2.67e-01 | 7.34e-03 |
| model.layers.7.mlp.gate | 6.476 | 0.787 | 8.11e-01 | 8.10e-03 |
| model.layers.8.mlp.gate | 6.384 | 1.044 | 7.78e-01 | 8.16e-03 |
| model.layers.9.mlp.gate | 5.743 | 0.842 | 8.43e-01 | 7.67e-03 |
| model.layers.10.mlp.gate | 5.591 | 0.885 | 5.09e-01 | 7.26e-03 |
| model.layers.11.mlp.gate | 6.024 | 0.806 | 5.09e-01 | 7.82e-03 |
| model.layers.12.mlp.gate | 6.012 | 0.94 | 4.04e-01 | 7.75e-03 |
| model.layers.13.mlp.gate | 6.641 | 1.044 | 4.07e-01 | 8.52e-03 |
| model.layers.14.mlp.gate | 6.049 | 1.135 | 5.18e-01 | 8.17e-03 |
| model.layers.15.mlp.gate | 6.824 | 1.239 | 7.05e-01 | 9.12e-03 |

## C. Batch invariance on mps (BF16, batch 4 vs 1)

| router | tokens_with_different_logits | expert_set_changed | max_logit_diff |
|---|---:|---:|---:|
| model.layers.0.mlp.gate | 0 | 0 | 0.00e+00 |
| model.layers.1.mlp.gate | 7859 | 24 | 1.56e-02 |
| model.layers.2.mlp.gate | 12688 | 145 | 1.04e-01 |
| model.layers.3.mlp.gate | 14855 | 272 | 2.34e-01 |
| model.layers.4.mlp.gate | 15841 | 435 | 2.70e-01 |
| model.layers.5.mlp.gate | 16150 | 505 | 2.73e-01 |
| model.layers.6.mlp.gate | 16207 | 549 | 4.30e-01 |
| model.layers.7.mlp.gate | 16268 | 649 | 5.07e-01 |
| model.layers.8.mlp.gate | 16285 | 692 | 7.19e-01 |
| model.layers.9.mlp.gate | 16296 | 722 | 6.88e-01 |
| model.layers.10.mlp.gate | 16299 | 727 | 3.77e-01 |
| model.layers.11.mlp.gate | 16301 | 747 | 5.23e-01 |
| model.layers.12.mlp.gate | 16303 | 753 | 6.78e-01 |
| model.layers.13.mlp.gate | 16307 | 821 | 6.33e-01 |
| model.layers.14.mlp.gate | 16313 | 856 | 5.73e-01 |
| model.layers.15.mlp.gate | 16330 | 840 | 4.34e-01 |

## D. Boundary ties and near-ties (reference logits)

| router | fp32_key_ties | bf16_key_ties | at_risk_bf16_logits |
|---|---:|---:|---:|
| model.layers.0.mlp.gate | 0 | 731 | 867 |
| model.layers.1.mlp.gate | 0 | 983 | 1486 |
| model.layers.2.mlp.gate | 0 | 793 | 1070 |
| model.layers.3.mlp.gate | 0 | 686 | 1031 |
| model.layers.4.mlp.gate | 0 | 735 | 1182 |
| model.layers.5.mlp.gate | 0 | 662 | 1162 |
| model.layers.6.mlp.gate | 0 | 579 | 1058 |
| model.layers.7.mlp.gate | 0 | 543 | 1100 |
| model.layers.8.mlp.gate | 0 | 527 | 1107 |
| model.layers.9.mlp.gate | 0 | 438 | 994 |
| model.layers.10.mlp.gate | 0 | 376 | 921 |
| model.layers.11.mlp.gate | 0 | 402 | 1024 |
| model.layers.12.mlp.gate | 0 | 433 | 982 |
| model.layers.13.mlp.gate | 0 | 443 | 1145 |
| model.layers.14.mlp.gate | 0 | 431 | 1249 |
| model.layers.15.mlp.gate | 0 | 421 | 1265 |
