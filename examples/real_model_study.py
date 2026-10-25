"""Measure router numerics on a real MoE model with real text.

    pip install "moe-routerlab[torch] @ git+https://github.com/upalchowdhury/routerlab" transformers pyarrow
    python examples/real_model_study.py allenai/OLMoE-1B-7B-0924 --tokens 8192 --out results/olmoe

Text: the WikiText-2 test split, cut into 512-token sequences. Runs:

  reference   whole model in FP32 on CPU (router inputs and logits captured)
  bf16-cpu    whole model in BF16 on CPU, same batch
  bf16-gpu    whole model in BF16 on the accelerator (CUDA or Apple MPS), batched
  bf16-gpu-1  the same, one sequence at a time (batch invariance)

Analyses (results/<name>/summary.json + summary.md):

  A. router-local precision sweep on the reference router inputs
  B. end-to-end routing divergence, BF16 model vs FP32 model, per layer; split into
     the router's own BF16 arithmetic vs upstream hidden-state drift
  C. batch invariance: batched vs one-at-a-time on the accelerator
  D. exact top-k boundary ties with FP32 / BF16 selection keys
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routerlab import (
    RouterSpec,
    boundary_ties,
    certify,
    compare_routing,
    precision_sweep,
    project,
    projection_error_bound,
)
from routerlab.capture import RouterCapture


def wikitext_sequences(tokenizer, tokens: int, length: int) -> torch.Tensor:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    count = tokens // length
    if ids.numel() < count * length:
        raise SystemExit(f"Corpus has only {ids.numel()} tokens")
    return ids[: count * length].reshape(count, length)


def run(model_id, dtype, device, batches, max_tokens):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    started = time.time()
    with RouterCapture(model, max_tokens=max_tokens) as capture, torch.inference_mode():
        for batch in batches:
            model(input_ids=batch.to(device))
    records = capture.records()
    spec = RouterSpec.from_hf_config(model.config)
    elapsed = time.time() - started
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return records, spec, elapsed


def layer_table(rows, columns):
    head = "| " + " | ".join(columns) + " |\n|" + "|".join("---:" if i else "---" for i in range(len(columns))) + "|"
    return "\n".join([head] + ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4, help="sequences per forward pass")
    parser.add_argument("--accelerator", default="cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "none"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", help="model name for the report (default: the model argument)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sequences = wikitext_sequences(tokenizer, args.tokens, args.seq_len)
    batches = list(sequences.split(args.batch))
    singles = list(sequences.split(1))
    n = sequences.numel()
    print(f"{args.model}: {sequences.shape[0]} sequences x {args.seq_len} tokens")

    ref, spec, t_ref = run(args.model, torch.float32, "cpu", batches, n)
    print(f"reference FP32 run: {t_ref:.0f}s, {len(ref)} routers, spec: {spec.describe()}")
    runs = {"bf16-cpu": run(args.model, torch.bfloat16, "cpu", batches, n)[0]}
    if args.accelerator != "none":
        try:
            runs["bf16-gpu"] = run(args.model, torch.bfloat16, args.accelerator, batches, n)[0]
            runs["bf16-gpu-1"] = run(args.model, torch.bfloat16, args.accelerator, singles, n)[0]
        except (RuntimeError, NotImplementedError) as exc:
            print(f"accelerator runs skipped: {exc}")

    def layer_spec(rec):
        return spec.with_bias(rec.correction_bias) if spec.use_correction_bias else spec

    title = args.name or args.model
    summary = {"model": title, "tokens": n, "spec": spec.to_dict(), "layers": {}, "sweep": {}}
    rows_b, rows_c, rows_d = [], [], []
    sweep_totals: dict[str, list[int]] = {}
    for name, rec in ref.items():
        s = layer_spec(rec)
        exact = project(rec.hidden_states, rec.weight, rec.linear_bias)          # float64 reference logits
        layer = {"router": name}

        # A. router-local precision sweep on real inputs
        sweep = precision_sweep(rec.hidden_states, rec.weight, s, linear_bias=rec.linear_bias)
        for row in sweep.rows:
            if "report" in row:
                total = sweep_totals.setdefault(row["config"].name, [0, 0, 0])
                total[0] += row["report"].mismatched_tokens
                total[1] += row["report"].num_tokens
                total[2] += row["ties"]
        layer["radius_quantiles"] = sweep.radius_quantiles

        # B. end to end: BF16 model vs FP32 model
        if "bf16-cpu" in runs:
            cand = runs["bf16-cpu"][name]
            e2e = compare_routing(exact, cand.logits, s, key_precisions=())
            # Router-local share: the BF16 router GEMM (fp32 accumulate, bf16 out) on the *reference* inputs.
            local_logits = project(rec.hidden_states, rec.weight, rec.linear_bias, input_format="bf16",
                                   weight_format="bf16", accumulate="fp32", output_format="bf16")
            local = compare_routing(exact, local_logits, s, key_precisions=())
            rows_b.append({"router": name, "e2e_changed_pct": round(100 * e2e.mismatch_rate, 3),
                           "router_local_changed_pct": round(100 * local.mismatch_rate, 3),
                           "e2e_max_logit_err": f"{e2e.logit_max_abs_error:.2e}",
                           "moved_weight_mean": f"{e2e.moved_weight_mean:.2e}"})
            layer["e2e"] = e2e.to_dict()
            layer["router_local_bf16"] = local.to_dict()

        # C. batch invariance on the accelerator
        if "bf16-gpu" in runs and "bf16-gpu-1" in runs:
            batched, single = runs["bf16-gpu"][name].logits, runs["bf16-gpu-1"][name].logits
            inv = compare_routing(single.astype(np.float64), batched.astype(np.float64), s, key_precisions=())
            rows_c.append({"router": name, "tokens_with_different_logits":
                           int((single != batched).any(axis=1).sum()), "expert_set_changed": inv.mismatched_tokens,
                           "max_logit_diff": f"{inv.logit_max_abs_error:.2e}"})
            layer["batch_invariance"] = inv.to_dict()

        # D. boundary ties on real logits
        ties = {p: int(boundary_ties(exact, s, p).sum()) for p in ("fp32", "bf16")}
        # Tokens a BF16-output router GEMM (fp32 accumulation) could legitimately re-route.
        bound = projection_error_bound(rec.hidden_states, rec.weight, rec.linear_bias, accumulate="fp32",
                                       output_format="bf16")
        at_risk = int((~certify(exact, s, bound)).sum())
        rows_d.append({"router": name, "fp32_key_ties": ties["fp32"], "bf16_key_ties": ties["bf16"],
                       "at_risk_bf16_logits": at_risk})
        layer["boundary_ties"] = ties
        summary["layers"][name] = layer
        print(f"  {name}: done")

    summary["sweep"] = {k: {"changed": v[0], "routings": v[1], "pct": round(100 * v[0] / v[1], 4),
                            "fp32_key_ties": v[2]} for k, v in sweep_totals.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    md = [f"# {title}: router numerics on WikiText-2 ({n} tokens)", "", f"Router: `{spec.describe()}`", "",
          "## A. Router-local precision (all layers)", "",
          layer_table([{"precision": k, **v} for k, v in summary["sweep"].items()],
                      ["precision", "changed", "routings", "pct", "fp32_key_ties"])]
    if rows_b:
        md += ["", "## B. End to end: BF16 model vs FP32 model (CPU)", "",
               layer_table(rows_b, ["router", "e2e_changed_pct", "router_local_changed_pct", "e2e_max_logit_err",
                                    "moved_weight_mean"])]
    if rows_c:
        md += ["", f"## C. Batch invariance on {args.accelerator} (BF16, batch {args.batch} vs 1)", "",
               layer_table(rows_c, ["router", "tokens_with_different_logits", "expert_set_changed",
                                    "max_logit_diff"])]
    md += ["", "## D. Boundary ties and near-ties (reference logits)", "",
           layer_table(rows_d, ["router", "fp32_key_ties", "bf16_key_ties", "at_risk_bf16_logits"])]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
