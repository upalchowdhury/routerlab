"""CUDA microbenchmark of router projection kernels (BF16 X/W -> FP32 logits).

--help works without torch; execution requires Linux, CUDA PyTorch, and Triton.
The Triton candidates in routerlab.gpu_kernels are experimental and have not
been validated on GPU hardware by the maintainers; every row is checked
against the float64 reference before it is timed.

Examples:
 routerlab bench --suite smoke --mode eager --output results/eager.json
 routerlab bench --suite smoke --mode graph --output results/graph.json
 routerlab bench --adapter my_engine:router --engine-commit SHA --mode graph
"""

import argparse
import datetime
import functools
import importlib
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np

from .compare import compare_routing
from .projection import project
from .routing import RouterSpec


def _measure(torch, fn, mode, iterations, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    calls_per_iteration = 1
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        calls_per_iteration = 16
        with torch.cuda.graph(graph):
            for _ in range(calls_per_iteration):
                fn()
        run = graph.replay
        for _ in range(10):
            run()
        torch.cuda.synchronize()
    else:
        run = fn
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (iterations * calls_per_iteration))
    return {
        "median_us": float(np.median(samples)),
        "p25_us": float(np.quantile(samples, .25)),
        "p75_us": float(np.quantile(samples, .75)),
        "window_average_samples_us": samples,
        "calls_per_iteration": calls_per_iteration,
    }


def _numerics(reference, candidate, k):
    report = compare_routing(reference, candidate, RouterSpec(top_k=k, name="raw-logit top-k"))
    return {"max_abs_error": report.logit_max_abs_error,
            "relative_l2_error": report.logit_relative_l2_error,
            "topk_set_match_fraction": 1.0 - report.mismatch_rate,
            "changed_tokens": report.mismatched_tokens,
            "min_margin_radius": report.min_radius}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="routerlab bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--mode", choices=["eager", "graph"], default="eager")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter", help="Optional pinned-engine callable module:function(x,w,out)")
    parser.add_argument("--engine-commit", help="Required revision label when --adapter is used")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.repeats < 3:
        parser.error("Use iterations >= 1 and repeats >= 3")
    if args.adapter and not args.engine_commit:
        parser.error("Record --engine-commit when supplying an engine adapter")
    try:
        import torch
        import triton
    except ImportError as exc:
        raise SystemExit("Install CUDA-enabled PyTorch and a compatible Triton on Linux first.") from exc
    if not torch.cuda.is_available() or torch.version.hip:
        raise SystemExit("A supported NVIDIA CUDA GPU is required; CPU demo: python -m routerlab.demo")
    from .gpu_kernels import rowdot, tiled
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                          "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    result = {
        "schema_version": 2,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {"gpu": torch.cuda.get_device_name(),
                        "capability": list(torch.cuda.get_device_capability()),
                        "torch": torch.__version__, "triton": triton.__version__,
                        "cuda": torch.version.cuda, "driver": driver,
                        "python": platform.python_version(), "project_commit": commit,
                        "engine_commit": args.engine_commit},
        "measurement": {"mode": args.mode, "cache_regime": "warm_repeated_inputs_and_weights",
                        "scope": "projection_only; reusable output; casts excluded for all timed backends",
                        "timing": "CUDA-event window averages; not individual request percentiles",
                        "tf32": False, "bf16_reduced_precision_reduction": False,
                        "seed": args.seed},
        "validation": {"reference": "CPU float64 products of identical BF16 values",
                       "atol": 1e-3, "rtol": 1e-3,
                       "routing_scope": "raw logits top-k diagnostic; not full model routing"},
        "rows": [],
    }
    cases = [(1, 1024, 64), (8, 4096, 128), (32, 8192, 256)]
    if args.suite == "full":
        cases = [(m, h, e) for m in [1, 2, 4, 8, 16, 32, 128]
                 for h in [1024, 4096, 8192] for e in [64, 128, 256]]
    backends = {
        "torch_mm_fp32out": lambda x, w, out: torch.mm(x, w.T, out_dtype=torch.float32, out=out),
        "triton_rowdot": rowdot, "triton_tiled": tiled,
    }
    if args.adapter:
        module, function = args.adapter.split(":", 1)
        backends["engine_adapter"] = getattr(importlib.import_module(module), function)
    order_rng = random.Random(args.seed)
    with torch.inference_mode():
        for m, h, e in cases:
            # Unit variance output keeps initial tolerances interpretable.
            x = torch.randn(m, h, device="cuda").to(torch.bfloat16)
            w = (torch.randn(e, h, device="cuda") / h**.5).to(torch.bfloat16)
            reference = project(x.float().cpu().numpy(), w.float().cpu().numpy())
            order = list(backends)
            order_rng.shuffle(order)
            for name in order:
                row = {"backend": name, "m": m, "h": h, "e": e,
                       "dtype": "bf16_inputs_bf16_weights_fp32_output"}
                try:
                    out = torch.full((m, e), float("nan"), device="cuda", dtype=torch.float32)
                    fn = functools.partial(backends[name], x, w, out)
                    actual = fn()
                    if (actual is None or actual.data_ptr() != out.data_ptr()
                            or actual.shape != out.shape or actual.dtype != torch.float32):
                        raise ValueError("Backend must fill and return the supplied FP32 output")
                    candidate = out.cpu().numpy()
                    row["numerics"] = _numerics(reference, candidate, k=8)
                    if not np.allclose(candidate, reference, atol=1e-3, rtol=1e-3):
                        raise ValueError("Projection tolerance failed; candidate will not be timed")
                    alone = torch.empty((1, e), device="cuda", dtype=torch.float32)
                    backends[name](x[:1], w, alone)
                    alone_values = alone.cpu().numpy()
                    row["same_token_alone_vs_batched"] = {
                        "bitwise_equal": bool(np.array_equal(candidate[:1], alone_values)),
                        "max_abs_difference": float(np.abs(candidate[:1] - alone_values).max())}
                    row["timing"] = _measure(torch, fn, args.mode, args.iterations, args.repeats)
                    # Validate again after all graph/eager replay, outside the timed region.
                    if not np.allclose(out.cpu().numpy(), reference, atol=1e-3, rtol=1e-3):
                        raise ValueError("Post-timing validation failed")
                    row["status"] = "passed"
                except Exception as exc:
                    row["status"] = "failed_or_unavailable"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                result["rows"].append(row)
                print(f"M={m} H={h} E={e} {name}: {row['status']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
    if not any(row["status"] == "passed" for row in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
