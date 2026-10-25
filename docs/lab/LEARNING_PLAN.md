# Learn by building RouterLab

Spend roughly 30% of a session reading and 70% running or explaining experiments. Keep a three-line note: prediction, observation, explanation.

## Day 1: understand the problem on your Mac

Run `python -m routerlab.demo`. Write out the dimensions of a two-token, four-expert projection by hand. Change the nearby scores and predict BF16 rounding. Trace `topk_indices` and explain its explicit tie rule.

Deliverable: a one-page explanation of why output precision matters, with your own numbers. Do not generalize the toy example into a model-quality claim.

## Week 1: GPU execution and memory

Learn program instances, blocks, warps, offsets, strides, boundary masks, and global memory. Run the official Triton vector-add tutorial on Linux/NVIDIA, then recreate it without copying. Test a length not divisible by the block size.

Read the starter row-dot kernel. For one program, draw which token and expert rows it owns. Calculate its load addresses for H=10 and a small reduction block.

Deliverable: a correct vector-add exercise and an annotated explanation of the row-dot mapping. CPU simulation teaches numerics; only the GPU run teaches GPU behavior.

## Week 2: reductions and the first router kernel

Learn row sum and FP32 accumulation. Reimplement the row-dot kernel yourself in a separate experiment, initially for one expert per program. Start with M=1, H=65, E=17 so masks are exercised.

Run the GPU unit tests. Fix any compilation, indexing, dtype, or graph-replay failures before benchmarking. Inspect CPU float64 oracle differences rather than loosening tolerances automatically.

Deliverable: your first validated router projection and a trace of one reduction.

## Week 3: numerical correctness as engineering

Study cancellation, output downcasts, correction bias, near ties, and the top-k margin bound. Construct examples beyond random normal inputs. Run one token alone and with unrelated companion tokens, across the token-count thresholds of the actual engine when you get there.

Distinguish run-to-run repeatability from batch invariance. A function can be repeatable for a fixed shape but yield different rounding at another batch size. Do not promise bitwise invariance simply because an allclose check passes.

Deliverable: an expanded numerical suite and a short error report.

## Week 4: matrix tiling and tensor cores

Work through the official Triton matrix-multiplication tutorial. Explain the M/N/K tiles, weight reuse, and why padding a tiny M can waste work. Read and modify the supplied tiled candidate.

Compare the tiled and row-dot implementations against CUDA `torch.mm` with FP32 output. Keep the raw JSON results. Record one shape where your candidate loses and explain a plausible reason.

Deliverable: first complete precision/performance matrix.

## Week 5: profiling and one hypothesis

Select one expensive or surprising shape. Use Nsight Systems to inspect launches and Nsight Compute for register pressure, memory access, and instruction utilization.

Choose one hypothesis: larger expert tiles reuse X, a smaller reduction tile reduces register pressure, a different matrix tile handles small M better, or a stock library is already best. Make one change, revalidate, then remeasure.

Deliverable: a before/after profile with a causal explanation. Keep negative results.

## Week 6: match a real engine

Pin an SGLang commit and inspect the exact gate that matches your hardware and precision contract. Identify existing coverage and concurrent implementation work. Build a thin operation adapter using the callable interface in CONTRIBUTING_PATH.md.

Run the same inputs through the actual stock path and your candidate. Verify correction bias, score transforms, top-k outputs and weights, layout, graph execution, and deterministic settings as applicable. Preserve FP32 weights where the model requires them; that is a new contract, not permission to downcast.

Deliverable: a report comparing against the engine's actual implementation.

## Week 7: produce your first useful contribution

Prepare a small reproducible report or benchmark/test patch that fills a verified gap. Reference existing implementation work and explain exactly what your evidence adds. If a new kernel does not win, numerical coverage or benchmark evidence remains a useful result.

Deliverable: a concrete reviewable artifact. Coordinate scope with maintainers when you decide to post; the starter does not send anything or open a PR for you.

## Week 8: integration and explanation

If the operation-level evidence justifies it, add a conservative shape-specific path to one compatible model. Profile to verify it is invoked. Compare fixed workloads before and after with the same settings, and run the required output-quality checks.

Deliverable: an upstream-ready narrow patch and a public-style technical report. If full-model hardware is unavailable, finish with a strong operation-level contribution and state that end-to-end validation is pending.

## Read only what supports the next experiment

| Topic | Primary resource |
| --- | --- |
| First kernel | https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html |
| Reduction and fusion | https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html |
| Matrix multiplication | https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html |
| GPU architecture | https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html |
| Profiling workflow | https://docs.nvidia.com/nsight-compute/ComputeTriage/ |
| PyTorch FP32-output GEMM | https://docs.pytorch.org/docs/stable/generated/torch.mm.html |
| Kernel lectures | https://github.com/gpu-mode/lectures |

Rust is optional for this NVIDIA path. Learn basic CUDA C++ after a few Triton kernels by rewriting vector addition and a reduction. Defer distributed communication until you can explain and measure a single GPU operation.
