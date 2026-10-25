# Contributing

Thanks for helping. Bug reports with a reproduction are the most valuable contribution. A dumped
`(logits, topk_ids, topk_weights)` triple plus the model config is usually enough.

## Development setup

```bash
python -m pip install -e ".[dev,torch]" transformers
python -m pytest                 # torch / transformers / GPU tests skip when unavailable
ruff check routerlab tests examples
```

## Adding a router family

1. Read the model's router in `transformers` (and in the engines you care about). Write down the
   scoring function, what the bias touches (selection only, or also the weights), the grouping
   rule, renormalization (eps or clamp), and scaling.
2. If it fits the `RouterSpec` pipeline, map its `model_type` in `routing.py`
   (`_spec_from_hf_config`). If it doesn't, extend `RouterSpec`. Then make sure `certify.py`
   still bounds the new selection scores soundly, and extend `tests/test_certify.py`.
3. Add a case to `tests/test_hf_parity.py`, which runs the real module side by side with the
   reference. Run it once with a deliberately wrong spec to confirm it fails.
4. Add the model type to the README table.

## GPU validation

The Triton kernels and `routerlab bench` haven't been validated on GPU hardware by the maintainers.
If you have an NVIDIA GPU, please run:

```bash
python -m pip install -e ".[gpu]"
python -m pytest tests/test_gpu.py -v
routerlab bench --suite smoke --mode graph --output results/graph.json
```

Then open an issue with the output, the JSON file, and your GPU, driver, CUDA, torch and Triton
versions. Failures are as useful as passes.

## Pull requests

- Keep changes focused, and include tests that fail without the change.
- Numerical claims need evidence in the PR description: which inputs, which tolerance, and why.
- Don't loosen a tolerance to make a test pass without explaining the error source.
