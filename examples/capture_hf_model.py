"""Capture router inputs from a HuggingFace MoE model, then run a precision sweep.

    pip install "moe-routerlab[torch] @ git+https://github.com/upalchowdhury/routerlab" transformers
    # (+ accelerate for --device-map auto)
    python examples/capture_hf_model.py allenai/OLMoE-1B-7B-0924 --tokens 4096 --out olmoe_routers.safetensors
    routerlab sweep olmoe_routers.safetensors

Works with any model whose routers RouterLab recognises (see README). Use a
prompt file that resembles your workload: routing margins depend on the data.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routerlab.capture import RouterCapture, find_router_modules

DEFAULT_PROMPTS = [
    "The mixture-of-experts architecture routes each token to a small subset of expert networks.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n",
    "Translate to French: The weather is lovely today, so we will walk to the market.",
    "In 1905 Albert Einstein published four papers that changed physics, including",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="model id or local directory")
    parser.add_argument("--prompts", help="text file, one prompt per line")
    parser.add_argument("--tokens", type=int, default=4096, help="max tokens kept per router")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", help="e.g. 'auto' to shard a large model across GPUs (needs accelerate)")
    parser.add_argument("--out", default="router_inputs.safetensors")
    args = parser.parse_args()

    prompts = open(args.prompts).read().splitlines() if args.prompts else DEFAULT_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model.eval()
    routers = find_router_modules(model)
    print(f"found {len(routers)} routers, e.g. {next(iter(routers))}")

    with RouterCapture(model, max_tokens=args.tokens) as capture, torch.no_grad():
        for prompt in prompts:
            batch = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**batch)
    capture.save(args.out)
    print(f"wrote {args.out}; next: routerlab sweep {args.out}")


if __name__ == "__main__":
    main()
