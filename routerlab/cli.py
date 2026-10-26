"""Command-line interface: ``routerlab <command> --help``.

Commands
  spec      Print the router spec RouterLab derives from a HuggingFace config.
  inspect   List every router in a checkpoint (shapes, dtypes, biases, stats).
  compare   Compare candidate router logits/selections with a reference dump.
  sweep     Precision sweep on captured (or checkpoint + supplied) hidden states.
  demo      Small self-contained demonstration (no files needed).
  bench     CUDA microbenchmark of projection kernels (Linux + NVIDIA GPU).
  report    Markdown table from a bench result file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from ._version import __version__
from .checkpoint import Checkpoint, SafetensorsFile
from .compare import RoutingReport, _jsonable, compare_routing, format_reports
from .precision import is_representable
from .remote import is_remote
from .routing import RouterSpec
from .sweep import DEFAULT_CONFIGS, LOWBIT_CONFIGS, PrecisionConfig, plan_precision, precision_sweep

_SUFFIXES = {"logits": "logits", "topk_ids": "ids", "expert_ids": "ids", "topk_weights": "weights",
             "weights": "weights", "correction_bias": "bias", "e_score_correction_bias": "bias"}


# --------------------------------------------------------------------------- loading
def load_arrays(path) -> dict[str, np.ndarray]:
    """Load .npy/.npz/.safetensors/.pt into {key: array}."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return {"": np.load(path, allow_pickle=False)}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    if suffix == ".safetensors":
        file = SafetensorsFile(path)
        return {k: file.get(k) for k in file.keys()}
    if suffix in (".pt", ".pth", ".bin"):
        try:
            import torch
        except ImportError:
            raise SystemExit(f"Reading {path} requires torch (pip install torch)") from None
        from ._arrays import to_numpy
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if hasattr(obj, "detach"):
            return {"": to_numpy(obj)}
        if isinstance(obj, dict):
            return {str(k): to_numpy(v) for k, v in obj.items() if hasattr(v, "detach")}
        raise SystemExit(f"{path}: expected a tensor or a dict of tensors")
    raise SystemExit(f"Unsupported file type {path.suffix!r} (use .npy, .npz, .safetensors, or .pt)")


def load_capture_arrays(path, *, recompute: bool = False):
    """Capture file -> ({'NAME/logits': ..., 'NAME/correction_bias': ...}, RouterSpec | None).

    Uses the logits the captured run computed; with `recompute` (or when the
    capture holds no logits) recomputes exact float64 logits from its inputs."""
    from .capture import load_capture
    from .projection import project

    records, spec = load_capture(path)
    arrays = {}
    for name, rec in records.items():
        if rec.logits is not None and not recompute:
            arrays[f"{name}/logits"] = rec.logits
        elif rec.hidden_states is not None:
            arrays[f"{name}/logits"] = project(rec.hidden_states, rec.weight, rec.linear_bias)
        else:
            raise SystemExit(f"{path}: router {name} has neither logits nor hidden states")
        if rec.correction_bias is not None:
            arrays[f"{name}/correction_bias"] = rec.correction_bias
    return arrays, spec


def _load_inputs(path, recompute: bool):
    from .capture import is_capture_file

    if Path(path).suffix.lower() == ".safetensors" and is_capture_file(path):
        return load_capture_arrays(path, recompute=recompute)
    return load_arrays(path), None


def _single_array(path) -> np.ndarray:
    arrays = load_arrays(path)
    if len(arrays) != 1:
        raise SystemExit(f"{path} must contain exactly one array, found keys {sorted(arrays)[:5]}")
    return next(iter(arrays.values()))


def group_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Group keys like 'layer3/logits', 'layer3.topk_ids' into {'layer3': {'logits': ..., 'ids': ...}}."""
    groups: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        name, kind = key, None
        if key in _SUFFIXES:  # bare 'logits', 'topk_ids', ...: the unnamed router
            name, kind = "", _SUFFIXES[key]
        for sep in ("/", "."):
            head, _, tail = key.rpartition(sep)
            if kind is None and head and tail in _SUFFIXES:
                name, kind = head, _SUFFIXES[tail]
                break
        if kind is None:
            kind = "ids" if value.dtype.kind in "iu" else "logits"
        if kind == "ids" and value.ndim == 3:
            # vLLM routed_experts: [tokens, moe_layers, top_k] -> one group per MoE layer
            for j in range(value.shape[1]):
                groups.setdefault(f"{name or 'routed_experts'}[{j}]", {})["ids"] = value[:, j, :]
            continue
        groups.setdefault(name, {})[kind] = value
    return groups


_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_SLICE_INDEX = re.compile(r"\[(\d+)\]$")


def _layer_order(name: str):
    """Sort key: HF-style 'layers.N' first, else natural order ('layer2' before 'layer10')."""
    match = _LAYER_INDEX.search(name)
    if match:
        return (0, int(match.group(1)), ())
    return (1, 0, tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)))


def _spec_from_args(args, num_experts: int | None = None) -> RouterSpec:
    if args.config:
        spec = RouterSpec.from_hf_config(args.config)
    elif args.spec_json:
        spec = RouterSpec.from_dict(json.loads(Path(args.spec_json).read_text()))
    elif args.top_k:
        spec = RouterSpec(top_k=args.top_k, scoring=args.scoring, n_group=args.n_group,
                          topk_group=args.topk_group or args.n_group, group_score=args.group_score,
                          renormalize=args.renormalize, renorm_eps=args.renorm_eps,
                          scaling_factor=args.scaling_factor, use_correction_bias=bool(args.bias),
                          num_experts=args.num_experts or num_experts)
    else:
        raise SystemExit("Specify the router: --config config.json, --spec-json spec.json, or --top-k K [...]")
    for note in spec.notes:
        print(f"note: {note}", file=sys.stderr)
    return spec


def _add_spec_args(parser):
    group = parser.add_argument_group("router (choose one: --config, --spec-json, or --top-k and options)")
    group.add_argument("--config", help="HuggingFace config.json or model directory")
    group.add_argument("--spec-json", help="RouterSpec JSON (as printed by `routerlab spec --json`)")
    group.add_argument("--top-k", type=int)
    group.add_argument("--num-experts", type=int, help="needed when the reference is a selection, not logits")
    group.add_argument("--scoring", default="softmax", choices=["softmax", "sigmoid", "sqrtsoftplus"])
    group.add_argument("--renormalize", action="store_true")
    group.add_argument("--renorm-eps", type=float, default=0.0)
    group.add_argument("--n-group", type=int, default=1)
    group.add_argument("--topk-group", type=int)
    group.add_argument("--group-score", default="max", choices=["max", "top2_sum"])
    group.add_argument("--scaling-factor", type=float, default=1.0)
    group.add_argument("--bias", help="correction bias file (.npy etc.); applies to every router compared")


# --------------------------------------------------------------------------- commands
def cmd_spec(args) -> int:
    spec = RouterSpec.from_hf_config(args.config)
    print(json.dumps(spec.to_dict(), indent=2) if args.json else spec.describe())
    for note in spec.notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


def cmd_inspect(args) -> int:
    ckpt = Checkpoint(args.path)
    routers = ckpt.find_routers()
    out = {"path": str(args.path), "tensors": len(ckpt.names()), "routers": [], "spec": None, "notes": []}
    if ckpt.config is not None:
        try:
            spec = ckpt.router_spec()
            out["spec"] = spec.to_dict()
            out["notes"].extend(spec.notes)
        except ValueError as exc:
            out["notes"].append(str(exc))
    def stats(router):
        weight, bias, _ = ckpt.load_router(router)
        row = {"weight_abs_max": float(np.abs(weight).max()), "weight_abs_mean": float(np.abs(weight).mean())}
        if router.dtype in ("F32", "F64"):
            row["weight_bf16_representable"] = bool(is_representable(weight, "bf16").all())
        if bias is not None:
            row.update(_bias_stats(bias))
        if args.lowbit:
            row.update(_lowbit_stats(weight))
        return row

    import math
    total_params = sum(math.prod(ckpt.info(name)[1]) for name in ckpt.names()) if routers else 0
    router_params = sum(r.num_experts * r.hidden_size for r in routers)
    if total_params:
        out["parameters"] = {"total": total_params, "routers": router_params,
                             "router_fraction": router_params / total_params}
    if (args.stats or args.lowbit) and routers:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16 if is_remote(args.path) else 4) as pool:
            all_stats = list(pool.map(stats, routers))
    else:
        all_stats = [{} for _ in routers]
    for router, extra in zip(routers, all_stats, strict=True):
        out["routers"].append({**router.to_dict(), **extra})
    if not routers:
        out["notes"].append("No MoE router weights found (expected names ending in .gate/.router/.wg.weight)")
    biased = [row for row in out["routers"] if "bias_bf16_merged_pairs" in row]
    if biased:
        worst = max(biased, key=lambda row: row["bias_bf16_merged_pairs"])
        if worst["bias_bf16_merged_pairs"] > 0.01:
            out["notes"].append(
                f"Casting the correction bias to BF16 merges up to {worst['bias_bf16_merged_pairs']:.0%} of expert "
                f"pairs (layer {worst['layer']}: {worst['bias_distinct']} distinct values -> "
                f"{worst['bias_distinct_bf16']}); keep it in FP32")
        ulp = max(row["bias_fp32_spacing"] for row in biased)
        out["notes"].append(f"With FP32 selection keys, score differences below ~{ulp / 2:.1e} vanish when the bias is "
                            "added; tokens whose top-k boundary is that close tie (check with "
                            "routerlab.boundary_ties on real logits)")
    hashed = [r.layer for r in routers if r.hash_table]
    if hashed:
        out["notes"].append(f"Layers {hashed} use token-id hash routing (tid2eid); only their weights are learned")
    dtypes = sorted({r.dtype for r in routers})
    if dtypes and set(dtypes) <= {"BF16", "F16", "F8_E4M3", "F8_E5M2"}:
        out["notes"].append(f"Router weights are stored as {'/'.join(dtypes)}: FP32 math cannot recover "
                            "precision lost at storage time")
    if out["spec"] and out["spec"]["use_correction_bias"] and routers and \
            not all(r.correction_bias for r in routers):
        out["notes"].append("Spec uses a correction bias but some routers have none in the checkpoint")
    if args.json:
        print(json.dumps(_jsonable(out), indent=2))
        return 0
    print(f"{args.path}: {out['tensors']} tensors, {len(routers)} routers")
    if "parameters" in out:
        par = out["parameters"]
        print(f"router weights: {par['routers']:,} of {par['total']:,} parameters "
              f"({100 * par['router_fraction']:.3f}%); keeping them in BF16 costs "
              f"{par['routers'] * 2 / 1e6:.1f} MB")
    if out["spec"]:
        print("spec:", RouterSpec.from_dict(out["spec"]).describe())
    if routers:
        print(f"\n{'layer':>5}  {'shape':>13}  {'dtype':5}  {'corr. bias':10}  {'lin. bias':9}  weight")
        for r, row in zip(routers, out["routers"], strict=True):
            stats = (f"  |w|max {row['weight_abs_max']:.3g}" if "weight_abs_max" in row else "")
            if "bias_min" in row:
                stats += f"  bias [{row['bias_min']:.3g}, {row['bias_max']:.3g}]"
            if "ternary_rel_error" in row:
                stats += (f"  rel.err ternary {row['ternary_rel_error']:.2f} int4 {row['int4_rel_error']:.3f} "
                          f"int8 {row['int8_rel_error']:.4f}  ternary zeros {row['ternary_sparsity']:.0%}")
            print(f"{'-' if r.layer is None else r.layer:>5}  {str([r.num_experts, r.hidden_size]):>13}  "
                  f"{r.dtype:5}  {'yes' if r.correction_bias else 'no':10}  {'yes' if r.linear_bias else 'no':9}"
                  f"  {r.weight}{stats}")
    for note in out["notes"]:
        print(f"note: {note}")
    return 0


def _lowbit_stats(weight: np.ndarray) -> dict:
    """Relative weight error (Frobenius) of post-hoc low-bit quantization, and ternary sparsity."""
    from .lowbit import quantize_int, ternarize
    norm = float(np.linalg.norm(weight)) or 1.0
    ternary = ternarize(weight, "channel")
    return {"ternary_rel_error": float(np.linalg.norm(ternary.dequantize() - weight)) / norm,
            "ternary_sparsity": ternary.sparsity,
            "int4_rel_error": float(np.linalg.norm(quantize_int(weight, 4).dequantize() - weight)) / norm,
            "int8_rel_error": float(np.linalg.norm(quantize_int(weight, 8).dequantize() - weight)) / norm}


def _bias_stats(bias: np.ndarray) -> dict:
    from .precision import quantize
    bias = bias.astype(np.float64).reshape(-1)
    as_bf16 = quantize(bias, "bf16")
    same = as_bf16[:, None] == as_bf16[None, :]
    distinct_before = bias[:, None] == bias[None, :]
    pairs = np.triu_indices(len(bias), 1)
    return {"bias_min": float(bias.min()), "bias_max": float(bias.max()),
            "bias_distinct": int(len(np.unique(bias))), "bias_distinct_bf16": int(len(np.unique(as_bf16))),
            # pairs with different FP32 biases that become equal in BF16
            "bias_bf16_merged_pairs": float((same & ~distinct_before)[pairs].mean()) if len(bias) > 1 else 0.0,
            # FP32 spacing at the bias magnitude: score differences below ~half of it vanish in fp32 adds
            "bias_fp32_spacing": float(np.spacing(np.float32(np.abs(bias).max())))}


def _pair_groups(ref, cand):
    common = [name for name in ref if name in cand]
    if not common and len(ref) == 1 and len(cand) == 1:
        return [(next(iter(ref)), next(iter(cand)))]
    sliced = sorted((n for n in cand if _SLICE_INDEX.search(n)), key=lambda n: int(_SLICE_INDEX.search(n).group(1)))
    if not common and sliced:
        routers = sorted((n for n in ref if "logits" in ref[n] or "ids" in ref[n]), key=_layer_order)
        if len(routers) != len(sliced):
            raise SystemExit(f"Candidate has {len(sliced)} MoE layers but the reference has {len(routers)} routers; "
                             "select matching routers with --layers")
        return list(zip(routers, sliced, strict=True))
    if not common:
        raise SystemExit(f"No matching keys between reference ({sorted(ref)[:5]}) and candidate "
                         f"({sorted(cand)[:5]})")
    return [(name, name) for name in common]


def _has_spec_args(args) -> bool:
    return bool(args.config or args.spec_json or args.top_k)


def cmd_compare(args) -> int:
    ref_arrays, ref_spec = _load_inputs(args.reference, args.recompute)
    cand_arrays, _ = _load_inputs(args.candidate, False)
    ref_groups, cand_groups = group_arrays(ref_arrays), group_arrays(cand_arrays)
    if args.layers:
        wanted = set(args.layers.split(","))
        ref_groups = {name: group for name, group in ref_groups.items()
                      if name in wanted or str(_layer_order(name)[1]) in wanted and _layer_order(name)[0] == 0}
        if not ref_groups:
            raise SystemExit(f"--layers {args.layers} matched no reference routers")
    pairs = _pair_groups(ref_groups, cand_groups)
    extra_bias = _single_array(args.bias) if args.bias else None
    tolerance = None if args.tolerance is None else float(args.tolerance)
    reports: dict[str, RoutingReport] = {}
    base_spec = None
    for ref_name, cand_name in pairs:
        ref, cand = ref_groups[ref_name], cand_groups[cand_name]
        reference = ref.get("logits")
        if reference is None:
            reference = (ref["ids"], ref.get("weights")) if "ids" in ref else None
        if reference is None:
            continue
        num_experts = reference.shape[-1] if isinstance(reference, np.ndarray) else None
        if base_spec is None:
            base_spec = ref_spec if ref_spec is not None and not _has_spec_args(args) else \
                _spec_from_args(args, num_experts)
        spec = base_spec
        bias = ref.get("bias", cand.get("bias", extra_bias))
        if bias is not None:
            spec = spec.with_bias(bias)
        candidate = cand.get("logits")
        if candidate is None and "ids" in cand:
            candidate = (cand["ids"], cand.get("weights"))
        if candidate is None:
            raise SystemExit(f"Candidate group {cand_name!r} has neither logits nor topk ids")
        reports[ref_name or "router"] = compare_routing(reference, candidate, spec, tolerance=tolerance)
    if not reports:
        raise SystemExit("Nothing to compare")
    if len(reports) == 1:
        print(next(iter(reports.values())).summary(examples=args.examples))
    else:
        print(format_reports(reports))
        total = sum(r.num_tokens for r in reports.values())
        changed = sum(r.mismatched_tokens for r in reports.values())
        print(f"\n{len(reports)} routers, {changed} of {total} token-routings changed "
              f"({100 * changed / max(total, 1):.4g}%)")
    if args.json:
        Path(args.json).write_text(json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=2) + "\n")
        print(f"wrote {args.json}")
    unchecked = [name for name, r in reports.items() if r.passed is None]
    if unchecked:
        print(f"note: {len(unchecked)} routers differ but cannot be judged without --tolerance "
              "(the expected logit error, e.g. from projection_error_bound)", file=sys.stderr)
    if args.fail_on == "any" and any(r.mismatched_tokens or r.passed is not True for r in reports.values()):
        return 1
    if args.fail_on == "unexplained" and any(r.passed is not True for r in reports.values()):
        return 1
    return 0


def _parse_configs(text: str | None):
    if not text or text == "default":
        return DEFAULT_CONFIGS
    if text == "lowbit":
        return LOWBIT_CONFIGS
    if text == "all":
        names = {c.name for c in DEFAULT_CONFIGS}
        return DEFAULT_CONFIGS + tuple(c for c in LOWBIT_CONFIGS if c.name not in names)
    allowed = {f for f in PrecisionConfig.__dataclass_fields__ if f != "name"}
    configs = []
    for item in filter(None, (part.strip() for part in text.split(";"))):
        fields = {}
        for part in filter(None, (p.strip() for p in item.split(","))):
            key, sep, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if not sep or (key != "name" and key not in allowed):
                raise SystemExit(f"--configs: cannot parse {part!r}; use name=...,{'|'.join(sorted(allowed))}=...")
            fields[key] = value
        name = fields.pop("name", item)
        configs.append(PrecisionConfig(name=name, **fields))
    return tuple(configs)


def cmd_sweep(args) -> int:
    from .capture import load_capture

    configs = _parse_configs(args.configs)
    jobs = []  # (name, hidden, weight, bias, linear_bias)
    spec = None
    if args.capture:
        records, captured_spec = load_capture(args.capture)
        spec = captured_spec
        for name, rec in records.items():
            if rec.hidden_states is None:
                raise SystemExit(f"{args.capture}: router {name} has no captured inputs (it was recorded with "
                                 "inputs=False); a precision sweep needs the router inputs")
            jobs.append((name, rec.hidden_states, rec.weight, rec.correction_bias, rec.linear_bias))
    elif args.checkpoint:
        ckpt = Checkpoint(args.checkpoint)
        if args.layer is None:
            raise SystemExit("--checkpoint needs --layer")
        weight, bias, linear = ckpt.load_router(args.layer)
        if args.hidden:
            hidden = _single_array(args.hidden)
        elif args.synthetic:
            hidden = _synthetic_hidden(ckpt, args.layer, args.synthetic, weight.shape[1], args.seed)
            print("note: SYNTHETIC Gaussian hidden states; real activations have outliers and structure, "
                  "so treat these results as indicative only", file=sys.stderr)
        else:
            raise SystemExit("--checkpoint needs --hidden FILE (or --synthetic N)")
        jobs.append((f"layer {args.layer}", hidden, weight, bias, linear))
        if ckpt.config is not None and not (args.config or args.spec_json or args.top_k):
            spec = ckpt.router_spec()
    else:
        raise SystemExit("Give a capture file, or --checkpoint DIR --layer N --hidden FILE")
    if args.config or args.spec_json or args.top_k:
        spec = _spec_from_args(args, jobs[0][2].shape[0])
    if spec is None:
        raise SystemExit("No router spec in the capture; pass --config config.json")
    if args.bias:
        extra = _single_array(args.bias)
        jobs = [(name, hidden, weight, extra, linear) for name, hidden, weight, _, linear in jobs]
    if args.layers:
        wanted = set(args.layers.split(","))
        jobs = [j for j in jobs if j[0] in wanted or str(_layer_of(j[0])) in wanted]
    results = {}
    for name, hidden, weight, bias, linear in jobs:
        layer_spec = spec.with_bias(bias) if bias is not None else spec
        if layer_spec.use_correction_bias and layer_spec.correction_bias is None:
            raise SystemExit(f"{name}: the spec needs a correction bias but none was captured/found")
        result = precision_sweep(hidden, weight, layer_spec, configs, linear_bias=linear)
        results[name] = result
        print(f"## {name}\n{result.table()}\n")
    if len(results) > 1:
        print("## all routers (token-routings changed)")
        for config in configs:
            reports = [r.reports().get(config.name) for r in results.values()]
            reports = [r for r in reports if r is not None]
            if reports:
                total = sum(r.num_tokens for r in reports)
                changed = sum(r.mismatched_tokens for r in reports)
                print(f"- {config.name}: {changed}/{total} ({100 * changed / max(total, 1):.4g}%)")
    output = {k: v.to_dict() for k, v in results.items()}
    if args.budget is not None:
        plan = plan_precision(results, args.budget / 100)
        weights = {name: weight.size for name, _, weight, *_ in jobs}
        print(f"## precision plan (routing-change budget {args.budget:g}% per router)")
        total_bits = sum(weights[n] * p["weight_bits"] for n, p in plan.items())
        for name, entry in plan.items():
            rate = "-" if entry["change_rate"] is None else f"{100 * entry['change_rate']:.3f}%"
            print(f"- {name}: {entry['config'] or 'keep stored precision (nothing within budget)'} "
                  f"[{entry['weight_bits']:g} bits/weight, {rate} changed]")
        bf16_bits = 16 * sum(weights[n] for n in plan)
        print(f"router weights: {total_bits / 8 / 1e6:.2f} MB vs {bf16_bits / 8 / 1e6:.2f} MB in BF16 "
              f"({total_bits / bf16_bits:.1%})")
        output["_plan"] = plan
    if args.json:
        Path(args.json).write_text(json.dumps(_jsonable(output), indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


def _layer_of(name: str):
    from .capture import _LAYER_RE
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _synthetic_hidden(ckpt: Checkpoint, layer: int, tokens: int, hidden_size: int, seed: int):
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((tokens, hidden_size))
    norm = [n for n in ckpt.names() if n.endswith(f"layers.{layer}.post_attention_layernorm.weight")]
    if norm:
        hidden = hidden * ckpt.load(norm[0]).astype(np.float64)
    return hidden.astype(np.float32)


def cmd_demo(args) -> int:
    from .demo import main as demo_main
    demo_main()
    return 0


_PASSTHROUGH = {"bench": "routerlab.bench", "report": "routerlab.report"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routerlab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"routerlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("spec", help="print the router spec derived from a HuggingFace config")
    p.add_argument("config", help="config.json or model directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_spec)

    p = sub.add_parser("inspect", help="list the routers in a checkpoint")
    p.add_argument("path", help="model directory or .safetensors file")
    p.add_argument("--stats", action="store_true", help="load weights and report magnitudes/representability")
    p.add_argument("--lowbit", action="store_true",
                   help="load weights and report ternary/INT4/INT8 quantization error and ternary sparsity")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("compare", help="compare candidate router outputs with a reference",
                       description="Inputs are .npy/.npz/.safetensors/.pt or routerlab capture files. Per "
                       "router, provide float logits (key 'NAME' or 'NAME/logits') and/or 'NAME/topk_ids' "
                       "(+ 'NAME/topk_weights'); 'NAME/correction_bias' attaches a bias. A [tokens, layers, k] "
                       "integer array (vLLM routed_experts) is split per MoE layer and matched to the "
                       "reference routers in layer order. The reference should contain logits.")
    p.add_argument("reference")
    p.add_argument("candidate")
    _add_spec_args(p)
    p.add_argument("--tolerance", type=float,
                   help="expected logit error; flips at tokens with a larger margin are UNEXPLAINED")
    p.add_argument("--layers", help="comma-separated keys to compare")
    p.add_argument("--fail-on", choices=["unexplained", "any", "never"], default="unexplained",
                   help="exit status 1 on unexplained flips (default), on any flip, or never")
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--recompute", action="store_true",
                   help="for a reference capture file: use exact float64 logits recomputed from its inputs")
    p.add_argument("--json", help="write full reports to this JSON file")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="how often would routing change under BF16/FP16/FP8?",
                       description="Configs: 'default', 'lowbit', 'all', or 'name=X,weight_format=bf16,"
                       "output_format=bf16;...' with fields input_format, weight_format (also ternary, int4, int8), "
                       "weight_scaling (tensor, channel, group<N>), accumulate, output_format.")
    p.add_argument("capture", nargs="?", help="capture file from routerlab.capture.RouterCapture")
    p.add_argument("--checkpoint", help="read the router weight from this checkpoint instead")
    p.add_argument("--layer", type=int, help="layer index (with --checkpoint)")
    p.add_argument("--hidden", help="hidden states [N, H] (with --checkpoint)")
    p.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic Gaussian tokens (indicative only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", help="comma-separated router names or layer indices")
    p.add_argument("--configs", help="'default', 'lowbit' (ternary/INT), 'all', or custom (see description)")
    p.add_argument("--budget", type=float, metavar="PCT",
                   help="plan: cheapest config per router that changes at most PCT%% of tokens' experts")
    _add_spec_args(p)
    p.add_argument("--json", help="write results to this JSON file")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("demo", help="self-contained demonstration")
    p.set_defaults(func=cmd_demo)

    # Delegated to their own parsers by main(); listed here for --help.
    sub.add_parser("bench", help="CUDA projection microbenchmark (Linux + NVIDIA GPU)", add_help=False)
    sub.add_parser("report", help="Markdown table from a bench JSON file", add_help=False)
    return parser


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in _PASSTHROUGH:
        import importlib
        importlib.import_module(_PASSTHROUGH[argv[0]]).main(argv[1:])
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, KeyError, OSError) as exc:  # OSError covers missing files, HTTP and permission errors
        print(f"routerlab {args.command}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
