"""Precision sweep: how often would routing change under each precision choice?

Given real hidden states and router weights, `precision_sweep` computes the
float64 reference routing and, for each `PrecisionConfig`, emulates the
projection at that precision and compares routing. Use it to decide whether a
router can be stored or computed in BF16/FP16/FP8 before running a model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .certify import robustness_radius
from .compare import RoutingReport, _jsonable, compare_routing
from .keys import boundary_ties
from .precision import quantize
from .projection import project
from .routing import RouterSpec

__all__ = ["PrecisionConfig", "DEFAULT_CONFIGS", "LOWBIT_CONFIGS", "precision_sweep", "SweepResult",
           "plan_precision"]

_WEIGHT_BITS = {None: 16.0, "fp64": 64.0, "fp32": 32.0, "tf32": 19.0, "bf16": 16.0, "fp16": 16.0,
                "fp8_e4m3fn": 8.0, "fp8_e5m2": 8.0, "int8": 8.0, "int4": 4.0, "int3": 3.0, "int2": 2.0,
                "ternary": 1.58}


@dataclass(frozen=True)
class PrecisionConfig:
    """One way of computing router logits. None means 'use the stored values'."""

    name: str
    input_format: str | None = None
    weight_format: str | None = None
    weight_scaling: str | None = None
    accumulate: str = "fp32"
    output_format: str | None = None

    @property
    def weight_bits(self) -> float:
        """Nominal storage bits per router weight (stored values assumed BF16 when unset)."""
        return _WEIGHT_BITS.get(self.weight_format, 16.0)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "weight_bits": self.weight_bits}


DEFAULT_CONFIGS = (
    PrecisionConfig("fp32 math on stored values", input_format="fp32", weight_format="fp32"),
    PrecisionConfig("fp32 math, bf16 logits", input_format="fp32", weight_format="fp32", output_format="bf16"),
    PrecisionConfig("bf16 operands, fp32 logits", input_format="bf16", weight_format="bf16"),
    PrecisionConfig("bf16 operands, bf16 logits", input_format="bf16", weight_format="bf16", output_format="bf16"),
    PrecisionConfig("fp16 operands, fp16 logits", input_format="fp16", weight_format="fp16", output_format="fp16"),
    PrecisionConfig("fp8-e4m3 weights (per-channel scale)", input_format="bf16",
                    weight_format="fp8_e4m3fn", weight_scaling="channel"),
)


LOWBIT_CONFIGS = (
    PrecisionConfig("bf16 operands, fp32 logits", input_format="bf16", weight_format="bf16"),
    PrecisionConfig("int8 weights (per-channel)", weight_format="int8"),
    PrecisionConfig("int8 weights + int8 activations", input_format="int8", weight_format="int8"),
    PrecisionConfig("int4 weights (per-channel)", weight_format="int4"),
    PrecisionConfig("int4 weights (group 128)", weight_format="int4", weight_scaling="group128"),
    PrecisionConfig("ternary weights (per-tensor absmean)", weight_format="ternary"),
    PrecisionConfig("ternary weights (per-channel)", weight_format="ternary", weight_scaling="channel"),
    PrecisionConfig("ternary weights (group 128)", weight_format="ternary", weight_scaling="group128"),
    PrecisionConfig("ternary + int8 activations (BitNet b1.58)", input_format="int8", weight_format="ternary"),
)


@dataclass
class SweepResult:
    router: str
    num_tokens: int
    radius_quantiles: dict[str, float]
    zero_margin_tokens: int
    rows: list[dict[str, Any]] = field(default_factory=list)  # {"config", "report" | "error"}

    def reports(self) -> dict[str, RoutingReport]:
        return {row["config"].name: row["report"] for row in self.rows if "report" in row}

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "router": self.router, "num_tokens": self.num_tokens,
            "reference_radius_quantiles": self.radius_quantiles,
            "zero_margin_tokens": self.zero_margin_tokens,
            "configs": [{"config": row["config"].to_dict(),
                         **({"report": row["report"].to_dict(), "fp32_key_ties": row["ties"]}
                            if "report" in row else {"error": row["error"]})}
                        for row in self.rows]})

    def table(self) -> str:
        lines = [f"Router {self.router}, {self.num_tokens} tokens. Reference margin (logit units): "
                 + ", ".join(f"{k} {v:.3e}" for k, v in self.radius_quantiles.items())
                 + (f"; {self.zero_margin_tokens} exact ties" if self.zero_margin_tokens else ""), "",
                 "| precision | max logit err | tokens changed | changed % | moved weight (mean) | "
                 "max load shift | fp32-key ties |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in self.rows:
            name = row["config"].name
            if "error" in row:
                lines.append(f"| {name} | {row['error']} | | | | | |")
                continue
            r = row["report"]
            lines.append(f"| {name} | {r.logit_max_abs_error:.2e} | {r.mismatched_tokens} | "
                         f"{100 * r.mismatch_rate:.3f} | {r.moved_weight_mean:.2e} | {r.max_load_shift} | "
                         f"{row['ties']} |")
        return "\n".join(lines)


def plan_precision(results: dict[str, SweepResult], max_change_rate: float) -> dict[str, dict[str, Any]]:
    """Per router: the configuration with the fewest weight bits whose routing-change rate
    stays within `max_change_rate` (e.g. 0.005 = 0.5% of tokens). Routers where nothing
    qualifies get None (keep the stored precision)."""
    plan = {}
    for name, result in results.items():
        candidates = [(row["config"], row["report"]) for row in result.rows if "report" in row
                      and row["report"].mismatch_rate <= max_change_rate]
        if not candidates:
            plan[name] = {"config": None, "weight_bits": 16.0, "change_rate": None}
            continue
        config, report = min(candidates, key=lambda c: (c[0].weight_bits, c[1].mismatch_rate))
        plan[name] = {"config": config.name, "weight_bits": config.weight_bits, "change_rate": report.mismatch_rate}
    return plan


def precision_sweep(hidden, weight, spec: RouterSpec, configs: Iterable[PrecisionConfig] = DEFAULT_CONFIGS,
                    *, linear_bias=None) -> SweepResult:
    """Compare float64 reference routing against each precision configuration.

    The reference is the exact product of the *stored* hidden states and
    weights. A linear bias is cast to each config's accumulation precision.
    Configs that fail (e.g. FP16 overflow) are reported, not raised.
    """
    reference = project(hidden, weight, linear_bias, accumulate="fp64")
    radius = robustness_radius(reference, spec)
    finite = radius[np.isfinite(radius)]
    quantiles = ({f"p{p}": float(np.percentile(finite, p)) for p in (1, 5, 50)} if finite.size
                 else {"p1": float("inf"), "p5": float("inf"), "p50": float("inf")})
    result = SweepResult(router=spec.name, num_tokens=reference.shape[0], radius_quantiles=quantiles,
                         zero_margin_tokens=int((radius == 0).sum()))
    for config in configs:
        try:
            bias = None if linear_bias is None else quantize(linear_bias, config.accumulate)
            logits = project(hidden, weight, bias, input_format=config.input_format,
                             weight_format=config.weight_format, weight_scaling=config.weight_scaling,
                             accumulate=config.accumulate, output_format=config.output_format)
        except (FloatingPointError, ValueError) as exc:
            result.rows.append({"config": config, "error": str(exc)})
            continue
        result.rows.append({"config": config, "ties": int(boundary_ties(logits, spec, "fp32").sum()),
                            "report": compare_routing(reference, logits, spec, key_precisions=())})
    return result
