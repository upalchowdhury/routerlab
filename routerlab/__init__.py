"""RouterLab: numerical auditing for mixture-of-experts routers.

Reference routers for the major MoE architectures, margin certificates that
separate rounding effects from real bugs, tolerance-aware test assertions,
precision emulation, and checkpoint/model tooling. NumPy only; torch optional.
"""

from ._version import __version__
from .certify import certify, robustness_radius, selection_bounds
from .checkpoint import Checkpoint, RouterTensors, SafetensorsFile, save_safetensors
from .compare import RoutingReport, as_routing, compare_routing, format_reports
from .precision import FORMATS, FloatFormat, is_representable, quantize, quantize_scaled, resolve_format, round_bf16
from .projection import project, projection_error_bound
from .routing import SUPPORTED_MODEL_TYPES, RouterSpec, Routing, route, topk_indices
from .sweep import DEFAULT_CONFIGS, PrecisionConfig, SweepResult, precision_sweep
from .testing import assert_routing_close

__all__ = [
    "__version__",
    "RouterSpec", "Routing", "route", "topk_indices", "SUPPORTED_MODEL_TYPES",
    "certify", "robustness_radius", "selection_bounds",
    "RoutingReport", "compare_routing", "as_routing", "format_reports",
    "assert_routing_close",
    "project", "projection_error_bound",
    "FloatFormat", "FORMATS", "quantize", "quantize_scaled", "round_bf16", "resolve_format", "is_representable",
    "PrecisionConfig", "DEFAULT_CONFIGS", "precision_sweep", "SweepResult",
    "Checkpoint", "RouterTensors", "SafetensorsFile", "save_safetensors",
]
