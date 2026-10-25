"""Reference MoE routing functions (float64, deterministic tie-breaking).

Every router in the HuggingFace `transformers` MoE zoo is an instance of one
pipeline, parameterised by `RouterSpec`:

    logits  = hidden @ W.T (+ linear bias)              # computed elsewhere
    scores  = f(logits)          f in {softmax, sigmoid, sqrtsoftplus}
    choice  = scores + correction_bias                   # optional, selection only
    groups  = top `topk_group` of `n_group` by max / top-2-sum of choice   # optional
    experts = top `top_k` of choice within the selected groups
    weights = scores[experts]                            # never the biased value
    weights = weights / (sum + eps)  or  / max(sum, min) # optional renormalize
    weights = weights * scaling_factor

Ties are broken towards the lower expert (and group) index. This is a
documented convention: engines' `torch.topk` tie order is unspecified, and
`routerlab.compare` treats exact ties as zero-margin tokens.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._arrays import as_float64_2d, to_numpy

__all__ = ["RouterSpec", "Routing", "route", "topk_indices", "score_function", "SUPPORTED_MODEL_TYPES"]

SCORING_FUNCTIONS = ("softmax", "sigmoid", "sqrtsoftplus")
GROUP_SCORES = ("max", "top2_sum")


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _sqrtsoftplus(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


_SCORE_FUNCTIONS = {"softmax": _softmax, "sigmoid": _sigmoid, "sqrtsoftplus": _sqrtsoftplus}


def score_function(name: str):
    """Return the float64 NumPy implementation of a scoring function."""
    try:
        return _SCORE_FUNCTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown scoring function {name!r}; choose from {SCORING_FUNCTIONS}") from None


def topk_indices(scores, k: int) -> np.ndarray:
    """Indices of the k largest values per row, descending, lower index first on exact ties."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] == 0 or not 1 <= k <= scores.shape[1]:
        raise ValueError(f"Expected non-empty scores[M, E] and 1 <= k <= E, got shape {scores.shape}, k={k}")
    if np.isnan(scores).any():
        raise ValueError("Scores must not contain NaN")
    return np.argsort(-scores.astype(np.float64), axis=1, kind="stable")[:, :k]


@dataclass(frozen=True, eq=False)
class RouterSpec:
    """Configuration of one MoE router. See the module docstring for the pipeline.

    Build one directly, or from a model config with `RouterSpec.from_hf_config`.
    Routers that use an expert-score correction bias (DeepSeek-V3 style) need
    the per-layer bias attached with `spec.with_bias(bias)` before routing.
    """

    top_k: int
    scoring: str = "softmax"
    num_experts: int | None = None
    n_group: int = 1
    topk_group: int = 1
    group_score: str = "max"
    use_correction_bias: bool = False
    renormalize: bool = False
    renorm_eps: float = 0.0
    renorm_min: float | None = None
    scaling_factor: float = 1.0
    name: str = "custom"
    notes: tuple[str, ...] = ()
    correction_bias: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.scoring not in SCORING_FUNCTIONS:
            raise ValueError(f"scoring must be one of {SCORING_FUNCTIONS}, got {self.scoring!r}")
        if self.group_score not in GROUP_SCORES:
            raise ValueError(f"group_score must be one of {GROUP_SCORES}, got {self.group_score!r}")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be >= 1")
        if int(self.n_group) < 1 or not 1 <= int(self.topk_group) <= int(self.n_group):
            raise ValueError("Require n_group >= 1 and 1 <= topk_group <= n_group")
        if self.renorm_eps < 0 or (self.renorm_min is not None and self.renorm_min <= 0):
            raise ValueError("renorm_eps must be >= 0 and renorm_min > 0")
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "n_group", int(self.n_group))
        object.__setattr__(self, "topk_group", int(self.topk_group))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.correction_bias is not None:
            bias = to_numpy(self.correction_bias).astype(np.float64).reshape(-1)
            if not np.isfinite(bias).all():
                raise ValueError("correction_bias must be finite")
            bias.setflags(write=False)
            object.__setattr__(self, "correction_bias", bias)
            object.__setattr__(self, "use_correction_bias", True)
        if self.num_experts is not None:
            self.validate(int(self.num_experts))

    # ------------------------------------------------------------------ helpers
    @property
    def grouped(self) -> bool:
        """True when group-limited selection actually restricts the choice."""
        return self.n_group > 1 and self.topk_group < self.n_group

    @property
    def selects_on_logits(self) -> bool:
        """True when selection is a per-row monotone function of the raw logits.

        Then top-k on the scores equals top-k on the logits, which is exact and
        avoids float saturation (e.g. sigmoid(40) == sigmoid(50) in float64).
        """
        return not self.use_correction_bias and (not self.grouped or self.group_score == "max")

    def validate(self, num_experts: int) -> None:
        if self.num_experts is not None and int(self.num_experts) != num_experts:
            raise ValueError(f"Spec '{self.name}' expects {self.num_experts} experts, got {num_experts}")
        if self.top_k > num_experts:
            raise ValueError(f"top_k={self.top_k} exceeds the number of experts {num_experts}")
        if num_experts % self.n_group:
            raise ValueError(f"{num_experts} experts cannot be split into {self.n_group} groups")
        group_size = num_experts // self.n_group
        if self.grouped and self.group_score == "top2_sum" and group_size < 2:
            raise ValueError("group_score='top2_sum' needs at least 2 experts per group")
        if self.grouped and self.topk_group * group_size < self.top_k:
            raise ValueError("topk_group * group_size must be >= top_k")
        if self.use_correction_bias and self.correction_bias is not None and \
                self.correction_bias.shape[0] != num_experts:
            raise ValueError(f"correction_bias has {self.correction_bias.shape[0]} entries, "
                             f"expected {num_experts}")

    def with_bias(self, bias) -> RouterSpec:
        """Return a copy with this layer's expert-score correction bias attached."""
        return dataclasses.replace(self, correction_bias=bias, use_correction_bias=True)

    def replace(self, **changes) -> RouterSpec:
        return dataclasses.replace(self, **changes)

    def describe(self) -> str:
        parts = [f"{self.scoring} scoring", f"top-{self.top_k}"
                 + (f" of {self.num_experts} experts" if self.num_experts else "")]
        if self.grouped:
            parts.append(f"group-limited ({self.topk_group} of {self.n_group} groups by {self.group_score})")
        if self.use_correction_bias:
            parts.append("correction bias" + ("" if self.correction_bias is not None else " (not attached)"))
        if self.renormalize:
            parts.append("renormalized")
        if self.scaling_factor != 1.0:
            parts.append(f"scaled x{self.scaling_factor:g}")
        return f"{self.name}: " + ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable description (the bias array is not included)."""
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "correction_bias"}
        data["notes"] = list(self.notes)
        data["has_correction_bias"] = self.correction_bias is not None
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouterSpec:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @classmethod
    def from_hf_config(cls, config) -> RouterSpec:
        """Build the spec from a HuggingFace config: a dict, a config.json path, a
        model directory, or a `transformers.PretrainedConfig`. Raises ValueError
        for architectures whose routing is not modelled."""
        return _spec_from_hf_config(_load_config(config))


@dataclass(frozen=True, eq=False)
class Routing:
    """Selected experts (and optionally their weights) for each token."""

    expert_ids: np.ndarray            # int64 [M, k]
    weights: np.ndarray | None        # float64 [M, k], aligned with expert_ids
    num_experts: int

    def __post_init__(self):
        ids = to_numpy(self.expert_ids)
        if ids.dtype.kind not in "iu":
            if ids.dtype.kind == "f" and np.all(np.isfinite(ids)) and np.all(ids == np.round(ids)):
                ids = ids.astype(np.int64)
            else:
                raise TypeError("expert_ids must be integers")
        ids = ids.astype(np.int64).reshape(-1, ids.shape[-1]) if ids.ndim else ids.reshape(1, 1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_experts):
            bad = int(((ids < 0) | (ids >= self.num_experts)).sum())
            raise ValueError(f"{bad} expert ids are outside [0, {self.num_experts}); "
                             "padding/dropped-token markers must be removed before comparison")
        object.__setattr__(self, "expert_ids", ids)
        if self.weights is not None:
            w = to_numpy(self.weights).astype(np.float64).reshape(ids.shape)
            object.__setattr__(self, "weights", w)
        object.__setattr__(self, "num_experts", int(self.num_experts))

    @property
    def num_tokens(self) -> int:
        return self.expert_ids.shape[0]

    @property
    def top_k(self) -> int:
        return self.expert_ids.shape[1]

    def selection_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_tokens, self.num_experts), dtype=bool)
        np.put_along_axis(mask, self.expert_ids, True, axis=1)
        return mask

    def dense_weights(self) -> np.ndarray:
        """Weights scattered to [M, E] (zero for unselected experts)."""
        if self.weights is None:
            raise ValueError("This routing has no weights")
        dense = np.zeros((self.num_tokens, self.num_experts))
        rows = np.repeat(np.arange(self.num_tokens), self.top_k)
        np.add.at(dense, (rows, self.expert_ids.ravel()), self.weights.ravel())
        return dense

    def duplicate_rows(self) -> np.ndarray:
        """Tokens that list the same expert twice (always a bug in a top-k router)."""
        ordered = np.sort(self.expert_ids, axis=1)
        return (np.diff(ordered, axis=1) == 0).any(axis=1)

    def expert_load(self) -> np.ndarray:
        """Number of tokens routed to each expert."""
        return np.bincount(self.expert_ids.ravel(), minlength=self.num_experts)

    def to_dict(self) -> dict[str, Any]:
        return {"expert_ids": self.expert_ids.tolist(),
                "weights": None if self.weights is None else self.weights.tolist(),
                "num_experts": self.num_experts}


def _group_scores(values: np.ndarray, spec: RouterSpec) -> np.ndarray:
    """[M, E] -> [M, n_group] group scores (monotone non-decreasing in every element)."""
    m, e = values.shape
    grouped = values.reshape(m, spec.n_group, e // spec.n_group)
    if spec.group_score == "max":
        return grouped.max(axis=-1)
    top2 = np.partition(grouped, grouped.shape[-1] - 2, axis=-1)[..., -2:]
    return top2[..., 0] + top2[..., 1]


def selection_scores(logits: np.ndarray, spec: RouterSpec) -> np.ndarray:
    """Scores the router ranks experts by (logits themselves when that is equivalent)."""
    if spec.selects_on_logits:
        return logits
    scores = score_function(spec.scoring)(logits)
    if spec.use_correction_bias:
        if spec.correction_bias is None:
            raise ValueError(
                f"Router '{spec.name}' uses an expert-score correction bias; attach the layer's bias "
                "with spec.with_bias(bias) (pass zeros only if the bias is genuinely zero)")
        scores = scores + spec.correction_bias
    return scores


def _select(logits: np.ndarray, spec: RouterSpec):
    """Return (expert_ids [M,k], group_mask [M,G] or None)."""
    choice = selection_scores(logits, spec)
    group_mask = None
    if spec.grouped:
        groups = topk_indices(_group_scores(choice, spec), spec.topk_group)
        group_mask = np.zeros((logits.shape[0], spec.n_group), dtype=bool)
        np.put_along_axis(group_mask, groups, True, axis=1)
        eligible = np.repeat(group_mask, logits.shape[1] // spec.n_group, axis=1)
        choice = np.where(eligible, choice, -np.inf)
    return topk_indices(choice, spec.top_k), group_mask


def route(logits, spec: RouterSpec) -> Routing:
    """Route float logits [..., E] with `spec`, in float64. Returns a `Routing`."""
    logits = as_float64_2d(logits, "logits")
    spec.validate(logits.shape[1])
    ids, _ = _select(logits, spec)
    scores = score_function(spec.scoring)(logits)
    weights = np.take_along_axis(scores, ids, axis=1)
    if spec.renormalize:
        total = weights.sum(axis=1, keepdims=True)
        total = np.maximum(total, spec.renorm_min) if spec.renorm_min is not None else total + spec.renorm_eps
        weights = weights / total
    return Routing(ids, weights * spec.scaling_factor, logits.shape[1])


# ---------------------------------------------------------------------------
# HuggingFace config mapping. Each entry was checked against the router module
# in transformers 5.x (see tests/test_hf_parity.py, which runs them side by side).

_SOFTMAX_ALWAYS_RENORM = {"mixtral", "minimax", "gpt_oss", "granitemoe", "granitemoeshared",
                          "granitemoehybrid", "hunyuan_v1_moe", "qwen3_vl_moe_text", "qwen3_5_moe_text"}
_SOFTMAX_CONFIG_RENORM = {"qwen2_moe", "qwen3_moe", "qwen3_next", "olmoe", "flex_olmo"}
_SOFTMAX_NO_RENORM = {"jamba"}
_DEEPSEEK_V3_FAMILY = {"deepseek_v3", "kimi_k2", "glm4_moe", "dots1", "deepseek_v32",
                       "exaone_moe", "kimi_linear"}
_OTHER = {"deepseek_v2", "deepseek_v4", "llama4_text", "llama4", "ernie4_5_moe", "minimax_m2", "lfm2_moe"}
SUPPORTED_MODEL_TYPES = tuple(sorted(_SOFTMAX_ALWAYS_RENORM | _SOFTMAX_CONFIG_RENORM | _SOFTMAX_NO_RENORM
                                     | _DEEPSEEK_V3_FAMILY | _OTHER))

_EXPERT_KEYS = ("n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts")
_TOPK_KEYS = ("num_experts_per_tok", "moe_k", "experts_per_token", "moe_topk")


# Canonical routing fields and the alternative names some configs serialise them under.
_ALIASES = {"n_group": ("num_expert_group",), "num_experts_per_tok": ("num_experts_per_token",),
            "norm_topk_prob": ("moe_renormalize",)}
_FIELDS = ("model_type", "n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts",
           "num_experts_per_tok", "moe_k", "experts_per_token", "moe_topk", "n_group", "topk_group",
           "topk_method", "norm_topk_prob", "routed_scaling_factor", "scoring_func", "moe_norm_min",
           "use_expert_bias", "mlp_layer_types", "hidden_size")


def _normalise(cfg: dict) -> dict:
    for canonical, aliases in _ALIASES.items():
        if cfg.get(canonical) is None:
            for alias in aliases:
                if cfg.get(alias) is not None:
                    cfg[canonical] = cfg[alias]
                    break
    if isinstance(cfg.get("text_config"), Mapping):
        cfg["text_config"] = _normalise(dict(cfg["text_config"]))
    return cfg


def _load_config(config) -> dict:
    if isinstance(config, Mapping):
        return _normalise(dict(config))
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.is_dir():
            path = path / "config.json"
        return _normalise(json.loads(path.read_text()))
    if hasattr(config, "to_dict"):
        # PretrainedConfig: attribute access applies the class's attribute_map.
        data = config.to_dict()
        for name in _FIELDS:
            try:
                value = getattr(config, name)
            except AttributeError:
                continue
            if value is not None:
                data[name] = value
        text = getattr(config, "text_config", None)
        if text is not None and hasattr(text, "to_dict"):
            data["text_config"] = _load_config(text)
        return _normalise(data)
    raise TypeError("config must be a dict, a path to config.json or a model directory, "
                    "or a transformers PretrainedConfig")


def _first(cfg: Mapping, keys, default=None):
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return value
    return default


def _spec_from_hf_config(cfg: dict) -> RouterSpec:
    model_type = str(cfg.get("model_type", ""))
    text = cfg.get("text_config")
    if isinstance(text, Mapping) and _first(cfg, _EXPERT_KEYS) is None:
        cfg = dict(text)
        model_type = str(cfg.get("model_type") or model_type)
        if model_type == "llama4":
            model_type = "llama4_text"
    num_experts = _first(cfg, _EXPERT_KEYS)
    top_k = _first(cfg, _TOPK_KEYS)
    if isinstance(top_k, (list, tuple)):
        if len(set(top_k)) != 1:
            raise ValueError(f"Per-layer top_k values {top_k} are not supported; build RouterSpec per layer")
        top_k = top_k[0]
    if num_experts is None or top_k is None:
        raise ValueError(f"Config (model_type={model_type!r}) has no MoE expert count / top-k fields")
    common = dict(top_k=int(top_k), num_experts=int(num_experts), name=model_type or "custom")
    norm = bool(cfg.get("norm_topk_prob", False))
    scaling = float(cfg.get("routed_scaling_factor") or 1.0)

    if model_type in _SOFTMAX_ALWAYS_RENORM:
        return RouterSpec(scoring="softmax", renormalize=True, **common)
    if model_type in _SOFTMAX_CONFIG_RENORM:
        return RouterSpec(scoring="softmax", renormalize=norm, **common)
    if model_type in _SOFTMAX_NO_RENORM:
        return RouterSpec(scoring="softmax", **common)
    if model_type in _DEEPSEEK_V3_FAMILY or (model_type not in _OTHER and cfg.get("topk_method") == "noaux_tc"):
        scoring = str(cfg.get("scoring_func") or "sigmoid")
        return RouterSpec(scoring=scoring, n_group=int(cfg.get("n_group") or 1),
                          topk_group=int(cfg.get("topk_group") or 1), group_score="top2_sum",
                          use_correction_bias=True, renormalize=norm, renorm_eps=1e-20 if norm else 0.0,
                          scaling_factor=scaling, **common)
    if model_type == "deepseek_v2":
        notes = ()
        if norm:
            notes = ("norm_topk_prob is ignored, as in transformers' DeepseekV2TopkRouter",)
            warnings.warn(notes[0], stacklevel=3)
        method = cfg.get("topk_method") or "greedy"
        if method not in ("greedy", "group_limited_greedy"):
            raise ValueError(f"Unsupported deepseek_v2 topk_method {method!r}")
        groups = method == "group_limited_greedy"
        return RouterSpec(scoring="softmax", n_group=int(cfg.get("n_group") or 1) if groups else 1,
                          topk_group=int(cfg.get("topk_group") or 1) if groups else 1,
                          group_score="max", scaling_factor=scaling, notes=notes, **common)
    if model_type == "deepseek_v4":
        layer_types = cfg.get("mlp_layer_types") or []
        notes = ()
        if "hash_moe" in layer_types:
            hashed = [i for i, t in enumerate(layer_types) if t == "hash_moe"]
            notes = (f"Layers {hashed} use token-id hash routing; this spec models the learned 'moe' layers only",)
        return RouterSpec(scoring=str(cfg.get("scoring_func") or "sqrtsoftplus"), use_correction_bias=True,
                          renormalize=True, renorm_eps=1e-20, scaling_factor=scaling, notes=notes, **common)
    if model_type in ("llama4_text", "llama4"):
        return RouterSpec(scoring="sigmoid", **{**common, "name": "llama4"})
    if model_type == "ernie4_5_moe":
        return RouterSpec(scoring="softmax", use_correction_bias=True, renormalize=True,
                          renorm_min=float(cfg.get("moe_norm_min") or 1e-12), **common)
    if model_type == "minimax_m2":
        return RouterSpec(scoring="sigmoid", use_correction_bias=True, renormalize=True, **common)
    if model_type == "lfm2_moe":
        return RouterSpec(scoring="sigmoid", use_correction_bias=bool(cfg.get("use_expert_bias", True)),
                          renormalize=norm, renorm_eps=1e-6 if norm else 0.0, scaling_factor=scaling, **common)
    raise ValueError(f"Routing for model_type={model_type!r} is not modelled. Supported: "
                     f"{', '.join(SUPPORTED_MODEL_TYPES)}. Construct RouterSpec(...) manually instead.")
