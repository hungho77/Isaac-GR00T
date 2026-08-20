# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict GR00T-N1.7 layer discovery for HoloQ-style W4A4 quantization."""

from __future__ import annotations

from dataclasses import dataclass
import re

from torch import nn


_LLM_RE = re.compile(
    r"(?:^|.*\.)backbone\.model(?:\.model)?\.language_model\.layers\."
    r"(?P<layer>\d+)\."
    r"(?P<projection>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_DIT_RE = re.compile(
    r"(?:^|.*\.)action_head\.model\.transformer_blocks\."
    r"(?P<layer>\d+)\."
    r"(?P<projection>attn1\.(?:to_q|to_k|to_v|to_out\.0)|"
    r"ff\.net\.(?:0\.proj|2))$"
)

_LLM_PROJECTIONS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
_DIT_PROJECTIONS = {
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
}


@dataclass(frozen=True)
class QuantTarget:
    name: str
    module: nn.Linear
    scope: str
    layer_index: int
    projection: str


@dataclass(frozen=True)
class ScopeSummary:
    llm_layers: int
    dit_layers: int
    llm_linears: int
    dit_linears: int

    @property
    def total_linears(self) -> int:
        return self.llm_linears + self.dit_linears


def discover_n1d7_targets(model: nn.Module) -> list[QuantTarget]:
    """Return only paper-covered LLM and complete DiT transformer linears.

    The vision tower, embeddings, projectors, timestep/state/action encoders,
    output heads, and the four N1.7 VL self-attention blocks are intentionally
    outside this scope.
    """

    targets: list[QuantTarget] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = _LLM_RE.fullmatch(name)
        scope = "llm"
        if match is None:
            match = _DIT_RE.fullmatch(name)
            scope = "dit"
        if match is None:
            continue
        targets.append(
            QuantTarget(
                name=name,
                module=module,
                scope=scope,
                layer_index=int(match.group("layer")),
                projection=match.group("projection"),
            )
        )
    return sorted(targets, key=lambda target: target.name)


def _validate_layer_grid(
    targets: list[QuantTarget], scope: str, expected_projections: set[str]
) -> int:
    by_layer: dict[int, set[str]] = {}
    for target in targets:
        if target.scope == scope:
            by_layer.setdefault(target.layer_index, set()).add(target.projection)
    if not by_layer:
        raise ValueError(f"No GR00T-N1.7 {scope.upper()} quantization targets were found")

    layer_indices = sorted(by_layer)
    expected_indices = list(range(layer_indices[-1] + 1))
    if layer_indices != expected_indices:
        raise ValueError(
            f"{scope.upper()} layer indices are not contiguous from zero: {layer_indices}"
        )
    errors = []
    for index, projections in by_layer.items():
        missing = sorted(expected_projections - projections)
        extra = sorted(projections - expected_projections)
        if missing or extra:
            errors.append(f"layer {index}: missing={missing}, extra={extra}")
    if errors:
        raise ValueError(f"Invalid {scope.upper()} projection grid: " + "; ".join(errors))
    return len(by_layer)


def validate_n1d7_scope(
    targets: list[QuantTarget],
    *,
    expected_llm_layers: int | None = None,
    expected_dit_layers: int | None = None,
) -> ScopeSummary:
    """Validate full projection coverage and optional checkpoint layer counts."""

    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate quantization target names were discovered")

    llm_layers = _validate_layer_grid(targets, "llm", _LLM_PROJECTIONS)
    dit_layers = _validate_layer_grid(targets, "dit", _DIT_PROJECTIONS)
    if expected_llm_layers is not None and llm_layers != expected_llm_layers:
        raise ValueError(f"Expected {expected_llm_layers} LLM layers, discovered {llm_layers}")
    if expected_dit_layers is not None and dit_layers != expected_dit_layers:
        raise ValueError(f"Expected {expected_dit_layers} DiT layers, discovered {dit_layers}")

    summary = ScopeSummary(
        llm_layers=llm_layers,
        dit_layers=dit_layers,
        llm_linears=sum(target.scope == "llm" for target in targets),
        dit_linears=sum(target.scope == "dit" for target in targets),
    )
    expected_llm_linears = 7 * llm_layers
    expected_dit_linears = 6 * dit_layers
    if summary.llm_linears != expected_llm_linears:
        raise ValueError(
            f"Expected {expected_llm_linears} LLM linears, found {summary.llm_linears}"
        )
    if summary.dit_linears != expected_dit_linears:
        raise ValueError(
            f"Expected {expected_dit_linears} DiT linears, found {summary.dit_linears}"
        )
    return summary
