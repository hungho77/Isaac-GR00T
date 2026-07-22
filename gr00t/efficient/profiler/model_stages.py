# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable per-stage latency instrumentation for a loaded Gr00tN1d7 model.

Formalizes the ad-hoc ``wrap()`` timing pattern used throughout this session's
scratch profiling scripts (see ``day15_audit_report.md`` §14 and §18) into a
reversible, reusable utility so future hook investigations don't re-derive it.

Stages instrumented (attribute chain confirmed in the Day 15 audit):
- ``vision_encoder_ms``: ``model.backbone.model.model.visual.forward``
  (patch-embed + ViT blocks + merger call -- see note below on double-counting)
- ``visual_merger_ms``: ``model.backbone.model.model.visual.merger.forward``
  (a sub-portion of vision_encoder_ms; do not sum the two)
- ``policy_or_llm_ms``: ``model.backbone.model.model.language_model.forward``
- ``action_head_ms``: ``model.action_head.get_action``
- ``total_latency_per_action_ms``: ``model.get_action``

Does not require a real checkpoint to import or unit-test -- ``attach_stage_profiler``
only needs objects that expose the right attribute names and callables (see
``tests/gr00t/efficient/test_model_stages.py`` for a synthetic-object test).
"""

from __future__ import annotations

from typing import Any, Callable

from gr00t.efficient.profiler.latency import LatencyProfiler


STAGE_NAMES = (
    "vision_encoder_ms",
    "visual_merger_ms",
    "policy_or_llm_ms",
    "action_head_ms",
    "total_latency_per_action_ms",
)


def attach_stage_profiler(
    model: Any, profiler: LatencyProfiler | None = None
) -> tuple[LatencyProfiler, Callable[[], None]]:
    """Instrument one loaded model instance's known stages with a LatencyProfiler.

    Returns ``(profiler, detach)``. Call ``detach()`` to restore every patched
    ``forward``/``get_action`` to its original, unwrapped function -- this is
    the reversibility the efficient-inference framework requires of every hook.
    Missing attribute chains (e.g. a different backbone) are skipped, not
    raised, so this stays safe to call speculatively.
    """
    profiler = profiler or LatencyProfiler(synchronize_cuda=True)
    restores: list[tuple[Any, str, Any]] = []

    def _wrap(obj: Any, attr: str, name: str) -> None:
        if obj is None or not hasattr(obj, attr):
            return
        original = getattr(obj, attr)

        def timed(*args: Any, **kwargs: Any) -> Any:
            with profiler.profile(name):
                return original(*args, **kwargs)

        setattr(obj, attr, timed)
        restores.append((obj, attr, original))

    qwen_model = getattr(getattr(getattr(model, "backbone", None), "model", None), "model", None)
    visual = getattr(qwen_model, "visual", None)
    merger = getattr(visual, "merger", None)
    language_model = getattr(qwen_model, "language_model", None)
    action_head = getattr(model, "action_head", None)

    _wrap(visual, "forward", "vision_encoder_ms")
    _wrap(merger, "forward", "visual_merger_ms")
    _wrap(language_model, "forward", "policy_or_llm_ms")
    _wrap(action_head, "get_action", "action_head_ms")
    _wrap(model, "get_action", "total_latency_per_action_ms")

    def detach() -> None:
        for obj, attr, original in restores:
            setattr(obj, attr, original)
        restores.clear()

    return profiler, detach


def stage_summary_with_gpu_memory(profiler: LatencyProfiler) -> dict[str, Any]:
    """LatencyProfiler.summary() plus current/peak GPU memory, if CUDA is available."""
    from gr00t.efficient.profiler.memory import get_cuda_memory_stats

    summary: dict[str, Any] = dict(profiler.summary())
    summary["gpu_memory"] = get_cuda_memory_stats()
    return summary
