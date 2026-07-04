# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method registry for efficient inference benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gr00t.efficient.benchmark.methods import (
    BaselineMethod,
    DummyPruningMethod,
    EfficientInferenceMethod,
)


MethodFactory = Callable[..., EfficientInferenceMethod]

_METHOD_REGISTRY: dict[str, MethodFactory] = {}


def register_method(name: str, factory: MethodFactory) -> None:
    """Register a benchmark method factory by name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Method name must be non-empty.")
    _METHOD_REGISTRY[normalized_name] = factory


def get_method(name: str) -> MethodFactory:
    """Return a registered method factory."""
    normalized_name = name.strip().lower()
    try:
        return _METHOD_REGISTRY[normalized_name]
    except KeyError as exc:
        available = ", ".join(list_methods()) or "<none>"
        raise KeyError(f"Unknown method {name!r}. Available methods: {available}") from exc


def list_methods() -> list[str]:
    """List registered method names."""
    return sorted(_METHOD_REGISTRY)


def build_method(name: str, **kwargs: Any) -> EfficientInferenceMethod:
    """Instantiate a registered benchmark method."""
    factory = get_method(name)
    return factory(**kwargs)


def _register_default_methods() -> None:
    register_method("baseline", BaselineMethod)
    register_method("dummy", DummyPruningMethod)


_register_default_methods()

