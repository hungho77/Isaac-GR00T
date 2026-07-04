# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for collecting efficient benchmark result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load JSON result files from a directory without enforcing a schema yet."""
    root = Path(results_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records

