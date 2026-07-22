# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
generate_report.py used to hardcode "Mock LIBERO comparison" and "Real LIBERO
is not connected yet" regardless of what data it was actually summarizing --
now real_libero_adapter.py + run_comparison.py can produce real comparison
summaries too, so the report must say which one it is.
"""

import csv

from gr00t.efficient.benchmark.generate_report import generate_report
import pytest


def _write_summary_csv(path, rows):
    fieldnames = [
        "method",
        "num_records",
        "mean_success_rate",
        "mean_latency_per_action_ms",
        "mean_keep_ratio",
        "mean_token_reduction_ratio",
        "mean_action_l2_vs_baseline",
        "speedup_vs_baseline",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def summary_csv(tmp_path):
    path = tmp_path / "method_summary.csv"
    _write_summary_csv(
        path,
        [
            {
                "method": "baseline",
                "num_records": 5,
                "mean_success_rate": 1.0,
                "mean_latency_per_action_ms": 118.0,
                "mean_keep_ratio": 1.0,
                "mean_token_reduction_ratio": 0.0,
                "mean_action_l2_vs_baseline": 0.0,
                "speedup_vs_baseline": 1.0,
            },
            {
                "method": "specprune",
                "num_records": 5,
                "mean_success_rate": 1.0,
                "mean_latency_per_action_ms": 120.0,
                "mean_keep_ratio": 0.75,
                "mean_token_reduction_ratio": 0.25,
                "mean_action_l2_vs_baseline": 1.9,
                "speedup_vs_baseline": 0.98,
            },
        ],
    )
    return path


def test_mock_mode_keeps_original_wording(summary_csv, tmp_path):
    report = generate_report(summary_csv, tmp_path / "report.md", mode="mock")

    assert "Mock LIBERO comparison." in report
    assert "Real LIBERO is not connected in this run" in report
    assert "Mock speedup" in report


def test_real_mode_describes_real_checkpoint_run(summary_csv, tmp_path):
    report = generate_report(summary_csv, tmp_path / "report.md", mode="real")

    assert "Real GR00T N1.7 checkpoint + real LIBERO env comparison." in report
    assert "Mock LIBERO comparison." not in report
    assert "Real LIBERO is not connected" not in report
    assert "Real speedup" in report
    assert "smoke tests only" in report


def test_real_mode_surfaces_lowest_success_rate(summary_csv, tmp_path):
    report = generate_report(summary_csv, tmp_path / "report.md", mode="real")

    assert "Lowest success rate" in report
