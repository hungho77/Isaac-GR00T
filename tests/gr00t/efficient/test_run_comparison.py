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
Test the real-LIBERO comparison path added to run_comparison.py. No real
checkpoint or LIBERO env is used: subprocess.run is faked so these stay
CPU-safe, and just verify the argv built for run_libero.py and the dispatch
between --mock and real (per-method subprocess) modes.
"""

import json
from unittest.mock import patch

from gr00t.efficient.benchmark.run_comparison import (
    _real_run_argv,
    _run_one_real,
    build_parser,
    run_comparison,
)
import pytest


def _real_args(**overrides):
    args = build_parser().parse_args(["--mock"])  # start from defaults, flip mock off below
    args.mock = False
    args.model_path = "checkpoints/GR00T-N1.7-LIBERO/libero_10"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_real_run_argv_includes_method_and_keep_ratio(tmp_path):
    args = _real_args()
    json_path = tmp_path / "specprune_kr0p75_real.json"

    argv = _real_run_argv(args, "specprune", 0.75, json_path)

    assert "--method" in argv and argv[argv.index("--method") + 1] == "specprune"
    assert "--keep-ratio" in argv and argv[argv.index("--keep-ratio") + 1] == "0.75"
    assert "--model-path" in argv and argv[argv.index("--model-path") + 1] == args.model_path
    assert "--output" in argv and argv[argv.index("--output") + 1] == str(json_path)
    assert "--torch-compile" not in argv
    assert "--profile-stages" not in argv


def test_real_run_argv_adds_optional_flags_only_when_set(tmp_path):
    args = _real_args(torch_compile=True, profile_stages=True, save_actions=True)
    json_path = tmp_path / "baseline_real.json"

    argv = _real_run_argv(args, "baseline", 1.0, json_path)

    assert "--torch-compile" in argv
    assert "--profile-stages" in argv
    assert "--save-actions" in argv


def test_run_one_real_reads_back_run_libero_output(tmp_path):
    args = _real_args()
    output_dir = tmp_path

    fake_result = {
        "summary": {"success_rate": 1.0, "latency_per_action_ms": 118.0},
        "records": [{"method": "specprune", "episode_id": 0, "success": True}],
        "csv_output": str(output_dir / "libero_specprune_kr0p75_real.csv"),
    }

    def fake_subprocess_run(cmd, **kwargs):
        # run_libero.py writes --output before exiting; emulate that here.
        output_path = cmd[cmd.index("--output") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(fake_result, handle)

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        run_result = _run_one_real(args, "specprune", 0.75, output_dir)

    assert run_result["method"] == "specprune"
    assert run_result["keep_ratio"] == 0.75
    assert run_result["summary"]["success_rate"] == 1.0
    assert run_result["records"][0]["method"] == "specprune"


def test_run_one_real_raises_on_subprocess_failure(tmp_path):
    args = _real_args()

    def fake_subprocess_run(cmd, **kwargs):
        class _Completed:
            returncode = 1
            stdout = "some stdout"
            stderr = "CUDA out of memory"

        return _Completed()

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            _run_one_real(args, "specprune", 0.75, tmp_path)


def test_run_comparison_requires_model_path_without_mock():
    args = build_parser().parse_args(["--methods", "baseline"])
    args.mock = False
    args.model_path = ""

    with pytest.raises(ValueError, match="--model-path"):
        run_comparison(args)


def test_run_comparison_real_mode_dispatches_to_subprocess_per_run(tmp_path):
    args = _real_args(
        methods="baseline,specprune",
        keep_ratios="0.75",
        output_dir=str(tmp_path),
        num_episodes=2,
    )

    seen_methods = []

    def fake_subprocess_run(cmd, **kwargs):
        method = cmd[cmd.index("--method") + 1]
        seen_methods.append(method)
        output_path = cmd[cmd.index("--output") + 1]
        keep_ratio = float(cmd[cmd.index("--keep-ratio") + 1])
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": {"success_rate": 1.0},
                    "records": [
                        {
                            "method": method,
                            "episode_id": 0,
                            "success": True,
                            "success_rate": 1.0,
                            "keep_ratio": keep_ratio,
                            "latency_per_action_ms": 100.0,
                        }
                    ],
                    "csv_output": "",
                },
                handle,
            )

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        result = run_comparison(args)

    assert result["status"] == "real_ok"
    # baseline is always forced to keep_ratio=1.0 regardless of --keep-ratios.
    assert sorted(seen_methods) == ["baseline", "specprune"]
    assert result["num_runs"] == 2
    assert result["num_records"] == 2
