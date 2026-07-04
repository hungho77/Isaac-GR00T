# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a Markdown report from efficient benchmark method summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TITLE = "GR00T N1.7 Efficient Inference Benchmark — Day 7 Report"
METHODS = ["baseline", "dummy", "vlapruner", "specprune", "adp", "adp_vlapruner"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an efficient benchmark Markdown report.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    return parser


def _coerce(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _read_summary(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce(value) for key, value in row.items()} for row in reader]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Records | Success | Latency/action ms | Tokens After | Keep Ratio | Token Reduction | Action L2 | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {records} | {success} | {latency} | {tokens} | {keep} | {reduction} | {l2} | {speedup}x |".format(
                method=row.get("method", ""),
                records=_fmt(row.get("num_records"), digits=0),
                success=_fmt(row.get("mean_success_rate")),
                latency=_fmt(row.get("mean_latency_per_action_ms")),
                tokens=_fmt(row.get("mean_visual_token_count_after")),
                keep=_fmt(row.get("mean_keep_ratio")),
                reduction=_fmt(row.get("mean_token_reduction_ratio")),
                l2=_fmt(row.get("mean_action_l2_vs_baseline")),
                speedup=_fmt(row.get("speedup_vs_baseline")),
            )
        )
    return "\n".join(lines)


def _numeric_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row.get(field), float)]


def _observations(rows: list[dict[str, Any]]) -> list[str]:
    observations: list[str] = []
    latency_rows = _numeric_rows(rows, "mean_latency_per_action_ms")
    l2_rows = _numeric_rows(rows, "mean_action_l2_vs_baseline")
    reduction_rows = _numeric_rows(rows, "mean_token_reduction_ratio")
    speedup_rows = _numeric_rows(rows, "speedup_vs_baseline")

    if latency_rows:
        fastest = min(latency_rows, key=lambda row: row["mean_latency_per_action_ms"])
        observations.append(
            f"- Fastest method in mock: `{fastest['method']}` at "
            f"{_fmt(fastest['mean_latency_per_action_ms'])} ms/action."
        )
    if l2_rows:
        lowest_l2 = min(l2_rows, key=lambda row: row["mean_action_l2_vs_baseline"])
        observations.append(
            f"- Lowest action L2 in mock: `{lowest_l2['method']}` at "
            f"{_fmt(lowest_l2['mean_action_l2_vs_baseline'])}."
        )
    if reduction_rows:
        best_reduction = max(reduction_rows, key=lambda row: row["mean_token_reduction_ratio"])
        observations.append(
            f"- Best token reduction in mock: `{best_reduction['method']}` at "
            f"{_fmt(best_reduction['mean_token_reduction_ratio'])}."
        )
    if speedup_rows:
        best_speedup = max(speedup_rows, key=lambda row: row["speedup_vs_baseline"])
        achieved = best_speedup["speedup_vs_baseline"] >= 1.2
        observations.append(
            "- Mock speedup >= 1.2x: "
            f"{'yes' if achieved else 'no'}; best was `{best_speedup['method']}` at "
            f"{_fmt(best_speedup['speedup_vs_baseline'])}x."
        )
    if l2_rows:
        max_l2 = max(row["mean_action_l2_vs_baseline"] for row in l2_rows)
        observations.append(
            "- Action L2 remains low in mock: "
            f"{'yes' if max_l2 <= 0.01 else 'review needed'}; max was {_fmt(max_l2)}."
        )
    return observations


def generate_report(summary_csv: str | Path, output: str | Path, title: str = DEFAULT_TITLE) -> str:
    rows = _read_summary(summary_csv)
    methods_seen = {str(row.get("method")) for row in rows}
    missing_methods = [method for method in METHODS if method not in methods_seen]

    lines = [
        f"# {title}",
        "",
        "## Scope",
        "- Mock LIBERO comparison.",
        "- Methods tested: " + ", ".join(f"`{method}`" for method in METHODS if method in methods_seen) + ".",
        "- Metrics: success rate, latency, episode time, GPU memory, visual tokens, keep ratio, token reduction, and action L2.",
        "",
        "## Methods",
    ]
    lines.extend(f"- `{method}`" for method in METHODS)
    if missing_methods:
        lines.append("- Missing from summary: " + ", ".join(f"`{method}`" for method in missing_methods) + ".")

    lines.extend(
        [
            "",
            "## Summary Table",
            _summary_table(rows),
            "",
            "## Key Observations",
        ]
    )
    observations = _observations(rows)
    lines.extend(observations or ["- No numeric summary rows were available."])
    lines.extend(
        [
            "",
            "## Blockers",
            "- Real LIBERO is not connected yet unless discovered otherwise.",
            "- Real GR00T visual token/action-state hooks are not connected yet.",
            "- Mock metrics are for pipeline validation only.",
            "",
            "## Recommendation",
            "- Next: connect real LIBERO baseline.",
            "- Then: connect real visual token hook.",
            "- Then: evaluate VLA-Pruner on real episodes.",
            "- Then: run LIBERO-Plus robustness benchmark.",
            "",
        ]
    )

    report = "\n".join(lines)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_report(summary_csv=args.summary_csv, output=args.output, title=args.title)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
