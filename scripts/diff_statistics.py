# SPDX-License-Identifier: Apache-2.0
"""Diff two statistics.json files (as written by BaseProcessor.save_pretrained).

Why: the finetune run recomputes dataset statistics from *your* dataset and
overrides the pretrained ones (see the "Overriding statistics for embodiment"
log line). Those stats drive action/state normalization at train time AND
de-normalization at inference. If the converted dataset produced different
statistics than the base checkpoint was trained with, the policy emits actions
on the wrong scale -- loss still converges (it fits its own targets), but the
robot moves wrongly. This script localizes exactly which keys drifted.

Usage:
    uv run python scripts/diff_statistics.py \
        checkpoints/GR00T-N1.7-LIBERO/libero_object/statistics.json \
        checkpoints/<run_name>/checkpoint-8000/statistics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def walk(node, path=()):
    """Yield (path, list-of-floats) for every leaf list in a nested dict."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(node, list) and (not node or isinstance(node[0], (int, float))):
        yield path, node


def summarize(vals: list[float]) -> str:
    if not vals:
        return "[]"
    if len(vals) <= 4:
        return "[" + ", ".join(f"{v:.4g}" for v in vals) + "]"
    return (
        f"[{vals[0]:.4g}, {vals[1]:.4g}, ... {vals[-1]:.4g}] "
        f"(n={len(vals)}, min={min(vals):.4g}, max={max(vals):.4g})"
    )


def rel_drift(a: list[float], b: list[float]) -> float | None:
    """Max relative difference between two equal-length numeric lists."""
    if len(a) != len(b) or not a:
        return None
    worst = 0.0
    for x, y in zip(a, b):
        denom = max(abs(x), abs(y), 1e-8)
        worst = max(worst, abs(x - y) / denom)
    return worst


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("base", help="statistics.json from the base (NVIDIA) checkpoint")
    p.add_argument("finetuned", help="statistics.json from your finetune checkpoint")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="report keys whose max relative drift exceeds this (default 0.10 = 10%%)",
    )
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text())
    ft = json.loads(Path(args.finetuned).read_text())

    base_leaves = dict(walk(base))
    ft_leaves = dict(walk(ft))

    print(f"base      : {args.base}")
    print(f"finetuned : {args.finetuned}")
    print(f"leaves: base={len(base_leaves)}  finetuned={len(ft_leaves)}")
    print()

    only_base = sorted(set(base_leaves) - set(ft_leaves))
    only_ft = sorted(set(ft_leaves) - set(base_leaves))
    if only_base:
        print(f"--- ONLY in base ({len(only_base)}) ---")
        for path in only_base[:20]:
            print("  " + "/".join(path))
        if len(only_base) > 20:
            print(f"  ... and {len(only_base) - 20} more keys")
        print()
    if only_ft:
        print(f"--- ONLY in finetuned ({len(only_ft)}) ---")
        for path in only_ft[:20]:
            print("  " + "/".join(path))
        if len(only_ft) > 20:
            print(f"  ... and {len(only_ft) - 20} more keys")
        print()

    shared = sorted(set(base_leaves) & set(ft_leaves))
    drifted = []
    shape_mismatch = []
    for path in shared:
        a, b = base_leaves[path], ft_leaves[path]
        if len(a) != len(b):
            shape_mismatch.append((path, len(a), len(b)))
            continue
        d = rel_drift(a, b)
        if d is not None and d > args.threshold:
            drifted.append((d, path, a, b))

    if shape_mismatch:
        print(f"--- SHAPE MISMATCH ({len(shape_mismatch)}) -- critical ---")
        for path, la, lb in shape_mismatch:
            print(f"  {'/'.join(path)}: base n={la} vs finetuned n={lb}")
        print()

    drifted.sort(key=lambda x: -x[0])
    print(f"--- DRIFT > {args.threshold:.0%} ({len(drifted)}/{len(shared)} shared keys) ---")
    if not drifted:
        print("  (none) -> statistics match; the converted dataset looks correct scale-wise.")
    for d, path, a, b in drifted[:40]:
        print(f"  drift {d:7.1%}  {'/'.join(path)}")
        print(f"       base      = {summarize(a)}")
        print(f"       finetuned = {summarize(b)}")
    if len(drifted) > 40:
        print(f"  ... and {len(drifted) - 40} more keys")

    print()
    print("=" * 70)
    if shape_mismatch or drifted:
        print("CONCLUSION: statistics DIVERGE -> the converted dataset does not match")
        print("the distribution the base checkpoint was trained on. Suspect the data")
        print("conversion before blaming the model or the pruning configuration.")
    else:
        print("CONCLUSION: statistics MATCH -> action/state scale mismatch is ruled out.")
        print("Next step: eval the unpruned base checkpoint to validate the harness.")


if __name__ == "__main__":
    main()
