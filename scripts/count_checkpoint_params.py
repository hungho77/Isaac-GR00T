# SPDX-License-Identifier: Apache-2.0
"""Count parameters in one or more saved checkpoints, broken down by component.

Reads only the safetensors *headers* (shapes, not data), so it is fast and needs
no GPU and no model construction. Point it at any number of checkpoint
directories to get the accuracy-vs-compression table's parameter column.

Note on which directories work:
  * full-finetune checkpoints (checkpoint-N) contain the whole model
    -> reported exactly.
  * LoRA runs save only the adapter in checkpoint-N. Point this at the *merged*
    directory produced by scripts/merge_lora_checkpoint.py instead; a raw
    adapter dir is detected and reported as such rather than silently
    undercounted.

Usage:
    uv run python scripts/count_checkpoint_params.py \
        checkpoints/GR00T-N1.7-LIBERO/libero_object \
        checkpoints/<run_name>_merged \
        checkpoints/<run_name>/checkpoint-12000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open


def classify(key: str) -> str:
    """Bucket a state-dict key into the component it belongs to."""
    if "vl_self_attention" in key:
        return "action_head.vl_self_attention"
    if "transformer_blocks" in key and "action_head" in key:
        return "action_head.DiT_blocks"
    if key.startswith("backbone") or ".backbone." in key:
        if "visual" in key or "vision" in key:
            return "backbone.vision_tower"
        if "embed_tokens" in key or "lm_head" in key:
            return "backbone.embeddings"
        if "layers." in key:
            return "backbone.llm_layers"
        return "backbone.other"
    if "action_head" in key:
        return "action_head.other (projector/encoders/decoders)"
    return "other"


def count_dir(path: Path) -> tuple[dict[str, int], dict[str, int], int]:
    """Return (per-component totals, per-component tensor counts, grand total)."""
    files = sorted(path.glob("*.safetensors"))
    if not files:
        return {}, {}, 0

    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    grand = 0
    for f in files:
        with safe_open(str(f), framework="np") as fh:
            for key in fh.keys():
                shape = fh.get_slice(key).get_shape()
                n = 1
                for d in shape:
                    n *= d
                bucket = classify(key)
                totals[bucket] = totals.get(bucket, 0) + n
                counts[bucket] = counts.get(bucket, 0) + 1
                grand += n
    return totals, counts, grand


def n_dit_blocks(path: Path) -> int | None:
    """Recover how many DiT blocks a checkpoint actually has, from key indices."""
    files = sorted(path.glob("*.safetensors"))
    idxs = set()
    for f in files:
        with safe_open(str(f), framework="np") as fh:
            for key in fh.keys():
                if "transformer_blocks." in key and "vl_self_attention" not in key:
                    tail = key.split("transformer_blocks.", 1)[1]
                    head = tail.split(".", 1)[0]
                    if head.isdigit():
                        idxs.add(int(head))
    return len(idxs) if idxs else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dirs", nargs="+", help="checkpoint directories (merged, for LoRA runs)")
    args = p.parse_args()

    summary = []
    for d in args.dirs:
        path = Path(d)
        print("=" * 78)
        print(path)
        print("=" * 78)
        if not path.is_dir():
            print("  !! not a directory, skipping\n")
            continue
        if (path / "adapter_config.json").exists() and not list(path.glob("model*.safetensors")):
            print("  !! this is a LoRA adapter dir (adapter weights only).")
            print(
                "     Run scripts/merge_lora_checkpoint.py first, then point at the merged dir.\n"
            )
            continue

        totals, counts, grand = count_dir(path)
        if not totals:
            print("  !! no *.safetensors files found\n")
            continue

        for bucket in sorted(totals, key=lambda b: -totals[b]):
            n = totals[bucket]
            pct = 100.0 * n / grand if grand else 0.0
            print(f"  {bucket:46s} {n:>15,}  {pct:5.1f}%  ({counts[bucket]} tensors)")
        nb = n_dit_blocks(path)
        print(f"  {'-' * 46} {'-' * 15}")
        print(f"  {'TOTAL':46s} {grand:>15,}")
        if nb is not None:
            print(f"  (DiT blocks in checkpoint: {nb})")
        print()
        summary.append((path.name, grand, nb))

    if len(summary) > 1:
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        base = max(g for _, g, _ in summary)
        for name, grand, nb in summary:
            blocks = f"DiT={nb}" if nb is not None else "DiT=?"
            print(
                f"  {name:52s} {grand:>15,}  {blocks:8s}  {100.0 * grand / base:5.1f}% of largest"
            )


if __name__ == "__main__":
    main()
