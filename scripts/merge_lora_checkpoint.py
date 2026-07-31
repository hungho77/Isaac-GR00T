# SPDX-License-Identifier: Apache-2.0
"""Merge a LoRA-adapter-only checkpoint (produced by --lora_rank > 0) into a
full, standalone checkpoint that Gr00tPolicy / rollout_policy.py can load
directly for eval.

Why this is needed: HF Trainer + peft save only the adapter
(adapter_config.json + adapter_model.safetensors, a few tens of MB) under
--output-dir/checkpoint-N, not full model-*.safetensors. This script:
  1. Rebuilds the SAME pruned architecture used at train time (via
     Gr00tN1d7Pipeline, so it's guaranteed consistent with how training
     built it -- not a hand-rolled reimplementation).
  2. Loads the LoRA adapter from the training checkpoint on top of it.
  3. Merges the adapter into the base weights (merge_and_unload) and saves
     a full checkpoint.
  4. Copies over the processor/statistics/embodiment_id artifacts the
     training checkpoint already saved, so the merged directory is a
     complete, self-contained checkpoint.

Usage:
    uv run python scripts/merge_lora_checkpoint.py \
        --base-model-path checkpoints/GR00T-N1.7-LIBERO/libero_object \
        --adapter-checkpoint checkpoints/<run_name>/checkpoint-3000 \
        --output-dir checkpoints/<run_name>_merged \
        --kept-layer-idx-list-backbone "0,1,3,8,9,12,14" \
        --kept-layer-idx-list-dit "0,1,2,3,28,29,30,31" \
        --kept-layer-idx-list-vl-self-attn "0,1,2"

Then eval exactly as with any other checkpoint:
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path checkpoints/<run_name>_merged \
        --embodiment-tag LIBERO_PANDA \
        --use-sim-policy-wrapper
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from gr00t.configs.base_config import get_default_config
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from peft import PeftModel


def _parse_idx_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return sorted(int(x) for x in s.split(",") if x.strip() != "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-model-path", required=True, help="Same --base-model-path used at train time"
    )
    parser.add_argument(
        "--adapter-checkpoint",
        required=True,
        help="checkpoint-N dir containing adapter_config.json",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to write the merged, full checkpoint"
    )
    parser.add_argument("--kept-layer-idx-list-backbone", default=None)
    parser.add_argument("--kept-layer-idx-list-dit", default=None)
    parser.add_argument("--kept-layer-idx-list-vl-self-attn", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the minimal config needed to reconstruct the SAME pruned
    # architecture + load the SAME base weights used at train time.
    config = get_default_config()
    config.model.prune_model = bool(
        args.kept_layer_idx_list_backbone
        or args.kept_layer_idx_list_dit
        or args.kept_layer_idx_list_vl_self_attn
    )
    config.model.kept_layer_idx_list_backbone = _parse_idx_list(args.kept_layer_idx_list_backbone)
    config.model.kept_layer_idx_list_dit = _parse_idx_list(args.kept_layer_idx_list_dit)
    config.model.kept_layer_idx_list_vl_self_attn = _parse_idx_list(
        args.kept_layer_idx_list_vl_self_attn
    )
    config.training.start_from_checkpoint = args.base_model_path
    config.training.skip_weight_loading = False

    print(f"Rebuilding pruned base architecture from {args.base_model_path} ...")
    save_cfg_dir = output_dir / "_pipeline_cfg"
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Gr00tN1d7Pipeline(config, save_cfg_dir=save_cfg_dir)
    model = pipeline._create_model()

    print(f"Loading LoRA adapter from {args.adapter_checkpoint} ...")
    peft_model = PeftModel.from_pretrained(model, args.adapter_checkpoint)

    print("Merging LoRA adapter into base weights (merge_and_unload) ...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged checkpoint to {output_dir} ...")
    merged_model.save_pretrained(output_dir)

    # Copy the non-model artifacts the training checkpoint already saved
    # (processor config, dataset statistics, embodiment id mapping) --
    # Gr00tPolicy needs these alongside the model weights.
    for fname in ["processor_config.json", "statistics.json", "embodiment_id.json"]:
        src = Path(args.adapter_checkpoint) / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")
        else:
            print(f"WARNING: {fname} not found in {args.adapter_checkpoint}, skipping")

    print(f"\nDone. Merged checkpoint ready at: {output_dir}")
    print("Eval with:")
    print(
        f"  uv run python gr00t/eval/run_gr00t_server.py --model-path {output_dir} "
        "--embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper"
    )


if __name__ == "__main__":
    main()
