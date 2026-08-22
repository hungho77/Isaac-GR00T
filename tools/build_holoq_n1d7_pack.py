# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one GR00T-N1.7 LIBERO HoloQ W4A4 pack."""

from __future__ import annotations

import argparse

from gr00t.quantization.builder import build_holoq_pack
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--suite", required=True, choices=("object", "spatial", "goal", "long"))
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--scopes",
        default=None,
        help="Comma-separated scopes; defaults to the calibration manifest (llm,dit or llm,dit,vit)",
    )
    parser.add_argument(
        "--include-vit-mergers", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--include-vit-patch-embed", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--dit-activation-granularity",
        choices=("static-per-step-per-channel", "dynamic-per-token"),
        default="static-per-step-per-channel",
        help="Use dynamic-per-token for a native-compatible all-scope pack",
    )
    args = parser.parse_args()

    import gr00t.model  # noqa: F401

    model = AutoModel.from_pretrained(args.model_path).eval()
    output = build_holoq_pack(
        model,
        calibration_path=args.calibration_path,
        output_path=args.output_path,
        suite=args.suite,
        checkpoint=args.model_path,
        checkpoint_revision=args.checkpoint_revision,
        source_revision=args.source_revision,
        scopes=args.scopes,
        include_vit_mergers=args.include_vit_mergers,
        include_vit_patch_embed=args.include_vit_patch_embed,
        dit_activation_granularity=args.dit_activation_granularity,
    )
    print(f"Wrote strict HoloQ W4A4 pack: {output}")


if __name__ == "__main__":
    main()
