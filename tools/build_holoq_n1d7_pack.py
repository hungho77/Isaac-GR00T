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
    )
    print(f"Wrote strict HoloQ W4A4 pack: {output}")


if __name__ == "__main__":
    main()
