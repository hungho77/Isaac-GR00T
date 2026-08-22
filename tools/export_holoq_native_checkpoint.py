# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a loadable compact checkpoint with W4A4-owned BF16 tensors removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_metadata(source: Path, destination: Path, shard_names: set[str]) -> None:
    for item in source.iterdir():
        if (
            item.name in shard_names
            or item.name == "model.safetensors.index.json"
            or item.name == ".cache"
        ):
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def export_checkpoint(source: Path, pack_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite compact checkpoint: {output}")
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded safetensors index: {index_path}")
    pack = torch.load(pack_path, map_location="cpu", weights_only=True)
    records = pack.get("layers", {})
    manifest = pack.get("manifest", {})
    if manifest.get("runtime_compatible_backends") != ["fake", "native"]:
        raise ValueError("Pack is not strict native-compatible W4A4")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    excluded = {f"{name}.weight" for name in records}
    excluded.update(
        f"{name}.bias" for name, record in records.items() if record.get("bias") is not None
    )
    missing_from_source = sorted(excluded - set(weight_map))
    if missing_from_source:
        raise ValueError(
            f"Pack targets are absent from checkpoint index: {missing_from_source[:10]}"
        )
    shard_names = set(weight_map.values())
    kept_map = {key: shard for key, shard in weight_map.items() if key not in excluded}
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary_root:
        temporary = Path(temporary_root) / output.name
        temporary.mkdir(parents=True)
        _copy_metadata(source, temporary, shard_names)
        total_size = 0
        for shard_name in sorted(shard_names):
            kept_names = sorted(key for key, shard in kept_map.items() if shard == shard_name)
            if not kept_names:
                continue
            tensors = {}
            with safe_open(source / shard_name, framework="pt", device="cpu") as shard:
                shard_metadata = shard.metadata()
                for key in kept_names:
                    tensor = shard.get_tensor(key)
                    tensors[key] = tensor
                    total_size += tensor.numel() * tensor.element_size()
            save_file(tensors, temporary / shard_name, metadata=shard_metadata)
        compact_index = {
            "metadata": {**index.get("metadata", {}), "total_size": total_size},
            "weight_map": kept_map,
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(compact_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        deployment = {
            "schema_version": 1,
            "format": "holoq-native-compact-checkpoint",
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_bytes": int(index.get("metadata", {}).get("total_size", 0)),
            "residual_checkpoint_bytes": total_size,
            "pack_sha256": _sha256(pack_path),
            "pack_format_version": manifest.get("format_version"),
            "pack_suite": manifest.get("suite"),
            "source_revision": manifest.get("source_revision"),
            "checkpoint_revision": manifest.get("checkpoint_revision"),
            "excluded_state_keys": sorted(excluded),
            "excluded_tensor_count": len(excluded),
            "remaining_tensor_count": len(kept_map),
        }
        (temporary / "holoq_native_deployment.json").write_text(
            json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-path", required=True, type=Path)
    parser.add_argument("--pack-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_checkpoint(
        args.source_model_path.resolve(), args.pack_path.resolve(), args.output_path.resolve()
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
