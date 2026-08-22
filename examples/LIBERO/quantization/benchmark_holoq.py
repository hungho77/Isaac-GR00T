# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible BF16 versus native-W4A4 benchmarks on a captured LIBERO input."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.quantization.runtime import native_coverage
import numpy as np
import torch


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _latency_summary(values_ms: list[float]) -> dict[str, Any]:
    if not values_ms:
        raise ValueError("Latency benchmark produced no samples")
    mean = statistics.fmean(values_ms)
    return {
        "unit": "milliseconds",
        "samples": values_ms,
        "sample_count": len(values_ms),
        "mean": mean,
        "median": statistics.median(values_ms),
        "minimum": min(values_ms),
        "maximum": max(values_ms),
        "p95": _percentile(values_ms, 95),
        "p99": _percentile(values_ms, 99),
        "throughput_action_chunks_per_second": 1000.0 / mean,
    }


def _tensor_bytes(module: torch.nn.Module) -> dict[str, int]:
    parameter_bytes = sum(value.numel() * value.element_size() for value in module.parameters())
    buffer_bytes = sum(value.numel() * value.element_size() for value in module.buffers())
    return {
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "resident_tensor_bytes": parameter_bytes + buffer_bytes,
    }


def _load_replay(path: Path) -> dict[str, Any]:
    replay = torch.load(path, map_location="cpu", weights_only=False)
    if replay.get("schema_version") != 1 or not isinstance(replay.get("model_inputs"), dict):
        raise ValueError(f"Invalid HoloQ replay artifact: {path}")
    return replay["model_inputs"]


def _load_policy(args: argparse.Namespace, mode: str) -> Gr00tPolicy:
    kwargs: dict[str, Any] = {}
    if mode == "native_w4a4":
        kwargs.update(
            holoq_pack_path=str(args.pack_path),
            holoq_suite=args.suite,
            holoq_backend="native",
        )
    model_path = (
        args.native_model_path
        if mode == "native_w4a4" and args.native_model_path is not None
        else args.model_path
    )
    return Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(model_path),
        device="cuda",
        **kwargs,
    )


def _prepare(policy: Gr00tPolicy, replay_path: Path):
    model_inputs = _load_replay(replay_path)
    return policy.model.prepare_input(model_inputs)


def _forward(policy: Gr00tPolicy, prepared):
    backbone_inputs, action_inputs = prepared
    backbone_outputs = policy.model.backbone(backbone_inputs)
    return policy.model.action_head.get_action(backbone_outputs, action_inputs, None)


def profile_mode(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HoloQ benchmark requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    policy = _load_policy(args, args.mode)
    torch.cuda.synchronize()
    after_load = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    prepared = _prepare(policy, args.replay_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for _ in range(args.warmup):
        with torch.inference_mode():
            _forward(policy, prepared)
    torch.cuda.synchronize()
    steady_allocated = torch.cuda.memory_allocated()
    steady_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    cuda_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(args.iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        with torch.inference_mode():
            _forward(policy, prepared)
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_ms.append(float(start_event.elapsed_time(end_event)))
    inference_memory = {
        "steady_allocated_before_measurement_bytes": steady_allocated,
        "steady_reserved_before_measurement_bytes": steady_reserved,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated_bytes": (torch.cuda.max_memory_allocated() - steady_allocated),
    }
    coverage = native_coverage(policy.model) if args.mode == "native_w4a4" else None
    if coverage is not None and (
        coverage["native_modules"] == 0
        or coverage["native_modules"] != coverage["called_native_modules"]
        or coverage["non_native_modules"]
    ):
        raise RuntimeError(f"Incomplete native execution coverage: {coverage}")
    payload = {
        "schema_version": 1,
        "suite": args.suite,
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "pure_inference_cuda": _latency_summary(cuda_ms),
        "pure_inference_wall": _latency_summary(wall_ms),
        "vram": {
            "before_load": {
                "allocated_bytes": allocated_before,
                "reserved_bytes": reserved_before,
            },
            "after_load": after_load,
            "inference": inference_memory,
        },
        "live_model_storage": _tensor_bytes(policy.model),
        "native_coverage": coverage,
    }
    _atomic_json(args.output_path, payload)


def _resolve_module(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _tensor_leaves(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict) or hasattr(value, "items"):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def _flatten_output(value: Any) -> torch.Tensor:
    leaves = [item.detach().float().reshape(-1).cpu() for item in _tensor_leaves(value)]
    if not leaves:
        raise RuntimeError(f"Hook output contains no tensors: {type(value)}")
    return torch.cat(leaves)


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Cosine tensors have different shapes: {reference.shape} != {candidate.shape}"
        )
    denominator = float(reference.norm().double() * candidate.norm().double())
    if denominator == 0:
        return 1.0 if torch.equal(reference, candidate) else 0.0
    return float(torch.dot(reference.double(), candidate.double()) / denominator)


def _capture_outputs(
    policy: Gr00tPolicy,
    prepared,
    target_names: list[str],
    *,
    seed: int,
) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []
    for name in target_names:
        module = _resolve_module(policy.model, name)

        def capture(_module, _inputs, output, *, key=name):
            outputs[key].append(_flatten_output(output))

        handles.append(module.register_forward_hook(capture))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        result = _forward(policy, prepared)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    return dict(outputs), _flatten_output(result["action_pred"])


def cosine_similarity(args: argparse.Namespace) -> None:
    pack = torch.load(args.pack_path, map_location="cpu", weights_only=True)
    records = pack.get("layers", {})
    if not records:
        raise ValueError("W4A4 pack has no layer records")
    target_names = sorted(records)
    scopes = {name: str(records[name]["scope"]) for name in target_names}

    bf16 = _load_policy(args, "bf16")
    bf16_prepared = _prepare(bf16, args.replay_path)
    reference, reference_e2e = _capture_outputs(bf16, bf16_prepared, target_names, seed=args.seed)
    del bf16_prepared, bf16
    gc.collect()
    torch.cuda.empty_cache()

    native = _load_policy(args, "native_w4a4")
    native_prepared = _prepare(native, args.replay_path)
    candidate, candidate_e2e = _capture_outputs(
        native, native_prepared, target_names, seed=args.seed
    )
    module_results: dict[str, Any] = {}
    group_values: dict[str, list[float]] = defaultdict(list)
    for name in target_names:
        if len(reference.get(name, [])) != len(candidate.get(name, [])):
            raise RuntimeError(f"Invocation mismatch for {name}")
        invocation_cosines = [
            _cosine(left, right)
            for left, right in zip(reference[name], candidate[name], strict=True)
        ]
        value = statistics.fmean(invocation_cosines)
        module_results[name] = {
            "scope": scopes[name],
            "invocations": len(invocation_cosines),
            "mean_cosine": value,
            "minimum_cosine": min(invocation_cosines),
        }
        group_values[scopes[name]].append(value)
    groups = {
        scope: {
            "modules": len(values),
            "mean_cosine": statistics.fmean(values),
            "median_cosine": statistics.median(values),
            "minimum_cosine": min(values),
            "p05_cosine": _percentile(values, 5),
        }
        for scope, values in sorted(group_values.items())
    }
    coverage = native_coverage(native.model)
    if coverage["native_modules"] != coverage["called_native_modules"]:
        raise RuntimeError(f"Cosine pass did not exercise every native module: {coverage}")
    _atomic_json(
        args.output_path,
        {
            "schema_version": 1,
            "suite": args.suite,
            "reference": "bf16",
            "candidate": "native_w4a4",
            "seed": args.seed,
            "e2e_action_cosine": _cosine(reference_e2e, candidate_e2e),
            "groups": groups,
            "modules": module_results,
            "native_coverage": coverage,
        },
    )


def _index_size(model_path: Path) -> int:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return int(index["metadata"]["total_size"])


def _checkpoint_directory_bytes(model_path: Path) -> int:
    """Measure deployable checkpoint files without HF's local download cache."""

    return sum(
        path.stat().st_size
        for path in model_path.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_path).parts
    )


def storage(args: argparse.Namespace) -> None:
    pack = torch.load(args.pack_path, map_location="cpu", weights_only=True)
    records = pack["layers"]
    bf16_target_weight_bytes = 0
    packed_weight_bytes = 0
    scale_bytes = 0
    bias_bytes = 0
    transform_bytes = 0
    for record in records.values():
        rows, columns = (int(value) for value in record["weight_shape"])
        bf16_target_weight_bytes += rows * columns * 2
        packed_weight_bytes += (
            record["weight_packed"].numel() * record["weight_packed"].element_size()
        )
        scale_bytes += record["weight_scale"].numel() * record["weight_scale"].element_size()
        if record.get("bias") is not None:
            bias_bytes += record["bias"].numel() * record["bias"].element_size()
        for key in ("permutation", "rotation_blocks"):
            value = record.get(key)
            if isinstance(value, torch.Tensor):
                transform_bytes += value.numel() * value.element_size()
    base_tensor_bytes = _index_size(args.model_path)
    residual_tensor_bytes = _index_size(args.native_model_path)
    base_shard_bytes = sum(path.stat().st_size for path in args.model_path.glob("*.safetensors"))
    residual_shard_bytes = sum(
        path.stat().st_size for path in args.native_model_path.glob("*.safetensors")
    )
    pack_bytes = args.pack_path.stat().st_size
    base_directory_bytes = _checkpoint_directory_bytes(args.model_path)
    residual_directory_bytes = _checkpoint_directory_bytes(args.native_model_path)
    native_deployable_bytes = residual_directory_bytes + pack_bytes
    _atomic_json(
        args.output_path,
        {
            "schema_version": 1,
            "suite": args.suite,
            "bf16_checkpoint": {
                "indexed_tensor_bytes": base_tensor_bytes,
                "physical_safetensors_bytes": base_shard_bytes,
                "physical_deployable_directory_bytes": base_directory_bytes,
            },
            "native_w4a4_checkpoint": {
                "residual_indexed_tensor_bytes": residual_tensor_bytes,
                "residual_physical_safetensors_bytes": residual_shard_bytes,
                "residual_physical_directory_bytes": residual_directory_bytes,
                "physical_pack_bytes": pack_bytes,
                "physical_deployable_bytes": native_deployable_bytes,
                "physical_reduction_fraction_vs_bf16": 1.0
                - native_deployable_bytes / base_directory_bytes,
            },
            "quantized_scope": {
                "modules": len(records),
                "logical_bf16_weight_bytes": bf16_target_weight_bytes,
                "packed_int4_weight_bytes": packed_weight_bytes,
                "weight_scale_bytes": scale_bytes,
                "bias_bytes": bias_bytes,
                "transform_metadata_bytes": transform_bytes,
                "weight_only_reduction_fraction": 1.0
                - packed_weight_bytes / bf16_target_weight_bytes,
            },
            "note": (
                "Deployable storage is every compact checkpoint file except the Hugging Face "
                "download cache, plus the physical W4A4 pack. It does not double-count removed "
                "BF16 target weights."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("profile", "cosine", "storage"):
        child = subparsers.add_parser(command)
        child.add_argument("--suite", required=True, choices=("object", "spatial", "goal", "long"))
        child.add_argument("--model-path", required=True, type=Path)
        child.add_argument("--native-model-path", type=Path, default=None)
        child.add_argument("--pack-path", required=True, type=Path)
        if command != "storage":
            child.add_argument("--replay-path", required=True, type=Path)
        child.add_argument("--output-path", required=True, type=Path)
        child.add_argument("--seed", default=424242, type=int)
        if command == "profile":
            child.add_argument("--mode", required=True, choices=("bf16", "native_w4a4"))
            child.add_argument("--warmup", default=5, type=int)
            child.add_argument("--iterations", default=20, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "profile":
        profile_mode(args)
    elif args.command == "cosine":
        cosine_similarity(args)
    else:
        if args.native_model_path is None:
            raise ValueError("storage requires --native-model-path")
        storage(args)


if __name__ == "__main__":
    main()
