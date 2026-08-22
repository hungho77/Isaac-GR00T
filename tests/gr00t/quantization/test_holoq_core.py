# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

from gr00t.quantization.context import (
    get_dit_quant_step,
    get_dit_quant_total_steps,
    set_dit_quant_step,
)
from gr00t.quantization.packing import (
    SIGNED_QMAX,
    apply_input_transform,
    build_svd_hadamard_transform,
    gptq_quantize_blockwise,
    pack_signed_int4,
    symmetric_quantize_per_output_channel,
    unpack_signed_int4,
)
from gr00t.quantization.runtime import (
    HoloQLinear,
    HoloQPatchConv3d,
    NativeHoloQLinear,
    apply_holoq_pack,
    model_config_sha256,
    tensor_record_sha256,
)
from gr00t.quantization.scope import discover_n1d7_targets, validate_n1d7_scope
import pytest
import torch
from torch import nn
from torch.nn import functional as F


class _LanguageAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)


class _LanguageMlp(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width)
        self.up_proj = nn.Linear(width, width)
        self.down_proj = nn.Linear(width, width)


class _LanguageLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _LanguageAttention(width)
        self.mlp = _LanguageMlp(width)


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.ModuleList([nn.Linear(width, width)])


class _Geglu(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)


class _DitBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn1 = _Attention(width)
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([_Geglu(width), nn.Identity(), nn.Linear(width, width)])


class _VisionAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)


class _VisionMlp(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(width, 2 * width)
        self.linear_fc2 = nn.Linear(2 * width, width)


class _VisionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = _VisionAttention(width)
        self.mlp = _VisionMlp(width)


class _VisionMerger(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(width, width)
        self.linear_fc2 = nn.Linear(width, width)


class _PatchEmbed(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, width, kernel_size=(2, 2, 2), stride=(2, 2, 2))


class _N1d7Shape(nn.Module):
    def __init__(
        self, llm_layers: int, dit_layers: int, width: int = 8, vit_layers: int = 0
    ) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = nn.Module()
        self.backbone.model.model = nn.Module()
        self.backbone.model.model.language_model = nn.Module()
        self.backbone.model.model.language_model.layers = nn.ModuleList(
            [_LanguageLayer(width) for _ in range(llm_layers)]
        )
        if vit_layers:
            self.backbone.model.model.visual = nn.Module()
            self.backbone.model.model.visual.blocks = nn.ModuleList(
                [_VisionBlock(width) for _ in range(vit_layers)]
            )
            self.backbone.model.model.visual.merger = _VisionMerger(width)
            self.backbone.model.model.visual.deepstack_merger_list = nn.ModuleList(
                [_VisionMerger(width)]
            )
            self.backbone.model.model.visual.patch_embed = _PatchEmbed(width)
        self.action_head = nn.Module()
        self.action_head.model = nn.Module()
        self.action_head.model.transformer_blocks = nn.ModuleList(
            [_DitBlock(width) for _ in range(dit_layers)]
        )
        self.action_head.vl_self_attention = _Attention(width)
        self.action_head.num_inference_timesteps = 4
        self.config = _Config()


class _Config:
    def to_dict(self) -> dict[str, object]:
        return {"model_type": "Gr00tN1d7", "select_layer": 16, "dit_layers": 32}


def test_dit_context_is_nested_and_restored() -> None:
    assert get_dit_quant_step() is None
    assert get_dit_quant_total_steps() is None
    with set_dit_quant_step(0, total_steps=4):
        assert (get_dit_quant_step(), get_dit_quant_total_steps()) == (0, 4)
        with set_dit_quant_step(1, total_steps=4):
            assert (get_dit_quant_step(), get_dit_quant_total_steps()) == (1, 4)
        assert (get_dit_quant_step(), get_dit_quant_total_steps()) == (0, 4)
    assert get_dit_quant_step() is None


@pytest.mark.parametrize("step,total", [(-1, 4), (4, 4), (0, 0)])
def test_dit_context_rejects_invalid_steps(step: int, total: int) -> None:
    with pytest.raises(ValueError):
        with set_dit_quant_step(step, total_steps=total):
            pass


def test_n1d7_scope_is_complete_and_excludes_vl_self_attention() -> None:
    model = _N1d7Shape(llm_layers=2, dit_layers=3)
    targets = discover_n1d7_targets(model)
    summary = validate_n1d7_scope(targets, expected_llm_layers=2, expected_dit_layers=3)
    assert summary.llm_linears == 14
    assert summary.dit_linears == 18
    assert summary.total_linears == 32
    assert all("vl_self_attention" not in target.name for target in targets)


def test_n1d7_scope_covers_vit_blocks_mergers_and_optional_patch_embed() -> None:
    model = _N1d7Shape(llm_layers=2, dit_layers=3, vit_layers=2)
    targets = discover_n1d7_targets(
        model,
        scopes="llm,dit,vit",
        include_vit_mergers=True,
        include_vit_patch_embed=True,
    )
    summary = validate_n1d7_scope(
        targets,
        expected_llm_layers=2,
        expected_dit_layers=3,
        expected_vit_layers=2,
        scopes="llm,dit,vit",
    )
    assert summary.vit_layers == 2
    assert summary.vit_linears == 12  # 8 block + 2 merger + 2 DeepStack merger
    assert summary.vit_patch_convs == 1
    assert summary.total_modules == 45


def test_svd_randomized_hadamard_is_orthogonal_and_invariant() -> None:
    generator = torch.Generator().manual_seed(12)
    weight = torch.randn(16, 16, generator=generator)
    inputs = torch.randn(5, 16, generator=generator)
    permutation, rotations, rotated_weight = build_svd_hadamard_transform(
        weight, block_size=8, seed=3
    )
    identity = torch.eye(8)
    for rotation in rotations:
        torch.testing.assert_close(rotation.T @ rotation, identity, atol=2e-5, rtol=2e-5)
    transformed = apply_input_transform(inputs, permutation, rotations)
    torch.testing.assert_close(
        F.linear(transformed, rotated_weight),
        F.linear(inputs, weight),
        atol=2e-4,
        rtol=2e-4,
    )

    _, other_rotations, _ = build_svd_hadamard_transform(weight, block_size=8, seed=4)
    assert not torch.equal(rotations, other_rotations)


def test_weight_quantizer_uses_paper_signed_range() -> None:
    weight = torch.linspace(-10, 10, 128).reshape(8, 16)
    quantized, scale = symmetric_quantize_per_output_channel(weight)
    assert quantized.dtype == torch.int8
    assert int(quantized.min()) >= -SIGNED_QMAX
    assert int(quantized.max()) <= SIGNED_QMAX
    assert scale.shape == (8, 1)


def test_signed_int4_nibble_pack_is_bit_exact_and_padded() -> None:
    values = torch.tensor([[-7, -1, 0, 1, 7]], dtype=torch.int8)
    packed, logical_width = pack_signed_int4(values, pad_to=8)
    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 4)
    assert logical_width == 5
    restored = unpack_signed_int4(packed, logical_width=logical_width)
    torch.testing.assert_close(restored, values)


def test_gptq_uses_block_hessian_and_signed_int4() -> None:
    weight = torch.randn(8, 16, generator=torch.Generator().manual_seed(5))
    hessian_blocks = torch.eye(8).repeat(2, 1, 1)
    quantized, scale = gptq_quantize_blockwise(weight, hessian_blocks, block_size=8, damping=0.01)
    assert quantized.shape == weight.shape
    assert scale.shape == (8, 1)
    assert int(quantized.min()) >= -SIGNED_QMAX
    assert int(quantized.max()) <= SIGNED_QMAX


def _record(linear: nn.Linear, scope: str, steps: int = 4) -> dict:
    permutation, rotations, rotated = build_svd_hadamard_transform(
        linear.weight, block_size=8, seed=9
    )
    weight_q, weight_scale = symmetric_quantize_per_output_channel(rotated)
    return {
        "scope": scope,
        "solver": "rtn" if scope == "dit" else "gptq",
        "weight_q": weight_q,
        "weight_scale": weight_scale,
        "permutation": permutation,
        "rotation_blocks": rotations,
        "activation_scale": torch.ones(steps, linear.in_features) * 0.1 if scope == "dit" else None,
    }


def test_dit_runtime_requires_exact_step_table() -> None:
    linear = nn.Linear(16, 8)
    quantized = HoloQLinear(linear, name="dit", record=_record(linear, "dit"))
    inputs = torch.randn(2, 16)
    with pytest.raises(RuntimeError, match="active denoising-step context"):
        quantized(inputs)
    with set_dit_quant_step(0, total_steps=8):
        with pytest.raises(RuntimeError, match="runtime uses 8 denoising steps"):
            quantized(inputs)
    with set_dit_quant_step(0, total_steps=4):
        assert quantized(inputs).shape == (2, 8)


def test_native_runtime_rejects_non_factorable_dit_channel_scales() -> None:
    linear = nn.Linear(16, 8)
    record = _record(linear, "dit")
    weight_packed, logical_width = pack_signed_int4(record.pop("weight_q"), pad_to=64)
    record.update(
        {
            "weight_packed": weight_packed,
            "weight_shape": [8, logical_width],
            "padded_in_features": weight_packed.shape[1] * 2,
            "activation_granularity": "static-per-step-per-channel",
        }
    )
    with pytest.raises(ValueError, match="cannot use one native INT4 GEMM"):
        NativeHoloQLinear(linear, name="dit", record=record)


def test_llm_runtime_uses_dynamic_per_token_activation_scale() -> None:
    linear = nn.Linear(16, 8)
    quantized = HoloQLinear(linear, name="llm", record=_record(linear, "llm"))
    inputs = torch.randn(2, 3, 16)
    assert quantized(inputs).shape == (2, 3, 8)


def test_patch_conv3d_lowering_uses_same_fake_w4a4_matrix_core() -> None:
    conv = nn.Conv3d(3, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    matrix = conv.weight.detach().flatten(1)
    weight_q, weight_scale = symmetric_quantize_per_output_channel(matrix)
    weight_packed, logical_width = pack_signed_int4(weight_q, pad_to=64)
    record = {
        "scope": "vit",
        "module_kind": "conv3d_patch",
        "solver": "rtn",
        "weight_packed": weight_packed,
        "weight_shape": [8, logical_width],
        "padded_in_features": weight_packed.shape[1] * 2,
        "weight_scale": weight_scale,
        "transform": "identity",
        "permutation": None,
        "rotation_blocks": None,
        "activation_granularity": "dynamic-per-token",
        "activation_scale": None,
        "bias": conv.bias.detach(),
    }
    quantized = HoloQPatchConv3d(conv, name="patch", record=record)
    inputs = torch.randn(4, 3, 2, 2, 2)
    assert quantized(inputs).shape == (4, 8, 1, 1, 1)


def test_full_pack_round_trip_covers_exactly_304_linears(tmp_path) -> None:
    model = _N1d7Shape(llm_layers=16, dit_layers=32)
    targets = discover_n1d7_targets(model)
    records = {}
    for target in targets:
        weight_q, weight_scale = symmetric_quantize_per_output_channel(target.module.weight)
        record = {
            "scope": target.scope,
            "solver": "gptq" if target.scope == "llm" else "rtn",
            "weight_q": weight_q,
            "weight_scale": weight_scale,
            "permutation": torch.arange(target.module.in_features),
            "rotation_blocks": torch.eye(target.module.in_features).unsqueeze(0),
            "activation_scale": (
                torch.full((4, target.module.in_features), 0.1) if target.scope == "dit" else None
            ),
        }
        record["sha256"] = tensor_record_sha256(record)
        records[target.name] = record

    pack_path = tmp_path / "object-w4a4.pt"
    torch.save(
        {
            "manifest": {
                "format_version": 1,
                "algorithm": "holoq-vla",
                "model_type": "Gr00tN1d7",
                "suite": "object",
                "weight_bits": 4,
                "activation_bits": 4,
                "llm_layers": 16,
                "dit_layers": 32,
                "num_inference_timesteps": 4,
                "config_sha256": model_config_sha256(model),
            },
            "layers": records,
        },
        pack_path,
    )
    checksum = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    pack_path.with_suffix(".pt.sha256").write_text(checksum + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="does not match requested"):
        apply_holoq_pack(_N1d7Shape(16, 32), pack_path, expected_suite="long")
    summary = apply_holoq_pack(model, pack_path, expected_suite="object")
    assert summary.total_linears == 304
    assert sum(isinstance(module, HoloQLinear) for module in model.modules()) == 304


def test_v2_pack_round_trip_covers_llm_dit_and_complete_vit_scope(tmp_path) -> None:
    model = _N1d7Shape(llm_layers=1, dit_layers=1, vit_layers=1)
    targets = discover_n1d7_targets(
        model,
        scopes="llm,dit,vit",
        include_vit_mergers=True,
        include_vit_patch_embed=True,
    )
    summary = validate_n1d7_scope(targets, scopes="llm,dit,vit")
    records = {}
    for target in targets:
        weight = target.module.weight.detach()
        matrix = weight.flatten(1) if isinstance(target.module, nn.Conv3d) else weight
        weight_q, weight_scale = symmetric_quantize_per_output_channel(matrix)
        weight_packed, logical_width = pack_signed_int4(weight_q, pad_to=64)
        record = {
            "scope": target.scope,
            "module_kind": target.module_kind,
            "projection": target.projection,
            "solver": "rtn",
            "weight_packed": weight_packed,
            "weight_shape": [matrix.shape[0], logical_width],
            "padded_in_features": weight_packed.shape[1] * 2,
            "weight_storage": "signed-int4-low-high-nibble",
            "weight_scale": weight_scale,
            "transform": "identity",
            "permutation": None,
            "rotation_blocks": None,
            "activation_granularity": "dynamic-per-token",
            "activation_scale": None,
            "bias": None if target.module.bias is None else target.module.bias.detach(),
        }
        record["sha256"] = tensor_record_sha256(record)
        records[target.name] = record
    pack_path = tmp_path / "all-scope-w4a4.pt"
    torch.save(
        {
            "manifest": {
                "format_version": 2,
                "algorithm": "holoq-vla",
                "model_type": "Gr00tN1d7",
                "suite": "object",
                "weight_bits": 4,
                "activation_bits": 4,
                "scopes": ["llm", "dit", "vit"],
                "include_vit_mergers": True,
                "include_vit_patch_embed": True,
                "llm_layers": 1,
                "dit_layers": 1,
                "vit_layers": 1,
                "num_inference_timesteps": 4,
                "config_sha256": model_config_sha256(model),
            },
            "layers": records,
        },
        pack_path,
    )
    pack_path.with_suffix(".pt.sha256").write_text(
        hashlib.sha256(pack_path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    loaded_summary = apply_holoq_pack(model, pack_path, expected_suite="object")
    assert loaded_summary == summary
    assert sum(isinstance(module, HoloQLinear) for module in model.modules()) == 21
    assert sum(isinstance(module, HoloQPatchConv3d) for module in model.modules()) == 1
