# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.quantization.packing import pack_signed_int4
import pytest
from tools.build_holoq_native_extension import _validate_quantized_codes
import torch


def _packed_with_padding(codes: torch.Tensor, padded_k: int = 128) -> torch.Tensor:
    packed, _ = pack_signed_int4(codes, pad_to=padded_k)
    return packed


def test_native_quantizer_validator_accepts_sparse_one_lsb_rounding() -> None:
    reference = torch.zeros((10, 100), dtype=torch.int8)
    actual = reference.clone()
    actual.view(-1)[:5] = 1
    result = _validate_quantized_codes(
        _packed_with_padding(actual), reference, padded_k=128
    )
    assert result == {
        "max_code_error": 1,
        "mismatch_count": 5,
        "mismatch_fraction": 0.005,
    }


def test_native_quantizer_validator_rejects_excessive_mismatch_rate() -> None:
    reference = torch.zeros((10, 100), dtype=torch.int8)
    actual = reference.clone()
    actual.view(-1)[:6] = 1
    with pytest.raises(AssertionError, match="exceeds rounding tolerance"):
        _validate_quantized_codes(_packed_with_padding(actual), reference, padded_k=128)


def test_native_quantizer_validator_rejects_more_than_one_lsb() -> None:
    reference = torch.zeros((10, 100), dtype=torch.int8)
    actual = reference.clone()
    actual[0, 0] = 2
    with pytest.raises(AssertionError, match="max_code_error=2"):
        _validate_quantized_codes(_packed_with_padding(actual), reference, padded_k=128)


def test_native_quantizer_validator_rejects_nonzero_padding() -> None:
    reference = torch.zeros((10, 100), dtype=torch.int8)
    padded = torch.zeros((10, 128), dtype=torch.int8)
    padded[0, 100] = 1
    packed, _ = pack_signed_int4(padded, pad_to=128)
    with pytest.raises(AssertionError, match="nonzero padding"):
        _validate_quantized_codes(packed, reference, padded_k=128)
