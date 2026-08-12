# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for scalar and vector simulator success values."""

from gr00t.eval.rollout_policy import _coerce_success
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        (np.bool_(True), True),
        ([False, True], True),
        (np.array([False, False]), False),
        (None, False),
    ],
)
def test_coerce_success(value, expected):
    assert _coerce_success(value) is expected


def test_coerce_success_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown success dtype"):
        _coerce_success({"success": True})
