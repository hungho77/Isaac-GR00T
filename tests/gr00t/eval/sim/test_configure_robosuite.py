# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned robosuite 1.4.0 compatibility setup."""

from __future__ import annotations

import warnings

from gr00t.eval.sim.LIBERO.configure_robosuite import configure_robosuite


def test_configure_robosuite_is_targeted_and_idempotent(tmp_path):
    package_dir = tmp_path / "robosuite"
    package_dir.mkdir()
    init_path = package_dir / "__init__.py"
    macros_path = package_dir / "macros.py"
    init_path.write_text('__version__ = "1.4.0"\n__logo__ = """\\ logo"""\n', encoding="utf-8")
    macros_path.write_text("SIMULATION_TIMESTEP = 0.002\n", encoding="utf-8")

    first = configure_robosuite(package_dir)

    assert first == {"logo_patched": True, "macros_created": True}
    assert '__logo__ = r"""' in init_path.read_text(encoding="utf-8")
    assert (package_dir / "macros_private.py").read_bytes() == macros_path.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(init_path.read_text(encoding="utf-8"), str(init_path), "exec")

    second = configure_robosuite(package_dir)

    assert second == {"logo_patched": False, "macros_created": False}
