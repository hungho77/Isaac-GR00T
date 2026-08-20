# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply deterministic Python 3.12 compatibility fixes to robosuite 1.4.0.

LIBERO pins robosuite 1.4.0. That release emits a warning until a user-local
``macros_private.py`` exists, and its ASCII logo contains an invalid Python
escape sequence. The latter is reported as a ``SyntaxWarning`` on Python 3.12.

This helper runs after installing the isolated LIBERO environment and before
the first robosuite import. It deliberately patches only those two upstream
files and fails closed if their expected structure changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


_LOGO_ASSIGNMENT = '__logo__ = """'
_RAW_LOGO_ASSIGNMENT = '__logo__ = r"""'


def configure_robosuite(package_dir: Path) -> dict[str, bool]:
    """Configure one installed robosuite package and return applied changes."""

    package_dir = package_dir.resolve()
    init_path = package_dir / "__init__.py"
    macros_path = package_dir / "macros.py"
    macros_private_path = package_dir / "macros_private.py"
    for required in (init_path, macros_path):
        if not required.is_file():
            raise FileNotFoundError(f"Expected robosuite file is missing: {required}")

    init_source = init_path.read_text(encoding="utf-8")
    logo_patched = False
    if _RAW_LOGO_ASSIGNMENT not in init_source:
        occurrences = init_source.count(_LOGO_ASSIGNMENT)
        if occurrences != 1:
            raise RuntimeError(
                f"Expected exactly one robosuite logo assignment in {init_path}, found {occurrences}"
            )
        init_path.write_text(
            init_source.replace(_LOGO_ASSIGNMENT, _RAW_LOGO_ASSIGNMENT, 1),
            encoding="utf-8",
        )
        logo_patched = True

    macros_created = not macros_private_path.is_file()
    if macros_created:
        # Equivalent to robosuite/scripts/setup_macros.py, but does not import
        # robosuite first and therefore does not emit the warning it fixes.
        shutil.copyfile(macros_path, macros_private_path)

    return {"logo_patched": logo_patched, "macros_created": macros_created}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()
    changes = configure_robosuite(args.package_dir)
    print(
        "Configured robosuite:",
        f"package={args.package_dir.resolve()}",
        f"logo_patched={changes['logo_patched']}",
        f"macros_created={changes['macros_created']}",
    )


if __name__ == "__main__":
    main()
