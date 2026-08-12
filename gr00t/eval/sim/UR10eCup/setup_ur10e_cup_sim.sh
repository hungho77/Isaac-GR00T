#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep the core GR00T lockfile unchanged: MuJoCo is needed only by this proxy
# simulator.  Resolve the project interpreter explicitly because `uv pip`
# otherwise selected Kaggle's `/usr` environment even when
# UV_PROJECT_ENVIRONMENT pointed at the GR00T venv.
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  project_python="${UV_PROJECT_ENVIRONMENT}/bin/python"
else
  project_python="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
fi

if [[ ! -x "${project_python}" ]]; then
  echo "GR00T project Python is missing or not executable: ${project_python}" >&2
  exit 1
fi

uv pip install --python "${project_python}" "mujoco>=3.1,<4"

"${project_python}" -c \
  "import gymnasium, mujoco; print('UR10e Cup simulator dependencies are ready')"
