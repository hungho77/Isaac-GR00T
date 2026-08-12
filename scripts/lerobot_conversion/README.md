# Converting from LeRobot v3 to v2

## Setup

### 1. Create and Activate Virtual Environment

Run these from the `scripts/lerobot_conversion` directory (it has its own
`pyproject.toml`; installing from the repo root would install the `gr00t`
package instead):
```bash
cd scripts/lerobot_conversion
uv venv
source .venv/bin/activate
uv pip install -e . --verbose
```

### 2. Run Conversion Script

Inside the uv environment, run:
```bash
python convert_v3_to_v2.py --repo-id BobShan/double_folding_towel_v3.0
```

## Validated robot-specific recipes

The generic converter reconstructs LeRobot v2 layout but cannot infer a
robot's action semantics or GR00T modality slices. For UR10e + Robotiq cup
picking, use the checked wrapper in
[`examples/UR10eCup`](../../examples/UR10eCup/README.md). It pins the source
revision, installs `modality.json`, and verifies the absolute action alignment
for every converted episode.

> **Note:** You may need to install lerobot with `GIT_LFS_SKIP_SMUDGE=1`:
> 
> ```bash
> GIT_LFS_SKIP_SMUDGE=1 uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
> ```
