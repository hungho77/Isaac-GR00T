# Native CKA pruning and deployment for GR00T N1.7

This branch supports the manifest contract emitted by the UR10e CKA notebooks:

```json
{
  "schema_version": 1,
  "model_type": "Gr00tN1d7",
  "modules": {
    "action_dit": {"original_depth": 32, "keep_indices": []},
    "backbone_language": {"original_depth": 16, "keep_indices": []},
    "vl_self_attention": {"original_depth": 4, "keep_indices": []}
  }
}
```

## Load-order contract

- Recovery from a full NVIDIA checkpoint: load full weights first, then apply the manifest.
- Loading a saved pruned checkpoint: its constructor reads `cka_pruning_manifest`, builds the
  reduced architecture, then Transformers loads the reduced state dict.
- `AlternateVLDiT` uses `_gr00t_original_index` after pruning. Arbitrary retained counts such
  as 6 DiT blocks are therefore supported, provided text-cross, image-cross and self-attention
  roles are all retained.

The implementation is deliberately not the group-of-four schema from the `yennt` branch.
Do not replace `cka_pruning_manifest` with `prune_model`/`kept_layer_idx_list_*` in a checkpoint.

## Offline checkpoint gate

Run this before moving the checkpoint to a robot server:

```bash
uv run python scripts/cka_n1d7/verify_native_checkpoint.py \
  --model-path /path/to/checkpoint \
  --expected-action-dit 6 \
  --expected-backbone-language 4 \
  --expected-vl-self-attention 2 \
  --expected-action-horizon 16
```

Change the three retained depths for 4/6/2, 4/4/4 or another notebook variant. The command
fails on missing/unexpected/mismatched weights, incorrect original-index routing, a wrong
variant, missing processor files, or an incompatible action horizon.

## Real-robot server

Use the fail-closed entry point instead of plain `run_gr00t_server.py`:

```bash
uv run python gr00t/eval/run_gr00t_server_cka_native_duc.py \
  --model-path /path/to/checkpoint \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/UR10eCup/ur10e_cup_config.py \
  --expected-action-dit 6 \
  --expected-backbone-language 4 \
  --expected-vl-self-attention 2 \
  --expected-action-horizon 16 \
  --host 0.0.0.0 --port 8791
```

The server refuses to start for an unpruned checkpoint or a checkpoint that does not match
the expected architecture. It prints a JSON preflight report before opening the ZMQ server.

The checkpoint processor remains the source of runtime modalities. For the included UR10e
pipeline the expected action contract is 16 steps with seven channels: six arm-joint targets
followed by the gripper target. Robot-side joint order, units and absolute/relative semantics
must still be checked against the controller before live motion.
