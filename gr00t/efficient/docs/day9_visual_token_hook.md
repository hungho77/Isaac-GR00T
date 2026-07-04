# Day 9 — Visual Token Hook

## Located Hook Path
- File: `gr00t/model/modules/qwen3_backbone.py`
- Class: `Qwen3Backbone`
- Function: `forward(self, vl_input)`
- Exposed tensor variable: `outputs = outputs.hidden_states[-1]`
- Returned field: `backbone_features`

The returned `BatchFeature` contains:
- `backbone_features`: VLM hidden states, shape `[B, T, D]`.
- `backbone_attention_mask`: valid-token mask, shape `[B, T]`.
- `image_mask`: image-token positions, shape `[B, T]`.

## Action Head Consumption
- File: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Class: `Gr00tN1d7ActionHead`
- Function: `_encode_features(...)`
- Consumed variable: `vl_embeds = backbone_output.backbone_features`
- The diffusion/action head uses `vl_embeds` as `encoder_hidden_states`.

## Tensor Notes
- Shape is `[B, T, D]`, not a pure visual-only `[B, N, D]` tensor.
- `T` includes both language and image-token positions.
- Visual-token positions are identified by `image_mask`.
- Dtype/device match the loaded model, typically bf16/fp16/fp32 on CUDA for real inference.
- Positional handling has already happened inside Qwen3-VL before this tensor is exposed.

## Pure Visual Token Availability
The repo wrapper does not name a pure post-projector visual tensor. Qwen3-VL internally computes image features before scattering them into language embeddings, but `Qwen3Backbone.forward()` calls `self.model(**vl_input, output_hidden_states=True)` directly and only exposes the final hidden states plus `image_mask`.

Deployment code in `scripts/deployment/trt_model_forward.py` shows the lower-level path:
- `inner_model.get_image_features(...)`
- `inputs_embeds.masked_scatter(image_mask, image_embeds)`
- language model forward

That path is useful for future deeper integration, but the safest initial hook is the exposed `backbone_features` plus `image_mask` just before the action head consumes it.

## Safest Initial Hook Point
Use an instance-level hook around `Gr00tN1d7ActionHead.process_backbone_output(...)`.

Contract:
```text
backbone_output.backbone_features
  -> select visual positions with backbone_output.image_mask
  -> EfficientInferenceMethod.process_visual_tokens()
  -> gather backbone_features, backbone_attention_mask, image_mask with matching indices
  -> action head
```

## Risk Notes
- Pruning only visual positions requires gathering all masks with the same full-sequence indices.
- Text tokens must be preserved.
- Baseline must keep the hook disabled and preserve exact behavior.
- If Qwen3-VL changes its output layout, the hook should fail closed and leave tensors unchanged.
- Because positional embeddings are already applied, pruning here removes tokens from the action head context rather than saving vision-encoder compute.
- If latency does not improve, the hook is likely after the dominant vision/VLM compute.
