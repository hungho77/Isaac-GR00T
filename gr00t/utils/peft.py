# SPDX-License-Identifier: Apache-2.0
"""LoRA wrapping for Gr00tN1d7, adapted from the equivalent utility in the
CLP_VLA (GR00T-N1.5) fork. Ported here so `--lora_rank > 0` can train small
low-rank adapters on the attention projections instead of the full weights,
cutting optimizer-state VRAM drastically compared to full finetuning.
"""

from peft import LoraConfig, get_peft_model
import torch


def _wrap_forward(model):
    """PeftModel dispatches through **kwargs; Gr00tN1d7.forward takes a single
    positional `inputs: dict`. Re-point forward at the original model's own
    logic so training still goes through prepare_input -> backbone -> action_head.
    """

    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        return model.action_head(backbone_outputs, action_inputs)

    model.forward = _forward
    return model


# Attention Q/K/V(+out): q_proj/k_proj/v_proj/o_proj is Qwen3-VL backbone's
# HF-style naming; to_q/to_k/to_v/to_out is diffusers' Attention module, used
# by both DiT and the vl_self_attention adapter.
_ATTN_PATTERNS = ["q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out"]

# FFN: "mlp" covers Qwen3's gate_proj/up_proj/down_proj (all live under an
# "...mlp...." prefix); "ff.net" covers diffusers FeedForward's GEGLU + output
# Linear inside BasicTransformerBlock (DiT / vl_self_attention).
_FFN_PATTERNS = ["mlp", "ff.net"]

# AdaLN modulation (DiT / vl_self_attention only, when norm_type="ada_norm"):
# AdaLayerNorm.linear projects the timestep embedding into per-block
# scale/shift. Qwen's plain RMSNorm has no such linear, so nothing to match
# on the backbone side.
_ADA_NORM_PATTERNS = ["norm1.linear"]


def get_lora_model(
    model,
    rank=32,
    lora_alpha=16,
    lora_dropout=0.1,
    action_head_only=True,
    include_ffn=True,
    include_ada_norm=True,
):
    """Wrap `model` (a Gr00tN1d7 instance, already pruned if --prune_model was
    used) with LoRA adapters.

    action_head_only=True (default) restricts LoRA to gr00t.model.action_head
    (DiT + vl_self_attention) -- pass False to also target the backbone
    (only useful combined with letting the backbone actually receive
    gradients some other way, e.g. tune_llm=True, since PeftModel freezes
    every non-LoRA param regardless of tune_* flags).

    include_ffn / include_ada_norm widen coverage beyond plain attention
    Q/K/V(+out): a heavily pruned module (e.g. DiT cut from 32 to 8 layers)
    may need more than attention-only LoRA to re-adapt -- the FFN typically
    holds most of a transformer block's parameters, and AdaLN modulation
    controls how strongly each block's timestep conditioning is applied.
    """
    patterns = list(_ATTN_PATTERNS)
    if include_ffn:
        patterns += _FFN_PATTERNS
    if include_ada_norm:
        patterns += _ADA_NORM_PATTERNS

    target_modules = []

    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue

        if isinstance(module, torch.nn.Linear):
            if any(p in name for p in patterns):
                target_modules.append(name)

    if not target_modules:
        raise ValueError(
            "get_lora_model found no matching Linear layers to wrap -- "
            f"action_head_only={action_head_only}. Check that the model has "
            "already been constructed (and pruned, if applicable) before "
            f"calling this, and that target module names still contain one of: {patterns}."
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        # None (not "CAUSAL_LM"): Gr00tN1d7 isn't a text-generation model and
        # has no prepare_inputs_for_generation, which PeftModelForCausalLM
        # requires at construction time. None -> generic PeftModel wrapper.
        task_type=None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = _wrap_forward(model)

    return model
