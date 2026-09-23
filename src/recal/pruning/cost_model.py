from __future__ import annotations

from typing import Any


def estimate_pruning_cost(config: Any, retained_intermediate_size: int) -> dict[str, float | int]:
    hidden = int(config.hidden_size)
    original = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    vocab = int(config.vocab_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    ffn_original = layers * 3 * hidden * original
    ffn_retained = layers * 3 * hidden * retained_intermediate_size
    # Decoder parameter proxy excluding small norms/biases.
    attention = layers * (hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2)
    embeddings = vocab * hidden
    total_proxy = embeddings + attention + ffn_original
    total_retained_proxy = embeddings + attention + ffn_retained
    # Per-token matmul FLOPs are two times parameter multiplies.
    ffn_flops_original = 2 * 3 * hidden * original * layers
    ffn_flops_retained = 2 * 3 * hidden * retained_intermediate_size * layers
    attention_decode_flops_proxy = 2 * attention
    total_decode_original = attention_decode_flops_proxy + ffn_flops_original
    total_decode_retained = attention_decode_flops_proxy + ffn_flops_retained
    return {
        "ffn_parameters_original": ffn_original,
        "ffn_parameters_retained": ffn_retained,
        "actual_total_parameter_reduction_proxy": 1.0 - total_retained_proxy / total_proxy,
        "theoretical_ffn_flops_reduction": 1.0 - ffn_flops_retained / ffn_flops_original,
        "theoretical_total_decode_flops_reduction_proxy": 1.0
        - total_decode_retained / total_decode_original,
    }


def exact_parameter_breakdown(model) -> dict[str, Any]:
    groups = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norm": 0,
        "lm_head": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if "embed_tokens" in name:
            key = "embeddings"
        elif any(part in name for part in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            key = "attention"
        elif any(part in name for part in ["gate_proj", "up_proj", "down_proj"]):
            key = "ffn"
        elif "norm" in name:
            key = "norm"
        elif "lm_head" in name:
            key = "lm_head"
        else:
            key = "other"
        groups[key] += count
    total = sum(groups.values())
    return {
        "total_parameters": total,
        "groups": groups,
        "fractions": {key: value / total for key, value in groups.items()},
    }

