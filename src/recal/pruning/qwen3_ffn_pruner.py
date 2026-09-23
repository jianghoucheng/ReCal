from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recal.common import directory_fingerprint, dump_json, load_json, sha256_text, utc_now
from recal.pruning.cost_model import exact_parameter_breakdown


def discover_mlp_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Discover dense gated MLPs with gate/up/down projections.

    The implementation is architecture-structural rather than tied to a Qwen
    model class, so other dense decoder families with the same gated-MLP
    topology can share the pruning pipeline.
    """
    layers = []
    for name, module in model.named_modules():
        if all(hasattr(module, part) for part in ["gate_proj", "up_proj", "down_proj"]):
            gate, up, down = module.gate_proj, module.up_proj, module.down_proj
            if all(isinstance(piece, torch.nn.Linear) for piece in [gate, up, down]):
                if gate.out_features == up.out_features == down.in_features:
                    layers.append((name, module))
    if not layers:
        raise RuntimeError("Could not discover gated gate_proj/up_proj/down_proj MLP layers")
    return layers


def translate_original_to_local(
    current_original_ids: list[int],
    target_original_ids: list[int],
) -> list[int]:
    lookup = {original: local for local, original in enumerate(current_original_ids)}
    missing = [value for value in target_original_ids if value not in lookup]
    if missing:
        raise ValueError(f"Target mask is not a subset of source channels; first missing IDs: {missing[:16]}")
    return [lookup[value] for value in target_original_ids]


def _slice_linear_rows(module: torch.nn.Linear, indices: list[int]) -> None:
    index = torch.tensor(indices, dtype=torch.long, device=module.weight.device)
    module.weight = torch.nn.Parameter(module.weight.detach().index_select(0, index).contiguous())
    module.out_features = len(indices)
    if module.bias is not None:
        module.bias = torch.nn.Parameter(module.bias.detach().index_select(0, index).contiguous())


def _slice_linear_columns(module: torch.nn.Linear, indices: list[int]) -> None:
    index = torch.tensor(indices, dtype=torch.long, device=module.weight.device)
    module.weight = torch.nn.Parameter(module.weight.detach().index_select(1, index).contiguous())
    module.in_features = len(indices)


def prune_mlp_module(module: torch.nn.Module, local_indices: list[int]) -> None:
    gate, up, down = module.gate_proj, module.up_proj, module.down_proj
    if not (gate.out_features == up.out_features == down.in_features):
        raise ValueError("Inconsistent source MLP shapes")
    _slice_linear_rows(gate, local_indices)
    _slice_linear_rows(up, local_indices)
    _slice_linear_columns(down, local_indices)
    if hasattr(module, "intermediate_size"):
        module.intermediate_size = len(local_indices)
    if not (gate.out_features == up.out_features == down.in_features == len(local_indices)):
        raise AssertionError("Post-pruning MLP shapes are inconsistent")


def _source_original_mapping(
    source_model_path: Path,
    num_layers: int,
    current_size: int,
) -> dict[int, list[int]]:
    manifest_path = source_model_path / "pruning_manifest.json"
    if not manifest_path.exists():
        return {idx: list(range(current_size)) for idx in range(num_layers)}
    manifest = load_json(manifest_path)
    values = manifest["retained_original_channel_ids_per_layer"]
    mapping = {int(idx): list(map(int, ids)) for idx, ids in values.items()}
    if len(mapping) != num_layers or any(len(ids) != current_size for ids in mapping.values()):
        raise ValueError("Source pruning_manifest does not match checkpoint architecture")
    return mapping


def _cast_model(model, dtype: torch.dtype) -> None:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)


def prune_qwen3_ffn_width(
    source_model_path: str,
    mask_path: str,
    output_model_path: str,
    output_dtype: str = "bfloat16",
) -> None:
    source = Path(source_model_path)
    output = Path(output_model_path)
    mask = load_json(mask_path)
    dtype = getattr(torch, output_dtype)
    config_kwargs: dict[str, Any] = {}
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        **config_kwargs,
    )
    if getattr(model.config, "quantization_config", None):
        raise ValueError("Physical width pruning of a quantized checkpoint is unsupported; dequantize first")
    source_model_type = str(getattr(model.config, "model_type", "unknown"))
    source_breakdown = exact_parameter_breakdown(model)
    parent_manifest_path = source / "pruning_manifest.json"
    if parent_manifest_path.exists():
        parent_manifest = load_json(parent_manifest_path)
        base_breakdown = parent_manifest.get(
            "base_parameter_breakdown",
            parent_manifest.get("source_parameter_breakdown", source_breakdown),
        )
    else:
        base_breakdown = source_breakdown
    layers = discover_mlp_layers(model)
    current_size = layers[0][1].gate_proj.out_features
    if any(module.gate_proj.out_features != current_size for _, module in layers):
        raise ValueError("This implementation requires a uniform current FFN width across layers")
    current_mapping = _source_original_mapping(source, len(layers), current_size)
    target_mapping = {
        int(idx): list(map(int, ids))
        for idx, ids in mask["retained_original_channel_ids_per_layer"].items()
    }
    if set(target_mapping) != set(range(len(layers))):
        raise ValueError(f"Mask layers {sorted(target_mapping)} do not match checkpoint layers 0..{len(layers)-1}")
    new_size = int(mask["intermediate_size"])
    if any(len(ids) != new_size for ids in target_mapping.values()):
        raise ValueError("Mask has inconsistent per-layer retained sizes")
    layer_shapes = {}
    for idx, (name, module) in enumerate(layers):
        local_indices = translate_original_to_local(current_mapping[idx], target_mapping[idx])
        prune_mlp_module(module, local_indices)
        layer_shapes[str(idx)] = {
            "module": name,
            "gate_proj": list(module.gate_proj.weight.shape),
            "up_proj": list(module.up_proj.weight.shape),
            "down_proj": list(module.down_proj.weight.shape),
        }
    model.config.intermediate_size = new_size
    if hasattr(model, "generation_config"):
        model.generation_config._from_model_config = False
    _cast_model(model, dtype)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    tokenizer.save_pretrained(output)
    generation_config = source / "generation_config.json"
    if generation_config.exists() and not (output / "generation_config.json").exists():
        shutil.copy2(generation_config, output / "generation_config.json")
    # Preserve standalone templates or processor metadata not emitted by tokenizer.save_pretrained.
    for name in ["chat_template.jinja", "tokenizer_config.json", "special_tokens_map.json"]:
        candidate = source / name
        if candidate.exists() and not (output / name).exists():
            shutil.copy2(candidate, output / name)
    breakdown = exact_parameter_breakdown(model)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_model_path": str(source),
        "source_model_type": source_model_type,
        "source_checkpoint_hash": directory_fingerprint(source),
        "mask_path": mask_path,
        "mask": {
            key: mask.get(key)
            for key in [
                "ratio_basis",
                "target_total_parameter_reduction",
                "required_ffn_width_pruning_ratio",
                "actual_ffn_pruning_ratio",
                "actual_total_parameter_reduction",
                "original_intermediate_size",
                "intermediate_size",
                "hardware_multiple",
            ]
        },
        "output_dtype": output_dtype,
        "config_sha256": sha256_text(json.dumps(model.config.to_dict(), sort_keys=True)),
        "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
        "num_hidden_layers": len(layers),
        "intermediate_size_before": current_size,
        "intermediate_size_after": new_size,
        "retained_original_channel_ids_per_layer": {
            str(idx): ids for idx, ids in target_mapping.items()
        },
        "layer_shapes": layer_shapes,
        "parameter_breakdown": breakdown,
        "source_parameter_breakdown": source_breakdown,
        "base_parameter_breakdown": base_breakdown,
        "actual_total_parameter_reduction_from_source": 1.0
        - breakdown["total_parameters"] / source_breakdown["total_parameters"],
        "actual_total_parameter_reduction_from_original": 1.0
        - breakdown["total_parameters"] / base_breakdown["total_parameters"],
        "dense_physical_checkpoint": True,
        "mask_wrapper_present": False,
    }
    dump_json(manifest, output / "pruning_manifest.json")
    # Verify the written checkpoint can be reconstructed and its shapes match.
    del model
    reloaded = AutoModelForCausalLM.from_pretrained(
        output,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    reloaded_layers = discover_mlp_layers(reloaded)
    if len(reloaded_layers) != len(layers):
        raise AssertionError("Reloaded checkpoint layer count changed")
    for _, module in reloaded_layers:
        if not (
            module.gate_proj.out_features
            == module.up_proj.out_features
            == module.down_proj.in_features
            == new_size
        ):
            raise AssertionError("Reloaded checkpoint has an invalid FFN shape")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--output-dtype", default="bfloat16")
    args = parser.parse_args()
    prune_qwen3_ffn_width(
        args.source_model_path,
        args.mask_path,
        args.output_model_path,
        args.output_dtype,
    )


if __name__ == "__main__":
    main()
