from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recal.common import (
    dump_json,
    load_yaml,
    render_chat,
    set_seed,
    sha256_file,
    sha256_text,
    supports_enable_thinking,
    utc_now,
)
from recal.modes import as_messages
from recal.pruning.importance import FFNImportanceAccumulator, register_down_projection_hooks


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


@torch.inference_mode()
def prefill_and_collect(
    model,
    tokenizer,
    prompts: list[str],
    accumulator: FFNImportanceAccumulator,
    *,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Run ModelOpt-equivalent calibration prefills.

    The official Minitron path performs one logits-free prefill over each
    tokenized calibration sequence. It does not autoregressively generate a
    response. We retain that behavior here while using Hugging Face modules.
    """
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    accumulator.set_token_mask(attention_mask)
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    accumulator.set_token_mask(None)
    token_counts = attention_mask.sum(dim=1).detach().cpu().tolist()
    return [
        {
            "calibration_tokens": int(count),
            "truncated": len(
                tokenizer(prompt, add_special_tokens=False).input_ids
            )
            > sequence_length,
        }
        for prompt, count in zip(prompts, token_counts)
    ]


def model_shape_summary(model) -> dict[str, Any]:
    config = model.config
    total = sum(p.numel() for p in model.parameters())
    ffn = 0
    module_counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        root = name.split(".")[-2] if "." in name else name
        module_counts[root] = module_counts.get(root, 0) + param.numel()
        if any(piece in name for piece in [".gate_proj.", ".up_proj.", ".down_proj."]):
            ffn += param.numel()
    return {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "total_parameters": total,
        "ffn_parameters": ffn,
        "ffn_fraction": ffn / total,
        "module_parameter_counts": module_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    pruning = cfg.get("pruning", {})
    importance_cfg = pruning.get("importance", {})
    model_cfg = cfg["model"]
    generation_cfg = importance_cfg.get("generation", {})
    scoring_cfg = importance_cfg.get("scoring", {})
    calibration_path = pruning.get(
        "calibration_path",
        cfg.get("data", {}).get("calibration_path"),
    )
    output_path = pruning.get(
        "importance_dir",
        cfg.get("output_dir", "artifacts/pruning_importance"),
    )
    if not calibration_path:
        raise ValueError("pruning.calibration_path is required")
    rank, world_size, local_rank = init_distributed()
    set_seed(int(cfg.get("seed", 42)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    model_path = model_cfg.get("path", model_cfg.get("student_path"))
    revision = model_cfg.get("revision")
    tokenizer_revision = model_cfg.get("tokenizer_revision", revision)
    thinking_enabled = bool(model_cfg.get("thinking_enabled", True))
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if thinking_enabled and not supports_enable_thinking(tokenizer):
        raise RuntimeError("Official tokenizer chat template does not support enable_thinking=True")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=importance_cfg.get("attn_implementation", "eager"),
    ).to(device)
    model.eval()
    summary = model_shape_summary(model)
    source_metadata_path = Path(model_path) / "recal_source.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.exists() else {}
    )
    layer_sizes = {
        idx: module.in_features
        for idx, (_name, module) in enumerate(
            (item for item in model.named_modules() if item[0].endswith("down_proj"))
        )
    }
    accumulator_dtype = (
        torch.float64
        if scoring_cfg.get("accumulator_dtype", "float32") == "float64"
        else torch.float32
    )
    accumulator = FFNImportanceAccumulator(layer_sizes, device=device, accumulator_dtype=accumulator_dtype)
    handles, hook_names = register_down_projection_hooks(model, accumulator)

    frame = pd.read_parquet(calibration_path)
    shard = frame.iloc[rank::world_size].reset_index(drop=True)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_records = output_dir / f"calibration_records_rank{rank:03d}.jsonl"
    sequence_length = int(
        generation_cfg.get(
            "sequence_length",
            generation_cfg.get("max_new_tokens", 512),
        )
    )
    batch_size = int(generation_cfg.get("batch_size_per_gpu", 1))
    collected_tokens = 0
    collected_samples = 0
    truncated_samples = 0
    with local_records.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(shard), batch_size), disable=rank != 0):
            batch = shard.iloc[start : start + batch_size].to_dict("records")
            prompts = [
                render_chat(
                    tokenizer,
                    # `or` cannot be used to default this column. Arrow round-trips
                    # `messages` as a numpy array of dicts, and `array or fallback`
                    # evaluates the array's truth value -- "The truth value of an
                    # array with more than one element is ambiguous". An older
                    # calibration parquet happened to store the column as a Python
                    # list, so the same line worked there and fails here on data
                    # that is equally valid. as_messages normalizes both shapes.
                    as_messages(row["messages"])
                    if row.get("messages") is not None
                    else [{"role": "user", "content": row["prompt"]}],
                    thinking_enabled=thinking_enabled,
                )
                for row in batch
            ]
            records = prefill_and_collect(
                model,
                tokenizer,
                prompts,
                accumulator,
                sequence_length=sequence_length,
            )
            for row, record in zip(batch, records):
                record = {
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    **record,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected_samples += 1
                collected_tokens += record["calibration_tokens"]
                truncated_samples += int(record["truncated"])
            f.flush()

    for handle in handles:
        handle.remove()
    accumulator.all_reduce()
    stats = accumulator.finalize()
    local_summary = torch.tensor(
        [collected_samples, collected_tokens, truncated_samples],
        dtype=torch.long,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_summary, op=dist.ReduceOp.SUM)
    if rank == 0:
        orders = {}
        for idx, layer in stats.items():
            payload = {
                **layer,
                "layer_idx": idx,
                "module_name": hook_names[idx],
                "original_channel_ids": torch.arange(
                    layer[scoring_cfg.get("primary_score", "modelopt_minitron")].numel(),
                    dtype=torch.int64,
                ),
            }
            torch.save(payload, output_dir / f"layer_{idx:03d}.pt")
            score_name = scoring_cfg.get("primary_score", "modelopt_minitron")
            order = torch.argsort(layer[score_name], descending=True).tolist()
            orders[str(idx)] = order
        dump_json(
            {
                "schema_version": 1,
                "primary_score": scoring_cfg.get(
                    "primary_score", "modelopt_minitron"
                ),
                "pruning_method": str(pruning.get("method", "minitron")),
                "layers": orders,
            },
            output_dir / "channel_order.json",
        )
        dump_json(
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "model_path": model_path,
                "model_revision": source_metadata.get("revision", revision),
                "tokenizer_revision": source_metadata.get("tokenizer_revision", tokenizer_revision),
                "config_sha256": sha256_text(json.dumps(model.config.to_dict(), sort_keys=True)),
                "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
                "calibration_path": calibration_path,
                "calibration_sha256": sha256_file(calibration_path),
                "world_size": world_size,
                "calibration_rows": int(local_summary[0].item()),
                "calibration_tokens": int(local_summary[1].item()),
                "truncated_rows": int(local_summary[2].item()),
                "sequence_length": sequence_length,
                "records": "calibration_records_rank*.jsonl",
                "generation": generation_cfg,
                "importance": scoring_cfg,
                "model_summary": summary,
                "hook_modules": hook_names,
            },
            output_dir / "calibration_manifest.json",
        )
        print(json.dumps(summary, indent=2))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
