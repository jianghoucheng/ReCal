"""Collect LLM-Pruner's Taylor channel ranking.

    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      -m recal.pruning.llm_pruner_collector --config <arm.yaml>

Writes the same ``channel_order.json`` contract as the activation collectors, so the mask
builder and pruner are shared and the width target rounds identically across arms.

Unlike those collectors this one needs a **backward** pass: the criterion is
``|dL/dw * w|``, so gradients of the FFN projections are the quantity being measured. Two
consequences shape the loop below.

**Shorter sequences.** Backward retains activations, so the 32k length the forward-only
collectors use will not fit. ``pruning.importance.gradient.sequence_length`` defaults to
2,048 -- LLM-Pruner itself calibrates on 128-token segments, so this is already generous,
and the manifest records it so the difference from the other arms is on the record rather
than buried.

**Loss is the model's own LM loss.** The calibration pool stores prompts without reference
responses, and LLM-Pruner likewise backprops next-token loss over raw calibration text, so
nothing is missing. Labels are the input ids with padding masked to ``-100`` -- leaving
padding in would let the loss reward predicting pad tokens and pollute the gradient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from recal.pruning.minitron_collector import init_distributed, model_shape_summary
from recal.pruning.llm_pruner_criteria import (
    SCORES,
    TaylorImportanceAccumulator,
    discover_ffn_groups,
    freeze_all_but_ffn,
)

# LLM-Pruner calibrates on 128-token segments; 2,048 is well above that and still fits a
# backward pass beside an 8B model on one 80 GB device.
DEFAULT_GRADIENT_SEQUENCE_LENGTH = 2048


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    pruning = cfg.get("pruning", {})
    importance_cfg = pruning.get("importance", {})
    scoring_cfg = importance_cfg.get("scoring", {})
    gradient_cfg = importance_cfg.get("gradient", {})
    model_cfg = cfg["model"]

    primary_score = scoring_cfg.get("primary_score", "llm_pruner_taylor")
    if primary_score not in SCORES:
        raise ValueError(
            f"pruning.importance.scoring.primary_score={primary_score!r} is not one of "
            f"{SCORES}; this collector produces the LLM-Pruner criterion only"
        )

    # Stamped onto channel_order.json so the mask carries its own provenance.
    pruning_method = str(pruning.get("method", primary_score))

    calibration_path = pruning.get(
        "calibration_path", cfg.get("data", {}).get("calibration_path")
    )
    if not calibration_path:
        raise ValueError("pruning.calibration_path is required")
    output_dir = Path(
        pruning.get("importance_dir", cfg.get("output_dir", "artifacts/pruning_importance"))
    )

    rank, world_size, local_rank = init_distributed()
    set_seed(int(cfg.get("seed", 42)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model_path = model_cfg.get("path", model_cfg.get("student_path"))
    revision = model_cfg.get("revision")
    tokenizer_revision = model_cfg.get("tokenizer_revision", revision)
    thinking_enabled = bool(model_cfg.get("thinking_enabled", True))

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if thinking_enabled and not supports_enable_thinking(tokenizer):
        raise RuntimeError(
            "Official tokenizer chat template does not support enable_thinking=True"
        )

    # float32 for the backward pass. In bf16 the product w*grad underflows for the
    # smallest-magnitude channels, which are exactly the ones the ranking must order.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation=importance_cfg.get("attn_implementation", "eager"),
    ).to(device)
    model.eval()
    # Checkpointing trades compute for the activation memory a backward pass needs.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    summary = model_shape_summary(model)
    source_metadata_path = Path(model_path) / "recal_source.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )

    groups = discover_ffn_groups(model)
    freeze_all_but_ffn(model, groups)
    accumulator = TaylorImportanceAccumulator(
        groups,
        device=device,
        accumulator_dtype=(
            torch.float64
            if scoring_cfg.get("accumulator_dtype", "float32") == "float64"
            else torch.float32
        ),
    )

    frame = pd.read_parquet(calibration_path)
    shard = frame.iloc[rank::world_size].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_length = int(
        gradient_cfg.get("sequence_length", DEFAULT_GRADIENT_SEQUENCE_LENGTH)
    )
    batch_size = int(gradient_cfg.get("batch_size_per_gpu", 1))

    collected_samples = 0
    collected_tokens = 0
    truncated_samples = 0
    for start in tqdm(range(0, len(shard), batch_size), disable=rank != 0):
        batch = shard.iloc[start : start + batch_size].to_dict("records")
        prompts = [
            render_chat(
                tokenizer,
                as_messages(row["messages"])
                if row.get("messages") is not None
                else [{"role": "user", "content": row["prompt"]}],
                thinking_enabled=thinking_enabled,
            )
            for row in batch
        ]
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
        # -100 on padding: without it the LM loss rewards predicting pad tokens and the
        # gradient reflects the padding pattern rather than the calibration text.
        labels = input_ids.masked_fill(attention_mask == 0, -100)

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        accumulator.accumulate_from_gradients(batch_size=len(batch))
        model.zero_grad(set_to_none=True)

        collected_samples += len(batch)
        collected_tokens += int(attention_mask.sum().item())
        truncated_samples += sum(
            len(tokenizer(prompt, add_special_tokens=False).input_ids) > sequence_length
            for prompt in prompts
        )

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
                "module_name": f"layer_{idx}_mlp",
                "original_channel_ids": torch.arange(
                    layer[primary_score].numel(), dtype=torch.int64
                ),
            }
            torch.save(payload, output_dir / f"layer_{idx:03d}.pt")
            orders[str(idx)] = torch.argsort(
                layer[primary_score], descending=True
            ).tolist()
        dump_json(
            {
                "schema_version": 1,
                "primary_score": primary_score,
                "available_scores": list(SCORES),
                "layers": orders,
                # Provenance for the mask built from this order. nested_masks.py copies
                # it onto the mask and preflight refuses an arm whose mask was built by
                # a different method -- without it the key defaults to "activation" and
                # this arm's mask is indistinguishable from the Minitron baseline's
                # after 60 GPU-hours of recovery.
                "pruning_method": pruning_method,
            },
            output_dir / "channel_order.json",
        )
        dump_json(
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "criterion": primary_score,
                # Record the integration choices used by the shared setup.
                "llm_pruner_variant_notes": {
                    "recovery": "shared SFT+OPD budget",
                    "uniform_layer_width": True,
                    "gradient_sequence_length": sequence_length,
                    "reason": (
                        "every arm prunes to the same per-layer width and spends the same "
                        "recovery budget, so the table compares pruning criteria rather "
                        "than whole pipelines; the shorter sequence length is a backward-"
                        "pass memory constraint"
                    ),
                },
                "model_path": model_path,
                "model_revision": source_metadata.get("revision", revision),
                "tokenizer_revision": source_metadata.get(
                    "tokenizer_revision", tokenizer_revision
                ),
                "config_sha256": sha256_text(
                    json.dumps(model.config.to_dict(), sort_keys=True)
                ),
                "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
                "calibration_path": calibration_path,
                "calibration_sha256": sha256_file(calibration_path),
                "world_size": world_size,
                "calibration_rows": int(local_summary[0].item()),
                "calibration_tokens": int(local_summary[1].item()),
                "truncated_rows": int(local_summary[2].item()),
                "sequence_length": sequence_length,
                "gradient": gradient_cfg,
                "importance": scoring_cfg,
                "model_summary": summary,
            },
            output_dir / "calibration_manifest.json",
        )
        print(json.dumps({"criterion": primary_score, **summary}, indent=2))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
