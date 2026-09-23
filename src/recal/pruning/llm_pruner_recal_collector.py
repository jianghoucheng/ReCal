"""Collect LLM-Pruner's Taylor ranking under ReCal token weighting.

    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      -m recal.pruning.llm_pruner_recal_collector --config <arm.yaml>

The third plug-and-play cell: the weighting is applied to the *loss*, not to an
activation statistic. LLM-Pruner's criterion is ``|dL/dw * w|`` summed over the FFN
group a channel owns, so what the token weighting changes is which positions the
training loss listens to. Everything downstream -- gradient accumulation, the Taylor
estimator, the mask builder, the pruner -- is shared with the unweighted arm, so the
only difference between the two rankings is the weighting. That is exactly the ablation.

Three passes per batch, mirroring the ReCal collector:

1. **Divergence.** The dense teacher and the probe score the same calibration
   prefixes; per-position ``KL(teacher || probe)`` comes out of
   ``batch_divergences``. Only the probe differs from the other weighted arms --
   it is the *llm_pruner* criterion's own pruned model, so the weighting measures
   "what this pruning criterion breaks", not some other arm's damage.

2. **Weighting.** ``trajectory_token_weights`` normalizes the divergences per
   trajectory, exactly as for the activation-weighted arms.

3. **Weighted loss.** One forward+backward on the dense teacher with per-token CE
   weighted by those weights: ``L = Σ_t w_t · CE_t / Σ_t w_t`` over response
   positions. The weight at position ``t`` scales the loss at position ``t``, the
   same slot the activation-weighted arms install it in -- so the gradient side of
   this ablation and the statistic side weight identical information.

The alignment is the subtle part, so it is pinned by tests: the divergence
computation shifts weights onto the activation position they describe, the CE at
position ``t`` uses the label at ``t+1``, and both use the same index ``t``. No
further shift is applied here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recal.common import (
    dump_json,
    load_yaml,
    set_seed,
    sha256_file,
    sha256_text,
    supports_enable_thinking,
    utc_now,
)
from recal.pruning.minitron_collector import init_distributed, model_shape_summary
from recal.pruning.recal_weighting import trajectory_token_weights
from recal.pruning.recal_collector import (
    batch_divergences,
    encode_batch,
    sample_calibration_trajectories,
)
from recal.pruning.llm_pruner_criteria import (
    SCORES,
    TaylorImportanceAccumulator,
    discover_ffn_groups,
    freeze_all_but_ffn,
)

DEFAULT_TOP_K = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    pruning = config["pruning"]
    settings = pruning.get("recal", {})
    importance_cfg = pruning.get("importance", {})
    scoring_cfg = importance_cfg.get("scoring", {})
    model_cfg = config["model"]

    primary_score = scoring_cfg.get("primary_score", "llm_pruner_taylor")
    if primary_score not in SCORES:
        raise ValueError(
            f"pruning.importance.scoring.primary_score={primary_score!r} is not one of "
            f"{SCORES}; this collector produces the weighted-Taylor criterion only"
        )
    pruning_method = str(pruning.get("method", "llm_pruner_recal"))
    output_dir = Path(settings.get("statistics_dir", pruning["importance_dir"]))

    probe_path = settings.get("probe_path")
    if not probe_path:
        raise ValueError(
            "pruning.recal.probe_path must point at the llm_pruner criterion's "
            "pruned checkpoint; the weighting is measured against *its* damage"
        )
    if not (Path(probe_path) / "config.json").exists():
        raise FileNotFoundError(f"Probe checkpoint is missing: {probe_path}")

    teacher_path = config["data"]["sft_teacher_path"]
    frame = pd.read_parquet(teacher_path)
    calibration = sample_calibration_trajectories(
        frame,
        per_domain=int(settings.get("trajectories_per_domain", 256)),
        seed=int(config.get("seed", 42)),
    )

    rank, world_size, local_rank = init_distributed()
    shard = calibration.iloc[rank::world_size].reset_index(drop=True)
    set_seed(int(config.get("seed", 42)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model_path = model_cfg.get("path", model_cfg.get("teacher_path"))
    revision = model_cfg.get("revision")
    tokenizer_revision = model_cfg.get("tokenizer_revision", revision)
    thinking = bool(model_cfg.get("thinking_enabled", True))

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if thinking and not supports_enable_thinking(tokenizer):
        raise RuntimeError("Tokenizer chat template does not support enable_thinking=True")

    attn = settings.get("attn_implementation", "sdpa")
    # fp32 on the teacher: the gradient pass needs it (bf16 products w*grad underflow
    # for the small-magnitude channels the ranking must order), and divergence logits
    # on the same model come from the same load rather than a second one.
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    ).to(device)
    teacher_model.eval()
    if hasattr(teacher_model, "gradient_checkpointing_enable"):
        teacher_model.gradient_checkpointing_enable()
    probe_model = AutoModelForCausalLM.from_pretrained(
        probe_path,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    ).to(device)
    probe_model.eval()
    summary = model_shape_summary(teacher_model)
    source_metadata_path = Path(model_path) / "recal_source.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )

    groups = discover_ffn_groups(teacher_model)
    freeze_all_but_ffn(teacher_model, groups)
    accumulator = TaylorImportanceAccumulator(
        groups,
        device=device,
        accumulator_dtype=(
            torch.float64
            if scoring_cfg.get("accumulator_dtype", "float32") == "float64"
            else torch.float32
        ),
    )

    top_k = int(settings.get("top_k", DEFAULT_TOP_K))
    logit_chunk = int(settings.get("logit_chunk", 512))
    # The *gradient* length governs, not `forward_kl.max_calibration_length`. Two
    # reasons, and they point the same way:
    #
    #   Correctness. This arm is an ablation against the unweighted llm_pruner arm,
    #   whose Taylor sums were accumulated at `gradient.sequence_length` (2048). If the
    #   weighted arm integrated over 8192-token prefixes instead, the two rankings would
    #   differ by sequence length as well as by the weighting, and the ablation would be
    #   confounded -- the one thing it exists to isolate.
    #
    #   Memory. Unlike the activation-weighted arms, this one runs a *backward* pass on
    #   an fp32 teacher while a bf16 probe sits on the same device. At 8192 the fp32
    #   logits are ~4.9 GB, doubled again by the contiguous view cross_entropy needs,
    #   and rank 5 died with "Tried to allocate 3.16 GiB ... 2.92 GiB is free" one batch
    #   in. The ReCal default of 8192 is right for the forward-only collectors and
    #   wrong here.
    #
    # `max_calibration_length` still applies as a ceiling, so an arm may shorten the
    # window but not silently widen it past what the gradient pass was budgeted for.
    gradient_length = int(
        importance_cfg.get("gradient", {}).get("sequence_length", 2048)
    )
    calibration_ceiling = settings.get("max_calibration_length")
    max_length = (
        min(gradient_length, int(calibration_ceiling))
        if calibration_ceiling
        else gradient_length
    )
    batch_size = int(settings.get("batch_size_per_gpu", 1))
    pad_id = tokenizer.pad_token_id

    collected_samples = 0
    collected_tokens = 0
    truncated_samples = 0
    for start in tqdm(range(0, len(shard), batch_size), disable=rank != 0):
        batch = shard.iloc[start : start + batch_size].to_dict("records")
        input_ids, attention_mask, response_mask, lengths = encode_batch(
            tokenizer,
            batch,
            thinking_enabled=thinking,
            max_length=max_length,
            pad_id=pad_id,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        response_mask = response_mask.to(device)

        divergences = batch_divergences(
            teacher_model,
            probe_model,
            input_ids,
            attention_mask,
            response_mask,
            top_k=top_k,
            logit_chunk=logit_chunk,
        )
        weights = trajectory_token_weights(divergences, response_mask)

        # Weighted next-token loss on the dense teacher. CE[t] uses label[t+1]; the
        # weight W[t] is already shifted onto the activation position it describes,
        # so the same index t aligns them -- do NOT shift the weights again.
        teacher_model.zero_grad(set_to_none=True)
        logits = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        labels = input_ids.masked_fill(~response_mask, -100)
        per_token = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(input_ids.shape[0], -1)
        ce_weights = weights[:, :-1]
        denom = ce_weights.sum().clamp_min(1e-9)
        loss = (per_token * ce_weights).sum() / denom
        loss.backward()
        accumulator.accumulate_from_gradients(batch_size=len(batch))
        teacher_model.zero_grad(set_to_none=True)
        del logits, per_token, loss

        collected_samples += len(batch)
        collected_tokens += int(response_mask.sum().item())
        # Truncation is measured from the *encoded* lengths, not by re-tokenizing a
        # `prompt` field: the calibration records carry `messages`, and a literal
        # `record["prompt"]` is invalid here because calibration records store
        # structured messages rather than a separate prompt field.
        truncated_samples += sum(1 for length in lengths if length >= max_length)

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
        # Create the directory BEFORE the save loop, not after it. This sat below the
        # `torch.save` below and every rank-0 write died with "Parent directory ...
        # llm_pruner_recal_statistics does not exist" -- after all 128 calibration batches
        # had already been collected. The gradient pass is the expensive part, so the
        # bug threw away a full 5-minute 8-GPU run at the very last step, twice.
        output_dir.mkdir(parents=True, exist_ok=True)
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
                "pruning_method": pruning_method,
                "available_scores": list(SCORES),
                "layers": orders,
            },
            output_dir / "channel_order.json",
        )
        dump_json(
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "criterion": primary_score,
                "pruning_method": pruning_method,
                "weighted_loss": "sum(w_t * CE_t) / sum(w_t) over response positions",
                "probe_path": probe_path,
                "model_path": model_path,
                "model_revision": source_metadata.get("revision", revision),
                "tokenizer_revision": source_metadata.get(
                    "tokenizer_revision", tokenizer_revision
                ),
                "config_sha256": sha256_text(
                    json.dumps(teacher_model.config.to_dict(), sort_keys=True)
                ),
                "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
                "calibration_path": teacher_path,
                "calibration_sha256": sha256_file(teacher_path),
                "world_size": world_size,
                "calibration_rows": int(local_summary[0].item()),
                "calibration_tokens": int(local_summary[1].item()),
                "truncated_rows": int(local_summary[2].item()),
                "sequence_length": max_length,
                "top_k": top_k,
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
