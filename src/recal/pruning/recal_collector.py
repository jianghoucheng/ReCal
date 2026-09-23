"""Calibrate FFN channel importance weighted by pruning-induced forward KL.

The Minitron baseline in ``minitron_collector.py`` is left untouched. This
module changes exactly one thing about it: how much each calibration position
counts toward a channel's score.

**The question being asked.** Baseline activation pruning ranks channels by mean
absolute activation over calibration tokens, weighting every position equally.
That treats all reasoning states as equally worth spending FFN capacity on. But
every pruned model here is followed by SFT and OPD, so the objective is a student
that recovers well -- and the states worth protecting are the ones ordinary pruning
*already breaks*. Measuring that requires a probe: a throwaway model pruned by the
baseline criterion at the target ratio, whose disagreement with the dense teacher
localizes the damage.

**Three passes over the same prefixes.** Pass 1 collects dense teacher logits,
pass 2 collects probe logits on byte-identical prefixes, and pass 3 re-runs the
dense model to accumulate activations with the resulting weights. The prefixes must
match exactly across passes or the divergence would be attributed to the wrong
state, so they are tokenized once and reused. Pass 3 is a separate forward rather
than a hook on pass 1 because the weights are not known until both logit passes
have completed.

**Why teacher forcing and never student rollout.** A probe left to generate its own
continuations would pick its own easy states: a repetition loop scores high
agreement precisely where pruning did the most damage. Conditioning both models on
the dense teacher's own trajectories fixes the state distribution across arms and
makes the signal independent of probe quality.

Calibration data is the existing SFT teacher parquet -- the same trajectories the
SFT stage trains on -- so no generation pass is needed and the calibration
distribution is by construction the recovery distribution.
"""

from __future__ import annotations

import argparse
import json
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
    set_seed,
    sha256_file,
    sha256_text,
    supports_enable_thinking,
    utc_now,
)
from recal.modes import as_messages
from recal.pruning.minitron_collector import init_distributed, model_shape_summary
from recal.pruning.recal_weighting import (
    DEFAULT_TOP_K,
    token_forward_kl,
    trajectory_token_weights,
)
from recal.pruning.baseline_criteria import (
    SCORES as BASELINE_SCORES,
    BaselineImportanceAccumulator,
    down_projection_column_norms,
)
from recal.pruning.importance import (
    FFNImportanceAccumulator,
    register_down_projection_hooks,
)
from recal.pruning.trajectory_calibration import (
    response_token_mask,
    sample_calibration_trajectories,
)


def encode_batch(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    thinking_enabled: bool,
    max_length: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Tokenize a batch once so every pass conditions on identical prefixes."""
    encoded = [
        response_token_mask(
            tokenizer,
            as_messages(record["messages"])[:-1]
            if as_messages(record["messages"])[-1].get("role") == "assistant"
            else as_messages(record["messages"]),
            record["teacher_response"],
            thinking_enabled=thinking_enabled,
            max_length=max_length,
        )
        for record in records
    ]
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), width), dtype=torch.long)
    response_mask = torch.zeros((len(records), width), dtype=torch.bool)
    for row, (ids, mask) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        response_mask[row, : len(mask)] = torch.tensor(mask, dtype=torch.bool)
    return input_ids, attention_mask, response_mask, [len(ids) for ids, _ in encoded]


@torch.inference_mode()
def batch_divergences(
    teacher_model,
    probe_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    top_k: int,
    logit_chunk: int,
) -> torch.Tensor:
    """Per-position ``KL(teacher || probe)`` for one batch of teacher prefixes.

    Logits are consumed in sequence chunks: a full ``[batch, seq, 151k]`` fp32
    tensor at 8k context is tens of gigabytes, and only the teacher's top-k plus a
    residual are ever needed. Chunking keeps peak memory proportional to
    ``logit_chunk`` instead of sequence length.
    """
    device = input_ids.device
    batch, width = input_ids.shape
    divergences = torch.zeros((batch, width), dtype=torch.float32, device=device)

    teacher_out = teacher_model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits
    probe_out = probe_model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits
    if teacher_out.shape[:2] != probe_out.shape[:2]:
        raise RuntimeError(
            "Teacher and probe disagree on sequence layout: "
            f"{tuple(teacher_out.shape)} vs {tuple(probe_out.shape)}"
        )

    for start in range(0, width, logit_chunk):
        stop = min(start + logit_chunk, width)
        divergences[:, start:stop] = token_forward_kl(
            teacher_out[:, start:stop, :],
            probe_out[:, start:stop, :],
            top_k=top_k,
        )
    del teacher_out, probe_out
    # The divergence at a position predicts the *next* token, so the weight for
    # scoring position t must come from the prefix ending at t-1. Shifting keeps the
    # weight aligned with the activation that produced the mismatched prediction.
    shifted = torch.zeros_like(divergences)
    shifted[:, 1:] = divergences[:, :-1]
    return shifted * response_mask


@torch.inference_mode()
def collect_weighted_statistics(
    teacher_model,
    probe_model,
    tokenizer,
    records: list[dict[str, Any]],
    accumulator: FFNImportanceAccumulator,
    *,
    thinking_enabled: bool,
    max_length: int,
    batch_size: int,
    top_k: int,
    logit_chunk: int,
    progress: bool = False,
) -> list[dict[str, Any]]:
    device = next(teacher_model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    diagnostics: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(records), batch_size), disable=not progress):
        batch = records[start : start + batch_size]
        input_ids, attention_mask, response_mask, lengths = encode_batch(
            tokenizer,
            batch,
            thinking_enabled=thinking_enabled,
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

        # Third pass: the same dense model, same prefixes, now with hooks live and
        # the KL weights installed. Only assistant positions are scored, so padding
        # is excluded implicitly -- a padded position is never a response position.
        accumulator.set_token_mask(response_mask)
        accumulator.set_token_weights(weights)
        teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        accumulator.set_token_weights(None)
        accumulator.set_token_mask(None)

        row_sums = (divergences * response_mask).sum(dim=1)
        counts = response_mask.sum(dim=1)
        for row, record in enumerate(batch):
            n = int(counts[row].item())
            total = float(row_sums[row].item())
            diagnostics.append(
                {
                    "sample_id": record.get("sample_id"),
                    "domain": record.get("domain"),
                    "response_tokens": n,
                    "kl_sum": total,
                    "kl_mean": total / n if n else 0.0,
                    "kl_max": float((divergences[row] * response_mask[row]).max().item())
                    if n
                    else 0.0,
                    # Recorded per sample because a run where most trajectories fell
                    # back is a run that did not test the hypothesis at all.
                    "uniform_fallback": bool(total <= 1e-9),
                    "truncated": lengths[row] >= max_length,
                }
            )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    pruning = config["pruning"]
    settings = pruning.get("recal", {})
    output_dir = Path(settings.get("statistics_dir", pruning["importance_dir"]))
    model_cfg = config["model"]
    # Stamped onto channel_order.json so the mask carries its own provenance and
    # preflight can refuse an arm pointed at another method's mask directory. Read from
    # the config because this collector now serves every weighted variant, not one arm.
    pruning_method = str(pruning.get("method", "minitron_recal"))

    rank, world_size, local_rank = init_distributed()
    set_seed(int(config.get("seed", 42)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    probe_path = settings.get("probe_path")
    if not probe_path:
        raise ValueError(
            "pruning.recal.probe_path must point at a provisional pruned model; "
            "it is the reference the divergence is measured against"
        )
    if not (Path(probe_path) / "config.json").exists():
        raise FileNotFoundError(f"Probe checkpoint is missing: {probe_path}")

    teacher_path = config["data"]["sft_teacher_path"]
    frame = pd.read_parquet(teacher_path)
    per_domain = int(settings.get("trajectories_per_domain", 256))
    # Same seed on every rank so all ranks select the same trajectories; only the
    # shard split below differs. Sampling per rank would make the statistic depend
    # on world size.
    calibration = sample_calibration_trajectories(
        frame, per_domain=per_domain, seed=int(config.get("seed", 42))
    )
    shard = calibration.iloc[rank::world_size].reset_index(drop=True)

    model_path = model_cfg.get("path", model_cfg.get("teacher_path"))
    revision = model_cfg.get("revision")
    thinking = bool(model_cfg.get("thinking_enabled", True))
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if thinking and not supports_enable_thinking(tokenizer):
        raise RuntimeError("Tokenizer chat template does not support enable_thinking=True")

    attn = settings.get("attn_implementation", "sdpa")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    ).to(device)
    teacher_model.eval()
    probe_model = AutoModelForCausalLM.from_pretrained(
        probe_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    ).to(device)
    probe_model.eval()
    summary = model_shape_summary(teacher_model)

    # Hooks go on the dense teacher: the statistic being weighted is the *teacher's*
    # activation, exactly as in the baseline. The probe contributes only logits.
    layer_sizes = {
        idx: module.in_features
        for idx, (_name, module) in enumerate(
            item for item in teacher_model.named_modules() if item[0].endswith("down_proj")
        )
    }
    # The weighting is a component, not a criterion: it replaces the uniform average
    # inside whichever score the config names. `modelopt_minitron` keeps the original
    # arm byte-identical; a name from baseline_criteria.SCORES weights that published
    # formula instead, which is what makes this plug-in rather than a fourth criterion.
    primary_score = str(
        pruning.get("importance", {}).get("scoring", {}).get(
            "primary_score", "modelopt_minitron"
        )
    )
    if primary_score == "modelopt_minitron":
        accumulator = FFNImportanceAccumulator(
            layer_sizes, device=device, accumulator_dtype=torch.float32
        )
    elif primary_score in BASELINE_SCORES:
        accumulator = BaselineImportanceAccumulator(
            layer_sizes,
            device=device,
            accumulator_dtype=torch.float32,
            # Opted in here and nowhere else: the unweighted arms construct the same
            # class without this flag and still refuse weights.
            allow_token_weights=True,
        )
    else:
        raise ValueError(
            f"pruning.importance.scoring.primary_score={primary_score!r} is neither "
            f"'modelopt_minitron' nor one of {BASELINE_SCORES}; the KL weighting can "
            "only replace the token average of an activation-based criterion"
        )
    handles, hook_names = register_down_projection_hooks(teacher_model, accumulator)

    top_k = int(settings.get("top_k", DEFAULT_TOP_K))
    weighting = str(settings.get("weighting", "forward_kl"))
    if weighting != "forward_kl":
        raise ValueError(
            "The release implementation supports only "
            "pruning.recal.weighting=forward_kl"
        )
    diagnostics = collect_weighted_statistics(
        teacher_model,
        probe_model,
        tokenizer,
        shard.to_dict("records"),
        accumulator,
        thinking_enabled=thinking,
        max_length=int(settings.get("max_calibration_length", 8192)),
        batch_size=int(settings.get("batch_size_per_gpu", 1)),
        top_k=top_k,
        logit_chunk=int(settings.get("logit_chunk", 512)),
        progress=rank == 0,
    )
    for handle in handles:
        handle.remove()

    accumulator.all_reduce()
    # BaselineImportanceAccumulator needs the weight side of its formulas, which is read
    # from the model rather than accumulated; the Minitron one is activation-only.
    if primary_score == "modelopt_minitron":
        layer_statistics = accumulator.finalize()
    else:
        layer_statistics = accumulator.finalize(
            down_projection_column_norms(teacher_model)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"calibration_records_rank{rank:03d}.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in diagnostics:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    totals = torch.tensor(
        [
            len(diagnostics),
            sum(record["response_tokens"] for record in diagnostics),
            sum(int(record["truncated"]) for record in diagnostics),
            sum(int(record["uniform_fallback"]) for record in diagnostics),
        ],
        dtype=torch.long,
        device=device,
    )
    kl_totals = torch.tensor(
        [sum(record["kl_sum"] for record in diagnostics)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(kl_totals, op=dist.ReduceOp.SUM)

    if rank == 0:
        rows, tokens, truncated, fallbacks = (int(v) for v in totals.tolist())
        channel_order = {
            str(idx): torch.argsort(stats[primary_score], descending=True).tolist()
            for idx, stats in layer_statistics.items()
        }
        # Schema mirrors the activation arm's channel_order.json exactly -- keys
        # `layers` and `primary_score` -- because nested_masks.py reads those names
        # and both arms must go through the same mask builder. `pruning_method` is
        # what lets the mask carry its own provenance downstream so preflight can
        # refuse an arm pointed at the other's mask directory.
        dump_json(
            {
                "schema_version": 1,
                "layers": channel_order,
                # The criterion that was weighted, and the fact that it was weighted,
                # are two separate facts: the results table needs both to say
                # "Wanda-sp + KL" rather than collapsing it into either name alone.
                "primary_score": primary_score,
                "pruning_method": pruning_method,
                "recal_weighted": True,
            },
            output_dir / "channel_order.json",
        )
        dump_json(
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "method": "recal",
                "model_path": model_path,
                "model_revision": revision,
                "probe_path": str(probe_path),
                "calibration_path": teacher_path,
                "calibration_sha256": sha256_file(teacher_path),
                "calibration_rows": rows,
                "response_tokens": tokens,
                "truncated_rows": truncated,
                # If most trajectories fell back to uniform weights this arm is the
                # baseline wearing a different name, and the comparison is void.
                "uniform_fallback_rows": fallbacks,
                "uniform_fallback_fraction": round(fallbacks / max(rows, 1), 6),
                "kl_mean_per_token": round(float(kl_totals.item()) / max(tokens, 1), 8),
                "top_k": top_k,
                "calibration_token_source": "assistant_response_only",
                "weighting": weighting,
                "normalization": "per_trajectory_sum_to_one",
                "trajectories_per_domain": per_domain,
                "max_calibration_length": int(settings.get("max_calibration_length", 8192)),
                "world_size": world_size,
                "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
                "config_sha256": sha256_text(json.dumps(config, sort_keys=True, default=str)),
                # Key name matches the activation arm's manifest, because
                # nested_masks.py reads `model_summary` to convert a total-parameter
                # target into an FFN width ratio. Naming it anything else fails the
                # mask build with "must contain exact total_parameters and
                # ffn_parameters" -- after calibration has already run.
                "model_summary": summary,
                "hook_modules": hook_names,
                "layer_token_counts": {
                    str(idx): stats["token_count"] for idx, stats in layer_statistics.items()
                },
            },
            output_dir / "calibration_manifest.json",
        )
        print(
            f"[ReCal] {rows} trajectories, {tokens} response tokens, "
            f"{fallbacks} uniform fallbacks, mean KL/token "
            f"{float(kl_totals.item()) / max(tokens, 1):.6f}",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
