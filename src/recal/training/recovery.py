from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from recal.common import (
    atomic_write_text,
    directory_fingerprint,
    dump_json,
    dump_yaml,
    load_json,
    utc_now,
)
from recal.data.contamination import disjointness_report
from recal.data.offline_teacher import generate_shared_teacher_sequences
from recal.data.sft_split import split_sft_and_opd
from recal.data.teacher_targets import build_sft_targets
from recal.modes import (
    as_messages,
    assert_data_matches_mode,
    thinking_enabled,
    thinking_report,
)
from recal.orchestration.manifests import stage_manifest
from recal.pruning.qwen3_ffn_pruner import prune_qwen3_ffn_width
from recal.training.boundary_eval import run_boundary_benchmark
from recal.training.checkpoint_export import copy_pruning_metadata, export_fsdp_checkpoint
from recal.training.opd import (
    _opd_step_budget,
    export_sft_checkpoint,
    latest_sft_checkpoint,
    latest_sft_step_dir,
    expected_sft_steps,
    launch_opd,
    launch_sft,
    verify_training_backend,
)


def _ratios(config: dict[str, Any]) -> list[float]:
    """Validate the stage ratios and return them in cascade order.

    Ratios are cumulative whole-model parameter targets in the *original* dense
    channel-ID space, so they must be strictly increasing: stage N prunes the
    stage N-1 checkpoint further until the original-model target is reached. A
    single ratio is the one-shot control arm, which goes straight from the dense
    model to the final width. Both arms end at the same ``intermediate_size`` for
    a given ratio; only the path differs.

    Any ratio with a corresponding mask is accepted. Requiring a fixed 10%-step
    grid rejected legitimate experiments (a one-shot 20% arm, or a finer sweep)
    for no reason -- the mask files are the real constraint, and the caller
    checks those.
    """
    ratios = list(map(float, config["pruning"]["ratios"]))
    if not ratios:
        raise ValueError("pruning.ratios must not be empty")
    normalized = [round(value, 4) for value in ratios]
    if any(not 0.0 < value < 1.0 for value in normalized):
        raise ValueError(f"pruning.ratios must lie strictly within (0, 1), got {normalized}")
    if normalized != sorted(set(normalized)):
        raise ValueError(
            "Cascade ratios must be strictly increasing cumulative whole-model "
            f"parameter targets, got {normalized}"
        )
    return normalized


def _stage_mask(config: dict[str, Any], ratio: float) -> str:
    return str(
        Path(config["pruning"]["mask_dir"])
        / f"width_prune_{round(ratio * 100):02d}.json"
    )


def _write_stage_status(
    stage_dir: Path,
    *,
    stage_id: int,
    ratio: float,
    phase: str,
    **extra: Any,
) -> None:
    dump_json(
        {
            "stage_id": stage_id,
            "target_ratio": ratio,
            "phase": phase,
            "updated_at": utc_now(),
            **extra,
        },
        stage_dir / "stage_status.json",
    )


def _completed_stage(stage_dir: Path, expected_source: str) -> Path | None:
    manifest_path = stage_dir / "stage_manifest.json"
    recovered = stage_dir / "recovered_hf"
    if not manifest_path.exists() or not recovered.exists():
        return None
    manifest = load_json(manifest_path)
    if manifest.get("status") != "completed":
        return None
    if manifest.get("source_checkpoint_hash") != directory_fingerprint(expected_source):
        raise RuntimeError(f"Stage source hash mismatch in {stage_dir}")
    return recovered


def _pruned_matches_mask(pruned: Path, mask_path: str) -> bool:
    """Whether an existing pruned checkpoint was actually built from `mask_path`.

    Artifact-gating on the manifest merely *existing* is not enough here: a mask can
    be rebuilt after its checkpoint was written, and then the stage silently trains on
    the previous mask's architecture. Channel IDs are the
    honest identity of a pruned model, so compare those rather than timestamps, which
    say nothing about content and go backwards across a restore.
    """
    manifest_path = pruned / "pruning_manifest.json"
    if not manifest_path.exists():
        return False
    key = "retained_original_channel_ids_per_layer"
    wanted = load_json(mask_path).get(key) or {}
    if not wanted:
        # Nothing to compare against, so there is no evidence of a mismatch. Discarding
        # a checkpoint here would cost a 12 GB re-prune and a re-benchmark to learn
        # nothing; the check exists to catch a mask that demonstrably disagrees.
        return True
    built = load_json(manifest_path).get(key) or {}
    if built.keys() != wanted.keys():
        return False
    return all(list(map(int, built[layer])) == list(map(int, wanted[layer])) for layer in wanted)


def _skipped_boundaries(config: dict[str, Any]) -> set[str]:
    """Boundary benchmarks this run should not spend GPUs on.

    Set via ``boundary_benchmark.skip``. The use case is a boundary being measured
    elsewhere: the after_prune checkpoints are evaluated at a 32k budget on a
    separate machine, and running them here too would cost ~2.5 GPU-hours per arm to
    reproduce a number that already exists.

    A skipped boundary records ``{"skipped": ...}`` in the stage outcome rather than
    a summary, so the cascade trajectory stays readable and nothing downstream
    mistakes a missing benchmark for a failed one.
    """
    configured = (config.get("boundary_benchmark") or {}).get("skip") or []
    if isinstance(configured, str):
        configured = [configured]
    return {str(name) for name in configured}


def _completed_boundary(stage_dir: Path, boundary: str) -> dict[str, Any] | None:
    output = stage_dir / "benchmarks" / boundary
    success = output / "_SUCCESS.json"
    summary = output / "summary.json"
    if not success.exists() or not summary.exists():
        return None
    payload = load_json(success)
    expected = ["aime", "gpqa_diamond", "ifbench", "livecodebench"]
    if payload.get("completed_tasks") != expected:
        return None
    return load_json(summary)


def _prepare_shared_training_data(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool,
) -> None:
    """Build and validate the SFT/OPD data and the shared offline teacher targets.

    Two data provenances are supported, distinguished by whether the config names a
    single ``source_path`` to split:

      split     one pool is divided into SFT and OPD halves here, by ``split_sft_and_opd``.
      prebuilt  the pools are built independently and to exact size by
          ``scripts/prepare_recovery_data.py``, which also runs the contamination gate and
          the domain quotas, so there is nothing to split. Teacher targets go
          through ``teacher_targets.build_sft_targets``, which adds the quality
          filters (empty, unclosed reasoning, missing answer format, degenerate
          repetition) on top of the generator's truncation filter.

    Beyond disjointness, the teacher data is validated against the configured
    thinking paradigm and its answer-format statistics are recorded. Both were
    previously verified by hand: a paradigm mismatch or a prompt-format drift is
    invisible until benchmark scores come back inexplicably low, so the check
    belongs in the run rather than in someone's notes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    data = config["data"]
    built_pools = "opd_num_prompts" in data
    if built_pools:
        for key in ("sft_prompt_path", "opd_path"):
            path = Path(data[key])
            if not path.exists():
                raise FileNotFoundError(
                    f"Recovery pool {path} is missing; build it with "
                    "scripts/prepare_recovery_data.py before starting the pipeline"
                )
        split_manifest = {
            "provenance": "prebuilt_recovery_pools",
            "manifest": str(Path(data["sft_prompt_path"]).parent / "pool_manifest.json"),
        }
        # The release uses the stable, single-command teacher-target builder.
        # It resumes from the raw parquet and applies the same length, format,
        # truncation, and repetition filters before SFT starts.
        teacher_manifest = build_sft_targets(config)
    else:
        split_manifest = split_sft_and_opd(config)
        teacher_manifest = generate_shared_teacher_sequences(config)

    teacher_frame = pd.read_parquet(config["data"]["sft_teacher_path"])
    sft_mode = assert_data_matches_mode(
        teacher_frame,
        expect_thinking=thinking_enabled(config, "sft"),
        source=str(config["data"]["sft_teacher_path"]),
    )
    modes = thinking_report(config)
    if not modes["consistent"]:
        raise RuntimeError(
            "Stages disagree on thinking mode, which would train and evaluate under "
            f"different paradigms: {modes['resolved']}"
        )

    sft_ids = set(teacher_frame["sample_id"].astype(str))
    opd_frame = pd.read_parquet(
        config["data"]["opd_path"], columns=["sample_id", "prompt"]
    )
    opd_ids = set(opd_frame["sample_id"].astype(str))
    overlap = sft_ids & opd_ids
    if overlap:
        raise RuntimeError(f"SFT teacher data overlaps OPD prompts: {len(overlap)}")
    # Ids can differ while the problem is the same -- two sources can carry one
    # question under different UUIDs. The normalized-text check is the one that binds.
    #
    # The teacher parquet has no `prompt` column: it stores prompt and response
    # together in `messages`, so the question has to be recovered from the user
    # turns. Reading `prompt` here worked on the OPD side and raised KeyError on the
    # SFT side.
    def _user_text(messages: Any) -> str:
        return " ".join(
            str(item.get("content", ""))
            for item in as_messages(messages)
            if item.get("role") == "user"
        )

    disjoint = disjointness_report(
        teacher_frame["messages"].map(_user_text).tolist(),
        opd_frame["prompt"].astype(str).tolist(),
    )
    if not disjoint["disjoint"]:
        raise RuntimeError(
            "SFT and OPD share "
            f"{disjoint['normalized_prompt_overlap']} normalized prompts; OPD would "
            "train on problems the student was already fit to"
        )
    dump_json(
        {
            "split": split_manifest,
            "offline_teacher": teacher_manifest,
            "thinking": modes,
            "sft_data_validation": sft_mode,
            "sample_id_overlap": 0,
            "normalized_prompt_disjointness": disjoint,
            "validated_at": utc_now(),
        },
        output_dir / "shared_training_data.json",
    )


def _aggregate_root(root: Path, ratios: list[float]) -> None:
    rows = []
    for stage_id, ratio in enumerate(ratios, 1):
        outcome = root / "stages" / f"stage_{stage_id:02d}_r{round(ratio * 100):02d}" / "stage_outcome.json"
        if outcome.exists():
            rows.append(load_json(outcome))
    if not rows:
        return
    frame = pd.json_normalize(rows)
    frame.to_parquet(root / "recovery_trajectory.parquet", index=False)
    frame.to_csv(root / "recovery_trajectory.csv", index=False)


def _opd_training_budget(stage_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Record how many OPD updates the stage actually used.

    OPD stops on convergence rather than at a fixed step, so the realized budget
    comes from the convergence report written by ``launch_opd``.
    """
    max_steps, _ = _opd_step_budget(config)
    budget: dict[str, Any] = {"opd_max_updates": max_steps}
    report_path = stage_dir / "opd_convergence.json"
    if report_path.exists():
        report = load_json(report_path)
        budget["opd_updates"] = report.get("final_step")
        budget["opd_stop_reason"] = report.get("stop_reason")
        budget["opd_final_overlap_ratio"] = report.get("final_overlap_ratio")
    return budget


def _stop_after(config: dict[str, Any]) -> str:
    """Where this arm deliberately stops, from ``pipeline.stop_after``.

    One of ``after_opd`` (the default, the full cascade), ``after_sft`` (skip OPD and
    its benchmark -- used for baseline rows whose SFT-stage comparison is the result
    and whose OPD time is not worth spending) or ``prune`` (stop immediately after
    pruning -- used for arms that exist only to produce a probe checkpoint for a
    weighted counterpart).

    The value is recorded in ``recovery_summary.json`` so a later full run can tell a
    deliberately truncated summary from a finished one.
    """
    stop_after = str((config.get("pipeline") or {}).get("stop_after", "after_opd"))
    if stop_after not in {"after_opd", "after_sft", "prune"}:
        raise ValueError(
            f"pipeline.stop_after={stop_after!r}; expected one of "
            "'after_opd', 'after_sft', 'prune'"
        )
    return stop_after


def run_recovery(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    start_stage: int | None = None,
) -> Path:
    verify_training_backend(config)
    ratios = _ratios(config)
    # Ratios are only meaningful if their masks exist; fail before any GPU work
    # rather than partway through a multi-hour cascade.
    missing = [ratio for ratio in ratios if not Path(_stage_mask(config, ratio)).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing nested pruning masks for ratios "
            f"{missing}; expected files under {config['pruning']['mask_dir']}"
        )
    _prepare_shared_training_data(config, output_dir, dry_run=dry_run)

    teacher = config["model"]["teacher_path"]
    previous_checkpoint = teacher
    final_checkpoint = Path(teacher)
    completed_ratios: list[float] = []

    for stage_id, ratio in enumerate(ratios, 1):
        stage_dir = output_dir / "stages" / f"stage_{stage_id:02d}_r{round(ratio * 100):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        completed = _completed_stage(stage_dir, previous_checkpoint)
        if completed is not None and (start_stage is None or stage_id < start_stage):
            previous_checkpoint = str(completed)
            final_checkpoint = completed
            completed_ratios.append(ratio)
            continue
        if start_stage is not None and stage_id < start_stage:
            raise RuntimeError(f"Requested stage {start_stage}, but stage {stage_id} is incomplete")

        mask = _stage_mask(config, ratio)
        stage_config = copy.deepcopy(config)
        stage_config["stage_id"] = stage_id
        stage_config["pruning"]["target_ratio"] = ratio
        stage_config["model"]["stage_source_path"] = previous_checkpoint
        dump_yaml(stage_config, stage_dir / "resolved_config.yaml")

        pruned = stage_dir / "pruned_initial"
        _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="pruning")
        # A checkpoint left over from a different mask is worse than no checkpoint: it
        # looks resumable and quietly changes what the comparison measures. Rebuild it,
        # and drop the after_prune benchmark with it since it scored the old weights.
        if (pruned / "pruning_manifest.json").exists() and not _pruned_matches_mask(pruned, mask):
            if not dry_run:
                shutil.rmtree(pruned)
                shutil.rmtree(stage_dir / "benchmarks" / "after_prune", ignore_errors=True)
        if not (pruned / "pruning_manifest.json").exists() and not dry_run:
            # Cascade semantics: every stage starts from the previously recovered
            # checkpoint, while masks are cumulative in the original dense
            # channel-ID space. The pruner translates original IDs to local
            # indices, so stage 2 removes channels from the stage-1 checkpoint
            # until the *original-model* cumulative 20% target is reached.
            prune_qwen3_ffn_width(
                previous_checkpoint,
                mask,
                str(pruned),
                config["model"].get("output_dtype", "bfloat16"),
            )

        _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="benchmark_after_prune")
        skip = _skipped_boundaries(config)
        stop_after = _stop_after(config)
        if stop_after == "prune":
            # Probe-only arm: the pruned checkpoint IS the product (a weighted
            # counterpart calibrates against it). No benchmark, no training.
            recovered = pruned
            after_prune = {"skipped": "pipeline.stop_after=prune"}
            after_sft = {"skipped": "pipeline.stop_after=prune"}
            after_opd = {"skipped": "pipeline.stop_after=prune"}
            sft_hf = pruned
        else:
            after_prune = _completed_boundary(stage_dir, "after_prune")
            if after_prune is None and "after_prune" in skip:
                after_prune = {"skipped": "measured separately at a 32k budget"}
            if after_prune is None:
                after_prune = run_boundary_benchmark(
                    stage_config,
                    model=str(pruned),
                    stage_dir=stage_dir,
                    boundary="after_prune",
                    dry_run=dry_run,
                )

            _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="sft")
            after_sft = _completed_boundary(stage_dir, "after_sft")
            # Reuse a finished SFT checkpoint regardless of whether its benchmark
            # completed. Keying reuse on the benchmark instead would retrain a stage
            # whose SFT had already succeeded just because its benchmark failed --
            # hours of recomputation for nothing.
            sft_hf = latest_sft_checkpoint(stage_dir, stage_config)
            if sft_hf is None:
                # SFT may have finished while only its HF export is missing -- the
                # pinned verl checkout writes the config but not the weights. Merge
                # the existing FSDP shards instead of retraining hours of work.
                #
                # Only when the shards are from a *finished* run, though. A crash leaves
                # shards at the last save_freq boundary, and merging those would hand a
                # partial model to the after_sft benchmark -- the same mistake as reusing
                # a partial HF export, just one step later.
                step_dir = latest_sft_step_dir(stage_dir)
                if step_dir is not None and not dry_run:
                    expected = expected_sft_steps(stage_config)
                    actual = int(step_dir.name.removeprefix("global_step_"))
                    if expected is None or actual >= expected:
                        sft_hf = export_sft_checkpoint(step_dir, stage_config)
                    else:
                        print(
                            f"[sft] {step_dir.name} holds {actual} of {expected} expected steps; "
                            "resuming training rather than merging a partial run",
                            flush=True,
                        )
            if sft_hf is None:
                sft_hf = launch_sft(
                    stage_config,
                    student_model=str(pruned),
                    stage_dir=stage_dir,
                    dry_run=dry_run,
                )
                if not dry_run:
                    copy_pruning_metadata(pruned, sft_hf)
            if after_sft is None and "after_sft" in skip:
                after_sft = {"skipped": "measured separately at a 32k budget"}
            if after_sft is None:
                _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="benchmark_after_sft")
                after_sft = run_boundary_benchmark(
                    stage_config,
                    model=str(sft_hf),
                    stage_dir=stage_dir,
                    boundary="after_sft",
                    dry_run=dry_run,
                )

        if stop_after in {"prune", "after_sft"}:
            # after_sft arms stop here: the SFT-stage model is the result and OPD is
            # deliberately not spent. `recovered` is whatever the last real stage
            # produced (sft_hf for after_sft, pruned for prune-only).
            if stop_after == "after_sft":
                recovered = sft_hf
                after_opd = {"skipped": "pipeline.stop_after=after_sft"}
        else:
            recovered = stage_dir / "recovered_hf"
            after_opd = _completed_boundary(stage_dir, "after_opd")
        if stop_after == "after_opd" and (
            after_opd is None or not (recovered / "config.json").exists()
        ):
            _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="opd")
            actor = launch_opd(
                stage_config,
                student_model=str(sft_hf),
                stage_dir=stage_dir,
                dry_run=dry_run,
            )
            if not dry_run:
                export_fsdp_checkpoint(
                    actor,
                    recovered,
                    python_executable=config.get("runtime", {}).get("python"),
                )
                copy_pruning_metadata(pruned, recovered)
            _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="benchmark_after_opd")
            after_opd = run_boundary_benchmark(
                stage_config,
                model=str(recovered),
                stage_dir=stage_dir,
                boundary="after_opd",
                dry_run=dry_run,
            )

        outcome = {
            "stage_id": stage_id,
            "target_ratio": ratio,
            "source_checkpoint": previous_checkpoint,
            "pruned_checkpoint": str(pruned),
            "sft_checkpoint": str(sft_hf),
            "recovered_checkpoint": str(recovered),
            "teacher": teacher,
            "thinking_enabled": thinking_enabled(config, "sft"),
            "sft_teacher_data": config["data"]["sft_teacher_path"],
            "opd_data": config["data"]["opd_path"],
            "benchmarks": {
                "after_prune": after_prune,
                "after_sft": after_sft,
                "after_opd": after_opd,
            },
            "finished_at": utc_now(),
        }
        if not dry_run:
            dump_json(outcome, stage_dir / "stage_outcome.json")
            dump_json(
                stage_manifest(
                    stage_id=stage_id,
                    stage_name=stage_dir.name,
                    source_checkpoint=previous_checkpoint,
                    output_checkpoint=str(recovered),
                    mask_path=mask,
                    target_ratio=ratio,
                    status="completed",
                    training_budget=_opd_training_budget(stage_dir, config),
                ),
                stage_dir / "stage_manifest.json",
            )
            _write_stage_status(stage_dir, stage_id=stage_id, ratio=ratio, phase="completed")
            _aggregate_root(output_dir, ratios[:stage_id])

        previous_checkpoint = str(recovered)
        final_checkpoint = recovered
        completed_ratios.append(ratio)

    if not dry_run:
        dump_json(
            {
                "status": "completed",
                # Recorded so a later full-budget run can tell this arm stopped early
                # on purpose rather than finishing.
                "stop_after": _stop_after(config),
                "completed_ratios": completed_ratios,
                "final_checkpoint": str(final_checkpoint),
                "teacher": teacher,
                "finished_at": utc_now(),
            },
            output_dir / "recovery_summary.json",
        )
        atomic_write_text(str(final_checkpoint) + "\n", output_dir / "final_checkpoint.txt")
    return final_checkpoint
