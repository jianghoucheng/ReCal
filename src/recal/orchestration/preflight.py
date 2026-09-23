"""Pre-flight validation for a ReCal recovery experiment.

The checks validate the resolved configuration, data provenance, masks, runtime,
benchmark scorers, and SFT/OPD commands before expensive GPU stages begin.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from recal.common import dump_json, load_yaml, utc_now
from recal.data.pool_sources import DOMAINS
from recal.modes import assert_data_matches_mode, thinking_enabled, thinking_report
from recal.training.recovery import _ratios, _stage_mask
from recal.training.opd import (
    _opd_step_budget,
    _python_executable,
    build_opd_command,
    build_sft_command,
    verify_training_backend,
    verify_opd_runtime,
    verify_sft_runtime,
)


def _extract(command: list[str], prefix: str) -> str | None:
    return next((item for item in command if item.startswith(prefix)), None)


def _validate_split_files(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    sft_path = Path(data["sft_prompt_path"])
    opd_path = Path(data["opd_path"])
    if not sft_path.exists() or not opd_path.exists():
        return {
            "prepared": False,
            "sft_rows": None,
            "opd_rows": None,
            "sample_id_overlap": None,
        }
    sft = pd.read_parquet(sft_path, columns=["sample_id"])
    opd = pd.read_parquet(opd_path, columns=["sample_id"])
    overlap = set(sft["sample_id"].astype(str)) & set(opd["sample_id"].astype(str))
    return {
        "prepared": True,
        "sft_rows": len(sft),
        "opd_rows": len(opd),
        "sample_id_overlap": len(overlap),
    }


def _validate_teacher_data(config: dict[str, Any]) -> dict[str, Any]:
    """Confirm the offline teacher data matches the configured paradigm.

    Absent teacher data is not a failure: the pipeline generates it as its first
    step. Present-but-mismatched data is, because every later stage would train
    against a convention the benchmark cannot read.
    """
    path = Path(config["data"]["sft_teacher_path"])
    if not path.exists():
        return {"prepared": False, "matches_mode": None}
    try:
        frame = pd.read_parquet(path)
        report = assert_data_matches_mode(
            frame,
            expect_thinking=thinking_enabled(config, "sft"),
            source=str(path),
        )
        return {"prepared": True, "matches_mode": True, **report}
    except RuntimeError as error:
        return {"prepared": True, "matches_mode": False, "error": str(error)}


def _format_alignment_warnings(teacher_data: dict[str, Any]) -> list[str]:
    """Flag teacher data whose answers do not speak the grader's format.

    The benchmark appends a format instruction to every prompt and scores by
    extracting that exact form -- ``\\boxed{}`` for math/science, a fenced block
    for code. Teacher data generated before those instructions were aligned
    trains the student toward a convention the grader cannot read, which is
    invisible until scores come back near zero. Reported rather than fatal:
    regenerating teacher data is a deliberate, expensive decision.
    """
    statistics = teacher_data.get("format")
    if not statistics:
        return []
    warnings: list[str] = []
    expectations = {
        "math": ("response_boxed", "prompt_boxed_instruction", "\\boxed{}"),
        "code": ("response_fenced", "prompt_fenced_instruction", "a fenced code block"),
    }
    for domain, (response_key, prompt_key, description) in expectations.items():
        entry = statistics["per_domain"].get(domain)
        if not entry or not entry["rows"]:
            continue
        if entry[prompt_key] < 0.9:
            warnings.append(
                f"{domain}: only {entry[prompt_key]:.0%} of prompts ask for {description}, "
                "so the training prompts disagree with the benchmark's instruction"
            )
        if entry[response_key] < 0.9:
            warnings.append(
                f"{domain}: only {entry[response_key]:.0%} of teacher responses produce "
                f"{description}, which the grader extracts; SFT would teach a form "
                "the scorer cannot read"
            )
    return warnings


def _wandb_is_usable(mode: str) -> dict[str, Any]:
    """Confirm the configured W&B mode can actually initialize.

    ``mode: online`` needs two things, and checking only the first is what let a
    week of runs fall back to offline: a credential, *and* egress to
    api.wandb.ai. On these nodes egress exists only through the HTTP proxy that
    ``scripts/activate_recal.sh`` exports; without it TCP 443 times out and
    looks exactly like a firewall block. Both are verified here, in seconds,
    rather than discovered inside the trainer after the model has loaded.
    """
    if mode != "online":
        return {"mode": mode, "usable": True, "reason": "offline/disabled needs no credential"}

    import os

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            import netrc

            key = netrc.netrc().hosts.get("api.wandb.ai", (None, None, None))[2]
        except Exception:
            key = None
    if not key:
        return {
            "mode": mode,
            "usable": False,
            "reason": "no WANDB_API_KEY and no netrc entry for api.wandb.ai",
        }

    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    try:
        import json as _json
        import urllib.request

        request = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=_json.dumps({"query": "{viewer{username entity}}"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        import base64

        token = base64.b64encode(f"api:{key}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = _json.loads(response.read())
        viewer = (payload.get("data") or {}).get("viewer") or {}
        if not viewer.get("username"):
            return {"mode": mode, "usable": False, "reason": f"unexpected response: {payload}", "proxy": proxy}
        return {
            "mode": mode,
            "usable": True,
            "reason": "credential accepted and api.wandb.ai reachable",
            "username": viewer.get("username"),
            "entity": viewer.get("entity"),
            "proxy": proxy,
        }
    except Exception as error:
        return {
            "mode": mode,
            "usable": False,
            "reason": (
                f"api.wandb.ai unreachable ({type(error).__name__}: {error}). "
                "Is the HTTP proxy exported? source scripts/activate_recal.sh"
            ),
            "proxy": proxy,
        }


def _benchmark_scorers_importable(config: dict[str, Any]) -> dict[str, Any]:
    """Confirm the benchmark scorers can load before any GPU work starts.

    IFBench's official scorer runs as a subprocess *after* generation finishes,
    so a missing dependency wastes the whole task: the model produced all 300
    responses, then scoring died on ``No module named 'syllapy'`` and the
    boundary failed. Importing it here costs milliseconds.
    """
    tasks = [str(task) for task in config.get("boundary_benchmark", {}).get("tasks", [])]
    result: dict[str, Any] = {"checked": [], "missing": []}
    if "ifbench" in tasks:
        result["checked"].append("ifbench")
        probe = subprocess.run(
            [
                _python_executable(config),
                "-c",
                (
                    "from recal.evaluation.ifbench import "
                    "run_eval, evaluation_lib, instructions_registry"
                ),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if probe.returncode != 0:
            result["missing"].append(
                "ifbench: " + (probe.stderr.strip().splitlines() or ["import failed"])[-1]
            )
    if "livecodebench" in tasks:
        result["checked"].append("livecodebench")
        probe = subprocess.run(
            [
                _python_executable(config),
                "-c",
                (
                    "from lcb_runner.benchmarks.code_generation import "
                    "CodeGenerationProblem; "
                    "from lcb_runner.evaluation.compute_code_generation_metrics "
                    "import codegen_metrics"
                ),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if probe.returncode != 0:
            result["missing"].append(
                "livecodebench: "
                + (probe.stderr.strip().splitlines() or ["import failed"])[-1]
            )
    result["usable"] = not result["missing"]
    return result


def _validate_recovery_pools(config: dict[str, Any]) -> dict[str, Any]:
    """Hard gate on pool contamination, domain balance, and SFT/OPD disjointness.

    The math pool includes DAPO-derived examples, so contamination filtering
    against every evaluation set is a hard requirement.

    It reads the report ``prepare_recovery_data.py`` wrote rather than rescanning: that
    build already verified the artifacts on disk and refused to write on any hit, so
    what matters here is that the verdict exists, says zero, and describes the pools
    this config actually points at. A missing report on such a config is itself a
    failure -- it means the pools were produced by something other than the gated
    builder.

    Configurations without independently constructed recovery pools are skipped.
    """
    data = config["data"]
    if "opd_num_prompts" not in data:
        return {"applicable": False, "passed": True, "reason": "not a recovery-pool config"}

    sft_path = Path(data["sft_prompt_path"])
    opd_path = Path(data["opd_path"])
    # Which manifest describes this config's pool is decided by the config, not by
    # which filename exists. Several pool generations share data/recovery_pools, so
    # preferring the newest name would hand an older arm the newer pool's manifest and
    # compare one pool's rows against another pool's intended composition -- reported as
    # a contamination-check failure, which is not what went wrong.
    manifest_name = data.get("pool_manifest", "pool_manifest.json")
    report_path = sft_path.parent / manifest_name
    if not report_path.exists():
        report_path = None
    if report_path is None:
        return {
            "applicable": True,
            "passed": False,
            "reason": (
                f"no pool manifest beside {sft_path}; build with "
                "scripts/prepare_recovery_data.py"
            ),
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))

    residual = report.get("residual_contamination") or {}
    disjoint = report.get("disjointness") or {}
    counts = report.get("prompt_count_by_domain") or {}

    # The intended domain mix is a config value, not a constant: earlier runs drew 25%
    # per domain, and later ones weight math and code higher because those carry the
    # recoverable pruning damage. The check reads ``data.opd_mix`` rather than assuming
    # an even split, or it rejects a correctly built pool for not being uniform.
    configured_mix = data.get("opd_mix")
    if configured_mix:
        expected_share = {str(key): float(value) for key, value in configured_mix.items()}
    else:
        expected_share = {domain: 1.0 / len(DOMAINS) for domain in DOMAINS}

    problems: list[str] = []
    if not sft_path.exists() or not opd_path.exists():
        problems.append("pool parquet files are missing")
    if any(int(value) for value in residual.values()):
        problems.append(f"contamination survives in the written pools: {residual}")
    if not disjoint.get("disjoint"):
        problems.append(
            f"SFT and OPD share {disjoint.get('normalized_prompt_overlap')} normalized prompts"
        )
    # Only the OPD pool's shape is asserted. OPD prompts are rolled out rather than
    # pre-generated, so nothing between the draw and training reshapes that pool and its
    # prompt counts must match the intended mix. The SFT pool is drawn deliberately
    # larger than the target row count because the quality and length filters delete
    # rather than truncate, and their bite is domain-dependent (49% of code responses
    # survive against 99.8% of instruction-following) -- so its prompt counts are
    # expected not to match, and the realized composition of the *filtered* product is
    # recorded in sft_teacher.quality.json.
    for split in ("sft", "opd"):
        per_domain = counts.get(split) or {}
        if sorted(per_domain) != sorted(DOMAINS):
            problems.append(f"{split} pool covers domains {sorted(per_domain)}")
            continue
        if split != "opd":
            continue
        total = sum(int(value) for value in per_domain.values())
        for domain, count in per_domain.items():
            share = int(count) / total if total else 0.0
            if abs(share - expected_share.get(domain, 0.0)) > 0.02:
                problems.append(
                    f"opd pool domain {domain} is {share:.1%} of the pool, "
                    f"expected {expected_share.get(domain, 0.0):.1%}"
                )

    return {
        "applicable": True,
        "passed": not problems,
        "problems": problems,
        "expected_domain_share": expected_share,
        "residual_contamination": residual,
        "benchmarks_scanned": report.get("benchmarks"),
        "benchmark_items": report.get("benchmark_items"),
        "rejected_rows": report.get("rejected_rows"),
        "prompt_count_by_domain": counts,
        "disjointness": disjoint,
        "manifest": str(report_path),
    }


def _validate_mask_method(config: dict[str, Any]) -> dict[str, Any]:
    """Confirm each mask was built by the method the config claims.

    The two comparison arms differ in exactly one thing, so pointing an arm at the other's
    mask directory would silently produce two identical runs -- and a null result
    that looks like "the method does not help" rather than "the experiment did not
    run". Masks record their own provenance, so it is checkable.
    """
    expected = str(config["pruning"].get("method", "minitron"))
    ratios = _ratios(config)
    findings: dict[str, Any] = {"expected_method": expected, "per_ratio": {}}
    problems: list[str] = []
    for ratio in ratios:
        path = Path(_stage_mask(config, ratio))
        if not path.exists():
            continue
        mask = json.loads(path.read_text(encoding="utf-8"))
        # Activation masks predate the field and carry no pruning_method key.
        found = str(mask.get("pruning_method", "minitron"))
        entry = {
            "method": found,
            "intermediate_size": mask.get("intermediate_size"),
            "actual_total_parameter_reduction": mask.get("actual_total_parameter_reduction"),
        }
        findings["per_ratio"][str(ratio)] = entry
        if found != expected:
            problems.append(
                f"ratio {ratio}: mask was built by '{found}' but the config declares '{expected}'"
            )
    findings["problems"] = problems
    findings["passed"] = not problems
    return findings


def build_preflight(config: dict[str, Any]) -> dict[str, Any]:
    ratios = _ratios(config)
    sft_count = int(config["data"]["sft_num_prompts"])
    # The recovery pools are built directly at their target sizes, so there is no single
    # source file to split. Split-derived configurations carry source_path and
    # are validated against it.
    source_path = config["data"].get("source_path")
    source_rows = (
        len(pd.read_parquet(source_path, columns=["sample_id"])) if source_path else None
    )
    opd_expected = (
        int(config["data"]["opd_num_prompts"])
        if "opd_num_prompts" in config["data"]
        else (source_rows - sft_count if source_rows else None)
    )
    stage_dir = Path("STAGE_OUTPUT")
    sft_command = build_sft_command(
        config,
        student_model="STAGE_PRUNED_CHECKPOINT",
        stage_dir=stage_dir,
    )
    opd_command = build_opd_command(
        config,
        student_model="STAGE_SFT_CHECKPOINT",
        stage_dir=stage_dir,
    )
    split_files = _validate_split_files(config)
    teacher_data = _validate_teacher_data(config)
    recovery_pools = _validate_recovery_pools(config)
    mask_method = _validate_mask_method(config)
    modes = thinking_report(config)
    wandb_mode = str(config.get("wandb", {}).get("mode", "online"))
    max_steps, save_freq = _opd_step_budget(config)
    requested_gpus = max(
        int(config["sft"].get("n_gpus_per_node", 8)),
        int(config["opd"].get("n_gpus_per_node", 8)),
    )

    checks: dict[str, Any] = {
        "checked_at": utc_now(),
        "arm": "one_shot" if len(ratios) == 1 else "cascade",
        "cuda_device_count": torch.cuda.device_count(),
        "enough_gpus": torch.cuda.device_count() >= requested_gpus,
        "requested_gpus": requested_gpus,
        "training_backend": verify_training_backend(config),
        "opd_runtime": verify_opd_runtime(config),
        "sft_runtime": verify_sft_runtime(config),
        "ratios": ratios,
        "masks": {
            str(ratio): {
                "path": _stage_mask(config, ratio),
                "exists": Path(_stage_mask(config, ratio)).exists(),
            }
            for ratio in ratios
        },
        "source_rows": source_rows,
        "sft_rows": sft_count,
        "opd_rows": opd_expected,
        "valid_split_size": (
            # A split-derived pool must leave room for both halves. A prebuilt pool
            # states both sizes outright, so the meaningful check is that each is
            # positive and that the pools on disk match (split_matches_config); an
            # oversized sft_num_prompts is caught there, by the row counts, rather
            # than against a source file that does not exist.
            0 < sft_count and (opd_expected is None or opd_expected > 0)
            if source_rows is None
            else 0 < sft_count < source_rows
        ),
        "prepared_split": split_files,
        "shared_teacher_path": config["data"]["sft_teacher_path"],
        "teacher_data": teacher_data,
        "recovery_pools": recovery_pools,
        "mask_method": mask_method,
        "thinking": modes,
        "wandb_mode": wandb_mode,
        "valid_wandb_mode": wandb_mode in {"online", "offline", "disabled"},
        "wandb_usable": _wandb_is_usable(wandb_mode),
        "benchmark_scorers": _benchmark_scorers_importable(config),
        # SFT runs on the pinned verl trainer, reading the teacher parquet
        # directly. Thinking must come from the per-row enable_thinking column,
        # never from apply_chat_template_kwargs -- passing both makes
        # apply_chat_template raise on the first batch.
        "verl_sft_backend": "verl.trainer.sft_trainer" in sft_command,
        "sft_reads_teacher_parquet": any(
            item.startswith("data.train_files=") and config["data"]["sft_teacher_path"] in item
            for item in sft_command
        ),
        "sft_thinking_from_data_column": (
            "data.enable_thinking_key=enable_thinking" in sft_command
            and not any("apply_chat_template_kwargs" in item for item in sft_command)
        ),
        "sft_exports_hf_model": "checkpoint.save_contents=[model,optimizer,extra,hf_model]" in sft_command,
        "boundary_names": ["after_prune", "after_sft", "after_opd"],
        "opd_no_internal_validation": all(
            item in opd_command
            for item in [
                "trainer.val_before_train=False",
                "trainer.test_freq=-1",
                "trainer.log_val_generations=0",
            ]
        ),
        "opd_convergence_budget": (
            _extract(opd_command, "trainer.total_training_steps=")
            == f"trainer.total_training_steps={max_steps}"
            and _extract(opd_command, "trainer.save_freq=") == f"trainer.save_freq={save_freq}"
            and save_freq <= max_steps
        ),
        # OPD prompts do need the kwarg: verl's RLHFDataset has no per-row
        # thinking column, unlike the SFT dataset.
        "opd_thinking_matches_mode": (
            f"+data.apply_chat_template_kwargs.enable_thinking={thinking_enabled(config, 'opd')}"
            in opd_command
        ),
        "rethink_topk_opd": all(
            item in opd_command
            for item in [
                "algorithm.adv_estimator=token_reward_direct",
                f"+actor_rollout_ref.rollout.log_prob_top_k={int(config['opd'].get('topk', 16))}",
                "+actor_rollout_ref.rollout.top_k_strategy=only_stu",
                "+actor_rollout_ref.rollout.reward_weight_mode=student_p",
                "actor_rollout_ref.actor.use_kl_loss=False",
                "algorithm.use_kl_in_reward=False",
            ]
        ),
        "opd_gpus": int(config["opd"]["n_gpus_per_node"]),
        "opd_rollout_n": int(config["opd"]["rollout_n"]),
        "teacher_micro_batch_size_per_gpu": int(
            config["opd"]["teacher_micro_batch_size_per_gpu"]
        ),
        "sft_command": sft_command,
        "opd_command": opd_command,
    }

    # Each entry is (name, ok). Reporting names makes a failure self-describing
    # instead of "preflight failed, go read the JSON".
    required = [
        ("enough_gpus", checks["enough_gpus"]),
        ("masks_exist", all(item["exists"] for item in checks["masks"].values())),
        ("valid_split_size", checks["valid_split_size"]),
        (
            "split_matches_config",
            not split_files["prepared"]
            or (
                split_files["sft_rows"] == sft_count
                and (opd_expected is None or split_files["opd_rows"] == opd_expected)
                and split_files["sample_id_overlap"] == 0
            ),
        ),
        # A recovery-pool config without a clean pool manifest never reaches a GPU.
        ("recovery_pools_clean", recovery_pools["passed"]),
        ("mask_method_matches_config", mask_method["passed"]),
        (
            "teacher_data_matches_mode",
            teacher_data["matches_mode"] is not False,
        ),
        ("thinking_mode_consistent", modes["consistent"]),
        ("valid_wandb_mode", checks["valid_wandb_mode"]),
        ("wandb_usable", checks["wandb_usable"]["usable"]),
        ("benchmark_scorers_importable", checks["benchmark_scorers"]["usable"]),
        ("verl_sft_backend", checks["verl_sft_backend"]),
        ("sft_reads_teacher_parquet", checks["sft_reads_teacher_parquet"]),
        ("sft_thinking_from_data_column", checks["sft_thinking_from_data_column"]),
        ("sft_exports_hf_model", checks["sft_exports_hf_model"]),
        ("opd_no_internal_validation", checks["opd_no_internal_validation"]),
        ("opd_convergence_budget", checks["opd_convergence_budget"]),
        ("opd_thinking_matches_mode", checks["opd_thinking_matches_mode"]),
        ("rethink_topk_opd", checks["rethink_topk_opd"]),
    ]
    checks["failed_checks"] = [name for name, ok in required if not ok]
    checks["format_alignment_warnings"] = _format_alignment_warnings(teacher_data)
    checks["passed"] = not checks["failed_checks"]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks = build_preflight(load_yaml(args.config))
    dump_json(checks, args.output)
    for warning in checks["format_alignment_warnings"]:
        print(f"[preflight warning] {warning}", flush=True)
    if not checks["passed"]:
        raise RuntimeError(
            f"Cascade preflight failed: {', '.join(checks['failed_checks'])}. "
            f"Inspect {args.output}"
        )
    print(
        json.dumps(
            {
                "passed": True,
                "arm": checks["arm"],
                "ratios": checks["ratios"],
                "thinking_enabled": checks["thinking"]["resolved"]["sft"],
                "format_alignment_warnings": len(checks["format_alignment_warnings"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
