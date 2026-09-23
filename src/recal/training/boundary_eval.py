from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from recal.common import dump_json, utc_now
from recal.modes import thinking_enabled


BOUNDARY_INDEX = {
    "after_prune": 0,
    "after_sft": 1,
    "after_opd": 2,
}


def run_boundary_benchmark(
    config: dict[str, Any],
    *,
    model: str,
    stage_dir: str | Path,
    boundary: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if boundary not in BOUNDARY_INDEX:
        raise ValueError(f"Unknown boundary: {boundary}")
    settings = config["boundary_benchmark"]
    thinking = thinking_enabled(config, "benchmark")
    output = Path(stage_dir) / "benchmarks" / boundary
    command = [
        sys.executable,
        "-m",
        "recal.evaluation.comprehensive_suite",
        "--model",
        model,
        "--output-dir",
        str(output),
        "--profile",
        str(settings.get("profile", "quick")),
        "--tensor-parallel-size",
        str(int(settings.get("tensor_parallel_size", 8))),
        "--aime-n",
        str(int(settings.get("aime_n", 1))),
        "--aime-temperature",
        str(float(settings.get("aime_temperature", 0.0))),
        "--aime-top-p",
        str(float(settings.get("aime_top_p", 1.0))),
        "--max-tokens",
        str(int(settings.get("max_tokens", 16384))),
        "--max-model-len",
        str(int(settings.get("max_model_len", 18432))),
        "--gpu-memory-utilization",
        str(float(settings.get("gpu_memory_utilization", 0.84))),
        "--thinking-enabled"
        if thinking
        else "--no-thinking-enabled",
    ]
    if settings.get("shard_size"):
        command.extend(["--shard-size", str(int(settings["shard_size"]))])
    tasks = settings.get("tasks")
    if tasks:
        command.extend(["--tasks", *map(str, tasks)])
    dump_json(
        {
            "boundary": boundary,
            "boundary_index": BOUNDARY_INDEX[boundary],
            "model": model,
            "thinking_enabled": thinking,
            "command": command,
            "started_at": utc_now(),
        },
        output / "boundary_launch.json",
    )
    if not dry_run:
        subprocess.run(command, check=True)
    summary_path = output / "summary.json"
    if dry_run:
        summary = {}
    elif summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        # The suite exited without writing its final summary, which means at
        # least one task never finished. Point at the status file it maintains
        # rather than raising a bare FileNotFoundError on summary.json.
        raise RuntimeError(
            f"Benchmark {boundary} produced no summary.json. "
            f"Inspect {output / 'benchmark_status.json'} for the failing task; "
            "completed tasks and generation shards are cached and will resume."
        )
    # Benchmark results are the authoritative record in summary.json; they are
    # deliberately not mirrored to W&B. A transient wandb.init network timeout
    # here previously aborted the whole cascade after training had finished.
    return summary
