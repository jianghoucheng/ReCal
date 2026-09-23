from __future__ import annotations

from pathlib import Path
from typing import Any

from recal.common import (
    directory_fingerprint,
    dump_json,
    dump_yaml,
    environment_snapshot,
    load_yaml,
    utc_now,
)


EXPECTED_OUTPUTS = [
    "resolved_config.yaml",
    "environment.json",
    "preflight.json",
    "shared_training_data.json",
    "recovery_trajectory.parquet",
    "recovery_trajectory.csv",
    "recovery_summary.json",
    "final_checkpoint.txt",
]


def initialize_experiment(config_path: str, output_dir: str | None = None) -> tuple[dict[str, Any], Path]:
    config = load_yaml(config_path)
    name = config.get("experiment", {}).get("name") or Path(config_path).stem
    root = Path(output_dir or config.get("experiment", {}).get("output_dir", f"outputs/{name}"))
    root.mkdir(parents=True, exist_ok=True)
    for child in ["stages", "logs"]:
        (root / child).mkdir(exist_ok=True)
    dump_yaml(config, root / "resolved_config.yaml")
    dump_json(environment_snapshot(), root / "environment.json")
    return config, root


def stage_manifest(
    *,
    stage_id: int,
    stage_name: str,
    source_checkpoint: str,
    output_checkpoint: str,
    mask_path: str,
    target_ratio: float,
    status: str,
    training_budget: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    source = Path(source_checkpoint)
    output = Path(output_checkpoint)
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "status": status,
        "created_at": utc_now(),
        "source_checkpoint": str(source),
        "source_checkpoint_hash": directory_fingerprint(source) if source.exists() else None,
        "output_checkpoint": str(output),
        "output_checkpoint_hash": directory_fingerprint(output) if output.exists() else None,
        "mask_path": mask_path,
        "target_ratio": target_ratio,
        "training_budget": training_budget or {},
        "error": error,
    }


def describe_method(config: dict[str, Any]) -> str:
    """One-line description of the arm this experiment actually runs.

    Derived from the config rather than hard-coded: the same code path serves a
    multi-stage cumulative cascade and a single-shot control arm, in thinking or
    non-thinking mode, and a summary that always claimed "cumulative 10%" was
    wrong for every run except the first.
    """
    ratios = [float(value) for value in config.get("pruning", {}).get("ratios", [])]
    thinking = bool(config.get("model", {}).get("thinking_enabled", True))
    mode = "thinking" if thinking else "non-thinking"
    percentages = ", ".join(f"{round(value * 100)}%" for value in ratios) or "unspecified"
    if len(ratios) == 1:
        shape = f"one-shot {percentages} whole-model parameter pruning via FFN width"
    else:
        shape = (
            f"cumulative whole-model parameter pruning via FFN width at {percentages}"
        )
    return f"{shape} -> shared offline-teacher SFT -> Top-k OPD ({mode} mode)"


def write_summary(experiment_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    root = Path(experiment_dir)
    config = load_yaml(root / "resolved_config.yaml") if (root / "resolved_config.yaml").exists() else {}
    completed = {name: (root / name).exists() for name in EXPECTED_OUTPUTS}
    stages = sorted((root / "stages").glob("*/stage_manifest.json")) if (root / "stages").exists() else []
    lines = [
        f"# {config.get('experiment', {}).get('name', root.name)}",
        "",
        f"- Method: `{describe_method(config)}`",
        f"- Teacher: `{config.get('model', {}).get('teacher_path', 'unknown')}`",
        f"- Student/source: `{config.get('model', {}).get('student_path', 'unknown')}`",
        f"- Pruning ratios: `{config.get('pruning', {}).get('ratios', [])}`",
        f"- Thinking enabled: `{config.get('model', {}).get('thinking_enabled', 'unknown')}`",
        f"- Stage manifests: `{len(stages)}`",
        "",
        "## Completion",
        "",
    ]
    lines.extend(f"- [{'x' if ok else ' '}] `{name}`" for name, ok in completed.items())
    if extra:
        lines += ["", "## Notes", ""]
        lines.extend(f"- **{k}**: {v}" for k, v in extra.items())
    path = root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
