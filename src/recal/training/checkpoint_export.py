from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from recal.common import directory_fingerprint, dump_json, utc_now


def find_latest_actor_checkpoint(root: str | Path) -> Path:
    root = Path(root)
    candidates = []
    for path in root.glob("global_step_*/actor"):
        match = re.search(r"global_step_(\d+)", str(path))
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No verl actor checkpoints under {root}")
    return max(candidates)[1]


def export_fsdp_checkpoint(
    actor_checkpoint: str | Path,
    output_model: str | Path,
    *,
    python_executable: str | None = None,
) -> None:
    actor_checkpoint = Path(actor_checkpoint)
    output_model = Path(output_model)
    actor_fingerprint = directory_fingerprint(actor_checkpoint)
    manifest_path = output_model / "recal_export.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads((output_model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        if (
            manifest.get("actor_fingerprint") == actor_fingerprint
            and (output_model / "config.json").stat().st_size > 0
            and shards
            and all((output_model / shard).stat().st_size > 0 for shard in shards)
        ):
            return
    except Exception:
        pass
    temporary_output = output_model.with_name(f".{output_model.name}.partial-{os.getpid()}")
    shutil.rmtree(temporary_output, ignore_errors=True)
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2])
        + __import__("os").pathsep
        + env.get("PYTHONPATH", "")
    )
    command = [
        python_executable or sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_checkpoint),
        "--target_dir",
        str(temporary_output),
    ]
    subprocess.run(command, check=True, env=env)
    index = json.loads((temporary_output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    if not shards or not all((temporary_output / shard).stat().st_size > 0 for shard in shards):
        raise RuntimeError(f"Incomplete exported checkpoint: {temporary_output}")
    dump_json(
        {
            "actor_checkpoint": str(actor_checkpoint),
            "actor_fingerprint": actor_fingerprint,
            "exported_at": utc_now(),
        },
        temporary_output / "recal_export.json",
    )
    shutil.rmtree(output_model, ignore_errors=True)
    os.replace(temporary_output, output_model)


def copy_pruning_metadata(source_checkpoint: str | Path, target_checkpoint: str | Path) -> None:
    source = Path(source_checkpoint) / "pruning_manifest.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing pruning metadata in {source_checkpoint}")
    target = Path(target_checkpoint)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / "pruning_manifest.json")
