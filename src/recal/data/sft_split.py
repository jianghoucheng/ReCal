from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recal.common import dump_json, load_yaml, sha256_file, utc_now


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _pythonize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [dict(message) for message in value]


def split_sft_and_opd(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    source_path = Path(data["source_path"])
    sft_path = Path(data["sft_prompt_path"])
    opd_path = Path(data["opd_path"])
    manifest_path = Path(data["split_manifest"])
    target = int(data.get("sft_num_prompts", 10000))
    seed = int(config.get("seed", 42))

    frame = pd.read_parquet(source_path)
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("source OPD data contains duplicate sample_id values")
    if not 0 < target < len(frame):
        raise ValueError(f"sft_num_prompts must be in (0, {len(frame)}), got {target}")

    source_sha256 = sha256_file(source_path)
    ranked = frame.assign(
        _split_rank=frame["sample_id"].astype(str).map(lambda value: _stable_rank(value, seed))
    ).sort_values("_split_rank")
    sft = ranked.iloc[:target].drop(columns=["_split_rank"]).copy()
    opd = ranked.iloc[target:].drop(columns=["_split_rank"]).copy()
    sft["split"] = "sft_teacher_prompt"
    opd["split"] = "opd_train"

    sft_ids = set(sft["sample_id"].astype(str))
    opd_ids = set(opd["sample_id"].astype(str))
    overlap = sft_ids & opd_ids
    if overlap:
        raise RuntimeError(f"SFT/OPD split overlap: {len(overlap)} samples")

    # Parquet byte output is not guaranteed to be stable across pyarrow
    # versions. Reuse a semantically identical split instead of rewriting it
    # and invalidating the already generated shared teacher data.
    if sft_path.exists() and opd_path.exists() and manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_sft_ids = pd.read_parquet(sft_path, columns=["sample_id"])[
                "sample_id"
            ].astype(str).tolist()
            existing_opd_ids = pd.read_parquet(opd_path, columns=["sample_id"])[
                "sample_id"
            ].astype(str).tolist()
            if (
                existing_manifest.get("seed") == seed
                and existing_manifest.get("source_sha256") == source_sha256
                and existing_sft_ids == sft["sample_id"].astype(str).tolist()
                and existing_opd_ids == opd["sample_id"].astype(str).tolist()
            ):
                return existing_manifest
        except Exception:
            pass

    for path in [sft_path, opd_path, manifest_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    sft.to_parquet(sft_path, index=False)
    opd.to_parquet(opd_path, index=False)
    manifest = {
        "created_at": utc_now(),
        "seed": seed,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_rows": len(frame),
        "sft_prompt_path": str(sft_path),
        "sft_rows": len(sft),
        "opd_path": str(opd_path),
        "opd_rows": len(opd),
        "sample_id_overlap": 0,
        "sft_domain_counts": {
            str(key): int(value) for key, value in sft["domain"].value_counts().items()
        },
        "opd_domain_counts": {
            str(key): int(value) for key, value in opd["domain"].value_counts().items()
        },
    }
    dump_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(split_sft_and_opd(config), indent=2))


if __name__ == "__main__":
    main()
