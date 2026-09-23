#!/usr/bin/env python
"""Draw the baseline calibration sample from the SFT prompt pool.

    python scripts/prepare_calibration_data.py --config <arm config>

The unweighted pruning baselines use this file. Minitron follows its standard recipe -- one
logits-free prefill per calibration prompt, statistics over prompt positions -- so
it needs a standalone file of prompts. ReCal instead reads the dense teacher's own
reasoning trajectories from the SFT target parquet and scores response positions.

Two choices worth stating, because both affect whether the comparison is fair.

**Drawn from the SFT half, not the OPD half.** The mask must not be shaped by
anything the OPD stage will later train on; keeping calibration inside the SFT half
means the OPD prompts stay untouched by mask selection.

**Balanced across domains, deterministically.** The pool is already interleaved
25% per domain, so a contiguous head would be balanced by luck of the stride;
sampling per domain makes it balanced by construction. The seed comes from the
config, so the same config always yields the same calibration set and therefore the
same mask -- a rerun must not silently produce a different baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

from recal.common import dump_json, load_yaml, sha256_file, utc_now  # noqa: E402


def build_calibration_sample(config: dict) -> dict:
    pruning = config["pruning"]
    target = Path(pruning["calibration_path"])
    rows = int(pruning.get("calibration_rows", 1024))
    seed = int(config.get("seed", 42))

    source = Path(config["data"]["sft_prompt_path"])
    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing; build the pools with scripts/prepare_recovery_data.py first"
        )
    frame = pd.read_parquet(source)

    domains = sorted(frame["domain"].astype(str).unique())
    per_domain, remainder = divmod(rows, len(domains))
    selected = []
    for index, domain in enumerate(domains):
        pool = frame[frame["domain"].astype(str) == domain]
        # Spread the remainder over the first few domains rather than dropping it,
        # so the file has exactly `rows` and the count is reproducible.
        take = min(per_domain + (1 if index < remainder else 0), len(pool))
        selected.append(pool.sample(n=take, random_state=seed))
    calibration = pd.concat(selected, ignore_index=True)
    if len(calibration) != rows:
        raise RuntimeError(
            f"Calibration sample has {len(calibration)} rows, expected {rows}; "
            "a domain is smaller than its share"
        )

    # The collector reads these three columns: sample_id for the record, prompt for
    # reporting, messages for the chat template.
    missing = {"sample_id", "prompt", "messages"} - set(calibration.columns)
    if missing:
        raise RuntimeError(f"Calibration sample lacks required columns: {sorted(missing)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_parquet(target, index=False)
    manifest = {
        "created_at": utc_now(),
        "source": str(source),
        "output": str(target),
        "output_sha256": sha256_file(target),
        "rows": len(calibration),
        "seed": seed,
        "domain_counts": {
            str(key): int(value)
            for key, value in calibration["domain"].value_counts().items()
        },
    }
    dump_json(manifest, target.with_suffix(".manifest.json"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build_calibration_sample(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
