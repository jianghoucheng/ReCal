#!/usr/bin/env python
"""Build the balanced SFT and OPD recovery pools: fetch, filter, sample, interleave, generate.

    python scripts/prepare_recovery_data.py --config configs/qwen3_8b/minitron.yaml

Runs as a sequence of artifact-gated steps, following this repo's convention that
every product is detected before it is rebuilt: a rerun after any interruption
redoes at most the step that was in flight. The expensive step by far is teacher
generation (12K thinking-mode responses at up to 32k tokens), and it already has
its own shard-level resume inside ``offline_teacher.py``.

Step order is load-bearing:

  1. extract        eight sources, prompts only -- every dataset's own assistant
                    answer is discarded, because the SFT target is by definition
                    what *this* dense Qwen3-8B produces.
  2. dedupe         within each pool, by source id or normalized text.
  3. contaminate    scan against AIME 24/25/26, GPQA-D, IFBench, LCB-v5.
  4. quota          exact 25% per domain, stratified where the source file is
                    sorted by the key we need spread over.
  5. disjoint       OPD drops anything sharing normalized text with SFT.
  6. interleave     deterministic round-robin so row order is domain-balanced.
  7. teacher        dense Qwen3-8B generates SFT targets; truncated, malformed,
                    and degenerate responses are filtered out.

Steps 1-6 are CPU-only and cost minutes. Step 7 needs all 8 GPUs, so it is opt-in
via ``--with-teacher`` and is normally driven by the pipeline instead.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

from recal.common import dump_json, dump_jsonl, load_json, load_yaml, utc_now  # noqa: E402
from recal.data.contamination import (  # noqa: E402
    ContaminationIndex,
    disjointness_report,
    load_benchmark_items,
)
from recal.data.pool_sampling import (  # noqa: E402
    drop_overlap_with,
    filter_contaminated,
    interleave_domains,
    normalized_key,
    sample_id_for,
    sample_quota,
)
from recal.data.pool_sources import (  # noqa: E402
    DOMAINS,
    STRATUM_KEY,
    SourceRow,
    extractors_for,
)


def _cache_path(cache_dir: Path, split: str, domain: str) -> Path:
    return cache_dir / f"raw_{split}_{domain}.jsonl"


def _load_cached(path: Path) -> list[SourceRow] | None:
    """Rehydrate a cached extraction, or None when absent/unreadable.

    A corrupt cache is treated as a miss rather than an error: re-extracting costs
    minutes, while a hard failure here would block the build on a file that can
    simply be rebuilt.
    """
    if not path.exists():
        return None
    rows: list[SourceRow] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append(
                    SourceRow(
                        domain=record["domain"],
                        source=record["source"],
                        source_id=record["source_id"],
                        prompt=record["prompt"],
                        messages=record["messages"],
                        strata=record.get("strata", {}),
                    )
                )
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    return rows or None


def _save_cache(rows: list[SourceRow], path: Path) -> None:
    dump_jsonl(
        [
            {
                "domain": row.domain,
                "source": row.source,
                "source_id": row.source_id,
                "prompt": row.prompt,
                "messages": row.messages,
                "strata": row.strata,
            }
            for row in rows
        ],
        path,
    )


def extract_all(cache_dir: Path, *, splits: tuple[str, ...]) -> dict[str, dict[str, list[SourceRow]]]:
    pools: dict[str, dict[str, list[SourceRow]]] = {}
    for split in splits:
        pools[split] = {}
        for domain, extractor in extractors_for(split).items():
            path = _cache_path(cache_dir, split, domain)
            cached = _load_cached(path)
            if cached is not None:
                print(f"[extract] {split}/{domain}: {len(cached)} rows (cached)", flush=True)
                pools[split][domain] = cached
                continue
            print(f"[extract] {split}/{domain}: fetching...", flush=True)
            rows = extractor()
            _save_cache(rows, path)
            print(f"[extract] {split}/{domain}: {len(rows)} rows", flush=True)
            pools[split][domain] = rows
    return pools


def build_pools(config: dict[str, Any], *, output_dir: Path, seed: int) -> dict[str, Any]:
    data = config["data"]
    sft_total = int(data["sft_num_prompts"])
    opd_total = int(data["opd_num_prompts"])
    if sft_total % len(DOMAINS) or opd_total % len(DOMAINS):
        raise ValueError(
            f"Pool sizes must divide evenly across {len(DOMAINS)} domains; "
            f"got sft={sft_total}, opd={opd_total}"
        )
    sft_quota = sft_total // len(DOMAINS)
    opd_quota = opd_total // len(DOMAINS)

    cache_dir = output_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pools = extract_all(cache_dir, splits=("sft", "opd"))

    print("[contamination] loading benchmark questions...", flush=True)
    index = ContaminationIndex(load_benchmark_items())
    print(
        f"[contamination] indexed {len(index)} items over {index.benchmarks}",
        flush=True,
    )

    report: dict[str, Any] = {
        "created_at": utc_now(),
        "seed": seed,
        "quotas": {"sft_per_domain": sft_quota, "opd_per_domain": opd_quota},
        "benchmark_items": len(index),
        "benchmarks": index.benchmarks,
        "splits": {},
    }
    rejected_all: list[dict[str, Any]] = []
    selected: dict[str, dict[str, list[SourceRow]]] = {"sft": {}, "opd": {}}

    # SFT first: it defines the reserved prompt set that OPD must avoid, so its
    # quota is never disturbed by the disjointness pass.
    for split, quota in (("sft", sft_quota), ("opd", opd_quota)):
        split_report: dict[str, Any] = {}
        reserved = (
            {normalized_key(row.prompt) for rows in selected["sft"].values() for row in rows}
            if split == "opd"
            else set()
        )
        for domain in DOMAINS:
            rows = pools[split][domain]
            clean, rejected = filter_contaminated(rows, index)
            for entry in rejected:
                entry["split"] = split
            rejected_all.extend(rejected)
            overlap_dropped = 0
            if split == "opd":
                clean, overlap_dropped = drop_overlap_with(clean, reserved)
            chosen, statistics = sample_quota(
                clean,
                quota,
                seed=seed,
                stratum_key=STRATUM_KEY.get((split, domain)),
            )
            selected[split][domain] = chosen
            split_report[domain] = {
                "extracted": len(rows),
                "contaminated_dropped": len(rejected),
                "sft_overlap_dropped": overlap_dropped,
                "clean_available": len(clean),
                **statistics,
                "sources": sorted({row.source for row in chosen}),
            }
            print(
                f"[{split}/{domain}] extracted={len(rows)} "
                f"contaminated={len(rejected)} overlap={overlap_dropped} "
                f"-> selected={len(chosen)}",
                flush=True,
            )
        report["splits"][split] = split_report

    frames: dict[str, pd.DataFrame] = {}
    for split in ("sft", "opd"):
        merged = interleave_domains(selected[split], seed=seed)
        records = []
        for row in merged:
            record = row.as_dict()
            record["sample_id"] = sample_id_for(row, split)
            record["split"] = split
            if split == "sft":
                # verl's MultiTurnSFTDataset reads these two column names
                # directly; renaming either breaks the trainer.
                record["enable_thinking"] = bool(
                    config.get("model", {}).get("thinking_enabled", True)
                )
            else:
                # The pinned verl calls compute_reward() unconditionally, before the
                # OPD-only loss replaces its output, and its naive reward manager
                # indexes non_tensor_batch["reward_model"]["ground_truth"] directly.
                # Without these columns OPD dies on KeyError: 'reward_model' at the
                # first training step -- after SFT has already spent hours. The values
                # are placeholders: token_reward_direct supervises against the teacher's
                # per-token distribution and never reads a scalar reward.
                record["data_source"] = str(record["domain"])
                record["reward_model"] = {
                    "ground_truth": "",
                    "style": "recal_zero_reward",
                }
                record["extra_info"] = {"original_id": str(record.get("source_id", ""))}
            records.append(record)
        frame = pd.DataFrame(records)
        if frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{split} pool has duplicate sample_id values")
        frames[split] = frame

    disjoint = disjointness_report(
        frames["sft"]["prompt"].tolist(), frames["opd"]["prompt"].tolist()
    )
    if not disjoint["disjoint"]:
        raise RuntimeError(
            f"SFT and OPD pools share {disjoint['normalized_prompt_overlap']} prompts"
        )
    report["disjointness"] = disjoint
    report["contamination_hits_remaining"] = 0
    report["rejected_rows"] = len(rejected_all)

    sft_path = Path(data["sft_prompt_path"])
    opd_path = Path(data["opd_path"])
    for path, frame in ((sft_path, frames["sft"]), (opd_path, frames["opd"])):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        print(f"[write] {path}: {len(frame)} rows", flush=True)

    # Verify the written pools are still clean. Scanning the artifact rather than
    # the in-memory rows is the point: this is the check the preflight gate later
    # repeats, and it must be answerable from disk alone.
    residual: dict[str, int] = {}
    for split, frame in frames.items():
        _, hits = filter_contaminated(
            [
                SourceRow(
                    domain=str(record["domain"]),
                    source=str(record["source"]),
                    source_id=str(record["source_id"]),
                    prompt=str(record["prompt"]),
                    messages=[],
                )
                for record in frame.to_dict("records")
            ],
            index,
        )
        residual[split] = len(hits)
    if any(residual.values()):
        raise RuntimeError(f"Contamination survived quota sampling: {residual}")
    report["residual_contamination"] = residual

    per_domain = {
        split: {
            str(key): int(value)
            for key, value in frames[split]["domain"].value_counts().items()
        }
        for split in frames
    }
    report["prompt_count_by_domain"] = per_domain
    for split, counts in per_domain.items():
        expected = sft_quota if split == "sft" else opd_quota
        if set(counts) != set(DOMAINS) or any(value != expected for value in counts.values()):
            raise RuntimeError(f"{split} pool is not domain-balanced: {counts}")

    dump_json(report, output_dir / "pool_manifest.json")
    dump_jsonl(rejected_all, output_dir / "contamination_rejected.jsonl")
    dump_json(
        {
            "label": "recovery_pools",
            "created_at": utc_now(),
            "rejected_by_benchmark": _count_by(rejected_all, "benchmark"),
            "rejected_by_domain": _count_by(rejected_all, "domain"),
            "rejected_by_level": _count_by(rejected_all, "level"),
            "total_rejected": len(rejected_all),
            "residual_contamination": residual,
            "passed": not any(residual.values()),
        },
        output_dir / "contamination_report.json",
    )
    return report


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="data/recovery_pools")
    parser.add_argument(
        "--with-teacher",
        action="store_true",
        help="also run dense-teacher SFT target generation (needs all 8 GPUs)",
    )
    parser.add_argument("--force", action="store_true", help="ignore existing pool artifacts")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pool_manifest.json"
    seed = int(config.get("seed", 42))

    sft_path = Path(config["data"]["sft_prompt_path"])
    opd_path = Path(config["data"]["opd_path"])
    if manifest_path.exists() and sft_path.exists() and opd_path.exists() and not args.force:
        print(f"[skip] pools already built; {manifest_path}", flush=True)
        report = load_json(manifest_path)
    else:
        report = build_pools(config, output_dir=output_dir, seed=seed)

    print(json.dumps(report.get("prompt_count_by_domain", {}), indent=2), flush=True)

    if args.with_teacher:
        from recal.data.teacher_targets import build_sft_targets

        summary = build_sft_targets(config)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
