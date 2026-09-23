"""Quota sampling, contamination gating, and domain interleaving for the recovery pools.

This module owns everything that happens between "raw rows from a source" and
"a parquet the trainer reads". The ordering of those steps is the part that
matters, and it is fixed:

    extract -> dedupe -> **contamination filter** -> quota sample -> interleave

Filtering before sampling, not after, is a correctness requirement rather than an
optimization. If contaminated rows were removed after quotas were filled, every
drop would shrink the pool below its target and the domain balance the whole
experiment rests on would quietly stop holding. Filtering first means rejects are
backfilled from the same stratum and the quota lands exactly.

The two arms of this experiment differ *only* in which channels get pruned. Any
nondeterminism here would leak into that comparison, so every choice is derived
from a seeded hash of the row's own identity: the same source data yields the same
pool, in the same order, on any machine, whether or not a stratum needed
backfilling.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Sequence

from recal.data.contamination import ContaminationIndex, Match
from recal.data.pool_sources import SourceRow


def stable_rank(seed: int, *parts: Any) -> str:
    """Deterministic sort key for one row.

    Hash-based rather than RNG-based so a row's position never depends on how
    many rows were drawn before it. That property is what makes backfilling a
    depleted stratum safe: refilling changes which rows are taken, not the order
    of the ones that were already chosen.
    """
    payload = ":".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_rank(row: SourceRow, seed: int) -> str:
    return stable_rank(seed, row.source, row.source_id, row.prompt[:256])


# --------------------------------------------------------------------------- #
# Contamination gate
# --------------------------------------------------------------------------- #


def filter_contaminated(
    rows: Sequence[SourceRow],
    index: ContaminationIndex,
) -> tuple[list[SourceRow], list[dict[str, Any]]]:
    """Split rows into clean and contaminated, recording what each hit matched.

    Runs against every benchmark the comparison table reports, LiveCodeBench-v5
    included -- V1 never checked LCB at all because its problem statements were
    never written to disk, so "clean" there was an assumption rather than a
    measurement.
    """
    clean: list[SourceRow] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        matches: list[Match] = index.matches(row.prompt)
        if not matches:
            clean.append(row)
            continue
        strongest = matches[0]
        rejected.append(
            {
                "domain": row.domain,
                "source": row.source,
                "source_id": row.source_id,
                "benchmark": strongest.benchmark,
                "benchmark_index": strongest.benchmark_index,
                "level": strongest.level,
                "score": strongest.score,
                "matched_characters": strongest.matched_characters,
                "total_matches": len(matches),
            }
        )
    return clean, rejected


# --------------------------------------------------------------------------- #
# Quota sampling
# --------------------------------------------------------------------------- #


def sample_quota(
    rows: Sequence[SourceRow],
    quota: int,
    *,
    seed: int,
    stratum_key: str | None = None,
) -> tuple[list[SourceRow], dict[str, Any]]:
    """Take exactly ``quota`` rows, spread evenly across strata when asked.

    Even spreading is the point of the stratum argument. Left unstratified, the
    code pool would come out almost entirely ``VERY_HARD`` and the science pool
    almost entirely Biology, because those source files are *sorted* by exactly
    those keys -- so an unstratified sample of them reflects file layout, not the
    task distribution.

    Strata are filled in a largest-remainder pass and any shortfall is
    redistributed to strata that still have rows. A stratum with fewer rows than
    its share therefore does not cost the pool its quota; it costs it only that
    stratum's balance, which is reported.
    """
    ordered = sorted(rows, key=lambda row: _row_rank(row, seed))
    if quota > len(ordered):
        raise ValueError(
            f"Quota {quota} exceeds the {len(ordered)} available rows "
            f"(domain={ordered[0].domain if ordered else '?'}); widen the source sample"
        )
    if not stratum_key:
        selected = ordered[:quota]
        return selected, {
            "quota": quota,
            "available": len(ordered),
            "stratified_by": None,
            "selected": len(selected),
        }

    buckets: dict[str, list[SourceRow]] = defaultdict(list)
    for row in ordered:
        buckets[str(row.strata.get(stratum_key, ""))].append(row)

    # Round-robin across strata rather than computing shares up front: it fills
    # evenly, degrades gracefully when a stratum runs dry, and needs no remainder
    # bookkeeping. Strata are visited in sorted order for determinism.
    names = sorted(buckets)
    cursors = {name: 0 for name in names}
    selected: list[SourceRow] = []
    while len(selected) < quota:
        progressed = False
        for name in names:
            if len(selected) >= quota:
                break
            bucket = buckets[name]
            cursor = cursors[name]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                cursors[name] = cursor + 1
                progressed = True
        if not progressed:
            break
    if len(selected) != quota:
        raise ValueError(
            f"Stratified sampling produced {len(selected)} of {quota} rows; "
            "the pool is smaller than the quota"
        )
    realized = defaultdict(int)
    for row in selected:
        realized[str(row.strata.get(stratum_key, ""))] += 1
    return selected, {
        "quota": quota,
        "available": len(ordered),
        "stratified_by": stratum_key,
        "selected": len(selected),
        "available_per_stratum": {name: len(buckets[name]) for name in names},
        "selected_per_stratum": dict(sorted(realized.items())),
    }


# --------------------------------------------------------------------------- #
# Cross-split disjointness
# --------------------------------------------------------------------------- #


def normalized_key(text: str) -> str:
    from recal.data.contamination import normalize

    return normalize(text)


def drop_overlap_with(
    rows: Sequence[SourceRow],
    reserved: Iterable[str],
) -> tuple[list[SourceRow], int]:
    """Remove rows whose normalized prompt is already claimed by the other split.

    Compared on normalized text, not id. Two sources can carry the same problem
    under different ids -- ``Nemotron-SFT-Math-v4`` and the DAPO-derived
    ``RL-Math-v2`` demonstrably do -- and V1's id-only check could not see it. An
    SFT/OPD leak would let OPD train on problems the student was already fit to,
    which reads as recovery.
    """
    claimed = set(reserved)
    kept: list[SourceRow] = []
    dropped = 0
    for row in rows:
        if normalized_key(row.prompt) in claimed:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Domain interleaving
# --------------------------------------------------------------------------- #


def interleave_domains(
    rows_by_domain: dict[str, Sequence[SourceRow]],
    *,
    seed: int,
) -> list[SourceRow]:
    """Deterministic round-robin over domains, so row order is domain-balanced.

    verl's SFT trainer wraps the dataset in ``DistributedSampler(shuffle=True)``
    and the domains have very different response lengths, so a contiguous
    single-domain run would show up as a drifting token mix across the epoch.
    Interleaving at write time fixes the ordering without modifying the
    integrated training runtime.
    """
    ordered = {
        domain: sorted(rows, key=lambda row: _row_rank(row, seed))
        for domain, rows in rows_by_domain.items()
    }
    names = sorted(ordered)
    cursors = {name: 0 for name in names}
    total = sum(len(rows) for rows in ordered.values())
    merged: list[SourceRow] = []
    while len(merged) < total:
        for name in names:
            cursor = cursors[name]
            bucket = ordered[name]
            if cursor < len(bucket):
                merged.append(bucket[cursor])
                cursors[name] = cursor + 1
    return merged


def sample_id_for(row: SourceRow, split: str) -> str:
    """Stable per-row training id.

    Namespaced by split and source so an id collision cannot make an SFT row and
    an OPD row look like the same sample to the disjointness check.
    """
    digest = hashlib.sha256(
        f"{split}:{row.source}:{row.source_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{split}_{row.domain}_{digest}"
