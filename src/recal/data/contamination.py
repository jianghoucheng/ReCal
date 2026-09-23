"""Evaluation-set contamination detection for the recovery training pools.

V1 shipped with no filter at all and leaked 47-60% of AIME verbatim into
training, which made every absolute AIME number unusable. The math pool is
*by design* DAPO-derived (``Nemotron-RL-Math-v2`` reconstructs 1,398 of its rows
from DAPO-Math-17k, the exact V1 leak source), so this module is the only thing
standing between the pools and the same failure. It runs before quota sampling,
not after, and ``cascade_preflight`` refuses to commit GPUs while any hit
remains.

Two levels, both on normalized text:

  exact       the whole benchmark question appears inside a training prompt, or
              vice versa. This is what caught V1: AIME 2026 Q1 shares 598
              normalized characters with a training row, character for
              character.
  near-dup    Jaccard similarity over 13-gram shingles. Catches the rewrites and
              reformattings that defeat containment -- a leak that changed
              "Find the number of" to "Determine how many" is still a leak.

Candidate generation is via an inverted shingle index and is *lossless* for the
exact level: if benchmark question q is contained in prompt p, then every
shingle of q is also a shingle of p, so any-shared-shingle cannot miss it.
Benchmark questions too short to produce a shingle are scanned directly.

``min_containment_chars`` is deliberately low. It exists only to stop a
degenerate training prompt from matching everything, and it is what separates
V1's real contamination from its reported figure. V1's check was

    if len(q) > 80 and any(q in k or k in q for k in training):

which length-guards the benchmark question but not the training prompt. One V1
row normalizes to the single character ``"4"``, a substring of nearly every AIME
question, and that row alone produces all 18 AIME-2024 hits, all 14 AIME-2025,
and 17 of the 18 AIME-2026. Requiring *both* sides to be substantial leaves
exactly one genuine leak: AIME 2026 Q1 (answer 277), whose 598 normalized
characters appear verbatim inside a training row -- V1's own worked example.

The near-duplicate level confirms it independently. Across all 90 AIME
questions the maximum 13-gram Jaccard against the V1 pool is 0.904 for that
question and then drops straight to 0.168, with nothing between. There is no
population of partial leaks hiding below the containment threshold, so V1's true
AIME contamination is 1/90, not 47-60%.

The floor is in characters, not words, because that is the unit the length
distribution separates in: the shortest scored benchmark question is 49
normalized characters (IFBench) and AIME-2025's shortest is 73, so any word
floor tight enough to kill the ``"4"`` pathology also drops real AIME questions.
The asymmetry favours over-filtering: a wrongly dropped prompt is refilled from
the same pool, while a missed leak invalidates every number in the paper.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Match the V1 reproduction recipe exactly (HANDOVER.md 6): drop everything that
# is not a lowercase alphanumeric or a space, then collapse whitespace. Anything
# fancier would stop reproducing the known-positive AIME 2026 hit, which is the
# only evidence this detector works at all.
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_WHITESPACE = re.compile(r"\s+")

DEFAULT_SHINGLE_SIZE = 13
DEFAULT_JACCARD_THRESHOLD = 0.6
# Measured shortest normalized benchmark question, by suite: IFBench 49 chars,
# AIME-2025 73, AIME-2024 100, GPQA-D 113, LCB-v5 402. A 48-character floor sits
# just under the real minimum, so nothing scored is excluded, while the V1
# pathology (a training prompt normalizing to "4") is cut decisively.
# Characters rather than words because that is the unit the distribution is
# separated in -- a word floor tight enough to matter also drops short AIME
# questions. See the module docstring.
DEFAULT_MIN_CONTAINMENT_CHARS = 48


def normalize(text: Any) -> str:
    """Canonical form for both levels of matching."""
    lowered = str(text).lower()
    return _WHITESPACE.sub(" ", _NON_ALNUM.sub(" ", lowered)).strip()


def prompt_hash(text: Any) -> str:
    """Stable identity for a prompt under normalization.

    Used for the SFT/OPD disjointness check. V1 compared ``sample_id`` only,
    which cannot see the same problem entering both splits from two sources.
    """
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(normalized: str, size: int = DEFAULT_SHINGLE_SIZE) -> set[str]:
    """Word-level n-gram set of already-normalized text."""
    tokens = normalized.split()
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


@dataclass(frozen=True)
class BenchmarkItem:
    benchmark: str
    index: int
    text: str
    answer: str | None = None

    @property
    def normalized(self) -> str:
        return normalize(self.text)


@dataclass
class Match:
    benchmark: str
    benchmark_index: int
    level: str
    score: float
    benchmark_answer: str | None
    matched_characters: int


def load_benchmark_items(
    *,
    cache_dir: str = ".cache/huggingface",
    include: Sequence[str] | None = None,
) -> list[BenchmarkItem]:
    """Question text for every benchmark the comparison table reports.

    Selection mirrors ``evaluation/comprehensive_suite.py`` so the filtered set
    is the set actually scored -- in particular LiveCodeBench is restricted to
    the same v5 contest-date window the suite uses. V1 never checked LCB at all
    because its problem statements were not written to disk; pulling them from
    the source dataset here removes that blind spot.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    wanted = set(include) if include else None

    def enabled(name: str) -> bool:
        return wanted is None or name in wanted

    items: list[BenchmarkItem] = []

    aime_specs = {
        "aime_2024": ("HuggingFaceH4/aime_2024", "problem", "answer"),
        "aime_2025": ("MathArena/aime_2025", "problem", "answer"),
        "aime_2026": ("MathArena/aime_2026", "problem", "answer"),
    }
    for name, (repo, problem_key, answer_key) in aime_specs.items():
        if not enabled(name):
            continue
        dataset = load_dataset(repo, split="train", cache_dir=cache_dir)
        for index, row in enumerate(dataset):
            items.append(
                BenchmarkItem(name, index, row[problem_key], str(row[answer_key]))
            )

    if enabled("gpqa_diamond"):
        dataset = load_dataset("fingertap/GPQA-Diamond", split="test", cache_dir=cache_dir)
        for index, row in enumerate(dataset):
            items.append(
                BenchmarkItem("gpqa_diamond", index, row["question"], str(row["answer"]))
            )

    if enabled("ifbench"):
        dataset = load_dataset("allenai/IFBench_test", split="train", cache_dir=cache_dir)
        for index, row in enumerate(dataset):
            items.append(BenchmarkItem("ifbench", index, row["prompt"], None))

    if enabled("livecodebench_v5"):
        rows: list[dict[str, Any]] = []
        for file_name in ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"]:
            path = hf_hub_download(
                "livecodebench/code_generation_lite",
                file_name,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        rows = sorted(rows, key=lambda row: row["contest_date"], reverse=True)
        rows = [row for row in rows if "2024-10-01" <= row["contest_date"][:10] <= "2025-02-28"]
        for index, row in enumerate(rows):
            items.append(
                BenchmarkItem("livecodebench_v5", index, row["question_content"], None)
            )

    if not items:
        raise RuntimeError("No benchmark items loaded; contamination filtering would be vacuous")
    return items


class ContaminationIndex:
    """Inverted 13-gram index over benchmark questions."""

    def __init__(
        self,
        items: Iterable[BenchmarkItem],
        *,
        shingle_size: int = DEFAULT_SHINGLE_SIZE,
        jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
        min_containment_chars: int = DEFAULT_MIN_CONTAINMENT_CHARS,
    ):
        self.shingle_size = int(shingle_size)
        self.jaccard_threshold = float(jaccard_threshold)
        self.min_containment_chars = int(min_containment_chars)
        self.items: list[BenchmarkItem] = list(items)
        self._normalized: list[str] = []
        self._shingles: list[set[str]] = []
        self._postings: dict[str, list[int]] = {}
        # Items with fewer tokens than one shingle produce no postings, so they
        # can never be reached through the index and must be scanned directly.
        self._unindexed: list[int] = []
        for position, item in enumerate(self.items):
            normalized = item.normalized
            item_shingles = shingles(normalized, self.shingle_size)
            self._normalized.append(normalized)
            self._shingles.append(item_shingles)
            if not item_shingles:
                self._unindexed.append(position)
                continue
            for shingle in item_shingles:
                self._postings.setdefault(shingle, []).append(position)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def benchmarks(self) -> list[str]:
        return sorted({item.benchmark for item in self.items})

    def matches(self, text: Any) -> list[Match]:
        """Every benchmark item this prompt is contaminated by.

        Returns the strongest level per benchmark item: an exact containment is
        reported as ``exact`` even when the near-dup threshold also fires, since
        the two levels are not independent evidence.
        """
        normalized = normalize(text)
        if not normalized:
            return []
        prompt_shingles = shingles(normalized, self.shingle_size)

        candidates: set[int] = set(self._unindexed)
        for shingle in prompt_shingles:
            posting = self._postings.get(shingle)
            if posting:
                candidates.update(posting)

        found: list[Match] = []
        for position in candidates:
            item_normalized = self._normalized[position]
            level: str | None = None
            score = 0.0
            matched_characters = 0
            # Containment either way: a training row may quote the benchmark
            # question with extra scaffolding, or be a bare substring of it.
            # Both sides must clear the floor -- guarding only the benchmark side
            # is the V1 bug.
            overlap = min(len(item_normalized), len(normalized))
            if overlap >= self.min_containment_chars and (
                item_normalized in normalized or normalized in item_normalized
            ):
                level = "exact"
                score = 1.0
                matched_characters = overlap
            else:
                similarity = jaccard(prompt_shingles, self._shingles[position])
                if similarity >= self.jaccard_threshold:
                    level = "near_duplicate"
                    score = similarity
                    matched_characters = overlap
            if level is None:
                continue
            item = self.items[position]
            found.append(
                Match(
                    benchmark=item.benchmark,
                    benchmark_index=item.index,
                    level=level,
                    score=round(score, 4),
                    benchmark_answer=item.answer,
                    matched_characters=matched_characters,
                )
            )
        found.sort(key=lambda match: (-match.score, match.benchmark, match.benchmark_index))
        return found


@dataclass
class ScanResult:
    """Which prompts are contaminated, and by what."""

    total_prompts: int = 0
    contaminated_indices: list[int] = field(default_factory=list)
    matches_by_prompt: dict[int, list[Match]] = field(default_factory=dict)

    @property
    def clean_indices(self) -> list[int]:
        contaminated = set(self.contaminated_indices)
        return [index for index in range(self.total_prompts) if index not in contaminated]

    def report(self, index: ContaminationIndex, *, label: str) -> dict[str, Any]:
        per_benchmark: dict[str, dict[str, Any]] = {}
        totals = {item.benchmark: 0 for item in index.items}
        for item in index.items:
            totals[item.benchmark] = totals.get(item.benchmark, 0)
        counts: dict[str, int] = {}
        for item in index.items:
            counts[item.benchmark] = counts.get(item.benchmark, 0) + 1
        hit_items: dict[str, set[int]] = {}
        hit_prompts: dict[str, set[int]] = {}
        levels: dict[str, dict[str, int]] = {}
        # Matches carried by a short span are the ones most likely to be shared
        # boilerplate rather than a leaked question (IFBench constraint
        # sentences). They are still dropped; recording them separately keeps
        # "we filtered 8 template collisions" distinguishable from "we filtered
        # 8 leaks" without needing a human to re-inspect.
        short_span: dict[str, int] = {}
        for prompt_index, matches in self.matches_by_prompt.items():
            for match in matches:
                hit_items.setdefault(match.benchmark, set()).add(match.benchmark_index)
                hit_prompts.setdefault(match.benchmark, set()).add(prompt_index)
                levels.setdefault(match.benchmark, {})
                levels[match.benchmark][match.level] = (
                    levels[match.benchmark].get(match.level, 0) + 1
                )
                if match.matched_characters < 200:
                    short_span[match.benchmark] = short_span.get(match.benchmark, 0) + 1
        for benchmark, total in sorted(counts.items()):
            hit = len(hit_items.get(benchmark, ()))
            per_benchmark[benchmark] = {
                "benchmark_items": total,
                "benchmark_items_hit": hit,
                # The headline number: what fraction of the *evaluation set* is
                # visible in training. This is the 47-60% V1 figure.
                "benchmark_hit_fraction": round(hit / total, 4) if total else 0.0,
                "training_prompts_hit": len(hit_prompts.get(benchmark, ())),
                "levels": levels.get(benchmark, {}),
                "short_span_hits": short_span.get(benchmark, 0),
                "hit_indices": sorted(hit_items.get(benchmark, ())),
            }
        return {
            "label": label,
            "total_prompts": self.total_prompts,
            "contaminated_prompts": len(self.contaminated_indices),
            "contaminated_fraction": (
                round(len(self.contaminated_indices) / self.total_prompts, 6)
                if self.total_prompts
                else 0.0
            ),
            "criterion": {
                "shingle_size": index.shingle_size,
                "jaccard_threshold": index.jaccard_threshold,
                "min_containment_chars": index.min_containment_chars,
            },
            "per_benchmark": per_benchmark,
        }


def scan_prompts(prompts: Sequence[Any], index: ContaminationIndex) -> ScanResult:
    """Flag every prompt matching any benchmark item."""
    result = ScanResult(total_prompts=len(prompts))
    for position, prompt in enumerate(prompts):
        matches = index.matches(prompt)
        if matches:
            result.contaminated_indices.append(position)
            result.matches_by_prompt[position] = matches
    return result


def disjointness_report(
    sft_prompts: Sequence[Any],
    opd_prompts: Sequence[Any],
) -> dict[str, Any]:
    """SFT and OPD must not share a problem, by normalized text not by id.

    The recovery setup requires disjoint SFT and OPD pools. Comparing
    ``sample_id`` alone cannot detect the same problem arriving in both splits
    from two different sources.
    """
    sft_hashes = {prompt_hash(prompt) for prompt in sft_prompts}
    opd_hashes = {prompt_hash(prompt) for prompt in opd_prompts}
    overlap = sft_hashes & opd_hashes
    return {
        "sft_prompts": len(sft_prompts),
        "opd_prompts": len(opd_prompts),
        "sft_unique_normalized": len(sft_hashes),
        "opd_unique_normalized": len(opd_hashes),
        "normalized_prompt_overlap": len(overlap),
        "disjoint": not overlap,
    }


def main() -> None:
    """Scan an existing prompt parquet, for the V1 self-check.

        python -m recal.data.contamination \\
            --prompts data/opd_train_aligned.parquet \\
            --output artifacts/pruning_comparison/contamination_v1_selfcheck.json

    Run this against V1 data before trusting any "clean" verdict on new pools: the
    detector must reproduce the known 47-60% AIME hit rate. A detector that
    reports V1 as clean is broken, and its verdict on the new pools is worthless.
    """
    import argparse

    import pandas as pd

    from recal.common import dump_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, help="parquet with a `prompt` column")
    parser.add_argument("--column", default="prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    parser.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    parser.add_argument(
        "--min-containment-chars", type=int, default=DEFAULT_MIN_CONTAINMENT_CHARS
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    frame = pd.read_parquet(args.prompts, columns=[args.column])
    prompts = frame[args.column].astype(str).tolist()
    if args.limit:
        prompts = prompts[: args.limit]
    index = ContaminationIndex(
        load_benchmark_items(include=args.benchmarks),
        shingle_size=args.shingle_size,
        jaccard_threshold=args.jaccard_threshold,
        min_containment_chars=args.min_containment_chars,
    )
    result = scan_prompts(prompts, index)
    report = result.report(index, label=str(args.prompts))
    report["benchmark_items"] = len(index)
    dump_json(report, Path(args.output))
    print(json.dumps(report["per_benchmark"], indent=2, sort_keys=True))
    print(
        f"{report['contaminated_prompts']}/{report['total_prompts']} prompts contaminated",
        flush=True,
    )


if __name__ == "__main__":
    main()
