"""Eight recovery-pool extractors, one per (split, domain) pair.

Every extractor returns the same row shape, so the pool builder can treat the
sources uniformly and the only source-specific knowledge lives here:

    {"domain":    math | code | science | instruction_following,
     "source":    HF repo id,
     "source_id": stable id within that source (uuid / question_id / hash_id / key),
     "prompt":    bare question text, no assistant content, no format instruction,
     "messages":  [{"role": "user", "content": prompt + canonical suffix}],
     ...}        plus per-source stratification keys (difficulty, topic, constraints)

Three source properties drive nearly every design decision below.

**The big files are sorted, not shuffled.** ``Nemotron-SFT-Science-v2/vendor.jsonl``
(26.6 GB) is clustered by topic -- its first 397 rows and its tail are entirely
Biology -- and the competitive-programming shards (24 GB each) are clustered by
difficulty, with the whole middle at ``VERY_HARD`` and the 1500-2700 rating bands
only at the two ends. Reading a prefix of either file does not sample it; it
selects a stratum. So ``sample_jsonl_offsets`` spreads HTTP Range reads across the
whole file and every extractor that touches a clustered source declares the
stratification key it needs coverage of.

**We need 12K + 24K prompts out of ~145 GB, on a disk that is 93% full.**
``load_dataset()`` on these repos would download tens of gigabytes to select a
fraction of a percent. Range reads pull a few hundred MB per source instead. Only
the four genuinely small sources (RL-Math 0.01 GB, RL-Science 0.27 GB, IF-OPD
0.07 GB, and the 12 SFT-Math parquet shards) are fetched whole.

**A Range read starts and ends mid-line.** Both boundary fragments are discarded
rather than repaired: a half-parsed question is worse than a missing one when the
pool is 10x larger than the quota.

One source is not directly usable at all. ``Nemotron-RL-Math-v2`` ships 7,732
rows of which 3,984 (51.5%) have an empty ``question`` -- the DAPO-Math-17k and
Skywork-OR1 rows are masked pending reconstruction from the public originals. The
repo's own ``fill_placeholders.py`` is a ``uv run`` script, so its semantics are
reimplemented in ``restore_rl_math_placeholders`` below rather than shelled out
to. That reconstruction is what puts DAPO -- V1's AIME contamination source --
back into the math pool, which is why ``contamination.py`` is a hard gate and
runs before quota sampling, not after.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from recal.data.prompt_formats import (
    align_prompt,
    messages_for,
    prompt_is_clean,
    strip_code_harness,
    strip_format_instructions,
)

# --------------------------------------------------------------------------- #
# Repo ids
# --------------------------------------------------------------------------- #

SFT_MATH_REPO = "nvidia/Nemotron-SFT-Math-v4"
SFT_CODE_REPO = "nvidia/Nemotron-SFT-Competitive-Programming-v2"
SFT_SCIENCE_REPO = "nvidia/Nemotron-SFT-Science-v2"
SFT_IF_REPO = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"

OPD_MATH_REPO = "nvidia/Nemotron-RL-Math-v2"
OPD_CODE_REPO = "nvidia/Nemotron-RL-coding-competitive_coding"
OPD_SCIENCE_REPO = "nvidia/Nemotron-RL-Science-v1"
OPD_IF_REPO = "allenai/IF_multi_constraints_upto5"

DOMAINS = ("math", "code", "science", "instruction_following")

# Science topics we keep. "Math" is present in the science source and is excluded
# so the science and math domains stay disjoint -- otherwise the 25%/25% split
# would silently be math-heavier than it claims.
SCIENCE_TOPICS = ("Physics", "Chemistry", "Biology")


# --------------------------------------------------------------------------- #
# Range-based sampling of very large JSONL files
# --------------------------------------------------------------------------- #


def _proxies() -> dict[str, str] | None:
    """External egress on these nodes only works through the HTTP proxy."""
    http = os.environ.get("http_proxy")
    https = os.environ.get("https_proxy") or http
    if not http and not https:
        return None
    return {"http": http or https, "https": https or http}


def file_size(repo: str, filename: str) -> int:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            if not sibling.size:
                raise RuntimeError(f"{repo}/{filename} reports no size; cannot plan Range reads")
            return int(sibling.size)
    raise FileNotFoundError(f"{repo} has no file {filename}")


def sample_jsonl_at(
    repo: str,
    filename: str,
    byte_offset: int,
    *,
    length: int = 8_000_000,
    timeout: int = 400,
) -> list[dict[str, Any]]:
    """Parse whatever complete JSONL rows live in one byte window.

    The first and last lines of the window are almost always cut mid-row, so both
    are dropped unconditionally (``[1:-1]``). Undecodable rows in between are
    skipped rather than raised on: these files are large enough that a single bad
    row should not fail a build, and the pool is far larger than the quota.
    """
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, filename, repo_type="dataset")
    response = requests.get(
        url,
        headers={"Range": f"bytes={byte_offset}-{byte_offset + length}"},
        proxies=_proxies(),
        timeout=timeout,
    )
    response.raise_for_status()
    rows: list[dict[str, Any]] = []
    for line in response.text.split("\n")[1:-1]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sample_jsonl_offsets(
    repo: str,
    filename: str,
    *,
    offsets: int = 16,
    length: int = 8_000_000,
    total_size: int | None = None,
) -> list[dict[str, Any]]:
    """Read ``offsets`` evenly spread windows from one large JSONL file.

    Even spreading is the whole point: these files are sorted by the very key we
    need variety in (topic, difficulty), so any contiguous read is a biased
    sample. Windows are placed across ``[0, size - length]`` inclusive of both
    ends, because on the code shards the easy-difficulty rows exist *only* at the
    two extremes.
    """
    size = int(total_size if total_size is not None else file_size(repo, filename))
    span = max(size - length, 0)
    if offsets <= 1 or span == 0:
        positions = [0]
    else:
        positions = [int(round(i * span / (offsets - 1))) for i in range(offsets)]
    rows: list[dict[str, Any]] = []
    for position in dict.fromkeys(positions):
        rows.extend(sample_jsonl_at(repo, filename, position, length=length))
    return rows


def _van_der_corput(index: int, base: int = 2) -> float:
    """Low-discrepancy point in [0, 1). Any prefix of the sequence is spread out.

    This is what lets sampling be both *incremental* and *spread*: reading windows
    at 0, 1/2, 1/4, 3/4, 1/8, ... means stopping after any number of windows still
    leaves the file evenly covered. A linear scan of offsets has neither property --
    stopping early would leave the tail of a difficulty-sorted file unread.
    """
    result = 0.0
    denominator = 1.0
    value = index
    while value > 0:
        denominator *= base
        value, remainder = divmod(value, base)
        result += remainder / denominator
    return result


def sample_jsonl_until(
    repo: str,
    filename: str,
    *,
    need: int,
    key_of: Callable[[dict[str, Any]], str | None],
    window: int = 32_000_000,
    max_windows: int = 128,
    total_size: int | None = None,
) -> list[dict[str, Any]]:
    """Read spread-out windows until ``need`` distinct keys are seen.

    Fixed window budgets cannot work across these sources because row sizes differ
    by three orders of magnitude. A competitive-programming row carries a full
    solution and averages ~285 KB, so an 8 MB window yields roughly 28 rows and
    reaching a 3,000-problem quota would need ~800 MB read -- while a 0.6 MB
    instruction-following row fills the same quota from a single window. Sizing by
    hand per source would encode those measurements as constants that silently rot
    when the publisher reshards.

    So the target is stated in rows and the reads expand to meet it: windows are
    drawn in van der Corput order (spread at every prefix length) until the
    distinct-key count reaches ``need`` or the window budget runs out. Returning
    short is not an error here -- the caller's quota check is the authority on
    whether the pool is large enough, and it reports the shortfall with context
    this function does not have.
    """
    size = int(total_size if total_size is not None else file_size(repo, filename))
    span = max(size - window, 0)
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    seen_offsets: set[int] = set()
    for index in range(max_windows):
        if len(keys) >= need:
            break
        offset = int(_van_der_corput(index) * span) if span else 0
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        for record in sample_jsonl_at(repo, filename, offset, length=window):
            key = key_of(record)
            if key is None:
                continue
            rows.append(record)
            keys.add(key)
    return rows


def download_jsonl(repo: str, filename: str) -> Iterator[dict[str, Any]]:
    """Stream a fully downloaded JSONL file, for sources small enough to fetch."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# --------------------------------------------------------------------------- #
# Shared row helpers
# --------------------------------------------------------------------------- #


@dataclass
class SourceRow:
    domain: str
    source: str
    source_id: str
    prompt: str
    messages: list[dict[str, str]]
    strata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "source": self.source,
            "source_id": str(self.source_id),
            "prompt": self.prompt,
            "messages": self.messages,
            **{f"stratum_{key}": value for key, value in self.strata.items()},
        }


# A usable prompt has to be long enough to be a real task. The floor is
# deliberately low -- it only rejects empty and stub rows (the masked RL-Math rows
# normalize to ""), not short-but-real questions.
_MIN_PROMPT_CHARS = 24


def _first_user_content(messages: Any) -> str:
    """User text from a messages list, ignoring system and assistant turns.

    Assistant content is dropped everywhere in this module: this pipeline regenerates every
    SFT target with the dense Qwen3-8B teacher, so a source's own answer is never
    used and must never leak into a prompt.
    """
    if messages is None:
        return ""
    items = list(messages)
    for item in items:
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content", ""))
    return ""


def _system_content(messages: Any) -> str:
    if messages is None:
        return ""
    for item in list(messages):
        if isinstance(item, dict) and item.get("role") == "system":
            return str(item.get("content", ""))
    return ""


def _text_domain_row(
    *,
    domain: str,
    source: str,
    source_id: Any,
    problem: str,
    strata: dict[str, Any] | None = None,
) -> SourceRow | None:
    """Build a row for the suffix-based domains (math, science, IF).

    Rows whose aligned prompt still carries a second, competing answer-format
    instruction are dropped rather than force-stripped. The stripper is
    deliberately conservative around LaTeX (truncating a question is worse than
    leaving a redundant sentence), so a residue survives on ~0.1% of math and
    ~2.5% of science rows. Dropping them costs nothing at this pool size and a
    prompt demanding two answer formats is exactly the V1 failure.
    """
    problem = str(problem or "").strip()
    if len(problem) < _MIN_PROMPT_CHARS:
        return None
    aligned = align_prompt(problem, domain)
    if not prompt_is_clean(aligned, domain):
        return None
    body = strip_format_instructions(problem)
    if len(body) < _MIN_PROMPT_CHARS:
        return None
    return SourceRow(
        domain=domain,
        source=source,
        source_id=str(source_id),
        prompt=body,
        messages=messages_for(body, domain),
        strata=dict(strata or {}),
    )


def _code_row(
    *,
    source: str,
    source_id: Any,
    question: str,
    strata: dict[str, Any] | None = None,
) -> SourceRow | None:
    """Build a code row under LiveCodeBench's official template.

    Nemotron's own harness wrapper is stripped first. Leaving it would stack two
    output contracts: theirs asks for a bare ```python block, ours is the LCB
    template the grader parses.
    """
    body = strip_code_harness(str(question or ""))
    body = strip_format_instructions(body).strip()
    if len(body) < _MIN_PROMPT_CHARS:
        return None
    return SourceRow(
        domain="code",
        source=source,
        source_id=str(source_id),
        prompt=body,
        # Starter code is absent from both Nemotron code sources (they are
        # stdin-style competitive programming), so the stdin branch is correct
        # here. build_code_prompt still takes the argument for LCB parity.
        messages=messages_for(body, "code"),
        strata=dict(strata or {}),
    )


def _dedupe(rows: Iterable[SourceRow], *, key: Callable[[SourceRow], str]) -> list[SourceRow]:
    """Keep the first row per key, preserving encounter order."""
    seen: set[str] = set()
    unique: list[SourceRow] = []
    for row in rows:
        identity = key(row)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(row)
    return unique


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalized_text_key(row: SourceRow) -> str:
    return _NORMALIZE_RE.sub(" ", row.prompt.lower()).strip()


def source_id_key(row: SourceRow) -> str:
    return f"{row.source}:{row.source_id}"


# --------------------------------------------------------------------------- #
# SFT extractors
# --------------------------------------------------------------------------- #


def extract_sft_math(*, shards: int = 3, shard_offset: int = 0) -> list[SourceRow]:
    """``Nemotron-SFT-Math-v4`` questions, deduplicated by normalized text.

    Twelve ~0.47 GB parquet shards; a few are plenty for a 3K quota, and each is
    read column-projected so the (large) reasoning responses never leave disk.
    That response is discarded by design -- the dense Qwen3-8B teacher regenerates
    every SFT target.

    ``shard_offset`` exists so the OPD supplement can read different shards than
    the SFT draw (see ``extract_opd_math``), keeping the two draws disjoint by
    construction rather than relying only on the downstream overlap filter.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows: list[SourceRow] = []
    for index in range(shard_offset, min(shard_offset + shards, 12)):
        path = hf_hub_download(
            SFT_MATH_REPO,
            f"data/train-{index:05d}-of-00012.parquet",
            repo_type="dataset",
        )
        try:
            frame = pd.read_parquet(path, columns=["problem", "uuid", "source", "subset"])
        except OSError as error:
            # A shard interrupted mid-download deserializes as "Invalid data" rather
            # than as a missing file, and the HF cache will happily serve the
            # truncated blob forever. Re-fetch once, then move on: this pool needs a
            # few thousand rows out of millions, so one unreadable shard must not
            # fail the build.
            print(f"[sft-math] shard {index} unreadable ({error}); re-fetching", flush=True)
            Path(path).unlink(missing_ok=True)
            try:
                path = hf_hub_download(
                    SFT_MATH_REPO,
                    f"data/train-{index:05d}-of-00012.parquet",
                    repo_type="dataset",
                    force_download=True,
                )
                frame = pd.read_parquet(
                    path, columns=["problem", "uuid", "source", "subset"]
                )
            except OSError:
                print(f"[sft-math] shard {index} still unreadable; skipping", flush=True)
                continue
        for record in frame.to_dict("records"):
            row = _text_domain_row(
                domain="math",
                source=SFT_MATH_REPO,
                source_id=record.get("uuid"),
                problem=record.get("problem"),
                strata={"subset": str(record.get("subset") or "")},
            )
            if row is not None:
                rows.append(row)
    return _dedupe(rows, key=normalized_text_key)


def extract_sft_code(*, need: int = 9_000) -> list[SourceRow]:
    """Python competitive-programming questions, one row per problem.

    Two things this must not do. It must not sample rows uniformly at random --
    the file holds several accepted solutions per problem, so row-uniform
    sampling would silently weight popular problems and admit near-duplicate
    prompts; dedup is by ``question_id`` *before* quota sampling. And it must not
    read contiguously -- the shard is sorted by difficulty with ``VERY_HARD``
    filling the middle, so the ratings spread only appears when windows are
    spread across the whole file.

    ``need`` overshoots the 3,000 quota because this domain loses the most rows
    downstream: contamination against LiveCodeBench-v5 is a live risk here (a
    third-party audit found DeepCoder 88% contaminated against LCB-v5), and code
    teacher responses are the longest of any domain, so the quality filter takes a
    larger bite. Overshooting a Range read costs bandwidth; running short costs a
    rebuild.
    """
    rows: list[SourceRow] = []
    per_shard = max(need // 2, 1)
    for shard in (
        "data/competitive_programming_python_00.jsonl",
        "data/competitive_programming_python_01.jsonl",
    ):
        for record in sample_jsonl_until(
            SFT_CODE_REPO,
            shard,
            need=per_shard,
            key_of=lambda record: (
                str(record.get("question_id") or record.get("uuid") or "") or None
            ),
        ):
            if record.get("tools"):
                continue
            row = _code_row(
                source=SFT_CODE_REPO,
                source_id=record.get("question_id") or record.get("uuid"),
                question=_first_user_content(record.get("messages")),
                strata={
                    "difficulty": str(record.get("difficulty") or ""),
                    "source_name": str(record.get("source") or ""),
                },
            )
            if row is not None:
                rows.append(row)
    return _dedupe(rows, key=source_id_key)


def extract_sft_science(*, need: int = 12_000) -> list[SourceRow]:
    """Physics/Chemistry/Biology open questions from ``Nemotron-SFT-Science-v2``.

    ``vendor.jsonl`` is topic-clustered (pure Biology at both ends, mixed only in
    the middle), so windows are spread across the file and ``rqa.jsonl`` is added
    as a second pool to keep the per-topic quotas fillable. Rows carrying
    ``tools`` are excluded: they expect a Python or search tool we do not provide
    at rollout, so the teacher would be answering a different task than the
    student ever sees.

    ``need`` is per file and counts rows before topic filtering, which discards
    everything tagged ``Math`` -- a large share of ``vendor.jsonl`` -- so the
    surviving Physics/Chemistry/Biology pool is well under the target.
    """
    rows: list[SourceRow] = []
    for filename in ("vendor.jsonl", "rqa.jsonl"):
        for record in sample_jsonl_until(
            SFT_SCIENCE_REPO,
            filename,
            need=need,
            key_of=lambda record: str(record.get("uuid") or "") or None,
        ):
            metadata = record.get("metadata") or {}
            topic = str(metadata.get("topic") or "")
            if topic not in SCIENCE_TOPICS:
                continue
            if str(metadata.get("question_format") or "") != "OpenQ":
                continue
            if record.get("tools"):
                continue
            row = _text_domain_row(
                domain="science",
                source=SFT_SCIENCE_REPO,
                source_id=record.get("uuid"),
                problem=_first_user_content(record.get("messages")),
                strata={"topic": topic, "subtopic": str(metadata.get("subtopic") or "")},
            )
            if row is not None:
                rows.append(row)
    return _dedupe(rows, key=normalized_text_key)


def extract_sft_if(*, need: int = 9_000) -> list[SourceRow]:
    """Instruction-following prompts from ``data/instruction_following.jsonl``.

    ``chat.jsonl`` (17 GB) is deliberately not used: it is open-ended chat, not
    verifiable constraint following, and this domain is scored by executing
    constraint verifiers.

    The source's system message is preserved. For every other domain the system
    turn is boilerplate, but here it frequently *carries part of the constraint*,
    so dropping it would change the task the teacher answers.
    """
    rows: list[SourceRow] = []
    for record in sample_jsonl_until(
        SFT_IF_REPO,
        "data/instruction_following.jsonl",
        need=need,
        key_of=lambda record: str(record.get("uuid") or "") or None,
    ):
        user = _first_user_content(record.get("messages"))
        if len(user.strip()) < _MIN_PROMPT_CHARS:
            continue
        system = _system_content(record.get("messages"))
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        # No format suffix for IF: an added instruction would itself be an
        # unrequested constraint the verifier could mark as a violation.
        messages.append({"role": "user", "content": user})
        metadata = record.get("metadata") or {}
        rows.append(
            SourceRow(
                domain="instruction_following",
                source=SFT_IF_REPO,
                source_id=str(record.get("uuid")),
                prompt=user,
                messages=messages,
                strata={"seed_dataset": str(metadata.get("seed_dataset") or "")},
            )
        )
    return _dedupe(rows, key=normalized_text_key)


# --------------------------------------------------------------------------- #
# OPD extractors
# --------------------------------------------------------------------------- #

# Reconstruction constants, copied verbatim from the dataset's own
# fill_placeholders.py so the restored text is byte-identical to what the
# publisher intended. Paraphrasing these would leave wrapper fragments in the
# question.
_DAPO_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
_SKYWORK_REPO = "Skywork/Skywork-OR1-RL-Data"
_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem."
)
_DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'
_PLACEHOLDER_KEY = "_hf_question_placeholder"


def _strip_dapo_wrapper(text: str) -> str:
    body = text
    if _DAPO_PREFIX in body:
        body = body.split(_DAPO_PREFIX, 1)[1]
    if _DAPO_SUFFIX in body:
        body = body.rsplit(_DAPO_SUFFIX, 1)[0]
    return body.strip()


def _bare_question(dataset: str, content: str) -> str:
    if dataset == _DAPO_REPO:
        return _strip_dapo_wrapper(content)
    return content.strip()


def _unwrap_answer(raw: Any) -> str:
    """Bare answer text. Skywork stores ``'["5"]'``; DAPO stores the value bare."""
    if not isinstance(raw, str):
        if isinstance(raw, list) and raw:
            return str(raw[0])
        return str(raw)
    text = raw.strip()
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    ):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(value, list) and value:
            return str(value[0])
        return str(value)
    return text


def _reconstruct_question(placeholder: dict[str, Any], bare: str) -> str:
    if placeholder.get("mode") == "canonical":
        # The publisher reformatted the original, so the bare public text is
        # re-wrapped in the stored NVIDIA scaffolding.
        return placeholder.get("lead", "") + bare + placeholder.get("trail", "")
    # "exact" (the default, and what all 3,984 masked rows use): the stored
    # prefix/suffix reproduce the original text literally.
    return placeholder.get("prefix", "") + bare + placeholder.get("suffix", "")


def restore_rl_math_placeholders(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill the 3,984 masked ``Nemotron-RL-Math-v2`` questions from public sources.

    Reimplements the repo's ``fill_placeholders.py`` (a ``uv run`` script, so not
    importable) with identical semantics: DAPO-wrapper stripping, exact/canonical
    reconstruction, and Skywork's JSON-list answer unwrapping.

    Without this, 51.5% of the math OPD pool is empty strings and the 6K quota
    cannot be met from the 3,748 unmasked rows. With it, DAPO-Math-17k text --
    V1's AIME leak source -- is present in the pool, which is precisely why
    contamination filtering is a hard preflight gate rather than a report.
    """
    from datasets import load_dataset

    needed = {
        (placeholder["dataset"], placeholder["split"])
        for row in rows
        if (placeholder := row.get(_PLACEHOLDER_KEY))
    }
    sources = {key: load_dataset(key[0], split=key[1]) for key in sorted(needed)}

    restored: list[dict[str, Any]] = []
    for row in rows:
        placeholder = row.get(_PLACEHOLDER_KEY)
        if not placeholder:
            restored.append(row)
            continue
        dataset = sources[(placeholder["dataset"], placeholder["split"])]
        record = dataset[int(placeholder["row"])]
        bare = _bare_question(placeholder["dataset"], record["prompt"][0]["content"])
        filled = dict(row)
        filled.pop(_PLACEHOLDER_KEY, None)
        filled["question"] = _reconstruct_question(placeholder, bare)
        filled["expected_answer"] = _unwrap_answer(
            (record.get("reward_model") or {}).get("ground_truth")
        )
        filled["restored_from"] = placeholder["dataset"]
        restored.append(filled)
    return restored


def extract_opd_math(*, supplement_shards: int = 4) -> list[SourceRow]:
    """Math OPD prompts: all of ``Nemotron-RL-Math-v2``, plus a supplement.

    ``RL-Math-v2`` cannot fill a 6,000-row quota on its own, and the reason is a
    property of the source rather than of our filters. Its 7,732 rows contain only
    **6,457 distinct questions** -- the DAPO-derived rows repeat, in groups of up to
    ten (measured: 907 pairs, 54 triples, 46 quadruples, and a 10-way group), each
    copy carrying a different ``uuid`` but identical question text and answer. After
    reconstruction and format filtering, 3,451 unique prompts survive.

    Deduplicating is not optional: a repeated prompt is trained on repeatedly, so
    keeping the duplicates would weight those problems 2-10x and put the same
    question in both the SFT and OPD splits through different uuids -- exactly the
    id-based blind spot that let V1's overlap check pass.

    So the shortfall is covered from ``Nemotron-SFT-Math-v4``, which is the same
    domain, is already a pool source, and is large enough (7M rows) that the SFT and
    OPD draws cannot collide. The pool builder enforces disjointness on normalized
    text afterwards, so a problem present in both sources can only land in one
    split. Order matters here: RL-Math rows come first so the seeded quota sample
    prefers the purpose-built RL pool and treats the supplement as fill.
    """
    rows = list(download_jsonl(OPD_MATH_REPO, "data/train.jsonl"))
    restored = restore_rl_math_placeholders(rows)
    extracted: list[SourceRow] = []
    for record in restored:
        row = _text_domain_row(
            domain="math",
            source=OPD_MATH_REPO,
            source_id=record.get("uuid"),
            problem=record.get("question"),
            strata={"restored_from": str(record.get("restored_from") or "native")},
        )
        if row is not None:
            extracted.append(row)
    extracted = _dedupe(extracted, key=normalized_text_key)

    if supplement_shards > 0:
        # Read the tail shards, while SFT math reads the head ones, so the two
        # draws start from different data even before disjointness filtering.
        supplement = extract_sft_math(
            shards=supplement_shards, shard_offset=12 - supplement_shards
        )
        for row in supplement:
            extracted.append(
                SourceRow(
                    domain="math",
                    source=row.source,
                    source_id=row.source_id,
                    prompt=row.prompt,
                    messages=row.messages,
                    strata={"restored_from": "supplement"},
                )
            )
        extracted = _dedupe(extracted, key=normalized_text_key)
    return extracted


def extract_opd_code(*, shards: int = 11) -> list[SourceRow]:
    """``Nemotron-RL-coding`` prompts, dug out of a nested params column.

    Unlike ``RL-Math-v2`` there is no ``question`` field: the prompt lives at
    ``responses_create_params.input[0].content`` and needs its own extraction
    path. ``verifier_metadata.unit_tests`` is not read -- OPD supervises against
    the teacher's top-k distribution with a zero reward, so no test execution is
    involved and the (large) test column stays on disk.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows: list[SourceRow] = []
    for index in range(shards):
        path = hf_hub_download(
            OPD_CODE_REPO,
            f"data/train-{index:05d}-of-00011.parquet",
            repo_type="dataset",
        )
        frame = pd.read_parquet(
            path, columns=["responses_create_params", "hash_id", "dataset", "source"]
        )
        for record in frame.to_dict("records"):
            params = record.get("responses_create_params")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    continue
            inputs = params.get("input") if isinstance(params, dict) else None
            if inputs is None or len(inputs) == 0:
                continue
            first = inputs[0]
            content = first.get("content") if hasattr(first, "get") else None
            if not content:
                continue
            row = _code_row(
                source=OPD_CODE_REPO,
                source_id=record.get("hash_id"),
                question=content,
                strata={"source_name": str(record.get("source") or "")},
            )
            if row is not None:
                rows.append(row)
    return _dedupe(rows, key=source_id_key)


def extract_opd_science() -> list[SourceRow]:
    """All 150,644 ``Nemotron-RL-Science-v1`` open questions (0.27 GB, fetched whole).

    Topic distribution is heavily skewed (Physics 121,216 / Chemistry 18,852 /
    Biology 10,576), so per-topic quotas are applied by the pool builder. Even the
    smallest topic clears a 2K quota, but nothing here should assume balance.
    """
    rows: list[SourceRow] = []
    for record in download_jsonl(OPD_SCIENCE_REPO, "so_openq.jsonl"):
        if str(record.get("question_type") or "") != "open":
            continue
        metadata = record.get("metadata") or {}
        topic = str(metadata.get("topic") or "")
        if topic not in SCIENCE_TOPICS:
            continue
        row = _text_domain_row(
            domain="science",
            source=OPD_SCIENCE_REPO,
            source_id=record.get("uuid"),
            problem=record.get("problem"),
            strata={"topic": topic},
        )
        if row is not None:
            rows.append(row)
    return _dedupe(rows, key=normalized_text_key)


def _constraint_count(constraint: Any) -> int:
    """Number of constraints on an ``IF_multi_constraints_upto5`` row.

    Tab-separated, verified against the source: 23,007 / 23,903 / 23,322 /
    18,038 / 7,103 rows at counts 1-5. Every band clears the 1,200 quota.
    """
    text = str(constraint or "")
    if not text.strip():
        return 0
    return len([piece for piece in text.split("\t") if piece.strip()])


def _constraint_families(ground_truth: Any) -> list[str]:
    """Verifier family ids from the ``ground_truth`` column.

    That column is a **Python repr string**, not JSON -- single-quoted keys and
    ``None`` rather than ``null`` -- so ``json.loads`` raises and
    ``ast.literal_eval`` is required. Failures degrade to "no families known"
    instead of dropping the row, since family data only diversifies sampling and
    is not needed for correctness.
    """
    try:
        parsed = ast.literal_eval(str(ground_truth))
    except (ValueError, SyntaxError):
        return []
    families: list[str] = []
    entries = parsed if isinstance(parsed, list) else [parsed]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        identifiers = entry.get("instruction_id") or []
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        families.extend(str(item) for item in identifiers)
    return families


def extract_opd_if() -> list[SourceRow]:
    """All 95,373 ``IF_multi_constraints_upto5`` prompts (0.07 GB, fetched whole)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        OPD_IF_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset"
    )
    frame = pd.read_parquet(path)
    rows: list[SourceRow] = []
    for record in frame.to_dict("records"):
        user = _first_user_content(record.get("messages"))
        if len(user.strip()) < _MIN_PROMPT_CHARS:
            continue
        count = _constraint_count(record.get("constraint"))
        if not 1 <= count <= 5:
            continue
        families = _constraint_families(record.get("ground_truth"))
        rows.append(
            SourceRow(
                domain="instruction_following",
                source=OPD_IF_REPO,
                source_id=str(record.get("key")),
                prompt=user,
                messages=[{"role": "user", "content": user}],
                strata={
                    "constraint_count": count,
                    # First family only: the stratifier needs one label per row,
                    # and sorting makes the choice order-independent.
                    "constraint_family": sorted(families)[0] if families else "",
                    "constraint_families": len(set(families)),
                },
            )
        )
    return _dedupe(rows, key=source_id_key)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

SFT_EXTRACTORS: dict[str, Callable[[], list[SourceRow]]] = {
    "math": extract_sft_math,
    "code": extract_sft_code,
    "science": extract_sft_science,
    "instruction_following": extract_sft_if,
}

OPD_EXTRACTORS: dict[str, Callable[[], list[SourceRow]]] = {
    "math": extract_opd_math,
    "code": extract_opd_code,
    "science": extract_opd_science,
    "instruction_following": extract_opd_if,
}

# Which stratum key each (split, domain) pool balances on when filling its quota.
# Absent from this map means "no stratification, take a seeded sample".
STRATUM_KEY: dict[tuple[str, str], str] = {
    ("sft", "code"): "difficulty",
    ("sft", "science"): "topic",
    ("opd", "science"): "topic",
    ("opd", "instruction_following"): "constraint_count",
}


def extractors_for(split: str) -> dict[str, Callable[[], list[SourceRow]]]:
    if split == "sft":
        return SFT_EXTRACTORS
    if split == "opd":
        return OPD_EXTRACTORS
    raise ValueError(f"Unknown split {split!r}; expected 'sft' or 'opd'")
