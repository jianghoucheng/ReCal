"""Thinking-mode resolution and data/mode consistency checks.

Qwen3 base models ship a chat template with a ``<think>`` reasoning block that
``enable_thinking`` switches on, while the Instruct-2507 variants have no such
block at all. The cascade therefore has to run in one of two mutually exclusive
paradigms, and *every* stage must agree on which one is active: the offline
teacher generates with it, SFT computes loss over it, OPD rolls out under it, and
the benchmark scores it. A mismatch is silent and expensive -- a student trained
on ``<think>`` traces but evaluated with thinking disabled looks broken for
reasons that have nothing to do with pruning.

``model.thinking_enabled`` is the single authoritative switch. A stage may
override it explicitly, but the override is reported rather than inferred, so
resolved configurations record exactly which paradigm each stage ran under.

The teacher parquet also carries a per-row ``enable_thinking`` column, which is
what verl's ``MultiTurnSFTDataset`` reads to pick template kwargs. That column
and the config must not disagree, and the generated responses themselves must
actually match: ``assert_data_matches_mode`` checks all three.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


STAGES = ("teacher", "sft", "opd", "benchmark")

# The config section that owns each stage's optional override.
_STAGE_SECTION = {
    "teacher": "offline_teacher",
    "sft": "sft",
    "opd": "opd",
    "benchmark": "boundary_benchmark",
}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Math/AIME/GPQA are scored by extracting the last \boxed{...}; LiveCodeBench by
# the last fenced block. These are the response-side counterparts of the prompt
# suffixes the benchmark appends, so they measure whether training data speaks
# the format the grader reads.
BOXED_PATTERN = re.compile(r"\\boxed\{")
FENCED_PATTERN = re.compile(r"```(?:python)?", re.IGNORECASE)

# Whether a *prompt* asks for code in a delimited block. Matched against the
# instruction LiveCodeBench's official template actually uses, not the phrase
# "fenced code block" -- that literal string appears nowhere in the template, so
# grepping for it reported 0% on prompts that were 100% correct and raised a
# format-drift warning about the one thing that was not drifting. The warning is
# supposed to catch a real train/eval disagreement; a false positive here trains the
# reader to ignore it, which is worse than not having it.
CODE_FORMAT_INSTRUCTION_PATTERN = re.compile(
    r"Enclose your code within delimiters"
    r"|use the provided format with backticks"
    r"|fenced code block",
    re.IGNORECASE,
)


def thinking_enabled(config: dict[str, Any], stage: str) -> bool:
    """Resolve whether ``stage`` runs with Qwen3 thinking enabled.

    ``model.thinking_enabled`` is the default for every stage. A stage section
    may set its own ``thinking_enabled`` to deviate deliberately.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    model_default = bool(config.get("model", {}).get("thinking_enabled", True))
    section = config.get(_STAGE_SECTION[stage], {}) or {}
    if "thinking_enabled" in section:
        return bool(section["thinking_enabled"])
    return model_default


def thinking_report(config: dict[str, Any]) -> dict[str, Any]:
    """Per-stage resolved thinking flags, plus which stages deviate."""
    model_default = bool(config.get("model", {}).get("thinking_enabled", True))
    resolved = {stage: thinking_enabled(config, stage) for stage in STAGES}
    return {
        "model_default": model_default,
        "resolved": resolved,
        "overridden_stages": sorted(
            stage
            for stage in STAGES
            if "thinking_enabled" in (config.get(_STAGE_SECTION[stage], {}) or {})
        ),
        "consistent": len(set(resolved.values())) == 1,
    }


def as_messages(value: Any) -> list[dict[str, str]]:
    """Normalize a parquet ``messages`` cell into plain dicts.

    Arrow round-trips the column as an ``ndarray`` of dict-likes, which
    ``apply_chat_template`` will not accept.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [dict(item) for item in value]


def _assistant_text(messages: Any) -> str:
    items = as_messages(messages)
    return "".join(
        str(item.get("content", "")) for item in items if item.get("role") == "assistant"
    )


def _user_text(messages: Any) -> str:
    items = as_messages(messages)
    return " ".join(
        str(item.get("content", "")) for item in items if item.get("role") == "user"
    )


def teacher_format_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether teacher responses speak the benchmark's answer format.

    Reported per domain because each domain is graded differently: math and
    science by ``\\boxed{}``, code by a fenced block, and the instruction domains
    by their own constraint checkers with no format suffix at all.
    """
    assistant = frame["messages"].map(_assistant_text)
    user = frame["messages"].map(_user_text)
    domains = frame["domain"].astype(str)

    def fraction(mask: "pd.Series[bool]") -> float:
        return round(float(mask.mean()), 4) if len(mask) else 0.0

    per_domain: dict[str, Any] = {}
    for domain in sorted(domains.unique()):
        selector = domains == domain
        responses = assistant[selector]
        prompts = user[selector]
        per_domain[domain] = {
            "rows": int(selector.sum()),
            "response_think_closed": fraction(responses.str.contains(THINK_CLOSE, regex=False)),
            "response_boxed": fraction(responses.str.contains(BOXED_PATTERN)),
            "response_fenced": fraction(responses.str.contains(FENCED_PATTERN)),
            "prompt_boxed_instruction": fraction(prompts.str.contains(r"\boxed{}", regex=False)),
            "prompt_fenced_instruction": fraction(prompts.str.contains(CODE_FORMAT_INSTRUCTION_PATTERN)),
        }
    statistics: dict[str, Any] = {"rows": int(len(frame)), "per_domain": per_domain}
    if "teacher_response_tokens" in frame.columns:
        tokens = frame["teacher_response_tokens"].astype(int)
        statistics["response_tokens"] = {
            "mean": round(float(tokens.mean()), 1),
            "median": int(tokens.median()),
            "max": int(tokens.max()),
        }
    return statistics


def assert_data_matches_mode(
    frame: pd.DataFrame,
    *,
    expect_thinking: bool,
    source: str,
    min_agreement: float = 0.99,
) -> dict[str, Any]:
    """Fail loudly when teacher data does not match the configured paradigm.

    Checks the ``enable_thinking`` column verl reads per row, and the responses
    themselves. ``min_agreement`` tolerates a small tail rather than demanding
    perfection, because a handful of malformed generations should not block a
    run the way a wholesale paradigm mismatch must.
    """
    if frame.empty:
        raise RuntimeError(f"{source} is empty; nothing to train on")

    problems: list[str] = []
    column_values: list[bool] | None = None
    if "enable_thinking" in frame.columns:
        column_values = sorted(set(bool(value) for value in frame["enable_thinking"]))
        if column_values != [expect_thinking]:
            problems.append(
                f"enable_thinking column holds {column_values} but the config resolves "
                f"thinking_enabled={expect_thinking}; verl reads this column per row, "
                "so the rendered template would contradict the configuration"
            )

    assistant = frame["messages"].map(_assistant_text)
    opened = assistant.str.contains(THINK_OPEN, regex=False)
    closed = assistant.str.contains(THINK_CLOSE, regex=False)
    paired_fraction = float((opened & closed).mean())
    any_think_fraction = float((opened | closed).mean())

    if expect_thinking:
        if paired_fraction < min_agreement:
            problems.append(
                f"only {paired_fraction:.1%} of responses carry a paired "
                f"{THINK_OPEN}...{THINK_CLOSE} block, below the required {min_agreement:.1%}; "
                "thinking-mode SFT would train on truncated or absent reasoning"
            )
    else:
        if any_think_fraction > (1.0 - min_agreement):
            problems.append(
                f"{any_think_fraction:.1%} of responses contain thinking markers even though "
                "thinking is disabled; the student would learn to emit tags its template "
                "never opens"
            )

    if problems:
        raise RuntimeError(
            f"Teacher data in {source} does not match thinking_enabled={expect_thinking}:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    return {
        "source": source,
        "expect_thinking": expect_thinking,
        "enable_thinking_column": column_values,
        "response_think_paired_fraction": round(paired_fraction, 4),
        "response_any_think_marker_fraction": round(any_think_fraction, 4),
        "format": teacher_format_statistics(frame),
    }
