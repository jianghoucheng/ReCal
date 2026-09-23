"""Shared calibration helpers for arms that score teacher *response* tokens.

The Minitron baseline follows its standard recipe: one prefill per calibration
prompt, statistics over prompt positions only. For a reasoning model that measures the
wrong distribution -- the capability being preserved is long-form chain-of-thought, and
the activations of a 200-token question are not those of an 8,000-token derivation. Arms
that calibrate on trajectories instead need two things: a balanced sample of the dense
teacher's own responses, and a mask marking which positions are the response.

Both live here rather than in one arm's collector because more than one arm needs them
and importing across sibling collectors made deleting a retired arm break a live one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from recal.common import supports_enable_thinking


def sample_calibration_trajectories(
    frame: pd.DataFrame,
    *,
    per_domain: int,
    seed: int,
) -> pd.DataFrame:
    """Take ``per_domain`` teacher trajectories from each domain.

    Equal counts per domain, and every trajectory weighted equally regardless of
    length. Weighting by tokens instead would let math and code -- whose traces run
    several times longer -- dominate the statistics, which is the token-share skew
    the balanced-data design exists to avoid; reintroducing it in the calibration step
    would defeat that.
    """
    selected = []
    for domain in sorted(frame["domain"].astype(str).unique()):
        pool = frame[frame["domain"].astype(str) == domain]
        take = min(per_domain, len(pool))
        selected.append(pool.sample(n=take, random_state=seed))
    return pd.concat(selected, ignore_index=True)


def response_token_mask(
    tokenizer: Any,
    messages: list[dict[str, str]],
    response: str,
    *,
    thinking_enabled: bool,
    max_length: int,
) -> tuple[list[int], list[bool]]:
    """Tokenize prompt+response and mark which positions are the response.

    The prompt is rendered with ``add_generation_prompt=True`` -- the exact string
    the model is conditioned on at generation time -- and the response tokenized
    separately and appended. The boundary is therefore a token count rather than a
    substring search over the rendered text, which would be ambiguous whenever the
    response happens to repeat template markup.

    Truncation keeps the head. A tail-truncated trajectory still consists of
    genuine response positions; dropping the sample entirely would bias the
    calibration set toward short responses, and long reasoning is the behavior
    being calibrated for.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **({"enable_thinking": thinking_enabled} if supports_enable_thinking(tokenizer) else {}),
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(str(response), add_special_tokens=False).input_ids
    input_ids = (prompt_ids + response_ids)[:max_length]
    mask = ([False] * len(prompt_ids) + [True] * len(response_ids))[: len(input_ids)]
    return input_ids, mask
