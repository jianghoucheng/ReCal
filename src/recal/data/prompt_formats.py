"""One authoritative prompt/answer contract per benchmark, shared by SFT, OPD, and eval.

V1's central data bug was a format disagreement nobody could see: training prompts
asked for ``Answer: X``, so the teacher closed 99.2% of math responses that way and
almost never emitted ``\\boxed{}`` -- while the grader extracts only ``\\boxed{}``.
Training taught a convention the scorer cannot read, and the symptom was
inexplicably low scores rather than an error. Prompt text is therefore not a
detail to be set per call site; it is an interface, and this module is its
definition.

Every constant here is traced to a reference implementation rather than invented,
so that "our training prompt" and "the number the benchmark reports" cannot drift:

  math      ``{problem} Please reason step by step, and put your final answer
            within \\boxed{{}}.``
            Verbatim ``PROMPT_TEMPLATE`` from the OPD paper's own evaluator
            reference. Its OpenThoughts3 OPD training split uses this exact
            suffix on 100% of rows, and Nemotron-SFT-Math-v4 independently
            ships the same sentence on all 23,605 of its prompts. Scored by the
            last ``\\boxed{}``.

  science   Same as math. Both the reference's GPQA split and our GPQA-Diamond
            source embed lettered options in the question text and score a
            boxed letter, so science needs no separate convention -- but see
            ``SCIENCE_ANSWER_IS_FREE_TEXT`` below, which is why science training
            targets cannot be verified by exact match.

  code      LiveCodeBench's official two-branch template: call-based problems
            must be shown their starter code, stdin problems must be told to
            read stdin. Ported from the LiveCodeBench reference evaluator.
            Scored by executing the last fenced block; for call-based problems
            the grader dispatches on ``metadata.func_name`` and imports that
            exact method, so a prompt that hides the starter code makes 6.6% of
            LCB-v5 unscoreable no matter how correct the solution is.

  IF        No format suffix. Scored by executing constraint verifiers against
            the response, so any added instruction would itself be a constraint
            violation.

The suffix is applied by *replacement*, never by appending: several sources ship
their own format instruction already (Nemotron-RL-Math-v2 carries "Make sure your
answer is inside \\boxed{}", and Nemotron-RL-Science-v1 uses at least five
different answer-format templates across its 150,644 rows -- boxed 35.8%,
"Answer: X" 14.7%, "**X**" 9.2%, square brackets 6.7%, other 33.6%). Stacking a
second, conflicting instruction on top of those is how a model learns to emit two
answers in two formats.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# Canonical suffixes
# --------------------------------------------------------------------------- #

# Verbatim from the OPD reference evaluator's PROMPT_TEMPLATE.
# Leading space, not newline: that is how the reference joins it.
MATH_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
SCIENCE_SUFFIX = MATH_SUFFIX

# From lcb_runner.prompts.code_generation.PromptConstants, quoted exactly so a
# LiveCodeBench upgrade that changes the wording shows up as a test failure
# rather than as a silently different task.
CODE_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
CODE_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)
CODE_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

# Instruction-following is scored by constraint verifiers; an added format
# instruction would be an extra unrequested constraint.
IF_SUFFIX = ""

SUFFIX_BY_DOMAIN = {
    "math": MATH_SUFFIX,
    "science": SCIENCE_SUFFIX,
    "instruction_following": IF_SUFFIX,
    # Code is structural, not a suffix -- use build_code_prompt.
    "code": None,
}

# Nemotron-RL-Science-v1 expected_answer values are LLM-judge-verified free text
# (median 198 characters, only 6.2% under 40 characters). Exact-match answer
# checking is therefore invalid on that source: a correct teacher response would
# be rejected for wording. Teacher-solvability filtering for science uses a
# recall-style check, not equality. See data/teacher_filters.py.
SCIENCE_ANSWER_IS_FREE_TEXT = True


# --------------------------------------------------------------------------- #
# Stripping pre-existing format instructions
# --------------------------------------------------------------------------- #
#
# An enumerated pattern list cannot do this job. Nemotron-RL-Math-v2's 3,748
# usable rows carry 1,703 distinct leading lines, all paraphrases of the same
# request: "Solve the following math problem. Make sure to put the answer (and
# only answer) inside \boxed{}." / "Determine the answer to the following math
# problem. Use \boxed{} for your final answer." / "Think through the following
# math problem. Write only the answer in \boxed{}." A regex per phrasing would
# miss the tail and give the model two competing instructions.
#
# The structure is regular even though the wording is not. Measured over that
# source: the instruction sits at the very start or the very end of the prompt,
# never inside the problem body. But it is not always its own *line* -- most
# often it is the final sentence of the last line, sharing that line with real
# problem text ("... determine the values of (x, y). Present your answer inside
# \boxed{}."). So the unit of removal is a sentence at the prompt boundary, not a
# line: we peel sentences from the head and tail while they look like format
# instructions, and stop at the first one that does not.
#
# Identification is by shape rather than phrasing: a format marker (\boxed,
# "Answer:", "**X**", square-bracket wording) plus directive language, short
# enough to be an instruction, and free of the LaTeX or filled-in \boxed{value}
# that marks it as mathematics instead. Deliberately conservative in one
# direction -- a sentence carrying problem content is kept even if it mentions
# \boxed, because losing part of a question is far worse than leaving a redundant
# instruction. Idempotence and no-body-loss are pinned by tests.

# Markers that indicate a sentence is talking about answer *format*.
_FORMAT_MARKER_RE = re.compile(
    r"\\boxed|"
    r"\banswer\s*:|"
    r"\*\*x\*\*|"
    r"square brackets?|"
    r"fenced code block|"
    r"```python|"
    r"\(answer:",
    flags=re.IGNORECASE,
)

# Directive language: an instruction tells the model what to do with its answer.
_DIRECTIVE_RE = re.compile(
    r"\b(?:put|place|enclose|wrap|present|report|express|write|give|provide|"
    r"return|use|contain|include|conclude|ensure|make sure|should be|must be|"
    r"final answer|your answer|your response|the answer|solve|answer the|"
    r"reason step by step|think step-by-step|step by step)\b",
    flags=re.IGNORECASE,
)

# A boundary sentence is only removable if it is short enough to be an
# instruction rather than a problem. The longest genuine instruction across all
# sources is LiveCodeBench's stdin paragraph at ~250 characters; 400 leaves
# margin without reaching into real question text.
_MAX_INSTRUCTION_CHARS = 400

# Split into sentences while keeping the delimiter, so rejoining is lossless.
# Splits after .!? or a closing LaTeX display block, when followed by whitespace.
# The \] case matters: many math prompts end with a display equation, so without
# it the appended suffix lands inside the equation's "sentence" and a second
# alignment pass would peel the equation away with it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?=\s)|(?<=\\\])(?=\s)")

# Non-format boilerplate that trails the real instruction in
# Nemotron-RL-Science-v1 ("... inside \boxed{}.\n\nProvide a precise and correct
# solution using your knowledge."). It carries no format marker, so it is not an
# instruction by the test above, yet it sits between the instruction and the end
# of the prompt and blocks boundary peeling from reaching it. These are pure
# filler: removing them loses no task content.
_TRAILING_FILLER_RE = re.compile(
    r"^(?:"
    r"can you help (?:me )?answer this question\??|"
    r"(?:please )?(?:provide|deliver|give|offer)\s+(?:a|an|your)?\s*"
    r"(?:precise|accurate|correct|clear|complete|detailed|comprehensive|thorough)"
    r"[^.]*\.?|"
    r"apply your (?:comprehensive )?knowledge[^.]*\.?|"
    r"use your (?:expertise|knowledge)[^.]*\.?|"
    r"(?:please )?(?:answer|solve|address)\s+(?:the|this)\s+(?:question|problem)"
    r"[^.]*\.?|"
    r"what is the correct answer to (?:this|the) question\??|"
    r"think (?:carefully|step by step)[^.]*\.?"
    r")\s*$",
    flags=re.IGNORECASE,
)

# RL-Science-v1 sometimes orders a prompt as
#   <format instruction>\n\n<question>\n\n<filler>
# so the instruction is neither the first nor the last fragment once the filler
# is peeled -- it is second. Peeling cannot reach it without risking the question
# itself. Handle exactly this shape: a format-instruction *line* anywhere in the
# prompt is removable, since a line that is wholly an instruction carries no task
# content by definition. Restricted to whole lines (never sentences) so a
# question sharing a line with an instruction is never touched mid-body.
def _drop_interior_instruction_lines(lines: list[str]) -> list[str]:
    if len(lines) <= 1:
        return lines
    kept = [
        line
        for line in lines
        if line.strip() and not _is_format_instruction(line)
    ]
    # Never return nothing: if every line looked like an instruction the prompt
    # has no body and validation downstream should reject it intact.
    return kept if kept else lines


def _is_removable_boundary(fragment: str) -> bool:
    return _is_format_instruction(fragment) or bool(
        _TRAILING_FILLER_RE.match(fragment.strip())
    )


def _is_format_instruction(fragment: str) -> bool:
    """Whether one boundary sentence is purely an answer-format instruction."""
    stripped = fragment.strip()
    if not stripped or len(stripped) > _MAX_INSTRUCTION_CHARS:
        return False
    if not _FORMAT_MARKER_RE.search(stripped):
        return False
    if not _DIRECTIVE_RE.search(stripped):
        return False
    # A filled-in box is an answer inside a problem statement, not a request for
    # one: "\boxed{42}" is content, "\boxed{}" is an instruction.
    if re.search(r"\\boxed\{[^}]+\}", stripped):
        return False
    # LaTeX-heavy fragments are mathematics, not prose. Checked after the marker
    # test so "Express the answer using \boxed{}." still qualifies, and with the
    # marker itself masked out so "\boxed" does not count as its own evidence.
    without_marker = _FORMAT_MARKER_RE.sub(" ", stripped)
    if without_marker.count("$") >= 2 or re.search(
        r"\\(?:frac|tfrac|dfrac|int|sum|sqrt|begin|large|operatorname|mathbb|"
        r"mathrm|left|right|partial|sigma|lim|log|bar)",
        without_marker,
    ):
        return False
    return True


def _peel(fragments: list[str]) -> list[str]:
    """Drop leading and trailing format-instruction fragments.

    Never empties the list: a prompt that is *only* an instruction has no problem
    body to keep, so it is left for the caller's validation to reject rather than
    silently reduced to nothing.
    """
    start, end = 0, len(fragments)
    while start < end - 1 and (
        not fragments[start].strip() or _is_removable_boundary(fragments[start])
    ):
        start += 1
    while end - 1 > start and (
        not fragments[end - 1].strip() or _is_removable_boundary(fragments[end - 1])
    ):
        end -= 1
    return fragments[start:end]


def strip_format_instructions(text: Any) -> str:
    """Remove answer-format instructions from a prompt's boundaries.

    Applied before the canonical suffix so instructions are replaced rather than
    stacked. Two conflicting instructions are worse than either alone: the model
    learns to emit both forms and the grader reads whichever comes last.

    Peels at line granularity first (standalone instruction lines, the common
    case in the science and code sources), then at sentence granularity within
    the surviving boundary lines (the common case in the math sources, where the
    instruction shares its line with the question). Repeated to a fixed point,
    because the sources nest the two shapes: RL-Science-v1 places the format
    instruction on its own line and *then* a content-free pleasantry after it
    ("... inside \\boxed{}.\\n\\nProvide a precise and correct solution using your
    knowledge."), so one pass removes the filler and the next reaches the
    instruction it was hiding.
    """
    body = str(text).replace("\r\n", "\n")
    # A fixed point is reached in a handful of iterations; the bound only stops a
    # pathological input from spinning.
    for _ in range(8):
        lines = _drop_interior_instruction_lines(_peel(body.split("\n")))
        if lines:
            # Re-split only the boundary lines by sentence; interior lines are
            # problem body by construction and must not be touched.
            lines[0] = "".join(_peel(_SENTENCE_SPLIT_RE.split(lines[0]))).strip()
            lines[-1] = "".join(_peel(_SENTENCE_SPLIT_RE.split(lines[-1]))).strip()
            lines = _peel(lines)
        candidate = "\n".join(lines)
        candidate = re.sub(r"[ \t]+", " ", candidate)
        candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()
        if candidate == body.strip():
            break
        body = candidate
    return body.strip()


def align_prompt(problem: Any, domain: str) -> str:
    """Canonical training prompt for ``domain``: strip, then apply the suffix.

    Idempotent by construction: an already-aligned prompt has its canonical
    suffix removed before stripping and re-appended after, so the stripper never
    sees it. Doing this by re-stripping instead was actively unsafe -- the suffix
    is itself a format-instruction sentence, so peeling it off the last line took
    the neighbouring question sentence with it and silently truncated the
    problem. Idempotence matters because alignment can legitimately run more than
    once: on a source that already carries the canonical suffix, and again if a
    pool is rebuilt from aligned data.
    """
    if domain not in SUFFIX_BY_DOMAIN:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {sorted(SUFFIX_BY_DOMAIN)}")
    if domain == "code":
        raise ValueError("Code prompts are structural; use build_code_prompt()")
    suffix = SUFFIX_BY_DOMAIN[domain]
    text = str(problem)
    if suffix:
        # Remove any number of already-applied canonical suffixes from the tail.
        stripped_tail = text.rstrip()
        while stripped_tail.endswith(suffix.strip()):
            stripped_tail = stripped_tail[: -len(suffix.strip())].rstrip()
        text = stripped_tail
    body = strip_format_instructions(text)
    if not suffix:
        return body
    return body.rstrip() + suffix


def build_code_prompt(question: Any, starter_code: Any = None) -> str:
    """LiveCodeBench's official user message, both branches.

    Mirrors ``get_generic_question_template_answer``. The branch matters for
    scoring, not just style: LiveCodeBench's grader reads
    ``metadata.func_name`` and, when it is present, imports that exact method
    from the submitted code (``grade_call_based``). 38.6% of the LCB-v5 window is
    call-based, so a prompt that omits the starter code asks for a stdin program
    and is graded as a missing method -- unscoreable regardless of correctness.
    Measured on V1's dense baseline: Qwen3-8B guessed the right signature in
    82.8% of call-based problems and lost the remaining 6.6% of the whole
    benchmark outright.
    """
    body = strip_format_instructions(strip_code_harness(question))
    prompt = f"### Question:\n{body}\n\n"
    starter = str(starter_code).strip() if starter_code else ""
    if starter:
        prompt += f"### Format: {CODE_FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter}\n```\n\n"
    else:
        prompt += f"### Format: {CODE_FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def code_messages(question: Any, starter_code: Any = None) -> list[dict[str, str]]:
    """Full chat messages for a code prompt, system message included."""
    return [
        {"role": "system", "content": CODE_SYSTEM_MESSAGE},
        {"role": "user", "content": build_code_prompt(question, starter_code)},
    ]


# Nemotron's competitive-programming sources ship their own harness wrapper
# around every question: a role preamble, a language restriction, and a fenced
# format block. Verified fixed rather than paraphrased -- the block below appears
# byte-identical on 16,083/16,083 rows of Nemotron-RL-coding-competitive_coding
# (100%), unlike the math sources' 1,703 paraphrases. Because it is exactly
# constant, an exact-prefix removal is both safe and complete here, where a
# structural rule is not: the fenced ```python block is legitimate content
# elsewhere, so the generic stripper must not learn to delete fenced blocks.
#
# It has to go: we re-wrap every code question in LiveCodeBench's official
# template, and leaving Nemotron's wrapper in place would stack two different
# harnesses with two different output contracts.
_CODE_HARNESS_PREFIXES = [
    "You are a helpful and harmless assistant. You should think step-by-step "
    "before responding to the instruction below.",
    "Please use python programming language only.",
    "Please use c++ programming language only.",
    "You must use ```python for just the final solution code block with the "
    "following format:\n```python\n# Your code here\n```",
    "You must use ```cpp for just the final solution code block with the "
    "following format:\n```cpp\n// Your code here\n```",
]


def strip_code_harness(text: Any) -> str:
    """Remove Nemotron's fixed code-prompt wrapper, leaving the bare question."""
    body = str(text).replace("\r\n", "\n").strip()
    changed = True
    while changed:
        changed = False
        for prefix in _CODE_HARNESS_PREFIXES:
            # Compare on collapsed whitespace so a differing blank-line count
            # between shards does not defeat the match.
            candidate = body.lstrip()
            if candidate.startswith(prefix):
                body = candidate[len(prefix) :].lstrip()
                changed = True
    return body.strip()


def messages_for(problem: Any, domain: str, *, starter_code: Any = None) -> list[dict[str, str]]:
    """Canonical chat messages for one training row."""
    if domain == "code":
        return code_messages(problem, starter_code)
    return [{"role": "user", "content": align_prompt(problem, domain)}]


# --------------------------------------------------------------------------- #
# Response-side expectations, i.e. what the grader will look for
# --------------------------------------------------------------------------- #

BOXED_RE = re.compile(r"\\boxed\{")
FENCED_RE = re.compile(r"```(?:python)?", re.IGNORECASE)


def response_has_expected_format(response: Any, domain: str) -> bool:
    """Whether a teacher response speaks the form its grader extracts.

    A teacher response failing this is unusable as an SFT target: it would teach
    the student a convention the scorer cannot read. Instruction-following has no
    format requirement, so it always passes here and is filtered by its own
    constraint verifiers instead.
    """
    text = str(response)
    if domain in {"math", "science"}:
        return bool(BOXED_RE.search(text))
    if domain == "code":
        return bool(FENCED_RE.search(text))
    if domain == "instruction_following":
        return True
    raise ValueError(f"Unknown domain {domain!r}")


# A *competing* answer-format instruction, i.e. one that survived stripping and
# now contradicts the canonical suffix. Deliberately narrower than the stripper's
# marker set: this asks "would the model be told two different things?", so
# incidental mentions of the word "answer" do not qualify.
_COMPETING_INSTRUCTION_RE = re.compile(
    r"answer is \[X\]|"
    r"\banswer:\s*X\b|"
    r"\(Answer:\s*X\)|"
    r"\*\*X\*\*|"
    r"(?:within|inside|in) square brackets|"
    r"(?:put|place|enclose|wrap|present|express|write|report|give)\b[^.]{0,60}"
    r"\\boxed\{\}|"
    r"answer on its own line",
    flags=re.IGNORECASE,
)


def competing_instruction_count(prompt: Any, domain: str) -> int:
    """How many non-canonical format instructions remain in an aligned prompt.

    Stripping is intentionally conservative around LaTeX -- a fragment carrying
    heavy math markup is treated as a problem statement, not prose, because
    truncating a question is far worse than leaving a redundant sentence. The
    residue is small but real: measured on the aligned pool sources, 0.11% of
    Nemotron-RL-Math-v2 and 2.50% of Nemotron-RL-Science-v1 keep a second
    instruction, almost all inside ``$$...$$`` blocks.

    Rather than escalate the regex until it starts eating mathematics, the pool
    builder calls this and *drops* the offending rows, refilling the quota from
    the same pool. Losing 2.5% of a 150,644-row source costs nothing; a prompt
    that asks for two answer formats is exactly the V1 failure.
    """
    aligned = str(prompt)
    suffix = SUFFIX_BY_DOMAIN.get(domain)
    if suffix:
        # The canonical suffix is not competition with itself.
        aligned = aligned.replace(suffix.strip(), " ")
    return len(_COMPETING_INSTRUCTION_RE.findall(aligned))


def prompt_is_clean(prompt: Any, domain: str) -> bool:
    """Whether an aligned prompt carries exactly one answer-format instruction."""
    return competing_instruction_count(prompt, domain) == 0
