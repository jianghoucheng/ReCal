from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from recal.common import (
    directory_fingerprint,
    dump_json,
    dump_jsonl,
    render_chat,
    to_jsonable,
    utc_now,
)
from recal.data.prompt_formats import code_messages, messages_for


SCHEMA_VERSION = 3
TASKS = [
    "aime",
    "gpqa_diamond",
    "ifbench",
    "livecodebench",
]


def boxed(text: str) -> str | None:
    """Contents of the last ``\\boxed{...}``, with brace matching.

    Uses brace matching rather than a regular expression, because a regex
    one that used to be here -- ``\\\\boxed\\{([^{}]+)\\}`` -- silently scored every
    nested-brace answer as wrong:

        \\boxed{277}          -> "277"          both agree
        \\boxed{\\frac{3}{4}}  -> None           regex fails, reference returns \\frac{3}{4}
        \\boxed{2\\sqrt{3}}    -> None           regex fails
        \\boxed{\\text{42}}    -> None           regex fails

    AIME answers are bare integers, but robust brace matching also supports
    fractions, radicals, and ``\\text{}`` wrappers. ``\\fbox`` is accepted too,
    as in the reference implementation.

    Scanning from the *last* marker matters for reasoning models: a thinking trace
    routinely boxes intermediate results, and the graded answer is the final one.
    """
    index = text.rfind("\\boxed")
    if index < 0:
        index = text.rfind("\\fbox")
        if index < 0:
            return None
    open_brace = text.find("{", index)
    if open_brace < 0:
        return None
    depth = 0
    for position in range(open_brace, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : position].strip()
    # An unclosed brace means the response was cut off mid-answer, which is a
    # truncation rather than a wrong answer; the diagnostics report it separately.
    return None


def normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"[\s,$]", "", str(value))
    if re.fullmatch(r"-?\d+", value):
        return str(int(value))
    return value.lower()


def final_letter(text: str, choices: str) -> str | None:
    value = boxed(text)
    if value and value.upper() in choices:
        return value.upper()
    found = re.findall(rf"\b([{choices}])\b", text.upper())
    return found[-1] if found else None


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    return blocks[-1].strip() if blocks else text.strip()


def normalize_integral_floats(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: normalize_integral_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_integral_floats(item) for item in value]
    return value


def _signature(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _task_signature(
    *,
    task: str,
    model_fingerprint: str,
    profile: str,
    prompts: list[str],
    labels: Any,
    sampling: dict[str, Any],
) -> str:
    prompt_digest = hashlib.sha256()
    for prompt in prompts:
        prompt_digest.update(prompt.encode("utf-8"))
        prompt_digest.update(b"\0")
    return _signature(
        SCHEMA_VERSION,
        task,
        model_fingerprint,
        profile,
        prompt_digest.hexdigest(),
        labels,
        sampling,
    )


def _task_result_path(output: Path, task: str) -> Path:
    return output / "task_results" / f"{task}.json"


def _load_task_result(output: Path, task: str, signature: str) -> dict[str, Any] | None:
    path = _task_result_path(output, task)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("signature") == signature
            and isinstance(payload.get("metrics"), dict)
        ):
            return payload["metrics"]
    except Exception:
        return None
    return None


def _save_task_result(output: Path, task: str, signature: str, metrics: dict[str, Any]) -> None:
    dump_json(
        {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "signature": signature,
            "metrics": metrics,
            "completed_at": utc_now(),
        },
        _task_result_path(output, task),
    )


def _load_generation_shard(
    data_path: Path,
    manifest_path: Path,
    *,
    signature: str,
    start: int,
    stop: int,
) -> list[dict[str, Any]] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "start": start,
            "stop": stop,
            "rows": stop - start,
        }:
            return None
        rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
        if len(rows) != stop - start:
            return None
        for expected_index, row in enumerate(rows, start):
            if row.get("index") != expected_index or not isinstance(row.get("outputs"), list):
                return None
        return rows
    except Exception:
        return None


def generate_cached(
    llm: Any,
    sampling_params_cls: Any,
    *,
    output: Path,
    task: str,
    prompts: list[str],
    sampling: dict[str, Any],
    signature: str,
    shard_size: int,
) -> list[dict[str, Any]]:
    task_dir = output / "generations" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), shard_size):
        stop = min(len(prompts), start + shard_size)
        data_path = task_dir / f"shard_{start:06d}_{stop:06d}.jsonl"
        manifest_path = task_dir / f"shard_{start:06d}_{stop:06d}.manifest.json"
        cached = _load_generation_shard(
            data_path,
            manifest_path,
            signature=signature,
            start=start,
            stop=stop,
        )
        if cached is not None:
            print(f"[resume] {task} prompts {start}:{stop}", flush=True)
            rows.extend(cached)
            continue
        print(f"[generate] {task} prompts {start}:{stop}", flush=True)
        generated = llm.generate(prompts[start:stop], sampling_params_cls(**sampling))
        shard_rows = []
        for index, result in enumerate(generated, start):
            shard_rows.append(
                {
                    "index": index,
                    "outputs": [
                        {
                            "text": candidate.text,
                            "token_count": len(candidate.token_ids),
                            "finish_reason": candidate.finish_reason,
                        }
                        for candidate in result.outputs
                    ],
                }
            )
        if len(shard_rows) != stop - start:
            raise RuntimeError(f"{task} returned {len(shard_rows)} rows for prompts {start}:{stop}")
        dump_jsonl(shard_rows, data_path)
        dump_json(
            {
                "schema_version": SCHEMA_VERSION,
                "signature": signature,
                "start": start,
                "stop": stop,
                "rows": stop - start,
            },
            manifest_path,
        )
        rows.extend(shard_rows)
    if len(rows) != len(prompts):
        raise RuntimeError(f"{task} cache has {len(rows)} rows, expected {len(prompts)}")
    dump_json(
        {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "rows": len(rows),
            "shard_size": shard_size,
            "completed_at": utc_now(),
        },
        task_dir / "_SUCCESS.json",
    )
    return rows


def generation_diagnostics(
    output: Path,
    tasks: list[str],
    *,
    thinking_enabled: bool = True,
) -> dict[str, Any]:
    """Per-task truncation and repetition statistics, read back from the shards.

    A pruned reasoning model can score near zero for two very different reasons:
    it answered wrongly, or it never finished reasoning and emitted no answer at
    all. Those demand opposite responses, and the accuracy number alone cannot
    tell them apart. ``finish_reason=length`` and a missing ``</think>`` separate
    them directly.

    Repetition is reported alongside them because it is the third distinct failure
    model whose looping and overthinking are *easier* for SFT+OPD to repair. A
    response can be untruncated, well-formed, and still be a loop, so truncation
    rate alone would score that as healthy. Measured with the same statistic the
    teacher-data filter uses, so training data and student output are judged by one
    definition.

    Derived from the shards rather than computed during generation so the numbers
    are available for boundaries that already finished.
    """
    from recal.data.teacher_targets import (
        DEFAULT_REPETITION_NGRAM,
        DEFAULT_REPETITION_THRESHOLD,
        repetition_rate,
    )

    diagnostics: dict[str, Any] = {}
    for task in tasks:
        total = 0
        truncated = 0
        unclosed_think = 0
        token_counts: list[int] = []
        repetition_rates: list[float] = []
        degenerate = 0
        for path in sorted((output / "generations" / task).glob("shard_*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                for candidate in json.loads(line)["outputs"]:
                    total += 1
                    token_counts.append(int(candidate.get("token_count", 0)))
                    if candidate.get("finish_reason") == "length":
                        truncated += 1
                    text = candidate.get("text", "")
                    if thinking_enabled and "</think>" not in text:
                        unclosed_think += 1
                    rate = repetition_rate(text, ngram=DEFAULT_REPETITION_NGRAM)
                    repetition_rates.append(rate)
                    if rate >= DEFAULT_REPETITION_THRESHOLD:
                        degenerate += 1
        if not total:
            continue
        diagnostics[f"{task}_truncated_fraction"] = round(truncated / total, 4)
        diagnostics[f"{task}_unclosed_think_fraction"] = (
            round(unclosed_think / total, 4) if thinking_enabled else None
        )
        diagnostics[f"{task}_mean_response_tokens"] = round(sum(token_counts) / total, 1)
        diagnostics[f"{task}_mean_repetition_rate"] = round(
            sum(repetition_rates) / total, 4
        )
        # The share of responses that are mostly loop. Mean repetition can stay low
        # while a minority of responses degenerate completely, and it is that
        # minority which drives the score.
        diagnostics[f"{task}_degenerate_repetition_fraction"] = round(degenerate / total, 4)
    return diagnostics


def run_aime(
    llm: Any,
    sampling_params_cls: Any,
    tokenizer: Any,
    output: Path,
    *,
    quick: bool,
    profile: str,
    model_fingerprint: str,
    aime_n: int,
    aime_temperature: float,
    aime_top_p: float,
    max_tokens: int,
    thinking_enabled: bool,
    shard_size: int | None = None,
) -> dict[str, Any]:
    specs = {
        "aime_2024": ("HuggingFaceH4/aime_2024", "problem", "answer"),
        "aime_2025": ("MathArena/aime_2025", "problem", "answer"),
        "aime_2026": ("MathArena/aime_2026", "problem", "answer"),
    }
    labels: list[tuple[str, Any]] = []
    prompts: list[str] = []
    for name, (repo, problem_key, answer_key) in specs.items():
        ds = load_dataset(repo, split="train", cache_dir=".cache/huggingface")
        if quick:
            ds = ds.select(range(min(10, len(ds))))
        for row in ds:
            labels.append((name, row[answer_key]))
            prompts.append(
                render_chat(
                    tokenizer,
                    # One authority for the prompt contract, shared with the
                    # training pools. This suffix is verbatim the reference
                    # evaluation template, and it is the same string the SFT and
                    # OPD prompts carry. The suffix used to be written out
                    # here by hand and differed from the training one
                    # ("\nPut the final answer in \boxed{}." vs " Please reason step
                    # by step, and put your final answer within \boxed{}."), so
                    # training and evaluation use one prompt convention.
                    messages_for(row[problem_key], "math"),
                    thinking_enabled=thinking_enabled,
                )
            )
    n = int(aime_n)
    sampling = {
        "temperature": float(aime_temperature),
        "top_p": float(aime_top_p),
        "max_tokens": int(max_tokens),
        "n": n,
    }
    signature = _task_signature(
        task="aime",
        model_fingerprint=model_fingerprint,
        profile=profile,
        prompts=prompts,
        labels=labels,
        sampling=sampling,
    )
    cached_result = _load_task_result(output, "aime", signature)
    if cached_result is not None:
        return cached_result
    generations = generate_cached(
        llm,
        sampling_params_cls,
        output=output,
        task="aime",
        prompts=prompts,
        sampling=sampling,
        signature=signature,
        shard_size=shard_size or len(prompts),
    )
    metrics: dict[str, Any] = {}
    for name in specs:
        question_results = []
        for (benchmark, gold), result in zip(labels, generations):
            if benchmark != name:
                continue
            question_results.append(
                [
                    normalize_number(boxed(candidate["text"])) == normalize_number(gold)
                    for candidate in result["outputs"]
                ]
            )
        metrics[f"{name}_avg{n}"] = sum(sum(values) for values in question_results) / (
            len(question_results) * n
        )
        metrics[f"{name}_pass1_first"] = sum(values[0] for values in question_results) / len(question_results)
        metrics[f"{name}_samples"] = len(question_results)
    _save_task_result(output, "aime", signature, metrics)
    return metrics


def run_gpqa(
    llm: Any,
    sampling_params_cls: Any,
    tokenizer: Any,
    output: Path,
    *,
    quick: bool,
    profile: str,
    model_fingerprint: str,
    max_tokens: int,
    thinking_enabled: bool,
    shard_size: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset("fingertap/GPQA-Diamond", split="test", cache_dir=".cache/huggingface")
    if quick:
        dataset = dataset.select(range(min(64, len(dataset))))
    records = [to_jsonable(row) for row in dataset]
    prompts = [
        render_chat(
            tokenizer,
            # Same canonical suffix as math, which is what the reference evaluator
            # does: it applies one PROMPT_TEMPLATE to every task, GPQA included.
            # The options are already embedded in the question text with lettered
            # labels (A-D) and the gold answer is a letter, so a boxed letter is
            # the natural answer form and no separate convention is needed.
            messages_for(row["question"], "science"),
            thinking_enabled=thinking_enabled,
        )
        for row in records
    ]
    sampling = {"temperature": 0.0, "max_tokens": int(max_tokens)}
    labels = [row["answer"] for row in records]
    signature = _task_signature(
        task="gpqa_diamond",
        model_fingerprint=model_fingerprint,
        profile=profile,
        prompts=prompts,
        labels=labels,
        sampling=sampling,
    )
    cached_result = _load_task_result(output, "gpqa_diamond", signature)
    if cached_result is not None:
        return cached_result
    generations = generate_cached(
        llm,
        sampling_params_cls,
        output=output,
        task="gpqa_diamond",
        prompts=prompts,
        sampling=sampling,
        signature=signature,
        shard_size=shard_size or len(prompts),
    )
    correct = [
        final_letter(result["outputs"][0]["text"], "ABCD") == row["answer"].upper()
        for row, result in zip(records, generations)
    ]
    metrics = {"gpqa_diamond_accuracy": sum(correct) / len(correct), "gpqa_diamond_samples": len(correct)}
    _save_task_result(output, "gpqa_diamond", signature, metrics)
    return metrics


def run_ifbench(
    llm: Any,
    sampling_params_cls: Any,
    tokenizer: Any,
    output: Path,
    *,
    quick: bool,
    profile: str,
    model_fingerprint: str,
    max_tokens: int,
    thinking_enabled: bool,
    shard_size: int | None = None,
) -> dict[str, Any]:
    frame = load_dataset("allenai/IFBench_test", split="train", cache_dir=".cache/huggingface").to_pandas()
    if quick:
        frame = frame.head(min(50, len(frame)))
    records = [to_jsonable(row) for row in frame.to_dict("records")]
    prompts = [
        render_chat(
            tokenizer,
            [{"role": "user", "content": row["prompt"]}],
            thinking_enabled=thinking_enabled,
        )
        for row in records
    ]
    sampling = {"temperature": 0.0, "max_tokens": int(max_tokens)}
    signature = _task_signature(
        task="ifbench",
        model_fingerprint=model_fingerprint,
        profile=profile,
        prompts=prompts,
        labels=records,
        sampling=sampling,
    )
    cached_result = _load_task_result(output, "ifbench", signature)
    if cached_result is not None:
        return cached_result
    generations = generate_cached(
        llm,
        sampling_params_cls,
        output=output,
        task="ifbench",
        prompts=prompts,
        sampling=sampling,
        signature=signature,
        shard_size=shard_size or len(prompts),
    )
    input_path = output / "ifbench_input.jsonl"
    response_path = output / "ifbench_responses.jsonl"
    dump_jsonl([normalize_integral_floats(row) for row in records], input_path)
    dump_jsonl(
        [
            {"prompt": row["prompt"], "response": result["outputs"][0]["text"]}
            for row, result in zip(records, generations)
        ],
        response_path,
    )
    eval_path = output / "ifbench_eval"
    temporary_eval = output / ".ifbench_eval.partial"
    shutil.rmtree(temporary_eval, ignore_errors=True)
    temporary_eval.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "recal.evaluation.ifbench.run_eval",
            f"--input_data={input_path.resolve()}",
            f"--input_response_data={response_path.resolve()}",
            f"--output_dir={temporary_eval.resolve()}",
        ],
        check=True,
    )
    shutil.rmtree(eval_path, ignore_errors=True)
    os.replace(temporary_eval, eval_path)
    metrics: dict[str, Any] = {"ifbench_samples": len(records)}
    for mode in ["strict", "loose"]:
        path = next(eval_path.glob(f"*eval_results_{mode}.jsonl"))
        values = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        metrics[f"ifbench_prompt_{mode}"] = sum(v["follow_all_instructions"] for v in values) / len(values)
        instructions = [item for value in values for item in value["follow_instruction_list"]]
        metrics[f"ifbench_instruction_{mode}"] = sum(instructions) / len(instructions)
    _save_task_result(output, "ifbench", signature, metrics)
    return metrics


def run_livecodebench(
    llm: Any,
    sampling_params_cls: Any,
    tokenizer: Any,
    output: Path,
    *,
    quick: bool,
    profile: str,
    model_fingerprint: str,
    max_tokens: int,
    thinking_enabled: bool,
    shard_size: int | None = None,
) -> dict[str, Any]:
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    rows = []
    file_names = ["test5.jsonl"] if quick else [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ]
    for file_name in file_names:
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            file_name,
            repo_type="dataset",
            cache_dir=".cache/huggingface",
        )
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows = sorted(rows, key=lambda row: row["contest_date"], reverse=True)
    rows = [
        row
        for row in rows
        if "2024-10-01" <= row["contest_date"][:10] <= "2025-02-28"
    ]
    if quick:
        rows = rows[:10]
    problems = [CodeGenerationProblem(**row) for row in rows]
    # LiveCodeBench's official template, via prompt_formats.code_messages, which is
    # also what the code training pool uses. The version here previously was a
    # paraphrase, and each divergence cost real score:
    #
    #   * the system message was reworded, so the model was primed differently than
    #     LCB's own harness primes it;
    #   * the stdin branch was missing entirely. LCB's grader dispatches on
    #     metadata.func_name: with starter code present it imports that exact method
    #     (grade_call_based), otherwise it feeds stdin and compares stdout
    #     (grade_stdio). A prompt that says neither leaves the model guessing which
    #     contract to satisfy, and a guess that is wrong is unscoreable no matter how
    #     correct the algorithm is. 38.6% of this window is call-based.
    #   * "Return the complete Python solution in a fenced code block" replaced the
    #     official "### Format:" block plus the "### Answer: (use the provided format
    #     with backticks)" trailer.
    prompts = [
        render_chat(
            tokenizer,
            code_messages(problem.question_content, problem.starter_code),
            thinking_enabled=thinking_enabled,
        )
        for problem in problems
    ]
    sampling = {"temperature": 0.0, "max_tokens": int(max_tokens)}
    evaluation_samples = [to_jsonable(problem.get_evaluation_sample()) for problem in problems]
    signature = _task_signature(
        task="livecodebench",
        model_fingerprint=model_fingerprint,
        profile=profile,
        prompts=prompts,
        labels=evaluation_samples,
        sampling=sampling,
    )
    cached_result = _load_task_result(output, "livecodebench", signature)
    if cached_result is not None:
        return cached_result
    generations = generate_cached(
        llm,
        sampling_params_cls,
        output=output,
        task="livecodebench",
        prompts=prompts,
        sampling=sampling,
        signature=signature,
        shard_size=shard_size or len(prompts),
    )
    codes = [[extract_code(result["outputs"][0]["text"])] for result in generations]
    lcb_metrics = codegen_metrics(
        [problem.get_evaluation_sample() for problem in problems],
        codes,
        k_list=[1],
        num_process_evaluate=8,
        timeout=10,
    )[0]
    metrics = {
        "livecodebench_v5_pass1": float(lcb_metrics.get("pass@1", 0.0)),
        "livecodebench_v5_samples": len(problems),
    }
    _save_task_result(output, "livecodebench", signature, metrics)
    return metrics


def _completed_suite(
    output: Path,
    *,
    model_fingerprint: str,
    profile: str,
    tasks: list[str],
) -> dict[str, Any] | None:
    try:
        completion = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if (
            completion.get("schema_version") == SCHEMA_VERSION
            and completion.get("model_fingerprint") == model_fingerprint
            and completion.get("profile") == profile
            and completion.get("completed_tasks") == tasks
            and isinstance(summary, dict)
        ):
            return summary
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--aime-n", type=int, default=16)
    parser.add_argument("--aime-temperature", type=float, default=0.6)
    parser.add_argument("--aime-top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=18432)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    # Generation is sharded only to checkpoint partial progress. Leave unset to
    # submit every prompt in one call so vLLM can batch continuously.
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--thinking-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    # Recompute truncation diagnostics from cached shards without loading a model.
    # Lets a boundary that finished before diagnostics existed be backfilled.
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.diagnostics_only:
        output = Path(args.output_dir)
        diagnostics = generation_diagnostics(
            output,
            list(args.tasks or TASKS),
            thinking_enabled=args.thinking_enabled,
        )
        dump_json(diagnostics, output / "generation_diagnostics.json")
        print(json.dumps(diagnostics, indent=2), flush=True)
        return
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_tasks = args.tasks or TASKS
    model_fingerprint = directory_fingerprint(args.model)
    if not args.force:
        completed = _completed_suite(
            output,
            model_fingerprint=model_fingerprint,
            profile=args.profile,
            tasks=selected_tasks,
        )
        if completed is not None:
            print(json.dumps(completed, indent=2), flush=True)
            return

    quick = args.profile == "quick"
    summary: dict[str, Any] = {}
    completed_tasks: list[str] = []
    dump_json(
        {
            "status": "running",
            "profile": args.profile,
            "model": args.model,
            "model_fingerprint": model_fingerprint,
            "completed_tasks": completed_tasks,
            "current_task": "model_load",
            "updated_at": utc_now(),
        },
        output / "benchmark_status.json",
    )
    llm = None
    try:
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            # Bound how many sequences are resident at once. Left unset, vLLM admits
            # the whole request list -- all 720 AIME sequences (90 problems x n=8) --
            # and every one of them then advances a single token per step. Nothing
            # reaches its stop condition until nearly all of them do, so the progress
            # progress bar may remain at 0/720 for a long time. A limit of 32 keeps
            # each step's batch small enough to avoid excessive KV-cache pressure.
            max_num_seqs=32,
            # Custom all-reduce stays off: multi-hour TP=8 generation hit "invalid
            # argument" aborts that killed individual workers. Plain NCCL is the
            # configuration that runs to completion. CUDA graphs are left enabled --
            # they were disabled together with this, but the capture failures were the
            # all-reduce path, and eager costs real throughput on decode-bound work.
            disable_custom_all_reduce=True,
            seed=20260728,
        )
        runners = [
            ("aime", run_aime),
            ("gpqa_diamond", run_gpqa),
            ("ifbench", run_ifbench),
            ("livecodebench", run_livecodebench),
        ]
        runners = [(task, runner) for task, runner in runners if task in selected_tasks]
        for task, runner in runners:
            dump_json(
                {
                    "status": "running",
                    "profile": args.profile,
                    "model": args.model,
                    "model_fingerprint": model_fingerprint,
                    "completed_tasks": completed_tasks,
                    "current_task": task,
                    "updated_at": utc_now(),
                },
                output / "benchmark_status.json",
            )
            metrics = runner(
                llm,
                SamplingParams,
                tokenizer,
                output,
                quick=quick,
                profile=args.profile,
                model_fingerprint=model_fingerprint,
                max_tokens=args.max_tokens,
                thinking_enabled=args.thinking_enabled,
                shard_size=args.shard_size,
                **(
                    {
                        "aime_n": args.aime_n,
                        "aime_temperature": args.aime_temperature,
                        "aime_top_p": args.aime_top_p,
                    }
                    if task == "aime"
                    else {}
                ),
            )
            summary.update(metrics)
            completed_tasks.append(task)
            dump_json(summary, output / "summary.partial.json")
        # Truncation diagnostics are recorded alongside the scores so a near-zero
        # result can be read as "did not finish reasoning" rather than "answered
        # wrongly". They are written to their own file to keep summary.json a
        # clean metric record.
        diagnostics = generation_diagnostics(
            output,
            completed_tasks,
            thinking_enabled=args.thinking_enabled,
        )
        dump_json(diagnostics, output / "generation_diagnostics.json")
        dump_json(summary, output / "summary.json")
        dump_json(
            {
                "schema_version": SCHEMA_VERSION,
                "model": args.model,
                "model_fingerprint": model_fingerprint,
                "profile": args.profile,
                "completed_tasks": completed_tasks,
                "completed_at": utc_now(),
            },
            output / "_SUCCESS.json",
        )
        dump_json(
            {
                "status": "completed",
                "profile": args.profile,
                "model": args.model,
                "model_fingerprint": model_fingerprint,
                "completed_tasks": completed_tasks,
                "current_task": None,
                "updated_at": utc_now(),
            },
            output / "benchmark_status.json",
        )
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        dump_json(
            {
                "status": "failed",
                "profile": args.profile,
                "model": args.model,
                "model_fingerprint": model_fingerprint,
                "completed_tasks": completed_tasks,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            },
            output / "benchmark_status.json",
        )
        raise
    finally:
        if llm is not None:
            del llm
        gc.collect()
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
