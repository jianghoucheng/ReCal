from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from recal.common import (
    directory_fingerprint,
    dump_json,
    load_yaml,
    render_chat,
    set_seed,
    sha256_file,
    sha256_text,
    utc_now,
)
from recal.modes import as_messages, thinking_enabled


def _messages(value: Any) -> list[dict[str, str]]:
    return as_messages(value)


def filter_complete_teacher_sequences(
    frame: pd.DataFrame,
    *,
    max_response_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove responses that reached the generation limit.

    A response ending with ``finish_reason=length`` is incomplete even though
    its stored token count equals, rather than exceeds, the configured limit.
    Treat all three signals as truncated so no partial teacher sequence reaches
    sequence-level SFT.
    """
    truncated = (
        frame["teacher_truncated"].astype(bool)
        | frame["teacher_finish_reason"].astype(str).eq("length")
        | frame["teacher_response_tokens"].astype(int).ge(int(max_response_length))
    )
    retained = frame.loc[~truncated].reset_index(drop=True)
    rejected = frame.loc[truncated].reset_index(drop=True)
    return retained, rejected


def generate_shared_teacher_sequences(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    settings = config["offline_teacher"]
    output = Path(data["sft_teacher_path"])
    success = output.with_suffix(".success.json")
    prompts_path = Path(data["sft_prompt_path"])
    frame = pd.read_parquet(prompts_path)
    expected_ids = frame["sample_id"].astype(str).tolist()
    expected_id_digest = sha256_text("\n".join(expected_ids))
    teacher = config["model"]["teacher_path"]
    thinking = thinking_enabled(config, "teacher")
    expected = {
        "teacher_fingerprint": directory_fingerprint(teacher),
        "input_sha256": sha256_file(prompts_path),
        "input_rows": len(expected_ids),
        "input_sample_ids_sha256": expected_id_digest,
        "seed": int(config.get("seed", 42)),
        "thinking_enabled": thinking,
        "temperature": float(settings.get("temperature", 0.6)),
        "top_p": float(settings.get("top_p", 0.95)),
        "max_response_length": int(settings.get("max_response_length", 16384)),
    }
    semantic_expected = {
        key: value for key, value in expected.items() if key != "input_sha256"
    }

    if output.exists() and success.exists():
        manifest = json.loads(success.read_text(encoding="utf-8"))
        existing = pd.read_parquet(
            output,
            columns=[
                "sample_id",
                "teacher_response_tokens",
                "teacher_finish_reason",
                "teacher_truncated",
            ],
        )
        retained, rejected = filter_complete_teacher_sequences(
            existing,
            max_response_length=expected["max_response_length"],
        )
        if (
            len(rejected) == 0
            and existing["sample_id"].astype(str).tolist() == retained["sample_id"].astype(str).tolist()
            and set(existing["sample_id"].astype(str)).issubset(expected_ids)
            and manifest.get("retained_rows") == len(existing)
            and manifest.get("output_sha256") == sha256_file(output)
            and all(manifest.get(key) == value for key, value in semantic_expected.items())
        ):
            return manifest

    batch_prompts = int(settings.get("batch_prompts", 128))
    raw_rows = frame.to_dict("records")
    shard_dir = output.parent / f".{output.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    missing_ranges = []
    for start in range(0, len(expected_ids), batch_prompts):
        stop = min(start + batch_prompts, len(expected_ids))
        shard = shard_dir / f"part_{start:06d}_{stop:06d}.parquet"
        try:
            cached = pd.read_parquet(shard, columns=["sample_id"])
            if cached["sample_id"].astype(str).tolist() == expected_ids[start:stop]:
                continue
        except Exception:
            pass
        missing_ranges.append((start, stop, shard))

    if missing_ranges:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        set_seed(int(config.get("seed", 42)))
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(teacher)
        rendered = [
            render_chat(
                tokenizer,
                _messages(row["messages"]),
                thinking_enabled=thinking,
            )
            for row in raw_rows
        ]
        llm = LLM(
            model=teacher,
            tensor_parallel_size=int(settings.get("tensor_parallel_size", 8)),
            dtype="bfloat16",
            max_model_len=int(settings.get("max_model_len", 17409)),
            gpu_memory_utilization=float(settings.get("gpu_memory_utilization", 0.82)),
            enable_prefix_caching=True,
            # Under TP=8 the custom all-reduce kernel aborts with "Cuda error
            # custom_all_reduce.cuh:453 'invalid argument'". Disabling that kernel is
            # what fixes it -- CUDA graphs are a separate setting and stay ON.
            #
            # They used to be disabled here alongside it, on the theory that graph
            # capture went down with the kernel and that generation is decode-bound
            # enough not to care. Both halves were wrong: capture succeeds on all eight
            # ranks with the kernel disabled, and eager costs most of the throughput --
            # measured on this generation, 830 tok/s eager against 2,170 tok/s captured.
            disable_custom_all_reduce=True,
            # Cap resident sequences. Left unset, vLLM admits the whole batch at once and
            # every decode step pays a device-to-host sync across all of them; the
            # benchmark suite measured 38 minutes per tick that way against 4m32s with
            # this set. The batch here is `batch_prompts` (256), so unset means 256
            # resident.
            max_num_seqs=int(settings.get("max_num_seqs", 32)),
            seed=int(config.get("seed", 42)),
        )
        sampling = SamplingParams(
            temperature=float(settings.get("temperature", 0.6)),
            top_p=float(settings.get("top_p", 0.95)),
            max_tokens=int(settings.get("max_response_length", 16384)),
            n=1,
        )
        for start, stop, shard in missing_ranges:
            generated = llm.generate(rendered[start:stop], sampling)
            records = []
            for row, result in zip(raw_rows[start:stop], generated):
                candidate = result.outputs[0]
                user_messages = _messages(row["messages"])
                records.append(
                    {
                        "sample_id": row["sample_id"],
                        "domain": row["domain"],
                        "source": row["source"],
                        "enable_thinking": thinking,
                        "messages": [
                            *user_messages,
                            {"role": "assistant", "content": candidate.text},
                        ],
                        "teacher_response": candidate.text,
                        "teacher_response_tokens": len(candidate.token_ids),
                        "teacher_finish_reason": candidate.finish_reason,
                        "teacher_truncated": candidate.finish_reason == "length",
                    }
                )
            temporary_shard = shard.with_suffix(".partial.parquet")
            pd.DataFrame(records).to_parquet(temporary_shard, index=False)
            os.replace(temporary_shard, shard)
        del llm

    shard_frames = []
    for start in range(0, len(expected_ids), batch_prompts):
        stop = min(start + batch_prompts, len(expected_ids))
        shard_frames.append(pd.read_parquet(shard_dir / f"part_{start:06d}_{stop:06d}.parquet"))
    completed = pd.concat(shard_frames, ignore_index=True)
    if completed["sample_id"].astype(str).tolist() != expected_ids:
        raise RuntimeError("Offline teacher shards do not match the deterministic SFT prompt order")

    retained, rejected = filter_complete_teacher_sequences(
        completed,
        max_response_length=expected["max_response_length"],
    )
    if retained.empty:
        raise RuntimeError("All offline teacher responses reached the response-length limit")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(".partial.parquet")
    retained.to_parquet(temporary_output, index=False)
    os.replace(temporary_output, output)
    rejected_output = output.with_suffix(".filtered_length.parquet")
    if len(rejected):
        temporary_rejected = rejected_output.with_suffix(".partial.parquet")
        rejected.to_parquet(temporary_rejected, index=False)
        os.replace(temporary_rejected, rejected_output)
    else:
        rejected_output.unlink(missing_ok=True)
    manifest = {
        "status": "completed",
        "created_at": utc_now(),
        "teacher": teacher,
        **expected,
        "input": str(prompts_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "raw_rows": len(completed),
        "retained_rows": len(retained),
        "filtered_length_rows": len(rejected),
        "filtered_length_fraction": len(rejected) / len(completed),
        "filtered_length_path": str(rejected_output) if len(rejected) else None,
        "sample_id_unique": int(retained["sample_id"].nunique()),
        "raw_response_tokens": int(completed["teacher_response_tokens"].sum()),
        "retained_response_tokens": int(retained["teacher_response_tokens"].sum()),
        "retained_max_response_tokens": int(retained["teacher_response_tokens"].max()),
    }
    dump_json(manifest, success)
    shutil.rmtree(shard_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(generate_shared_teacher_sequences(config), indent=2))


if __name__ == "__main__":
    main()
