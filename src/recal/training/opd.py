from __future__ import annotations

import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from packaging.version import Version

from recal.common import dump_json
from recal.modes import thinking_enabled
from recal.training.convergence import ConvergenceMonitor, parse_metric_line


SUPPORTED_VLLM_MIN = Version("0.8.5")
SUPPORTED_VLLM_MAX = Version("0.11.0")


def _python_executable(config: dict[str, Any]) -> str:
    configured = config.get("runtime", {}).get("python")
    executable = Path(configured).expanduser().resolve() if configured else Path(sys.executable).resolve()
    if not executable.exists():
        raise FileNotFoundError(f"Configured Python executable does not exist: {executable}")
    return str(executable)


def _integrated_source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_training_backend(config: dict[str, Any]) -> str:
    del config
    root = _integrated_source_root()
    required_symbols = {
        root / "verl/trainer/ppo/core_algos.py": 'register_adv_est("token_reward_direct")',
        root / "verl/trainer/ppo/ray_trainer.py": "log_prob_top_k",
        root / "verl/workers/fsdp_workers.py": "top_k_strategy",
        root / "verl/workers/actor/dp_actor.py": "reward_weight_mode",
    }
    for path, symbol in required_symbols.items():
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Missing integrated OPD/verl source: {path}") from error
        if symbol not in source:
            raise RuntimeError(
                f"Integrated OPD/verl feature {symbol!r} is missing from {path}"
            )
    return str(root / "verl")


def verify_opd_runtime(config: dict[str, Any]) -> dict[str, str]:
    python = _python_executable(config)
    root = _integrated_source_root()
    script = f"""
import json
from importlib import metadata
from packaging.version import Version
import torch
import transformers
import vllm
import verl
versions = {{}}
for name in ["torch", "vllm", "transformers", "ray", "tensordict", "flashinfer-python", "wandb"]:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
assert torch.cuda.is_available(), "CUDA is required"
assert Version(versions["vllm"]) >= Version("{SUPPORTED_VLLM_MIN}"), versions
assert Version(versions["vllm"]) <= Version("{SUPPORTED_VLLM_MAX}"), versions
assert Version(versions["transformers"]) < Version("5"), versions
assert str(verl.__file__).startswith({str(root / "verl")!r}), verl.__file__
print(json.dumps(versions, sort_keys=True))
"""
    result = subprocess.run(
        [python, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_framework_env(config),
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _logger_config(config: dict[str, Any]) -> str:
    if str(config.get("wandb", {}).get("mode", "online")) == "disabled":
        return "['console']"
    return "['console','wandb']"


def _ray_cpu_budget(gpus: int = 8, overhead: int = 8) -> int:
    """CPU slots to declare to Ray: enough for the placement group plus its overhead.

    Not the real core count. Ray's ``num_cpus`` is an accounting quantity used to admit
    bundles, and verl's resource pool requests one CPU per GPU bundle under
    ``STRICT_PACK`` -- so a budget equal to the core count leaves nothing for the
    TaskRunner actor and Ray's dashboard/runtime-env agents, and ``get_placement_groups``
    blocks forever on a bundle that can never be scheduled.

    Declaring more slots than cores does not oversubscribe the CPU: the OS still
    schedules the same 8 cores, and this work is GPU-bound. What must not happen is
    Ray *discovering* the host's 384 cores and prespawning a worker per core, which is
    what ``_visible_cpu_count`` exists to prevent.
    """
    return max(_visible_cpu_count(), gpus + overhead)


def _visible_cpu_count() -> int:
    """Cores this process may actually run on, not the machine's total.

    ``os.cpu_count()`` reports the host's cores and ignores container limits: on these
    nodes it returns 384 against 8 available. Ray sizes its worker pool from that
    number, so the wrong answer here is not a performance detail -- it prespawns one
    torch-importing worker per phantom core until the raylet's registration window
    expires, and the run hangs with hundreds of ``ray::IDLE`` processes and idle GPUs
    while the driver reports "Failed to register worker to Raylet".

    Sources are tried in order of how reliably each reflects the real limit *here*:

      ``nproc``            respects cpuset, cgroup quota and affinity together, and is
                           the only one of the three that returns 8 on these nodes.
      cgroup v2 ``cpu.max`` a quota, when one is set; absent under cgroup v1.
      cgroup v1 quota      ``-1`` here, meaning unlimited -- the limit is expressed as a
                           cpuset, not a quota, which is why quota checks alone miss it.
      ``sched_getaffinity`` returns 384 here despite ``nproc`` returning 8, so it is only
                           a last resort.
    """
    try:
        count = int(subprocess.run(["nproc"], capture_output=True, text=True, check=True).stdout)
        if count > 0:
            return count
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    for path, parse in (
        (Path("/sys/fs/cgroup/cpu.max"), lambda text: text.split()),
        (Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), None),
    ):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if parse is not None:
            quota, period = parse(text)
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
        elif text != "-1":
            period = int(
                Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8")
            )
            return max(1, int(int(text) / period))

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


def _hydra(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(repr(str(item)) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)


def _wandb_env(config: dict[str, Any], output_dir: str | Path, *, run_kind: str) -> dict[str, str]:
    output = Path(output_dir)
    run_id_file = output / f"wandb_{run_kind}_run_id.txt"
    if run_id_file.exists():
        run_id = run_id_file.read_text(encoding="utf-8").strip()
    else:
        try:
            import wandb

            run_id = wandb.util.generate_id()
        except Exception:
            run_id = secrets.token_hex(4)
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(run_id + "\n", encoding="utf-8")
    settings = config.get("wandb", {})
    env = {
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow",
        "WANDB_DIR": str(output.resolve()),
        "WANDB_MODE": str(settings.get("mode", "online")),
        "WANDB_INIT_TIMEOUT": str(int(settings.get("init_timeout", 300))),
        "WANDB_HTTP_TIMEOUT": str(int(settings.get("http_timeout", 300))),
    }
    if settings.get("entity"):
        env["WANDB_ENTITY"] = str(settings["entity"])
    return env


def _base_env(config: dict[str, Any], output_dir: str | Path, *, run_kind: str) -> dict[str, str]:
    env = _framework_env(config)
    env.update(_wandb_env(config, output_dir, run_kind=run_kind))
    return env


def _framework_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_integrated_source_root()),
            str(Path(".").resolve()),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_memory_usage_threshold"] = "0.99"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    env["NCCL_TIMEOUT"] = "7200"
    env["NCCL_DEBUG"] = "WARN"
    # vLLM's colocated sleep-mode allocator is incompatible with PyTorch
    # expandable segments. Keep the variable defined without that option so
    # Ray workers do not inherit the project shell's allocator setting.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    # W&B needs the HTTP proxy to reach api.wandb.ai, but Ray's dashboard and
    # the NCCL/gloo rendezvous all talk to localhost and this node's own IP.
    # Routing those through a proxy would break cluster startup, so pin them
    # into no_proxy whenever a proxy is configured.
    if env.get("http_proxy") or env.get("https_proxy"):
        local = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        try:
            import socket

            local.append(socket.gethostbyname(socket.gethostname()))
        except Exception:
            pass
        existing = [item for item in env.get("no_proxy", "").split(",") if item]
        env["no_proxy"] = ",".join(dict.fromkeys(existing + local))
        env["NO_PROXY"] = env["no_proxy"]
    # CUDA_LAUNCH_BLOCKING is removed for training because it serializes CUDA work.
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    return env


def verify_sft_runtime(config: dict[str, Any]) -> dict[str, str]:
    """SFT runs on the same pinned verl checkout and interpreter as OPD."""
    return verify_opd_runtime(config)


def _build_sft_command(
    config: dict[str, Any],
    *,
    student_model: str,
    stage_dir: str | Path,
) -> list[str]:
    """Build the verl sequence-level SFT command for one cascade stage.

    Targets ``verl.trainer.sft_trainer``, whose Hydra config is
    ``sft_trainer_engine`` -- ``data.messages_key`` is flat (not
    ``data.multiturn.*``), the model group is ``model@model: hf_model``, and
    ``checkpoint`` is top level rather than under ``trainer``.

    Thinking mode is deliberately *not* passed through
    ``apply_chat_template_kwargs``. ``MultiTurnSFTDataset`` already reads
    ``enable_thinking`` per row from the teacher parquet and forwards it to
    ``apply_chat_template`` as a named argument, so also supplying it as a kwarg
    raises ``TypeError: got multiple values for keyword argument
    'enable_thinking'`` on the first batch. The per-row column is authoritative;
    ``_prepare_shared_training_data`` verifies it agrees with the config.
    """
    sft = config["sft"]
    data = config["data"]
    checkpoint_dir = Path(stage_dir) / "sft_checkpoints"
    experiment = f"{config['experiment']['name']}-{Path(stage_dir).name}-sft"
    max_length = int(sft.get("max_length", 17408))
    max_token_len_per_gpu = int(sft.get("max_token_len_per_gpu", max_length))
    return [
        _python_executable(config),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={int(sft.get('n_gpus_per_node', 8))}",
        "-m",
        "verl.trainer.sft_trainer",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        # The offline teacher parquet is consumed directly; there is no
        # intermediate dataset conversion step to fall out of sync.
        f"data.train_files={_hydra([data['sft_teacher_path']])}",
        "data.val_files=null",
        f"data.train_batch_size={int(sft['train_batch_size'])}",
        f"data.micro_batch_size_per_gpu={int(sft.get('micro_batch_size_per_gpu', 1))}",
        f"data.max_token_len_per_gpu={max_token_len_per_gpu}",
        "data.use_dynamic_bsz=True",
        "data.messages_key=messages",
        "data.enable_thinking_key=enable_thinking",
        # Teacher responses span a few hundred to 16k tokens. Padding every
        # sample to the cutoff would spend most of the token budget on padding,
        # so pack without padding and let dynamic batching balance by token count.
        "data.pad_mode=no_padding",
        f"data.max_length={max_length}",
        "data.truncation=right",
        f"model.path={student_model}",
        f"model.trust_remote_code={_hydra(sft.get('trust_remote_code', True))}",
        f"model.use_remove_padding={_hydra(sft.get('use_remove_padding', True))}",
        f"model.enable_gradient_checkpointing={_hydra(sft.get('enable_gradient_checkpointing', True))}",
        f"model.use_liger={_hydra(sft.get('enable_liger_kernel', True))}",
        f"engine.strategy={str(sft.get('fsdp_strategy', 'fsdp'))}",
        f"engine.ulysses_sequence_parallel_size={int(sft.get('ulysses_sequence_parallel_size', 1))}",
        f"engine.param_offload={_hydra(sft.get('param_offload', False))}",
        f"engine.optimizer_offload={_hydra(sft.get('optimizer_offload', False))}",
        f"engine.forward_prefetch={_hydra(sft.get('forward_prefetch', True))}",
        # Master weights stay fp32 for the same reason as the OPD actor: FSDP
        # still runs the forward in bf16, but a bf16 master weight cannot
        # represent a 1e-5 step against a typical FFN weight without rounding
        # most of it away.
        f"engine.model_dtype={str(sft.get('model_dtype', 'fp32'))}",
        f"optim.lr={float(sft['learning_rate'])}",
        f"optim.weight_decay={float(sft.get('weight_decay', 0.0))}",
        f"optim.clip_grad={float(sft.get('gradient_clip', 1.0))}",
        f"optim.lr_scheduler_type={str(sft.get('lr_scheduler_type', 'cosine'))}",
        f"optim.lr_warmup_steps_ratio={float(sft.get('warmup_ratio', 0.05))}",
        # hf_model makes the final step write a merged HF checkpoint under
        # global_step_N/huggingface, which is what the benchmark and OPD load.
        "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "checkpoint.load_contents=[model,optimizer,extra]",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.project_name={sft['project_name']}",
        f"trainer.experiment_name={experiment}",
        f"trainer.total_epochs={int(sft.get('total_epochs', 1))}",
        # fit() always saves on the last step, so a periodic frequency is only
        # needed when intermediate checkpoints are wanted.
        f"trainer.save_freq={int(sft.get('save_freq', -1))}",
        "trainer.test_freq=-1",
        f"trainer.max_ckpt_to_keep={_hydra(sft.get('max_ckpt_to_keep', 1))}",
        "trainer.resume_mode=auto",
        f"trainer.logger={_logger_config(config)}",
        f"trainer.seed={int(config.get('seed', 42))}",
    ]


def build_sft_command(
    config: dict[str, Any],
    *,
    student_model: str,
    stage_dir: str | Path,
) -> list[str]:
    return _build_sft_command(config, student_model=student_model, stage_dir=stage_dir)


def _has_hf_weights(path: Path) -> bool:
    """True only if ``path`` is a loadable HF checkpoint.

    ``config.json`` alone is not enough. The pinned verl checkout writes the
    config and tokenizer into ``huggingface/`` on every save but silently skips
    the weights (see ``export_sft_checkpoint``), so a config-only directory looks
    complete and then fails inside vLLM with "Cannot find any model weights".
    """
    if not (path / "config.json").exists():
        return False
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            # A shard named by the index but absent or empty means a truncated
            # copy, which must not pass as loadable.
            return bool(shards) and all(
                (path / shard).is_file() and (path / shard).stat().st_size > 0
                for shard in shards
            )
        except Exception:
            return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def export_sft_checkpoint(step_dir: Path, config: dict[str, Any]) -> Path:
    """Ensure ``step_dir/huggingface`` holds merged weights, merging if needed.

    ``checkpoint.save_contents=[...,hf_model]`` should make verl write the merged
    model itself, but the pinned checkout drops it: its FSDP engine constructs
    ``FSDPCheckpointManager(checkpoint_contents=...)`` while the manager's
    parameter is ``checkpoint_config``. The stray keyword is absorbed by
    ``**kwargs``, so ``checkpoint_config`` stays ``None`` and falls back to
    ``["model","optimizer","extra"]`` -- ``hf_model`` is never honored regardless
    of what is requested.

    Rather than patch the pinned checkout, merge the sharded state here with the
    same ``verl.model_merger`` used to export OPD actors. The FSDP shards are
    complete, so this is a pure format conversion and no training is lost.
    """
    target = step_dir / "huggingface"
    if _has_hf_weights(target):
        return target
    if not any(step_dir.glob("model_world_size_*_rank_*.pt")):
        raise FileNotFoundError(
            f"{step_dir} has neither merged HF weights nor FSDP model shards to merge"
        )
    print(f"+ merging FSDP shards in {step_dir} -> {target}", flush=True)
    staging = step_dir / ".huggingface.merging"
    shutil.rmtree(staging, ignore_errors=True)
    command = [
        _python_executable(config),
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(step_dir),
        "--target_dir",
        str(staging),
    ]
    subprocess.run(command, check=True, env=_framework_env(config))
    if not _has_hf_weights(staging):
        raise RuntimeError(f"Merge produced no usable weights in {staging}")
    # Move the weights beside the config/tokenizer verl already wrote, so the
    # directory the rest of the pipeline points at becomes loadable in place.
    target.mkdir(parents=True, exist_ok=True)
    for item in staging.iterdir():
        destination = target / item.name
        if destination.exists():
            continue
        shutil.move(str(item), str(destination))
    shutil.rmtree(staging, ignore_errors=True)
    if not _has_hf_weights(target):
        raise RuntimeError(f"Incomplete SFT checkpoint after merge: {target}")
    return target


def expected_sft_steps(config: dict[str, Any]) -> int | None:
    """How many optimizer steps a finished SFT run should have taken.

    ``floor(rows / batch) * epochs``, matching verl's own step accounting: its
    DataLoader runs with ``drop_last=True``, so the final partial batch is dropped
    rather than counted as a step. A first version of this used ``ceil`` and then
    rejected a legitimately complete checkpoint (616 steps of 9,883 rows at batch 32
    x 2 epochs) as one step short, retraining the whole SFT run against it. Returns
    None when the teacher parquet is unreadable, in which case completeness cannot
    be judged and callers must not guess.

    This exists because "a checkpoint is present" and "training finished" are
    different claims, and conflating them silently changes what the experiment
    measured: a mid-run checkpoint scored as after_sft reports a half-trained model
    as the recovery result. That happened twice -- once at step 400 of 728, once at
    step 200 of 616 -- both times after a crash left an intermediate export behind.
    """
    try:
        rows = len(pd.read_parquet(config["data"]["sft_teacher_path"], columns=["sample_id"]))
    except Exception:
        return None
    batch = int(config["sft"]["train_batch_size"])
    epochs = int(config["sft"].get("total_epochs", 1))
    if rows <= 0 or batch <= 0 or epochs <= 0:
        return None
    return (rows // batch) * epochs


def latest_sft_checkpoint(stage_dir: str | Path, config: dict[str, Any] | None = None) -> Path | None:
    """Newest exported HF checkpoint written by verl SFT, if any.

    Requires actual weight files, not just a config: a config-only directory is
    what the pinned checkout leaves behind, and treating it as complete would
    skip SFT and then fail at model load.

    When ``config`` is supplied, a checkpoint is only returned if its step count
    reaches the expected total, so an intermediate ``save_freq`` export left behind
    by a crash resumes training instead of being scored as the finished model.
    """
    candidates: list[tuple[int, Path]] = []
    for path in (Path(stage_dir) / "sft_checkpoints").glob("global_step_*/huggingface"):
        try:
            step = int(path.parent.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if _has_hf_weights(path):
            candidates.append((step, path))
    if not candidates:
        return None
    step, path = max(candidates)
    if config is not None:
        expected = expected_sft_steps(config)
        # Strictly fewer steps than expected means the run was interrupted. Returning
        # it would skip the remaining training and score a partial model.
        if expected is not None and step < expected:
            print(
                f"[sft] ignoring {path.parent.name}: {step} of {expected} expected steps; "
                "resuming training instead of scoring a partial checkpoint",
                flush=True,
            )
            return None
    return path


def latest_sft_step_dir(stage_dir: str | Path) -> Path | None:
    """Newest ``global_step_N`` directory holding FSDP shards, merged or not."""
    candidates: list[tuple[int, Path]] = []
    for path in (Path(stage_dir) / "sft_checkpoints").glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if any(path.glob("model_world_size_*_rank_*.pt")):
            candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def launch_sft(
    config: dict[str, Any],
    *,
    student_model: str,
    stage_dir: str | Path,
    dry_run: bool = False,
) -> Path:
    verify_training_backend(config)
    if not dry_run:
        verify_sft_runtime(config)
    command = build_sft_command(config, student_model=student_model, stage_dir=stage_dir)
    output = Path(stage_dir)
    dump_json(
        {
            "command": command,
            "implementation": "verl_sft_trainer",
            "train_files": [config["data"]["sft_teacher_path"]],
            "thinking_enabled": thinking_enabled(config, "sft"),
            "total_epochs": int(config["sft"].get("total_epochs", 1)),
            "train_batch_size": int(config["sft"]["train_batch_size"]),
        },
        output / "sft_launch.json",
    )
    print("+", " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return output / "sft_checkpoints" / "global_step_FINAL" / "huggingface"
    subprocess.run(command, check=True, env=_base_env(config, output, run_kind="sft"))
    step_dir = latest_sft_step_dir(output)
    if step_dir is None:
        raise FileNotFoundError(
            f"No SFT checkpoint under {output / 'sft_checkpoints'} after training"
        )
    return export_sft_checkpoint(step_dir, config)


def _opd_step_budget(config: dict[str, Any]) -> tuple[int, int]:
    """Return ``(max_steps, save_freq)`` for OPD recovery.

    ``max_training_steps`` is a ceiling, not a target: training normally stops
    earlier once teacher/student overlap plateaus. Checkpoints are therefore
    written periodically instead of only on the final step, so an early stop
    always has a recent actor to export.
    """
    opd = config["opd"]
    max_steps = int(opd.get("max_training_steps", opd.get("total_training_steps", 200)))
    save_freq = int(opd.get("save_freq", 0)) or max(1, min(25, max_steps))
    if max_steps <= 0:
        raise ValueError("opd.max_training_steps must be positive")
    return max_steps, save_freq


def build_opd_command(
    config: dict[str, Any],
    *,
    student_model: str,
    stage_dir: str | Path,
) -> list[str]:
    opd = config["opd"]
    data = config["data"]
    teacher = config["model"]["teacher_path"]
    max_model_len = int(opd["max_prompt_length"]) + int(opd["max_response_length"]) + 1
    max_tokens_per_gpu = int(opd.get("max_token_len_per_gpu", 32768))
    # vLLM refuses to start with chunked prefill when max_num_batched_tokens is
    # below max_model_len, because a single sequence could then never be
    # scheduled. max_model_len carries a +1 over the prompt+response budget, so a
    # config that sets the batched-token cap to exactly prompt+response is short
    # by one and the rollout workers all abort at init. Raise it rather than
    # failing: the cap only bounds a scheduling batch, so one extra token costs
    # nothing, whereas a stage that dies here wastes the whole SFT that preceded it.
    rollout_max_num_batched_tokens = max(
        int(opd.get("rollout_max_num_batched_tokens", max_tokens_per_gpu)),
        max_model_len,
    )
    max_steps, save_freq = _opd_step_budget(config)
    experiment = f"{config['experiment']['name']}-{Path(stage_dir).name}-opd"
    reward_path = Path(__file__).with_name("zero_reward.py").resolve()
    thinking = thinking_enabled(config, "opd")
    topk = int(opd.get("topk", 16))
    strategy = str(opd.get("top_k_strategy", "only_stu"))
    reward_weight_mode = str(opd.get("reward_weight_mode", "student_p"))
    if topk <= 0:
        raise ValueError("Formal OPD requires a positive top-k")
    if strategy != "only_stu" or reward_weight_mode != "student_p":
        raise ValueError(
            "Formal reverse-KL-style Top-k OPD requires top_k_strategy=only_stu and reward_weight_mode=student_p"
        )
    return [
        _python_executable(config),
        "-m",
        "verl.trainer.main_ppo",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "algorithm.adv_estimator=token_reward_direct",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={_hydra([data['opd_path']])}",
        # v0.7 constructs a validation dataloader unconditionally. Reuse the
        # prompt file only for construction; validation is disabled below.
        f"data.val_files={_hydra([data['opd_path']])}",
        "data.prompt_key=messages",
        f"data.train_batch_size={int(opd['train_batch_size'])}",
        f"data.max_prompt_length={int(opd['max_prompt_length'])}",
        f"data.max_response_length={int(opd['max_response_length'])}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.shuffle={_hydra(data.get('shuffle', False))}",
        "data.return_raw_chat=True",
        # Unlike the SFT dataset, verl's RLHFDataset has no per-row thinking
        # column: apply_chat_template_kwargs is the only way to set the switch
        # for rollout prompts, so it is passed here and deliberately not in the
        # SFT command.
        (
            "+data.apply_chat_template_kwargs.enable_thinking=True"
            if thinking
            else "+data.apply_chat_template_kwargs.enable_thinking=False"
        ),
        f"actor_rollout_ref.model.path={student_model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_activation_offload=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={float(opd['learning_rate'])}",
        f"actor_rollout_ref.actor.optim.weight_decay={float(opd.get('weight_decay', 0.0))}",
        f"actor_rollout_ref.actor.grad_clip={float(opd.get('gradient_clip', 1.0))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(opd['ppo_mini_batch_size'])}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={int(opd['ppo_micro_batch_size_per_gpu'])}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_tokens_per_gpu}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={int(opd.get('ulysses_sequence_parallel_size', 1))}",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True",
        # The actor's master weights must stay fp32. FSDP already runs the
        # forward in bf16 via MixedPrecision(param_dtype=bf16), so this only
        # sets the dtype the optimizer updates. Under bf16 master weights the
        # representable gap near a typical FFN weight (~2e-2) is ~8e-5, which
        # is ~78x larger than a 1e-6 Adam step, so most updates round away and
        # top-k overlap never rises above its own noise floor. verl defaults
        # the actor to fp32 for exactly this reason.
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={str(opd.get('actor_model_dtype', 'fp32'))}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(opd.get('rollout_tp', 1))}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={float(opd.get('rollout_gpu_memory_utilization', 0.8))}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={int(opd.get('rollout_max_num_seqs', opd['train_batch_size']))}",
        f"actor_rollout_ref.rollout.n={int(opd.get('rollout_n', 1))}",
        f"actor_rollout_ref.rollout.temperature={float(opd.get('rollout_temperature', 1.0))}",
        f"actor_rollout_ref.rollout.top_p={float(opd.get('rollout_top_p', 1.0))}",
        f"actor_rollout_ref.rollout.repetition_penalty={float(opd.get('repetition_penalty', 1.0))}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_tokens_per_gpu}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"+actor_rollout_ref.rollout.log_prob_top_k={topk}",
        f"+actor_rollout_ref.rollout.top_k_strategy={strategy}",
        f"+actor_rollout_ref.rollout.reward_weight_mode={reward_weight_mode}",
        f"+actor_rollout_ref.rollout.teacher_temperature={float(opd.get('teacher_temperature', 1.0))}",
        "reward_model.enable=True",
        "reward_model.enable_resource_pool=False",
        "reward_model.strategy=fsdp",
        f"reward_model.model.path={teacher}",
        "reward_model.model.input_tokenizer=null",
        "reward_model.model.use_remove_padding=True",
        "reward_model.model.fsdp_config.param_offload=False",
        "reward_model.model.fsdp_config.forward_prefetch=True",
        "+reward_model.model.dtype=bfloat16",
        "reward_model.micro_batch_size=null",
        f"reward_model.micro_batch_size_per_gpu={int(opd.get('teacher_micro_batch_size_per_gpu', 1))}",
        "reward_model.use_dynamic_bsz=True",
        f"reward_model.forward_max_token_len_per_gpu={int(opd.get('teacher_max_token_len_per_gpu', max_tokens_per_gpu))}",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=zero_reward",
        "trainer.balance_batch=True",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.log_val_generations=0",
        "trainer.is_plot=False",
        f"trainer.project_name={opd['project_name']}",
        f"trainer.experiment_name={experiment}",
        f"trainer.default_local_dir={Path(stage_dir) / 'opd_checkpoints'}",
        f"trainer.n_gpus_per_node={int(opd.get('n_gpus_per_node', 8))}",
        "trainer.nnodes=1",
        f"trainer.total_training_steps={max_steps}",
        f"trainer.save_freq={save_freq}",
        # Retain every periodic actor so a stage can be re-exported from any
        # step. With keep=1 verl deletes the previous actor on each save, which
        # defeats the point of saving periodically: only the final step has
        # weights, leaving no candidates to benchmark or fall back to.
        f"trainer.max_actor_ckpt_to_keep={_hydra(opd.get('max_actor_ckpt_to_keep'))}",
        "trainer.resume_mode=auto",
        # Pin Ray's CPU count to the cores this container actually has. Left unset, Ray
        # reads os.cpu_count(), which reports the *host's* core count -- 384 here
        # against 8 available -- and prespawns one idle worker per core. Each imports
        # torch, the raylet's registration window expires, and every worker dies with
        # "Failed to register worker to Raylet: IOError: Failed to read data from the
        # socket". The driver then hangs with ~230 ray::IDLE processes and no GPU work,
        # which reads like a Ray bug rather than a miscounted core budget.
        #
        # Pin Ray's CPU budget. Left unset, Ray reads os.cpu_count(), which reports the
        # *host's* cores -- 384 here against 8 available -- and prespawns one idle worker
        # per core. Each imports torch, the raylet's registration window expires, and the
        # driver hangs with ~250 ray::IDLE processes, no GPU work, and
        # "Failed to register worker to Raylet: IOError: Failed to read data from the
        # socket". That reads like a Ray bug rather than a miscounted core budget.
        #
        # The budget is deliberately *larger* than the real core count. verl's placement
        # group asks for one CPU per GPU bundle under STRICT_PACK (8 here, since FSDP
        # runs max_colocate_count=1), and the TaskRunner actor plus Ray's dashboard and
        # runtime-env agents each hold a slot on top of that. Setting exactly 8 therefore
        # deadlocks in get_placement_groups() waiting for a bundle that can never be
        # scheduled -- observed hanging there for 6 minutes with idle GPUs. Ray's CPU
        # count is an accounting quantity, not an affinity mask, so oversubscribing it is
        # safe: the work is GPU-bound and the OS still schedules on the real 8 cores.
        #
        # `ray_kwargs.ray_init`, not `ray_init`: main_ppo reads
        # `config.ray_kwargs.get("ray_init", {})`, so the shorter path is accepted by
        # Hydra and then silently ignored, leaving `num_cpus: None` in the log.
        f"ray_kwargs.ray_init.num_cpus={_ray_cpu_budget()}",
        f"trainer.logger={_logger_config(config)}",
    ]


def _run_opd_with_convergence(
    command: list[str],
    *,
    config: dict[str, Any],
    output: Path,
    max_steps: int,
    save_freq: int,
) -> dict[str, Any]:
    """Run OPD, stopping once teacher/student overlap plateaus.

    verl has no early-stopping hook and the checkout is pinned by SHA, so the
    decision is made here: the child's console metrics are parsed as they stream,
    and once the monitor converges we wait for the next periodic checkpoint and
    then shut the trainer down.
    """
    settings = dict(config["opd"].get("convergence", {}) or {})
    enabled = bool(settings.pop("enabled", True))
    monitor = ConvergenceMonitor(
        min_steps=int(settings.get("min_steps", 100)),
        window=int(settings.get("window", 20)),
        patience=int(settings.get("patience", 3)),
        min_delta=float(settings.get("min_delta", 0.002)),
        ema_beta=float(settings.get("ema_beta", 0.9)),
    )
    process = subprocess.Popen(
        command,
        env=_base_env(config, output, run_kind="opd"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stop_reason = "max_steps"
    checkpoint_target: int | None = None
    returncode = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            parsed = parse_metric_line(line)
            if parsed is None:
                continue
            step, value = parsed
            if not enabled:
                monitor.update(step, value)
                continue
            if checkpoint_target is None and monitor.update(step, value):
                # Let training continue to the next checkpoint boundary so the
                # exported actor reflects the converged policy.
                checkpoint_target = min(max_steps, ((step // save_freq) + 1) * save_freq)
                print(
                    f"[convergence] overlap_ratio plateaued at step {step}; "
                    f"stopping after checkpoint at step {checkpoint_target}",
                    flush=True,
                )
            if checkpoint_target is not None and step >= checkpoint_target:
                if (output / "opd_checkpoints" / f"global_step_{checkpoint_target}").exists():
                    stop_reason = "converged"
                    print(f"[convergence] checkpoint {checkpoint_target} written; terminating OPD", flush=True)
                    process.terminate()
                    break
        returncode = process.wait(timeout=1800)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if stop_reason != "converged" and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    report = monitor.report(stop_reason=stop_reason, final_step=monitor.last_step)
    report["convergence_enabled"] = enabled
    report["max_training_steps"] = max_steps
    report["save_freq"] = save_freq
    dump_json(report, output / "opd_convergence.json")
    return report


def count_overlong_opd_prompts(config: dict[str, Any]) -> dict[str, Any]:
    """Report how many OPD prompts ``filter_overlong_prompts`` will discard.

    verl drops prompts longer than ``max_prompt_length`` silently. That is the
    right behavior -- a truncated prompt would change the task -- but an
    unreported drop looks like the full split was trained on. Counting here keeps
    the realized prompt count in the stage record.
    """
    import pandas as pd
    from transformers import AutoTokenizer

    from recal.modes import as_messages

    opd = config["opd"]
    max_prompt_length = int(opd["max_prompt_length"])
    frame = pd.read_parquet(config["data"]["opd_path"], columns=["messages", "domain"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["teacher_path"])
    thinking = thinking_enabled(config, "opd")
    lengths = [
        len(
            tokenizer.apply_chat_template(
                as_messages(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        )
        for messages in frame["messages"]
    ]
    overlong = [length > max_prompt_length for length in lengths]
    dropped = int(sum(overlong))
    per_domain: dict[str, int] = {}
    for domain, is_overlong in zip(frame["domain"].astype(str), overlong):
        if is_overlong:
            per_domain[domain] = per_domain.get(domain, 0) + 1
    return {
        "max_prompt_length": max_prompt_length,
        "rows": len(frame),
        "dropped_overlong_prompts": dropped,
        "dropped_fraction": round(dropped / len(frame), 4) if len(frame) else 0.0,
        "retained_rows": len(frame) - dropped,
        "dropped_per_domain": per_domain,
        "max_prompt_tokens": max(lengths) if lengths else 0,
    }


def launch_opd(
    config: dict[str, Any],
    *,
    student_model: str,
    stage_dir: str | Path,
    dry_run: bool = False,
) -> Path:
    verify_training_backend(config)
    if not dry_run:
        verify_opd_runtime(config)
    command = build_opd_command(config, student_model=student_model, stage_dir=stage_dir)
    output = Path(stage_dir)
    max_steps, save_freq = _opd_step_budget(config)
    dump_json(
        {
            "command": command,
            "implementation": "opd",
            "algorithm": "token_reward_direct",
            "topk": int(config["opd"].get("topk", 16)),
            "top_k_strategy": config["opd"].get("top_k_strategy", "only_stu"),
            "reward_weight_mode": config["opd"].get("reward_weight_mode", "student_p"),
            "thinking_enabled": thinking_enabled(config, "opd"),
            "max_training_steps": max_steps,
            "save_freq": save_freq,
        },
        output / "opd_launch.json",
    )
    print("+", " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return output / "opd_checkpoints" / f"global_step_{max_steps}" / "actor"
    try:
        dump_json(count_overlong_opd_prompts(config), output / "opd_prompt_budget.json")
    except Exception as error:  # pragma: no cover - accounting must not block training
        print(f"[warn] could not count overlong OPD prompts: {error}", flush=True)
    _run_opd_with_convergence(
        command,
        config=config,
        output=output,
        max_steps=max_steps,
        save_freq=save_freq,
    )
    candidates = []
    for path in (output / "opd_checkpoints").glob("global_step_*/actor"):
        try:
            candidates.append((int(path.parent.name.removeprefix("global_step_")), path))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No OPD actor checkpoint under {output / 'opd_checkpoints'}")
    # Recovery stops on convergence rather than at a fixed step, so the newest
    # checkpoint is authoritative; only its absence is an error.
    return max(candidates)[1]
