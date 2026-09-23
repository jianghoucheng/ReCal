from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(value).__name__}")
    base_path = value.pop("_base_", None)
    if base_path:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        value = _deep_merge(load_yaml(base_path), value)
    return value


def atomic_write_text(text: str, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def dump_yaml(value: Any, path: str | os.PathLike[str]) -> None:
    text = yaml.safe_dump(to_jsonable(value), sort_keys=False, allow_unicode=True)
    atomic_write_text(text, path)


def dump_json(value: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    text = json.dumps(to_jsonable(value), ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    atomic_write_text(text, path)


def dump_jsonl(values: Iterable[Any], path: str | os.PathLike[str]) -> None:
    text = "".join(json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n" for value in values)
    atomic_write_text(text, path)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(value: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: str | os.PathLike[str]) -> str:
    """Stable metadata/content fingerprint for a checkpoint directory.

    Large model shards are hashed by file size and their safetensors header/index
    metadata rather than reading every weight byte. Individual manifests and
    configuration files are fully hashed.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = child.relative_to(root).as_posix()
        stat = child.stat()
        digest.update(rel.encode())
        digest.update(str(stat.st_size).encode())
        if stat.st_size <= 16 << 20 or child.suffix in {".json", ".yaml", ".yml", ".txt", ".jinja"}:
            digest.update(sha256_file(child).encode())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha(path: str | os.PathLike[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def package_direct_url(name: str) -> dict[str, Any] | None:
    try:
        from importlib.metadata import distribution

        dist = distribution(name)
        for file in dist.files or []:
            if str(file).endswith("direct_url.json"):
                return json.loads(dist.locate_file(file).read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def environment_snapshot() -> dict[str, Any]:
    gpu = []
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in output.splitlines():
            idx, name, memory, driver = [x.strip() for x in line.split(",", 3)]
            gpu.append({"index": int(idx), "name": name, "memory_mib": int(memory), "driver": driver})
    except Exception:
        pass
    packages = [
        "torch",
        "transformers",
        "datasets",
        "safetensors",
        "vllm",
        "ray",
        "verl",
        "pyarrow",
        "pandas",
    ]
    return {
        "created_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": gpu,
        "packages": {name: package_version(name) for name in packages},
        "package_sources": {name: package_direct_url(name) for name in packages},
        "env": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "NCCL_IB_DISABLE",
                "NCCL_IB_HCA",
                "NCCL_SOCKET_IFNAME",
                "HF_HOME",
                "TRANSFORMERS_CACHE",
            ]
            if os.environ.get(key) is not None
        },
    }


def supports_enable_thinking(tokenizer: Any) -> bool:
    """Verify the tokenizer exposes Qwen's official chat-template switch.

    New tokenizer implementations generally accept ``**kwargs`` rather than
    listing ``enable_thinking`` explicitly, so an actual template render is the
    authoritative check.
    """
    fn = tokenizer.apply_chat_template
    signature = inspect.signature(fn)
    explicit = "enable_thinking" in signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if not (explicit or accepts_kwargs):
        return False
    try:
        fn(
            [{"role": "user", "content": "ping"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return True
    except (TypeError, ValueError):
        return False


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, thinking_enabled: bool) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if supports_enable_thinking(tokenizer):
        kwargs["enable_thinking"] = thinking_enabled
    elif thinking_enabled:
        raise RuntimeError(
            "The tokenizer/chat template does not accept enable_thinking=True. "
            "Refusing to hand-build a <think> template."
        )
    return tokenizer.apply_chat_template(messages, **kwargs)


def batched(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
