#!/usr/bin/env python
"""Download the two model revisions used in the reference experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "qwen3_8b": (
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        "models/Qwen3-8B",
    ),
    "qwen3_4b": (
        "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554",
        "models/Qwen3-4B-Instruct-2507",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(MODELS),
        default=sorted(MODELS),
    )
    args = parser.parse_args()

    for name in args.models:
        repo_id, revision, local_dir = MODELS[name]
        target = Path(local_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id}@{revision} to {target}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
        )


if __name__ == "__main__":
    main()
