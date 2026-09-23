#!/usr/bin/env python
"""Collect benchmark summaries produced by the release pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--output", default="results/reproduced_results.json")
    args = parser.parse_args()

    root = Path(args.outputs)
    rows = []
    for path in sorted(root.rglob("benchmarks/*/summary.json")):
        diagnostics_path = path.with_name("generation_diagnostics.json")
        metrics = load(path)
        aime = [
            metrics.get("aime_2024_avg8"),
            metrics.get("aime_2025_avg8"),
            metrics.get("aime_2026_avg8"),
        ]
        valid_aime = [float(x) for x in aime if x is not None]
        row = {
            "path": str(path),
            "boundary": path.parent.name,
            "aime_avg": sum(valid_aime) / len(valid_aime) if valid_aime else None,
            **metrics,
        }
        if diagnostics_path.exists():
            row["diagnostics"] = load(diagnostics_path)
        rows.append(row)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} benchmark rows to {destination}")


if __name__ == "__main__":
    main()
