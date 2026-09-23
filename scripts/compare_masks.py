#!/usr/bin/env python
"""Compare a baseline mask with its ReCal counterpart layer by layer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_mask(path: str) -> tuple[int, dict[int, set[int]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["retained_original_channel_ids_per_layer"]
    width = int(payload["original_intermediate_size"])
    return width, {int(layer): set(map(int, ids)) for layer, ids in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mask", required=True)
    parser.add_argument("--recal-mask", required=True)
    parser.add_argument("--output", default=None, help="optional layer-wise CSV")
    args = parser.parse_args()

    base_width, base = load_mask(args.base_mask)
    recal_width, recal = load_mask(args.recal_mask)
    if base_width != recal_width:
        raise ValueError("Masks use different original intermediate sizes")
    if base.keys() != recal.keys():
        raise ValueError("Masks do not cover the same layers")

    rows = []
    for layer in sorted(base):
        base_removed = set(range(base_width)) - base[layer]
        recal_removed = set(range(base_width)) - recal[layer]
        if len(base_removed) != len(recal_removed):
            raise ValueError(f"Layer {layer} masks prune different channel counts")
        overlap = len(base_removed & recal_removed) / max(len(base_removed), 1)
        rows.append(
            {
                "layer": layer,
                "removed_overlap": overlap,
                "removed_replacement": 1.0 - overlap,
                "removed_channels": len(base_removed),
            }
        )

    summary = {
        "layers": len(rows),
        "mean_removed_overlap": sum(r["removed_overlap"] for r in rows) / len(rows),
        "mean_removed_replacement": sum(r["removed_replacement"] for r in rows) / len(rows),
        "max_removed_replacement": max(r["removed_replacement"] for r in rows),
        "max_replacement_layer": max(rows, key=lambda r: r["removed_replacement"])["layer"],
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
