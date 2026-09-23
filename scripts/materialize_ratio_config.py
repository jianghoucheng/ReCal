#!/usr/bin/env python
"""Create a ratio-specific config from one of the public 25% templates."""

from __future__ import annotations

import argparse
from pathlib import Path

from recal.common import dump_yaml, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--full-recovery",
        action="store_true",
        help="run OPD after SFT; by default ratio sweeps stop after SFT",
    )
    args = parser.parse_args()

    if not 0.0 < args.ratio < 1.0:
        raise ValueError("--ratio must be in (0, 1)")

    config = load_yaml(args.config)
    tag = f"r{round(args.ratio * 100):02d}"
    method = str(config["pruning"]["method"])
    model = Path(args.config).parent.name

    config["pruning"]["ratios"] = [args.ratio]
    config["pruning"]["importance_dir"] = f"artifacts/ratio_sweep/{model}/{method}_{tag}_statistics"
    config["pruning"]["mask_dir"] = f"artifacts/ratio_sweep/{model}/{method}_{tag}_masks"
    if "recal" in config["pruning"]:
        base_method = method.removesuffix("_recal")
        config["pruning"]["recal"]["statistics_dir"] = config["pruning"]["importance_dir"]
        config["pruning"]["recal"]["probe_path"] = (
            f"outputs/ratio_sweep/{model}/{base_method}_{tag}/"
            f"stages/stage_01_{tag}/pruned_initial"
        )

    config["experiment"]["name"] = f"{method}-{model}-{tag}"
    config["experiment"]["output_dir"] = f"outputs/ratio_sweep/{model}/{method}_{tag}"
    config["pipeline"] = {
        "stop_after": "after_opd" if args.full_recovery else "after_sft"
    }
    dump_yaml(config, args.output)


if __name__ == "__main__":
    main()

