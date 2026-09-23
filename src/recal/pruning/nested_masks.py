from __future__ import annotations

import argparse
import math
from pathlib import Path

from transformers import AutoConfig

from recal.common import dump_json, load_json, utc_now
from recal.pruning.cost_model import estimate_pruning_cost


def rounded_keep_size(original: int, ratio: float, multiple: int) -> int:
    """Round an FFN-width pruning ratio to an efficient channel multiple."""
    raw = (1.0 - ratio) * original
    # Nearest hardware multiple, bounded to [multiple, original].
    rounded = int(math.floor(raw / multiple + 0.5) * multiple)
    return max(min(rounded, original), min(multiple, original))


def total_parameter_target_to_ffn_ratio(
    *,
    total_parameters: int,
    ffn_parameters: int,
    target_total_parameter_reduction: float,
) -> float:
    """Translate a whole-model parameter target into an FFN-width ratio.

    Only dense FFN channels are physically removed, so the requested total
    parameter reduction is divided by the fraction of model parameters that
    belong to gate/up/down projections.
    """
    if total_parameters <= 0 or ffn_parameters <= 0:
        raise ValueError("total_parameters and ffn_parameters must be positive")
    ratio = target_total_parameter_reduction * total_parameters / ffn_parameters
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"Total-parameter target {target_total_parameter_reduction:.2%} "
            f"requires FFN-width ratio {ratio:.2%}, which is not feasible"
        )
    return ratio


def build_nested_masks(
    channel_orders: dict[int, list[int]],
    ratios: list[float],
    *,
    original_size: int,
    hardware_multiple: int = 128,
) -> dict[float, dict[int, list[int]]]:
    ratios = sorted(set(float(r) for r in ratios))
    previous_sets = {idx: set(order) for idx, order in channel_orders.items()}
    masks: dict[float, dict[int, list[int]]] = {}
    previous_keep = original_size
    for ratio in ratios:
        keep = rounded_keep_size(original_size, ratio, hardware_multiple)
        keep = min(keep, previous_keep)
        layer_masks = {}
        for idx, order in channel_orders.items():
            selected = set(order[:keep])
            if not selected.issubset(previous_sets[idx]):
                raise AssertionError(f"Layer {idx} mask at ratio {ratio} is not nested")
            # Keep physical channels in original order. This makes cascade index
            # translation deterministic and exactly matches one-shot final weights.
            layer_masks[idx] = sorted(selected)
            previous_sets[idx] = selected
        masks[ratio] = layer_masks
        previous_keep = keep
    return masks


def validate_nested_masks(masks: dict[float, dict[int, list[int]]]) -> None:
    previous = None
    for ratio in sorted(masks):
        current = {idx: set(ids) for idx, ids in masks[ratio].items()}
        if previous is not None:
            for idx in current:
                if not current[idx].issubset(previous[idx]):
                    raise ValueError(f"Mask nesting violation at ratio={ratio}, layer={idx}")
        previous = current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance-dir", required=True)
    parser.add_argument("--ratios", nargs="+", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hardware-multiple", type=int, default=128, choices=[128, 256])
    parser.add_argument("--model", default=None, help="HF model/config path for exact parameter accounting")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--ratio-basis",
        choices=["total_parameters", "ffn_width"],
        default="total_parameters",
        help="Interpret --ratios as whole-model parameter reduction or FFN-width reduction",
    )
    args = parser.parse_args()
    importance = Path(args.importance_dir)
    order_payload = load_json(importance / "channel_order.json")
    orders = {int(idx): list(map(int, values)) for idx, values in order_payload["layers"].items()}
    sizes = {len(values) for values in orders.values()}
    if len(sizes) != 1:
        raise ValueError(f"Per-layer intermediate sizes differ: {sizes}")
    original_size = sizes.pop()
    config = AutoConfig.from_pretrained(args.model, revision=args.revision) if args.model else None
    calibration_manifest_path = importance / "calibration_manifest.json"
    calibration_manifest = load_json(calibration_manifest_path) if calibration_manifest_path.exists() else {}
    model_summary = calibration_manifest.get("model_summary", {})
    if args.ratio_basis == "total_parameters":
        if config is None:
            raise ValueError("--model is required when --ratio-basis=total_parameters")
        total_parameters = int(model_summary.get("total_parameters", 0))
        ffn_parameters = int(model_summary.get("ffn_parameters", 0))
        if not total_parameters or not ffn_parameters:
            raise ValueError(
                "calibration_manifest.json must contain exact total_parameters and ffn_parameters"
            )
        width_ratios = [
            total_parameter_target_to_ffn_ratio(
                total_parameters=total_parameters,
                ffn_parameters=ffn_parameters,
                target_total_parameter_reduction=ratio,
            )
            for ratio in args.ratios
        ]
    else:
        width_ratios = list(args.ratios)
    masks_by_width_ratio = build_nested_masks(
        orders,
        width_ratios,
        original_size=original_size,
        hardware_multiple=args.hardware_multiple,
    )
    validate_nested_masks(masks_by_width_ratio)
    masks = {
        float(target): masks_by_width_ratio[float(width_ratio)]
        for target, width_ratio in zip(args.ratios, width_ratios)
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for target_ratio, width_ratio in zip(args.ratios, width_ratios):
        layer_masks = masks[float(target_ratio)]
        keep = len(next(iter(layer_masks.values())))
        actual_ffn = 1.0 - keep / original_size
        cost = estimate_pruning_cost(config, keep) if config is not None else {}
        if config is not None and model_summary.get("total_parameters"):
            removed = (
                int(config.num_hidden_layers)
                * 3
                * int(config.hidden_size)
                * (original_size - keep)
            )
            cost["actual_total_parameter_reduction"] = removed / int(model_summary["total_parameters"])
        payload = {
            "schema_version": 1,
            "created_at": utc_now(),
            "importance_dir": str(importance),
            "importance_score": order_payload.get("primary_score"),
            # Carried through from the statistics so the mask states which method
            # ranked it. Preflight compares this against the config and refuses an
            # arm pointed at another arm's mask directory -- otherwise two arms run
            # identically and the null result reads as "the method does not help"
            # rather than "the experiment never ran". Defaults to activation because
            # masks built before this field existed were all activation masks.
            "pruning_method": order_payload.get("pruning_method", "minitron"),
            "hardware_multiple": args.hardware_multiple,
            "ratio_basis": args.ratio_basis,
            "original_intermediate_size": original_size,
            "intermediate_size": keep,
            "target_total_parameter_reduction": (
                float(target_ratio) if args.ratio_basis == "total_parameters" else None
            ),
            "required_ffn_width_pruning_ratio": float(width_ratio),
            "actual_ffn_pruning_ratio": actual_ffn,
            "retained_original_channel_ids_per_layer": {
                str(idx): ids for idx, ids in layer_masks.items()
            },
            **cost,
        }
        name = f"width_prune_{round(float(target_ratio) * 100):02d}.json"
        dump_json(payload, output / name)
        print(
            f"{name}: {original_size} -> {keep}; "
            f"FFN width -{actual_ffn:.4%}; "
            f"total params -{cost.get('actual_total_parameter_reduction', float('nan')):.4%}"
        )


if __name__ == "__main__":
    main()
