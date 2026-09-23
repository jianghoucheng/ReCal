from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class LayerAccumulator:
    squared_batch_mean_abs_sum: torch.Tensor
    batch_count: torch.Tensor
    token_count: torch.Tensor


class FFNImportanceAccumulator:
    """Streaming channel statistics for gated FFN intermediate activations."""

    def __init__(
        self,
        layer_sizes: dict[int, int],
        *,
        device: torch.device,
        accumulator_dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = accumulator_dtype
        self.layers = {
            idx: LayerAccumulator(
                squared_batch_mean_abs_sum=torch.zeros(
                    size, dtype=accumulator_dtype, device=device
                ),
                batch_count=torch.zeros((), dtype=torch.float64, device=device),
                token_count=torch.zeros((), dtype=torch.float64, device=device),
            )
            for idx, size in layer_sizes.items()
        }
        self.current_token_mask: torch.Tensor | None = None
        self.current_token_weights: torch.Tensor | None = None

    def set_token_mask(self, mask: torch.Tensor | None) -> None:
        """Set an optional ``[batch, sequence]`` mask for the next forwards."""
        self.current_token_mask = mask

    def set_token_weights(self, weights: torch.Tensor | None) -> None:
        """Set optional ``[batch, sequence]`` per-token weights for the next forwards.

        ``None`` (the default) reproduces the ModelOpt-Minitron statistic exactly:
        every unmasked position contributes ``1 / n_valid``. Supplying weights that
        sum to 1 within each row replaces that uniform average with a weighted one
        and changes nothing else, so a weighting scheme is the single variable
        separating a calibrated run from the baseline.

        Weights are consumed in the same normalization slot the uniform mean
        occupies rather than added as a second factor, because scaling the
        already-averaged value would leave the per-sample aggregation dependent on
        the weight magnitudes and make the two arms incomparable.
        """
        self.current_token_weights = weights

    @torch.no_grad()
    def update(self, layer_idx: int, activation: torch.Tensor) -> None:
        """Accumulate a ``[batch, sequence, intermediate]`` activation.

        NVIDIA ModelOpt's Minitron implementation receives Megatron tensors in
        ``[sequence, batch, intermediate]`` layout, computes mean absolute
        activation over sequence for every sample, then sums the squared values
        over samples and calibration batches. Hugging Face uses batch-first
        layout, so the equivalent operation is ``abs().mean(dim=1).pow(2).sum(0)``.

        Padding is excluded with ``current_token_mask``. Each sample is
        normalized by its own number of valid tokens, preserving ModelOpt's
        per-sample weighting for variable-length examples.
        """
        activation = activation.detach()
        if activation.ndim == 2:
            activation = activation.unsqueeze(0)
        if activation.ndim != 3:
            raise ValueError(
                f"Expected [batch, sequence, intermediate] activation, got {tuple(activation.shape)}"
            )
        if not torch.isfinite(activation).all():
            raise FloatingPointError(f"NaN/Inf in layer {layer_idx} FFN activation")
        activation = activation.to(self.dtype)
        batch_size, sequence_length, _ = activation.shape
        if self.current_token_mask is None:
            valid = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=activation.device,
            )
        else:
            valid = self.current_token_mask.to(
                activation.device, dtype=torch.bool
            )
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if tuple(valid.shape) != (batch_size, sequence_length):
                raise ValueError(
                    f"Token mask shape {tuple(valid.shape)} does not match "
                    f"activation batch/sequence {(batch_size, sequence_length)}"
                )
        valid_counts = valid.sum(dim=1)
        nonempty = valid_counts > 0
        if not nonempty.any():
            return
        if self.current_token_weights is None:
            sample_mean_abs = (
                activation.abs() * valid.unsqueeze(-1)
            ).sum(dim=1) / valid_counts.clamp_min(1).unsqueeze(-1)
        else:
            weights = self.current_token_weights.to(
                activation.device, dtype=self.dtype
            )
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            if tuple(weights.shape) != (batch_size, sequence_length):
                raise ValueError(
                    f"Token weight shape {tuple(weights.shape)} does not match "
                    f"activation batch/sequence {(batch_size, sequence_length)}"
                )
            if (weights < 0).any():
                raise ValueError("Token weights must be non-negative")
            # Zero out masked positions before renormalizing: a weight on a padded
            # or prompt position would otherwise consume part of the row's budget
            # and silently shrink every real position's contribution.
            weights = weights * valid
            row_sums = weights.sum(dim=1, keepdim=True)
            # A row whose weights vanish carries no usable signal about *which*
            # positions matter, so fall back to the uniform average rather than
            # dropping the sample -- dropping it would bias the calibration set
            # toward whatever property made the weights degenerate.
            uniform = valid.to(self.dtype) / valid_counts.clamp_min(1).unsqueeze(-1)
            weights = torch.where(row_sums > 0, weights / row_sums.clamp_min(1e-12), uniform)
            sample_mean_abs = (activation.abs() * weights.unsqueeze(-1)).sum(dim=1)
        sample_mean_abs = sample_mean_abs[nonempty]
        stats = self.layers[layer_idx]
        stats.squared_batch_mean_abs_sum.add_(
            sample_mean_abs.pow(2).sum(dim=0)
        )
        stats.batch_count.add_(int(nonempty.sum().item()))
        stats.token_count.add_(int(valid_counts[nonempty].sum().item()))

    @torch.no_grad()
    def all_reduce(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        for stats in self.layers.values():
            dist.all_reduce(
                stats.squared_batch_mean_abs_sum, op=dist.ReduceOp.SUM
            )
            dist.all_reduce(stats.batch_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats.token_count, op=dist.ReduceOp.SUM)

    def finalize(self) -> dict[int, dict[str, torch.Tensor | int]]:
        result = {}
        for idx, stats in self.layers.items():
            count = int(stats.token_count.item())
            if count <= 0:
                raise RuntimeError(f"Layer {idx} has no effective response tokens")
            batch_count = int(stats.batch_count.item())
            modelopt_minitron = torch.sqrt(stats.squared_batch_mean_abs_sum)
            if not torch.isfinite(modelopt_minitron).all():
                raise FloatingPointError(f"Non-finite finalized score in layer {idx}")
            result[idx] = {
                "modelopt_minitron": modelopt_minitron.detach().cpu(),
                "token_count": count,
                "batch_count": batch_count,
            }
        return result



def register_down_projection_hooks(
    model: torch.nn.Module,
    accumulator: FFNImportanceAccumulator,
):
    """Register pre-hooks on ``down_proj`` so the observed tensor is exactly
    SiLU(gate_proj(x)) * up_proj(x), without recomputing either projection.
    """
    handles = []
    discovered: dict[int, str] = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if not name.endswith("down_proj") or not isinstance(module, torch.nn.Linear):
            continue
        idx = layer_idx
        layer_idx += 1
        discovered[idx] = name

        def hook(_module, inputs, output, *, _idx=idx):
            accumulator.update(_idx, inputs[0])

        # Match ModelOpt's linear_fc2 forward hook (rather than a pre-hook).
        handles.append(module.register_forward_hook(hook))
    if not handles:
        raise RuntimeError("No FFN down_proj modules were found")
    if set(discovered) != set(accumulator.layers):
        raise RuntimeError(
            f"Hook/model layer mismatch: discovered={list(discovered)}, expected={list(accumulator.layers)}"
        )
    return handles, discovered
