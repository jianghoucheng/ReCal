"""Two width-pruning criteria scored at the shared FFN hook site.

FLAP and Wanda-sp both rank FFN intermediate channels from the tensor entering
``down_proj``. They and Minitron can therefore be collected in a single forward
pass over one calibration set, keeping the calibration data, hook site, and
width rounding identical across criteria.

**Wanda-sp** scores a weight by
``|W_ij| * ||X_j||_2``. Aggregating over the output dimension to make it structured
gives, for input channel ``j`` of ``down_proj``:

    S_j = ||W_:,j||_2 * ||X_j||_2

i.e. the weight column's norm times the RMS activation of that channel.

**FLAP** uses the ``WIFV`` metric to measure how much a
channel *fluctuates* around its own mean, weighted by the weight column norm:

    S_j = Var(X_j) * ||W_:,j||_2^2

The motivation is directly relevant to this project: a channel whose value barely varies
can be replaced by its mean and folded into a bias, so variance -- not magnitude -- is
what makes a channel irreplaceable. FLAP's ``WIFN`` variant substitutes the mean absolute
value for the variance and is also computed here, since it is the version the official
repository defaults to for MLP layers.

## What is deliberately *not* reproduced

FLAP ships two further components: adaptive per-layer sparsity (its ``AL-AM`` mode) and
baseline-bias compensation, which folds the pruned channels' mean contribution into the
surviving bias term. Both are omitted, and the omission is the point of the comparison:

  * **Uniform width.** Every arm in this study prunes to the same per-layer width so the
    pruned models have identical shapes and parameter counts. Letting one arm choose its
    own architecture would make its numbers incomparable to the rest of the table.
  * **No bias compensation.** It is a retraining-free trick, and every arm here is
    followed by SFT and OPD. Adding it to one arm only would confound "better channel
    ranking" with "better zero-shot starting point".

Both omissions are recorded in the manifest so the executed variant is explicit.

## Numerical notes

Variance is accumulated with the shifted-data (sum, sum-of-squares) form rather than
Welford's, because the per-channel counts here are large and identical across ranks, and
the sums reduce with a single ``all_reduce`` each. It is computed in float32 regardless
of model dtype: bf16 sum-of-squares over ~10^6 tokens loses enough precision to reorder
adjacent channels, which is exactly the quantity being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

# Criteria this module produces. Every name here is a valid `primary_score`, so an arm
# selects one by config alone and the mask builder needs no changes.
SCORES = ("wanda_sp", "flap_wifv", "flap_wifn")


@dataclass
class LayerBaselineStats:
    """Streaming per-channel sums for one FFN layer."""

    abs_sum: torch.Tensor
    square_sum: torch.Tensor
    value_sum: torch.Tensor
    token_count: torch.Tensor


class BaselineImportanceAccumulator:
    """Collect Wanda-sp and FLAP statistics for gated-FFN intermediate channels.

    Shares the ``update(layer_idx, activation)`` contract with
    ``FFNImportanceAccumulator`` so ``register_down_projection_hooks`` drives either one
    unchanged, and so both can observe the same forward pass.

    Unlike the Minitron statistic, these criteria are *token-level*: they estimate a
    per-channel norm or variance over the whole calibration set rather than averaging
    within each sample first. So padded positions are excluded, but no per-sample
    normalization is applied -- imposing one would change the estimator the published
    formulas define.
    """

    def __init__(
        self,
        layer_sizes: dict[int, int],
        *,
        device: torch.device,
        accumulator_dtype: torch.dtype = torch.float32,
        allow_token_weights: bool = False,
    ):
        self.device = device
        self.dtype = accumulator_dtype
        self.allow_token_weights = allow_token_weights
        self.layers = {
            idx: LayerBaselineStats(
                abs_sum=torch.zeros(size, dtype=accumulator_dtype, device=device),
                square_sum=torch.zeros(size, dtype=accumulator_dtype, device=device),
                value_sum=torch.zeros(size, dtype=accumulator_dtype, device=device),
                token_count=torch.zeros((), dtype=torch.float64, device=device),
            )
            for idx, size in layer_sizes.items()
        }
        self.current_token_mask: torch.Tensor | None = None
        self.current_token_weights: torch.Tensor | None = None

    def set_token_mask(self, mask: torch.Tensor | None) -> None:
        self.current_token_mask = mask

    def set_token_weights(self, weights: torch.Tensor | None) -> None:
        """Set optional ``[batch, sequence]`` per-token weights for the next forwards.

        Refused unless the accumulator was constructed with ``allow_token_weights=True``.
        The published formulas are unweighted by definition, and silently weighting one
        would report a variant of our method under a baseline's name -- so opting in is
        explicit and per-arm, and the criterion name the mask records still says which
        formula ran while ``pruning.method`` says whether it was weighted.

        Weights land in the same normalization slot the token count occupies rather than
        multiplying the finished score, matching ``FFNImportanceAccumulator``: a channel's
        statistic becomes an expectation under the weight distribution instead of under
        the uniform one. Scaling the averaged value instead would make the magnitude of
        the weights leak into the score and the arms incomparable.
        """
        if weights is not None and not self.allow_token_weights:
            raise ValueError(
                "Wanda-sp and FLAP are unweighted by definition; refusing to apply "
                "token weights, which would turn the baseline into a variant of ours. "
                "Construct with allow_token_weights=True for a deliberately weighted arm"
            )
        self.current_token_weights = weights

    @torch.no_grad()
    def update(self, layer_idx: int, activation: torch.Tensor) -> None:
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
                (batch_size, sequence_length), dtype=torch.bool, device=activation.device
            )
        else:
            valid = self.current_token_mask.to(activation.device, dtype=torch.bool)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if tuple(valid.shape) != (batch_size, sequence_length):
                raise ValueError(
                    f"Token mask shape {tuple(valid.shape)} does not match "
                    f"activation batch/sequence {(batch_size, sequence_length)}"
                )
        count = int(valid.sum().item())
        if count == 0:
            return

        stats = self.layers[layer_idx]
        if self.current_token_weights is None:
            keep = valid.unsqueeze(-1)
            masked = activation * keep
            stats.abs_sum.add_(masked.abs().sum(dim=(0, 1)))
            stats.square_sum.add_(masked.pow(2).sum(dim=(0, 1)))
            stats.value_sum.add_(masked.sum(dim=(0, 1)))
            stats.token_count.add_(float(count))
            return

        # Weighted variant. `finalize` divides every sum by `token_count`, so scaling the
        # per-token contributions and adding the same weight total to the count turns each
        # score into an expectation under the weight distribution -- E_w[|x|], E_w[x^2],
        # E_w[x] -- with the published formulas downstream untouched. Weight one token per
        # row and the result is that token's value, exactly as an unweighted single-token
        # batch would give, which is the invariant the tests pin.
        weights = self.current_token_weights.to(activation.device, dtype=self.dtype)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if tuple(weights.shape) != (batch_size, sequence_length):
            raise ValueError(
                f"Token weight shape {tuple(weights.shape)} does not match "
                f"activation batch/sequence {(batch_size, sequence_length)}"
            )
        if not torch.isfinite(weights).all():
            raise FloatingPointError(f"NaN/Inf in layer {layer_idx} token weights")
        if (weights < 0).any():
            raise ValueError(
                f"Layer {layer_idx}: negative token weight; a KL divergence cannot be "
                "negative, so this indicates a corrupted weight tensor"
            )
        weights = weights * valid.to(self.dtype)
        weight_total = float(weights.sum().item())
        if weight_total <= 0.0:
            # Every weight in this batch fell to zero (probe matched the teacher on all
            # scored positions). Contributing nothing is right: a uniform fallback here
            # would quietly mix unweighted batches into a weighted estimate.
            return
        scaled = activation * weights.unsqueeze(-1)
        stats.abs_sum.add_(scaled.abs().sum(dim=(0, 1)))
        stats.square_sum.add_((activation.pow(2) * weights.unsqueeze(-1)).sum(dim=(0, 1)))
        stats.value_sum.add_(scaled.sum(dim=(0, 1)))
        stats.token_count.add_(weight_total)

    @torch.no_grad()
    def all_reduce(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        for stats in self.layers.values():
            dist.all_reduce(stats.abs_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats.square_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats.value_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats.token_count, op=dist.ReduceOp.SUM)

    def finalize(
        self, weight_column_norms: dict[int, torch.Tensor]
    ) -> dict[int, dict[str, Any]]:
        """Turn the accumulated sums into the three published scores.

        ``weight_column_norms[idx]`` is ``||W_:,j||_2`` over ``down_proj``'s output
        dimension -- the weight side of both criteria. It is read from the model rather
        than accumulated, so it must be supplied by the caller.
        """
        result: dict[int, dict[str, Any]] = {}
        for idx, stats in self.layers.items():
            count = float(stats.token_count.item())
            if count <= 0:
                raise RuntimeError(f"Layer {idx} has no effective response tokens")
            norms = weight_column_norms[idx].to(self.abs_dtype(stats))
            if norms.shape != stats.abs_sum.shape:
                raise ValueError(
                    f"Layer {idx}: weight-norm shape {tuple(norms.shape)} does not match "
                    f"channel count {tuple(stats.abs_sum.shape)}"
                )

            mean = stats.value_sum / count
            mean_square = stats.square_sum / count
            # Var = E[x^2] - E[x]^2, clamped because catastrophic cancellation can push
            # a near-constant channel a hair below zero and sqrt/ranking would then
            # depend on floating-point noise.
            variance = (mean_square - mean.pow(2)).clamp_min(0.0)
            rms = mean_square.sqrt()
            mean_abs = stats.abs_sum / count

            scores = {
                # Wanda-sp: ||W_:,j||_2 * ||X_j||_2, with the activation norm expressed
                # as an RMS so the value does not grow with calibration-set size.
                "wanda_sp": norms * rms,
                # FLAP WIFV: Var(X_j) * ||W_:,j||_2^2.
                "flap_wifv": variance * norms.pow(2),
                # FLAP WIFN: the repository's MLP default, mean|X_j| * ||W_:,j||_2.
                "flap_wifn": mean_abs * norms,
            }
            for name, value in scores.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite {name} score in layer {idx}")

            result[idx] = {
                **{name: value.detach().cpu() for name, value in scores.items()},
                "activation_variance": variance.detach().cpu(),
                "activation_rms": rms.detach().cpu(),
                "activation_mean_abs": mean_abs.detach().cpu(),
                "weight_column_norm": norms.detach().cpu(),
                "token_count": int(count),
            }
        return result

    @staticmethod
    def abs_dtype(stats: LayerBaselineStats) -> torch.dtype:
        return stats.abs_sum.dtype


@torch.no_grad()
def down_projection_column_norms(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    """``||W_:,j||_2`` per FFN layer, indexed the way the hooks index layers.

    ``down_proj.weight`` is ``[hidden, intermediate]``, so the norm over dim 0 gives one
    value per intermediate channel -- the same axis the activation statistics index.
    Computed in float32: a bf16 norm over thousands of elements is not precise enough to
    order channels whose scores differ in the third decimal.
    """
    norms: dict[int, torch.Tensor] = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if not name.endswith("down_proj") or not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight.detach().to(torch.float32)
        norms[layer_idx] = weight.pow(2).sum(dim=0).sqrt()
        layer_idx += 1
    if not norms:
        raise RuntimeError("No FFN down_proj modules were found")
    return norms
