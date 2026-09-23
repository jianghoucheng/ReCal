"""LLM-Pruner Taylor channel importance adapted to FFN-width pruning.

The criterion scores a *group* of weights by how much removing it perturbs the
training loss, estimated by a Taylor expansion around the current weights.
Unlike activation-based criteria, it requires a backward pass.

## The group, and why it is these three matrices

Removing intermediate channel ``j`` of a gated FFN deletes exactly:

    gate_proj.weight[j, :]     one row
    up_proj.weight[j, :]       one row
    down_proj.weight[:, j]     one column

Nothing else changes shape, so the channel's importance is the summed contribution of
those weights. LLM-Pruner derives the same grouping from its dependency graph; here it is
written out directly because the FFN structure is fixed and known.

## The two estimators

With loss ``L`` and weight ``w``, expanding ``L(w=0)`` around ``w``:

    first order   |dL/dw * w|                 -- "param_first" in the released code
    second order  0.5 * (dL/dw)^2 * w^2       -- diagonal-Fisher stand-in for the Hessian

Both are accumulated. The first-order form is the default; the second-order
variant is retained as an optional alternative.

## Integration choices

**Shared recovery.** Every arm is followed by the same SFT + OPD budget so the
comparison isolates the pruning criterion.

**Uniform per-layer width.** LLM-Pruner prunes a contiguous block of layers and can vary
width; here every arm prunes to the same per-layer width so parameter counts match
exactly. The alternative -- letting one arm pick its own architecture -- would make its
row incomparable to the rest.

**Loss is next-token prediction on the calibration text.** The calibration pool holds
prompts without reference responses, so the gradient signal is the model's own LM loss on
those prompts. That is also what LLM-Pruner does (it backprops the LM loss over raw
calibration text), so no reference answers are needed.

## Memory

A backward pass retains activations, so the sequence length that the forward-only
collectors run at (32,768) will not fit. The gradient pass therefore has its own,
shorter, ``gradient.sequence_length``, and gradients are read and released per batch
rather than accumulated in autograd. Only the three FFN projections require grad;
everything else is frozen, which is both faster and the reason peak memory stays close to
the forward-only run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

SCORES = ("llm_pruner_taylor", "llm_pruner_taylor_second")


@dataclass
class LayerTaylorStats:
    """Summed per-channel Taylor contributions for one FFN layer."""

    first_order: torch.Tensor
    second_order: torch.Tensor
    batch_count: torch.Tensor


class TaylorImportanceAccumulator:
    """Accumulate LLM-Pruner group importance over calibration batches.

    Driven differently from the forward-hook accumulators: the caller runs a forward and
    backward, then calls :meth:`accumulate_from_gradients`, which reads ``.grad`` off the
    FFN projections. There is no hook, because the quantity needed is a gradient rather
    than an activation.
    """

    def __init__(
        self,
        layer_modules: dict[int, dict[str, torch.nn.Linear]],
        *,
        device: torch.device,
        accumulator_dtype: torch.dtype = torch.float32,
    ):
        self.layer_modules = layer_modules
        self.dtype = accumulator_dtype
        self.device = device
        self.layers = {
            idx: LayerTaylorStats(
                first_order=torch.zeros(
                    modules["down_proj"].in_features, dtype=accumulator_dtype, device=device
                ),
                second_order=torch.zeros(
                    modules["down_proj"].in_features, dtype=accumulator_dtype, device=device
                ),
                batch_count=torch.zeros((), dtype=torch.float64, device=device),
            )
            for idx, modules in layer_modules.items()
        }

    @torch.no_grad()
    def accumulate_from_gradients(self, batch_size: int = 1) -> None:
        """Fold the current ``.grad`` values into the running per-channel sums.

        Must be called after ``loss.backward()`` and before ``zero_grad()``. A missing
        gradient is an error rather than a skip: it means the backward pass did not reach
        this layer, and silently scoring it as zero would rank the whole layer last.
        """
        for idx, modules in self.layer_modules.items():
            stats = self.layers[idx]
            first = torch.zeros_like(stats.first_order)
            second = torch.zeros_like(stats.second_order)

            for name, module in modules.items():
                if module.weight.grad is None:
                    raise RuntimeError(
                        f"Layer {idx} {name}.weight has no gradient; the backward pass "
                        "did not reach it, and treating that as zero importance would "
                        "silently rank the entire layer last"
                    )
                weight = module.weight.detach().to(self.dtype)
                grad = module.weight.grad.detach().to(self.dtype)
                product = weight * grad

                # Reduce onto the intermediate-channel axis. gate_proj and up_proj are
                # [intermediate, hidden] so the channel is dim 0; down_proj is
                # [hidden, intermediate] so it is dim 1. Getting this backwards would
                # produce a hidden-sized vector, which is why the shape is asserted.
                axis = 1 if name == "down_proj" else 0
                reduce_dims = 0 if axis == 1 else 1
                first += product.abs().sum(dim=reduce_dims)
                second += 0.5 * product.pow(2).sum(dim=reduce_dims)

            if first.shape != stats.first_order.shape:
                raise ValueError(
                    f"Layer {idx}: reduced Taylor score has shape {tuple(first.shape)}, "
                    f"expected {tuple(stats.first_order.shape)}"
                )
            stats.first_order.add_(first)
            stats.second_order.add_(second)
            stats.batch_count.add_(float(batch_size))

    @torch.no_grad()
    def all_reduce(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        for stats in self.layers.values():
            dist.all_reduce(stats.first_order, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats.second_order, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats.batch_count, op=dist.ReduceOp.SUM)

    def finalize(self) -> dict[int, dict[str, object]]:
        result: dict[int, dict[str, object]] = {}
        for idx, stats in self.layers.items():
            count = float(stats.batch_count.item())
            if count <= 0:
                raise RuntimeError(f"Layer {idx} accumulated no calibration batches")
            scores = {
                # Averaged over batches so the magnitude does not depend on calibration
                # size; ranking is unaffected but cross-run comparison needs it.
                "llm_pruner_taylor": stats.first_order / count,
                "llm_pruner_taylor_second": stats.second_order / count,
            }
            for name, value in scores.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite {name} score in layer {idx}")
            result[idx] = {
                **{name: value.detach().cpu() for name, value in scores.items()},
                "batch_count": int(count),
            }
        return result


def discover_ffn_groups(model: torch.nn.Module) -> dict[int, dict[str, torch.nn.Linear]]:
    """Map layer index -> the three projections deleted together with one FFN channel.

    Indexed by ``down_proj`` encounter order, matching
    ``register_down_projection_hooks``, so a Taylor ranking and an activation ranking
    refer to the same layer numbering.
    """
    groups: dict[int, dict[str, torch.nn.Linear]] = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if not name.endswith("down_proj") or not isinstance(module, torch.nn.Linear):
            continue
        parent_name = name.rsplit(".", 1)[0]
        parent = model.get_submodule(parent_name)
        try:
            gate = parent.get_submodule("gate_proj")
            up = parent.get_submodule("up_proj")
        except AttributeError as error:
            raise RuntimeError(
                f"{parent_name} has a down_proj but not gate_proj/up_proj; this "
                "criterion assumes a gated FFN"
            ) from error
        if gate.out_features != module.in_features or up.out_features != module.in_features:
            raise RuntimeError(
                f"{parent_name}: gate/up output ({gate.out_features}/{up.out_features}) "
                f"does not match down_proj input ({module.in_features}); the three "
                "matrices would not be deleted along a common axis"
            )
        groups[layer_idx] = {"gate_proj": gate, "up_proj": up, "down_proj": module}
        layer_idx += 1
    if not groups:
        raise RuntimeError("No gated FFN blocks were found")
    return groups


def freeze_all_but_ffn(model: torch.nn.Module, groups: dict[int, dict[str, torch.nn.Linear]]) -> None:
    """Require grad on the FFN projections only.

    The Taylor score reads gradients of just these matrices, so allocating gradient
    buffers for embeddings and attention would multiply peak memory for values that are
    then discarded. Backward still traverses the whole graph -- it has to, to reach the
    early layers -- but only these leaves retain a ``.grad``.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for modules in groups.values():
        for module in modules.values():
            module.weight.requires_grad_(True)
