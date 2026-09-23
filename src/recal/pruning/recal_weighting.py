"""Token-level forward KL between the dense teacher and a pruned probe.

Standard activation pruning weights every calibration position equally, which
implicitly assumes each reasoning state is equally worth spending FFN capacity on.
This measures a different quantity: for each teacher state, *how much ordinary
pruning already broke the teacher's next-token distribution there*. States that a
provisional pruned model still predicts correctly need no protection; states it
destroys are where the remaining capacity should go.

The probe exists only to be measured. It is a throwaway model pruned by the
existing activation criterion at the target ratio, and its logits define the
mismatch signal -- it is never trained and never evaluated.

**Why the support is teacher top-k plus one bucket.** Full-vocabulary KL over
151k logits would be dominated by the tail, where both models are near zero and
differences are numerically meaningless. OPD already supervises against the
teacher's top-k, so scoring the same support keeps calibration and recovery
measuring the same object. Truncating to top-k alone would compare two
un-normalized vectors, so the residual mass ``1 - Σ top-k`` becomes an explicit
OTHER bucket: the resulting 17-dimensional distributions are proper, and mass the
probe leaks outside the teacher's top-k is charged rather than ignored.

**Why teacher prefixes only.** Scoring student rollouts would let a degenerate
probe choose its own easy states -- a repetition loop produces high agreement and
would read as "pruning did no damage" exactly where damage is worst. Teacher
forcing on the dense model's own trajectories keeps the state distribution fixed
across arms and independent of probe quality.
"""

from __future__ import annotations

import torch


# Matches opd.top_k: calibration and recovery score the same support, so a state
# that matters for the distillation objective is also a state that shapes the mask.
DEFAULT_TOP_K = 16

# Below this the trajectory's weights carry no information about which positions
# matter, and normalizing would amplify float noise into a mask decision.
DEGENERATE_KL_SUM = 1e-9


@torch.no_grad()
def token_forward_kl(
    teacher_logits: torch.Tensor,
    probe_logits: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> torch.Tensor:
    """Per-position ``KL(p_teacher || p_probe)`` over the teacher's top-k plus OTHER.

    Both tensors are ``[..., vocab]`` logits for the *same* prefixes. Returns
    ``[...]`` non-negative divergences.

    fp32 throughout: the models run in bf16, whose ~3 decimal digits cannot
    represent the small probability differences this signal consists of, and
    log-space subtraction of two bf16 values loses the result entirely.
    """
    if teacher_logits.shape != probe_logits.shape:
        raise ValueError(
            f"Teacher logits {tuple(teacher_logits.shape)} and probe logits "
            f"{tuple(probe_logits.shape)} must have the same shape"
        )
    vocab = teacher_logits.shape[-1]
    if not 0 < top_k <= vocab:
        raise ValueError(f"top_k must be in (0, {vocab}], got {top_k}")

    teacher = torch.softmax(teacher_logits.to(torch.float32), dim=-1)
    probe = torch.softmax(probe_logits.to(torch.float32), dim=-1)

    # The support is chosen by the teacher and applied to both, so the two
    # distributions are compared on identical coordinates.
    top_probs, top_idx = torch.topk(teacher, top_k, dim=-1)
    probe_top = probe.gather(-1, top_idx)

    # Residual mass as an explicit coordinate. Clamped at zero because softmax
    # summation error can make the complement marginally negative.
    teacher_other = (1.0 - top_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    probe_other = (1.0 - probe_top.sum(dim=-1, keepdim=True)).clamp_min(0.0)

    p = torch.cat([top_probs, teacher_other], dim=-1)
    q = torch.cat([probe_top, probe_other], dim=-1)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # p log(p/q) with the p→0 terms dropped, which is the correct limit (0 log 0 = 0)
    # and also avoids -inf * 0 = nan. q is floored, not dropped: a coordinate the
    # probe assigns zero mass while the teacher does not is precisely the damage
    # being measured, and must stay finite to be counted.
    ratio = torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12))
    divergence = (p * ratio).where(p > 0, torch.zeros_like(p)).sum(dim=-1)
    # KL is non-negative in exact arithmetic; clamp the float residue rather than
    # letting a -1e-8 propagate into a weight.
    return divergence.clamp_min(0.0)


@torch.no_grad()
def trajectory_token_weights(
    divergences: torch.Tensor,
    valid: torch.Tensor,
    *,
    degenerate_sum: float = DEGENERATE_KL_SUM,
) -> torch.Tensor:
    """Normalize per-position divergences into per-trajectory weights.

    ``divergences`` and ``valid`` are ``[batch, sequence]``. Returns weights that
    sum to 1 over each row's valid positions.

    Normalizing within a trajectory before averaging across trajectories is what
    keeps every calibration sample contributing equally. Normalizing globally
    instead would let a single high-divergence trajectory dominate the mask, which
    would make the statistic a function of which samples happen to be hardest
    rather than of which channels are worth keeping.
    """
    if divergences.shape != valid.shape:
        raise ValueError(
            f"Divergence shape {tuple(divergences.shape)} does not match "
            f"valid-mask shape {tuple(valid.shape)}"
        )
    weights = divergences.to(torch.float32).clamp_min(0.0) * valid
    row_sums = weights.sum(dim=1, keepdim=True)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    uniform = valid.to(torch.float32) / counts
    return torch.where(row_sums > degenerate_sum, weights / row_sums.clamp_min(1e-12), uniform)
