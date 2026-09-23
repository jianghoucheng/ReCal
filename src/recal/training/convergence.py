"""Convergence detection for OPD recovery training.

The number of updates a pruned student needs before it stops tracking the
teacher any better is not known in advance, so a fixed step budget either wastes
compute or stops early. verl already logs ``val-topk/overlap_ratio`` -- the
fraction of the student's top-k tokens that fall inside the teacher's top-k --
on every optimizer update, which is a direct measure of student/teacher
distribution alignment and costs nothing extra to read.

This module turns that per-step scalar into a stop decision. The raw metric is
noisy between updates, so it is smoothed with an EMA and progress is measured
between fixed-length windows of updates rather than between consecutive steps.
Improvement is judged *relatively*: the metric already starts around 0.84 on a
10%-pruned Qwen3-4B, so the useful headroom is small and an absolute delta
threshold would be scale-dependent across cascade stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# verl writes console metrics as "step:12 - key:value - key:value - ...".
STEP_PATTERN = re.compile(r"\bstep:(\d+)\b")
OVERLAP_PATTERN = re.compile(r"\bval-topk/overlap_ratio:([0-9.eE+-]+)")


def parse_metric_line(line: str) -> tuple[int, float] | None:
    """Extract ``(step, overlap_ratio)`` from a verl console line."""
    step_match = STEP_PATTERN.search(line)
    overlap_match = OVERLAP_PATTERN.search(line)
    if not step_match or not overlap_match:
        return None
    try:
        return int(step_match.group(1)), float(overlap_match.group(1))
    except ValueError:
        return None


@dataclass
class ConvergenceMonitor:
    """Stop when smoothed teacher/student overlap stops improving.

    Args:
        min_steps: Never converge before this many updates; lets the EMA settle
            and gives the student a floor of recovery training.
        window: Number of updates per comparison window.
        patience: Consecutive windows that must show no material improvement.
        min_delta: Minimum *relative* EMA gain over a window that still counts
            as progress (0.002 == 0.2%).
        ema_beta: EMA smoothing factor for the raw per-step metric.
    """

    min_steps: int = 100
    window: int = 20
    patience: int = 3
    min_delta: float = 0.002
    ema_beta: float = 0.9

    ema: float | None = None
    last_step: int = 0
    best_window_ema: float | None = None
    stagnant_windows: int = 0
    converged_at: int | None = None
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def update(self, step: int, value: float) -> bool:
        """Record one update's metric. Returns True once converged."""
        self.ema = value if self.ema is None else self.ema_beta * self.ema + (1.0 - self.ema_beta) * value
        self.last_step = step
        self.history.append((step, value, self.ema))
        if self.converged_at is not None:
            return True
        if step < self.min_steps or step % self.window != 0:
            return False

        # End of a window: compare its smoothed level against the best so far.
        if self.best_window_ema is None:
            self.best_window_ema = self.ema
            return False
        gain = (self.ema - self.best_window_ema) / max(abs(self.best_window_ema), 1e-8)
        if gain > self.min_delta:
            self.best_window_ema = self.ema
            self.stagnant_windows = 0
            return False
        self.stagnant_windows += 1
        if self.stagnant_windows >= self.patience:
            self.converged_at = step
            return True
        return False

    @property
    def converged(self) -> bool:
        return self.converged_at is not None

    def report(self, *, stop_reason: str, final_step: int) -> dict[str, object]:
        """Summarize the run for the stage record."""
        return {
            "stop_reason": stop_reason,
            "final_step": final_step,
            "converged_at": self.converged_at,
            "final_overlap_ratio": self.history[-1][1] if self.history else None,
            "final_overlap_ratio_ema": self.ema,
            "best_window_ema": self.best_window_ema,
            "stagnant_windows": self.stagnant_windows,
            "criterion": {
                "metric": "val-topk/overlap_ratio",
                "min_steps": self.min_steps,
                "window": self.window,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "ema_beta": self.ema_beta,
            },
            "history": [
                {"step": step, "overlap_ratio": value, "ema": ema}
                for step, value, ema in self.history
            ],
        }
