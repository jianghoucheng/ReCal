"""Zero-valued reward callback for pure OPD.

Official verl's rollout data path still materializes ``rm_scores`` even when
``use_task_rewards=False``. Returning zero here avoids invoking any task
grader/environment and makes the no-outcome-reward contract explicit.
"""


def zero_reward(*args, **kwargs) -> float:
    return 0.0

