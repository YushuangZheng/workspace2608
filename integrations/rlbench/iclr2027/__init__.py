"""ICLR 2027 task and evaluation integration for the pinned RLBench fork.

The package is deliberately separate from the reusable closed-loop algorithm
and from the archived phase-six development protocol.
"""

from integrations.rlbench.iclr2027.task_registry import (
    ExperimentTask,
    experiment_task,
    experiment_task_set,
    load_experiment_registry,
)

__all__ = [
    "ExperimentTask",
    "experiment_task",
    "experiment_task_set",
    "load_experiment_registry",
]
