"""Standard fixed-width task-frame view of the upstream RLBench WipeDesk.

The upstream task already creates dirt inside ``dirt_boundary``, executes its
authored wipe as one continuous ``CartesianPath``, and uses removal of the dirt
shapes as the success condition.  Its public low-dimensional state exposes only
the sponge pose.  This adapter adds the existing physical ``dirt_boundary``
pose in the same fixed-width format used by the other DynaMAC tasks.  It does
not alter the expert path, waypoints, success condition, controller, or runtime
task behavior.
"""

from __future__ import annotations

import numpy as np
from pyrep.objects.shape import Shape
from rlbench.tasks.wipe_desk import WipeDesk

LEGACY_WIPE_DESK_MODEL_NAME = "wipe_desk"


class WipeDeskTaskFrames(WipeDesk):
    """Expose the two existing physical WipeDesk task frames."""

    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot, name=LEGACY_WIPE_DESK_MODEL_NAME)

    def init_task(self) -> None:
        super().init_task()
        self._dirt_boundary = Shape("dirt_boundary")

    def get_low_dim_state(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.sponge.get_pose(), dtype=np.float64),
                np.asarray(self._dirt_boundary.get_pose(), dtype=np.float64),
            ]
        )
