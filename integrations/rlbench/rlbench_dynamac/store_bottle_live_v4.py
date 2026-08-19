"""Live V4 StoreBottle task view over the pinned, immutable legacy TTM.

This module is imported only by the simulator process, where the pinned
RLBench/PyRep environment is available.  Keeping it in the integration tree
makes the semantic task class part of the primary repository rather than an
untracked change inside the vendored RLBench checkout.
"""

from typing import Dict

import numpy as np
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.bimanual_tasks.bimanual_put_bottle_in_fridge import (
    BimanualPutBottleInFridge,
)

LEGACY_STORE_BOTTLE_MODEL_NAME = "bimanual_put_bottle_in_fridge"
BOTTLE_MOTION_ROOT_OBJECT = "fridge_root"
FRIDGE_MOTION_ROOT_OBJECT = "fridge_base"


class BimanualPutBottleInFridgeSemanticV4(BimanualPutBottleInFridge):
    """V4 StoreBottle semantics that reuse the immutable legacy TTM."""

    def __init__(self, pyrep, robot):
        # Task.load() derives the TTM path from ``self.name``.  Pinning the
        # legacy name deliberately reuses the audited binary without copying
        # or mutating it.
        super().__init__(pyrep, robot, name=LEGACY_STORE_BOTTLE_MODEL_NAME)

    def init_task(self) -> None:
        super().init_task()
        self._bottle_motion_root = Shape(BOTTLE_MOTION_ROOT_OBJECT)
        self._fridge_motion_root = Shape(FRIDGE_MOTION_ROOT_OBJECT)

    def bottle_motion_root(self) -> Object:
        """Return the root that moves the bottle interaction subtree only."""

        return self._bottle_motion_root

    def fridge_motion_root(self) -> Object:
        """Return the root of the physical fridge and its interaction subtree."""

        return self._fridge_motion_root

    def semantic_motion_roots(self) -> Dict[str, Object]:
        """Return explicit roots; callers must not infer them from boundary_root."""

        return {
            "bottle": self.bottle_motion_root(),
            "fridge": self.fridge_motion_root(),
        }

    def get_low_dim_state(self) -> np.ndarray:
        """Return the operated bottle and the true physical-fridge reference."""

        return np.concatenate(
            [
                self.bottle.get_pose(),
                self.fridge_motion_root().get_pose(),
            ]
        )
