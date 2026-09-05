"""Audited compact task-state adapters for the ICLR 2027 RLBench suite.

The adapters reuse the original task model, waypoints, success conditions and
demonstration generator.  They only expose a fixed-width low-dimensional task
state suitable for object-centric learning.  A wrapper keeps the original task
name so RLBench loads the unmodified public ``.ttm`` scene.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.tasks.close_jar import CloseJar
from rlbench.tasks.open_drawer import OpenDrawer
from rlbench.tasks.place_cups import PlaceCups
from rlbench.tasks.push_buttons import PushButtons
from rlbench.tasks.remove_cups import RemoveCups
from rlbench.tasks.sweep_to_dustpan import SweepToDustpan


def _flat(*values: Any) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64) for value in values])


class ICLRCloseJar(CloseJar):
    """Expose the lid and the variation-selected target jar."""

    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot, name="close_jar")

    def init_episode(self, index: int):
        self._iclr_target_jar = index % 2
        return super().init_episode(index)

    def get_low_dim_state(self) -> np.ndarray:
        return _flat(self.lid.get_pose(), self.jars[self._iclr_target_jar].get_pose())


class ICLROpenDrawer(OpenDrawer):
    """Expose the selected drawer pose and its internal joint coordinate."""

    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot, name="open_drawer")

    def get_low_dim_state(self) -> np.ndarray:
        joint = self._joints[self._current_index]
        return _flat(joint.get_pose(), [joint.get_joint_position()])


class _FixedVariationMixin:
    """Present one native repeat-count variation as a one-variation task."""

    _scene_task_name: str
    _base_variation: int

    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot, name=self._scene_task_name)

    def init_episode(self, index: int):
        if index != 0:
            raise ValueError(f"{type(self).__name__} exposes only variation 0")
        return super().init_episode(self._base_variation)

    def variation_count(self) -> int:
        return 1


class _PlaceCupsLevel(_FixedVariationMixin, PlaceCups):
    _scene_task_name = "place_cups"
    _target_count: int

    def get_low_dim_state(self) -> np.ndarray:
        entities = self._cups[: self._target_count] + self._spokes[: self._target_count]
        return _flat(*(entity.get_pose() for entity in entities))


class ICLRPlaceCups1(_PlaceCupsLevel):
    _base_variation = 0
    _target_count = 1


class ICLRPlaceCups2(_PlaceCupsLevel):
    _base_variation = 1
    _target_count = 2


class ICLRPlaceCups3(_PlaceCupsLevel):
    _base_variation = 2
    _target_count = 3


class ICLRSweepToDustpan(SweepToDustpan):
    """Retain the broom action frame and expose sparse task-scene entities."""

    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot, name="sweep_to_dustpan")

    def get_low_dim_state(self) -> np.ndarray:
        entities = [Shape("broom"), ProximitySensor("success"), *self.dirts]
        return _flat(*(entity.get_pose() for entity in entities))


class _RemoveCupsLevel(_FixedVariationMixin, RemoveCups):
    _scene_task_name = "remove_cups"
    _target_count: int

    def get_low_dim_state(self) -> np.ndarray:
        entities = []
        for index in range(self._target_count):
            entities.extend(
                (self.cups[index], self.spokes[index], self.success_detectors[index])
            )
        return _flat(*(entity.get_pose() for entity in entities))


class ICLRRemoveCups1(_RemoveCupsLevel):
    _base_variation = 0
    _target_count = 1


class ICLRRemoveCups2(_RemoveCupsLevel):
    _base_variation = 1
    _target_count = 2


class _PushButtonsLevel(_FixedVariationMixin, PushButtons):
    _scene_task_name = "push_buttons"
    _target_count: int

    def get_low_dim_state(self) -> np.ndarray:
        poses = [
            button.get_pose() for button in self.target_buttons[: self._target_count]
        ]
        joints = [
            joint.get_joint_position()
            for joint in self.target_joints[: self._target_count]
        ]
        return _flat(*poses, joints)


class ICLRPushButtons1(_PushButtonsLevel):
    _base_variation = 0
    _target_count = 1


class ICLRPushButtons2(_PushButtonsLevel):
    _base_variation = 1
    _target_count = 2


class ICLRPushButtons3(_PushButtonsLevel):
    _base_variation = 2
    _target_count = 3


__all__ = [name for name in globals() if name.startswith("ICLR")]
