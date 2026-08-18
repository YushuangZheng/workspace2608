from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from integrations.rlbench.rlbench_dynamac import unimanual_evaluate
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
    _prepare_low_dim_headless_scene,
)


def test_low_dim_scene_marks_all_base_sensors_before_normal_launch(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "task_design.ttt"
    source.write_bytes(b"source-scene")
    environment_module = SimpleNamespace(
        DIR_PATH=str(tmp_path),
        TTT_FILE=source.name,
    )
    events = []
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(
        unimanual_evaluate,
        "EXPECTED_UNIMANUAL_BASE_SCENE_SHA256",
        hashlib.sha256(b"source-scene").hexdigest(),
    )
    monkeypatch.setattr(
        unimanual_evaluate,
        "EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT",
        2,
    )

    class FakeSensor:
        def __init__(self, name):
            self.name = name
            self.explicit = 0

        def get_name(self):
            return self.name

        def set_explicit_handling(self, value):
            events.append(("explicit", self.name, value))
            self.explicit = value

        def get_explicit_handling(self):
            return self.explicit

    sensors = [FakeSensor("camera_b"), FakeSensor("camera_a")]

    class FakePyRep:
        def launch(self, scene, headless):
            events.append(("launch", scene, headless))

        def get_objects_in_tree(self, *, object_type):
            events.append(("enumerate", object_type))
            return sensors

        def export_scene(self, target):
            events.append(("export",))
            Path(target).write_bytes(b"derived-scene")

        def shutdown(self):
            events.append(("shutdown",))

    fake_sim = SimpleNamespace(
        simLoadScene=lambda path: events.append(("load", path))
    )
    pyrep_module = ModuleType("pyrep")
    pyrep_module.PyRep = FakePyRep
    backend_module = ModuleType("pyrep.backend")
    backend_module.sim = fake_sim
    const_module = ModuleType("pyrep.const")
    const_module.ObjectType = SimpleNamespace(VISION_SENSOR="vision-sensor")
    monkeypatch.setitem(sys.modules, "pyrep", pyrep_module)
    monkeypatch.setitem(sys.modules, "pyrep.backend", backend_module)
    monkeypatch.setitem(sys.modules, "pyrep.const", const_module)

    restore, metadata = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=True,
    )
    derived = Path(environment_module.TTT_FILE)

    assert events == [
        ("launch", "", True),
        ("load", str(source.resolve())),
        ("enumerate", "vision-sensor"),
        ("explicit", "camera_b", 1),
        ("explicit", "camera_a", 1),
        ("export",),
        ("shutdown",),
    ]
    assert metadata["protocol_id"] == LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID
    assert metadata["applied"] is True
    assert metadata["populated_scene_steps_before_patch"] == 0
    assert metadata["vision_sensors"] == ["camera_a", "camera_b"]
    assert metadata["vision_sensor_count"] == 2
    assert metadata["vision_sensor_handling"] == [
        {"name": "camera_a", "before": 0, "after": 1},
        {"name": "camera_b", "before": 0, "after": 1},
    ]
    assert metadata["qt_qpa_platform"] == "offscreen"
    assert metadata["qt_qpa_platform_defaulted"] is True
    assert derived.exists()
    assert derived != source

    restore()
    restore()
    assert environment_module.TTT_FILE == source.name
    assert not derived.exists()


def test_low_dim_scene_is_not_rewritten_for_windowed_evaluation(tmp_path) -> None:
    source = tmp_path / "task_design.ttt"
    source.write_bytes(b"source-scene")
    environment_module = SimpleNamespace(
        DIR_PATH=str(tmp_path),
        TTT_FILE=source.name,
    )

    restore, metadata = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=False,
    )

    assert metadata["protocol_id"] == LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID
    assert metadata["applied"] is False
    assert environment_module.TTT_FILE == source.name
    restore()
