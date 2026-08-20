#!/usr/bin/env python3
"""Dependency-light regression tests for the RACER isolation boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from compare_observation_fingerprints import first_differences
from isolated_simulator_proxy import SpawnIsolatedRLBenchSim
from observation_fingerprint import ObservationFingerprintError, snapshot


SCRIPT_DIR = Path(__file__).resolve().parent


class IsolationProtocolTest(unittest.TestCase):
    def test_round_trip_action_mutation_and_natural_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "RACER_ISOLATION_RUNTIME_DIR": temporary,
                "RACER_ISOLATION_WORKER_SELF_TEST": "1",
                "RACER_ISOLATION_RPC_TIMEOUT": "10",
            }
            with patch.dict(os.environ, environment, clear=False):
                proxy = SpawnIsolatedRLBenchSim("place_cups", "/dataset")
                obs_dict, observation = proxy.reset(0)
                self.assertEqual(obs_dict["task"], "place_cups")
                self.assertEqual(observation["raw"], obs_dict)
                self.assertEqual(proxy.task_goal, "goal:place_cups")
                action = np.array([2, 3, 4], dtype=np.int64)
                transition = proxy.step(action)
                np.testing.assert_array_equal(action, np.array([3, 3, 4]))
                np.testing.assert_array_equal(
                    transition["action"], np.array([3, 3, 4])
                )
                self.assertTrue(proxy.is_success())
                proxy.set_new_task("close_jar")
                proxy.set_new_dataset("/new-dataset")
                self.assertEqual(proxy.task_goal, "goal:close_jar")
                self.assertEqual(
                    proxy.get_video_frames(64, False),
                    [{"res": 64, "return_pil": False}],
                )
                proxy.close()
                self.assertEqual(proxy.worker_returncode, 0)
                self.assertFalse(proxy.socket_path.exists())


class ObservationFingerprintTest(unittest.TestCase):
    def test_equal_snapshot_has_no_differences(self) -> None:
        left = snapshot(
            {"rgb": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)},
            {"joint": np.array([1.25, 2.5], dtype=np.float32)},
        )
        right = snapshot(
            {"rgb": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)},
            {"joint": np.array([1.25, 2.5], dtype=np.float32)},
        )
        self.assertEqual(first_differences(left, right), [])

    def test_changed_array_is_rejected(self) -> None:
        left = snapshot({"rgb": np.array([1], dtype=np.uint8)}, None)
        right = snapshot({"rgb": np.array([2], dtype=np.uint8)}, None)
        differences = first_differences(left, right)
        self.assertTrue(any("sha256" in difference for difference in differences))

    def test_nonfinite_array_is_rejected(self) -> None:
        with self.assertRaises(ObservationFingerprintError):
            snapshot({"depth": np.array([np.nan], dtype=np.float32)}, None)

    def test_comparator_cli_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "schema": "racer_initial_observation_v1",
                "task": "place_cups",
                "episode": 0,
            }
            direct = root / "direct.json"
            isolated = root / "isolated.json"
            report = root / "report.json"
            direct.write_text(
                json.dumps({**common, "snapshot": {"value": 1}}), encoding="utf-8"
            )
            isolated.write_text(
                json.dumps({**common, "snapshot": {"value": 2}}), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "compare_observation_fingerprints.py"),
                    "--direct",
                    str(direct),
                    "--isolated",
                    str(isolated),
                    "--output",
                    str(report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(report.read_text(encoding="utf-8"))["ok"])


if __name__ == "__main__":
    unittest.main()
