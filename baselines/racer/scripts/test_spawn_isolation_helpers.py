#!/usr/bin/env python3
"""Dependency-light regression tests for the RACER isolation boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from compare_observation_fingerprints import first_differences
from isolated_simulator_proxy import SpawnIsolatedRLBenchSim
from observation_fingerprint import ObservationFingerprintError, snapshot
from point_cloud_evidence import CAMERAS, write_point_cloud_evidence


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
                lifecycle = json.loads(
                    proxy.worker_status_path.read_text(encoding="utf-8")
                )
                self.assertEqual(lifecycle["state"], "closed")
                self.assertTrue(lifecycle["natural_exit"])


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


class SingleEpisodeValidatorTest(unittest.TestCase):
    def _point_clouds(self):
        coordinate = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
        x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
        base = np.stack((x, y, x * np.float32(0.2) + y * np.float32(0.3)))
        return {
            f"{camera}_point_cloud": np.ascontiguousarray(
                base + np.float32(index * 0.01)
            )
            for index, camera in enumerate(CAMERAS)
        }

    def _fixture(self, root: Path, success: bool) -> tuple[Path, Path, Path]:
        output = root / "gate"
        episode = output / "place_cups" / "0"
        episode.mkdir(parents=True)
        metrics = output / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "place_cups": {
                        "0": {
                            "success": success,
                            "episode_len": 4,
                            "retry_times": 0,
                        }
                    },
                    "overall": {
                        "place_cups": 1.0 if success else 0.0,
                        "avg_success_rate": 1.0 if success else 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (episode / "episode_statistic.json").write_text("{}\n", encoding="utf-8")
        marker = "success_step4" if success else "failure_step4"
        (episode / marker).write_text(marker, encoding="utf-8")
        horizontal = np.arange(256, dtype=np.uint8)[None, :]
        for index, camera in enumerate(CAMERAS):
            pixels = np.zeros((346, 256, 3), dtype=np.uint8)
            pixels[:, :, 0] = horizontal
            pixels[:, :, 1] = np.uint8(20 + index)
            Image.fromarray(pixels, mode="RGB").save(
                episode / f"{camera}_rgb.gif", format="GIF"
            )
        actor_log = root / "actor.log"
        actor_log.write_text("natural evaluator exit\n", encoding="utf-8")
        evidence = root / "gate_point_cloud_evidence.json"
        write_point_cloud_evidence(
            self._point_clouds(),
            output=evidence,
            task_name="place_cups",
            episode_num=0,
        )
        return metrics, actor_log, evidence

    def _validate(
        self, metrics: Path, actor_log: Path, evidence: Path
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "validate_single_episode.py"),
                "--metrics",
                str(metrics),
                "--actor-log",
                str(actor_log),
                "--point-cloud-evidence",
                str(evidence),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_success_true_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, actor_log, evidence = self._fixture(Path(temporary), True)
            result = self._validate(metrics, actor_log, evidence)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_success_false_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, actor_log, evidence = self._fixture(Path(temporary), False)
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("success=true", result.stderr)

    def test_fake_gif_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, actor_log, evidence = self._fixture(Path(temporary), True)
            bad_gif = metrics.parent / "place_cups" / "0" / "front_rgb.gif"
            bad_gif.write_bytes(b"not-a-gif")
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid camera GIF", result.stderr)

    def test_wrong_size_gif_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, actor_log, evidence = self._fixture(Path(temporary), True)
            bad_gif = metrics.parent / "place_cups" / "0" / "front_rgb.gif"
            pixels = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
            Image.fromarray(pixels, mode="L").save(bad_gif, format="GIF")
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("size", result.stderr)

    def test_black_camera_with_nonblack_text_overlay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, actor_log, evidence = self._fixture(Path(temporary), True)
            bad_gif = metrics.parent / "place_cups" / "0" / "front_rgb.gif"
            pixels = np.zeros((346, 256, 3), dtype=np.uint8)
            pixels[300, 20, :] = 255
            Image.fromarray(pixels, mode="RGB").save(bad_gif, format="GIF")
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("camera pixels are degenerate", result.stderr)

    def test_wrong_shape_point_cloud_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, actor_log, evidence = self._fixture(root, True)
            npz_path = root / "gate_point_clouds.npz"
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["front_point_cloud"] = arrays["front_point_cloud"][:, :511, :]
            np.savez_compressed(npz_path, **arrays)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shape", result.stderr)

    def test_degenerate_point_cloud_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, actor_log, evidence = self._fixture(root, True)
            npz_path = root / "gate_point_clouds.npz"
            arrays = {
                f"{camera}_point_cloud": np.zeros((3, 512, 512), dtype=np.float32)
                for camera in CAMERAS
            }
            np.savez_compressed(npz_path, **arrays)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("degenerate", result.stderr)

    def test_nonfinite_point_cloud_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, actor_log, evidence = self._fixture(root, True)
            npz_path = root / "gate_point_clouds.npz"
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["front_point_cloud"][0, 0, 0] = np.nan
            np.savez_compressed(npz_path, **arrays)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            result = self._validate(metrics, actor_log, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-finite", result.stderr)


class ProcessBoundaryTest(unittest.TestCase):
    def test_owned_process_audit_detects_and_clears_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token = f"racer-test-{uuid.uuid4().hex}"
            environment = os.environ.copy()
            environment["RACER_OWNER_TOKEN"] = token
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                env=environment,
            )
            try:
                active_report = root / "active.json"
                active = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "audit_owned_processes.py"),
                        "--environment-key",
                        "RACER_OWNER_TOKEN",
                        "--value",
                        token,
                        "--output",
                        str(active_report),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(active.returncode, 1)
                report = json.loads(active_report.read_text(encoding="utf-8"))
                self.assertIn(child.pid, [item["pid"] for item in report["processes"]])
            finally:
                child.terminate()
                child.wait(timeout=5)

            clear_report = root / "clear.json"
            clear = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "audit_owned_processes.py"),
                    "--environment-key",
                    "RACER_OWNER_TOKEN",
                    "--value",
                    token,
                    "--output",
                    str(clear_report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(clear.returncode, 0, clear.stderr)
            self.assertEqual(
                json.loads(clear_report.read_text(encoding="utf-8"))["residual_count"],
                0,
            )

    def test_wall_clock_timeout_is_bounded(self) -> None:
        result = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=1s",
                "0.1s",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 124)


if __name__ == "__main__":
    unittest.main()
