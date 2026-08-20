import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout


BASELINE = Path(__file__).resolve().parents[1]


def load_script(name):
    path = BASELINE / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantProtocolTest(unittest.TestCase):
    def test_rollout_file_can_extend_contiguous_prefix(self):
        evaluate = load_script("logpzo_evaluate.py")
        payload = {
            "schema": "dynamac-fail-detect-logpzo-rollouts-v1",
            "protocol_label": evaluate.PROTOCOL_LABEL,
            "task": "transport",
            "policy_type": "diffusion",
            "modify": False,
            "start_seed": 100000,
            "requested_episodes": 10,
            "complete": True,
            "episodes": [{"seed": 100000 + index} for index in range(10)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            previous, records = evaluate.load_existing(path, False, 100000, 50)
        self.assertEqual(previous["requested_episodes"], 10)
        self.assertEqual(len(records), 10)

    def test_detection_confusion_and_time(self):
        summarize = load_script("summarize_logpzo.py")
        band = [1.0, 1.0, 1.0]
        records = [
            {"seed": 1, "domain": "id", "success": False, "logpzo": [0.0, 1.2, 1.3]},
            {"seed": 2, "domain": "ood", "success": False, "logpzo": [0.0, 0.2, 0.3]},
            {"seed": 3, "domain": "id", "success": True, "logpzo": [0.0, 0.2, 0.3]},
            {"seed": 4, "domain": "ood", "success": True, "logpzo": [1.1, 0.2, 0.3]},
        ]
        result = summarize.detection_stats(records, band)
        self.assertEqual(result["confusion"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(result["true_positive_detection_step_mean"], 8)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["true_positive_rate_wilson_95"], summarize.wilson_interval(1, 2))
        self.assertEqual(result["true_negative_rate_wilson_95"], summarize.wilson_interval(1, 2))
        self.assertIsNone(result["balanced_accuracy_interval"])
        self.assertIn("No single Wilson", result["balanced_accuracy_interval_note"])

    def test_wilson_interval_contains_rate(self):
        summarize = load_script("summarize_logpzo.py")
        lower, upper = summarize.wilson_interval(7, 10)
        self.assertLess(lower, 0.7)
        self.assertGreater(upper, 0.7)

    @unittest.skipUnless(shutil.which("timeout"), "GNU timeout is required")
    def test_timeout_exit_124_cannot_write_complete(self):
        deadline_runner = BASELINE / "scripts/deadline_runner.sh"
        status_script = BASELINE / "scripts/quant_status.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_file = root / "status.json"
            marker = root / "deadline.triggered"
            proc = subprocess.run(
                [
                    "bash", str(deadline_runner),
                    "--status-script", str(status_script),
                    "--status-file", str(status_file),
                    "--deadline-marker", str(marker),
                    "--duration", "0.2s",
                    "--kill-after", "1s",
                    "--", "bash", "-c", "trap 'exit 0' TERM; while :; do sleep 1; done",
                ],
                check=False,
                timeout=5,
            )
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        self.assertEqual(proc.returncode, 124)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["stage"], "deadline")
        self.assertNotIn("complete", [event["state"] for event in payload["history"]])

    @unittest.skipUnless(shutil.which("timeout"), "GNU timeout is required")
    def test_early_exit_137_is_failed_not_deadline(self):
        deadline_runner = BASELINE / "scripts/deadline_runner.sh"
        status_script = BASELINE / "scripts/quant_status.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_file = root / "status.json"
            marker = root / "deadline.triggered"
            proc = subprocess.run(
                [
                    "bash", str(deadline_runner),
                    "--status-script", str(status_script),
                    "--status-file", str(status_file),
                    "--deadline-marker", str(marker),
                    "--duration", "5s",
                    "--kill-after", "1s",
                    "--", "bash", "-c", "exit 137",
                ],
                check=False,
                timeout=5,
            )
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        self.assertEqual(proc.returncode, 137)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["stage"], "failed")
        self.assertFalse(marker.exists())

    def test_python_gate_treats_dead_panes_as_inactive(self):
        prepare = load_script("prepare_official_artifacts.py")
        with mock.patch.object(prepare.subprocess, "run") as run:
            run.side_effect = [
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=0, stdout="1\n"),
            ]
            self.assertFalse(prepare.tmux_session_active("finished"))

    def test_python_gate_treats_any_live_pane_as_active(self):
        prepare = load_script("prepare_official_artifacts.py")
        with mock.patch.object(prepare.subprocess, "run") as run:
            run.side_effect = [
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=0, stdout="1\n0\n"),
            ]
            self.assertTrue(prepare.tmux_session_active("running"))

    def test_generated_detector_states_are_distinct(self):
        artifact_state = load_script("generated_artifact_state.py")
        classify = artifact_state.classify_detector_metadata
        self.assertEqual(classify(200, 199, True), "complete")
        self.assertEqual(classify(73, 72, True), "resumable")
        self.assertEqual(classify(73, 71, True), "damaged")
        self.assertEqual(classify(200, 198, True), "damaged")
        self.assertEqual(classify(73, 72, False), "damaged")

    def test_feature_missing_and_nonempty_invalid_are_not_complete(self):
        artifact_state = load_script("generated_artifact_state.py")
        validator = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            with self.assertRaises(SystemExit) as missing, redirect_stdout(io.StringIO()):
                artifact_state.inspect_features(path, validator)
            self.assertEqual(missing.exception.code, artifact_state.EXIT_MISSING)
            path.write_bytes(b"partial")
            validator.validate_features.side_effect = RuntimeError("truncated")
            with self.assertRaises(SystemExit) as damaged, redirect_stdout(io.StringIO()):
                artifact_state.inspect_features(path, validator)
            self.assertEqual(damaged.exception.code, artifact_state.EXIT_DAMAGED)

    def test_only_one_compatibility_repair_can_be_registered(self):
        script = BASELINE / "scripts/compatibility_repair.py"
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "repairs.json"
            first = subprocess.run(
                [sys.executable, str(script), str(state), "register", "first repair"],
                check=False,
            )
            second = subprocess.run(
                [sys.executable, str(script), str(state), "register", "second repair"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            payload = json.loads(state.read_text(encoding="utf-8"))
        self.assertEqual(first.returncode, 0)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(len(payload["repairs"]), 1)


@unittest.skipUnless(shutil.which("tmux"), "tmux is required for the integration gate test")
class TmuxGateIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.socket = "fail-detect-test-" + uuid.uuid4().hex
        self.tmux = ["tmux", "-L", self.socket]
        self.script = BASELINE / "scripts/resource_gate.sh"
        self.env = dict(os.environ, FAIL_DETECT_TMUX_SOCKET=self.socket)

    def tearDown(self):
        subprocess.run(
            self.tmux + ["kill-server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def state(self, session):
        return subprocess.run(
            ["bash", str(self.script), "--session-active", session],
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode

    def test_alive_and_remain_on_exit_dead_sessions(self):
        subprocess.run(self.tmux + ["new-session", "-d", "-s", "alive", "sleep 30"], check=True)
        self.assertEqual(self.state("alive"), 0)

        subprocess.run(self.tmux + ["new-session", "-d", "-s", "dead"], check=True)
        subprocess.run(self.tmux + ["set-option", "-w", "-t", "dead", "remain-on-exit", "on"], check=True)
        subprocess.run(self.tmux + ["send-keys", "-t", "dead", "exit", "Enter"], check=True)
        for _ in range(30):
            pane_dead = subprocess.check_output(
                self.tmux + ["list-panes", "-t", "dead", "-F", "#{pane_dead}"],
                text=True,
            ).strip()
            if pane_dead == "1":
                break
            time.sleep(0.1)
        self.assertEqual(pane_dead, "1")
        self.assertEqual(self.state("dead"), 1)
        self.assertEqual(self.state("missing"), 1)


if __name__ == "__main__":
    unittest.main()
