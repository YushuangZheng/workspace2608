import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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

    def test_wilson_interval_contains_rate(self):
        summarize = load_script("summarize_logpzo.py")
        lower, upper = summarize.wilson_interval(7, 10)
        self.assertLess(lower, 0.7)
        self.assertGreater(upper, 0.7)

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
