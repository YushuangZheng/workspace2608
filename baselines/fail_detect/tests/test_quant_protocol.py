import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
