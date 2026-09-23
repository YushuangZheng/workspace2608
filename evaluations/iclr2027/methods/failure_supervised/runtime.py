"""Frozen-feature M4 runtime; no calibration data or evaluation labels loaded."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA
from evaluations.iclr2027.interfaces.runtime_monitor import (
    EpisodeContext,
    MonitorOutput,
    RuntimeMonitor,
)
from evaluations.iclr2027.methods.fail_detect.preprocessing import FeatureLayout, runtime_record

from .adapter import TorchGRUProbabilityScorer
from .training import load_training_checkpoint


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CanonicalFailureSupervisedMonitor(RuntimeMonitor):
    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any]
    ) -> CanonicalFailureSupervisedMonitor:
        task = mapping.get("_task_id")
        method_hash = mapping.get("_method_config_sha256")
        if not isinstance(task, str) or not task:
            raise ValueError("M4 monitor factory requires an injected task id")
        if not isinstance(method_hash, str) or len(method_hash) != 64:
            raise ValueError("M4 monitor factory requires the method-config identity")
        budget = int(mapping.get("training_budget", 200))
        seed = int(mapping.get("training_seed", 1103))
        held_out = mapping.get("held_out_family")
        root = Path(mapping["checkpoint_root"]) / task
        subdirectory = (
            f"budget_{budget}"
            if held_out is None
            else f"lofo_{held_out}"
        )
        return cls(
            root / subdirectory / f"seed_{seed}" / "model.pt",
            Path(mapping["backend_config"]),
            threshold_schedule=mapping.get("_threshold_schedule"),
            method_config_hash=method_hash,
        )

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path,
        *,
        device: str = "cpu",
        threshold_schedule: Any = None,
        method_config_hash: str | None = None,
    ) -> None:
        self.checkpoint_hash = sha256(Path(checkpoint))
        self.config_hash = sha256(Path(config))
        self.method_config_hash = (
            self.config_hash if method_config_hash is None else str(method_config_hash)
        )
        self.config = json.loads(Path(config).read_text())
        if (
            self.config.get("inference_device_policy")
            != "cpu_float32_for_calibration_golden_and_evaluation"
            or device != "cpu"
        ):
            raise ValueError(
                "this frozen method uses CPU float32 inference; GPU training is separate"
            )
        model, payload = load_training_checkpoint(checkpoint)
        self.metadata = payload["metadata"]
        if (
            self.metadata["config_sha256"] != self.config_hash
            or self.metadata["feature_schema"] != FEATURE_SCHEMA
        ):
            raise ValueError("checkpoint/config/schema identity mismatch")
        self.layout = FeatureLayout.from_dict(self.metadata["layout"])
        if model.input_dim != self.layout.input_dim:
            raise ValueError("checkpoint input dimension differs from frozen layout")
        self.mean = np.asarray(self.metadata["normalizer"]["mean"], dtype=np.float32)
        self.std = np.asarray(self.metadata["normalizer"]["std"], dtype=np.float32)
        self.clip = float(self.config["normalizer"]["clip"])
        if (
            self.mean.shape != (self.layout.input_dim,)
            or self.std.shape != self.mean.shape
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("invalid checkpoint normalizer")
        self.scorer = TorchGRUProbabilityScorer(model, device=device)
        self.schedule = threshold_schedule
        self.context = None
        self.last_cycle = None
        self.output = None

    def reset(self, episode_context: EpisodeContext) -> None:
        if not isinstance(episode_context, EpisodeContext):
            raise TypeError("reset requires A's EpisodeContext")
        if (
            episode_context.task_id != self.metadata["task"]
            or episode_context.feature_schema != FEATURE_SCHEMA
            or episode_context.method_id != "M4"
            or episode_context.bimanual != (len(self.layout.arms) == 2)
            or episode_context.method_config_hash != self.method_config_hash
            or episode_context.checkpoint_hash != self.checkpoint_hash
        ):
            raise ValueError("episode context disagrees with checkpoint identity")
        self.scorer.reset()
        self.context = episode_context
        self.last_cycle = None
        self.output = None
        self.count = 0
        self.first_alarm_cycle = None

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        if self.context is None:
            raise RuntimeError("reset must precede observe")
        record = runtime_record(observation, action, policy_state, self.context.episode_id)
        cycle = record["cycle"]
        if self.last_cycle is not None and cycle != self.last_cycle + 1:
            raise ValueError("non-contiguous cycle; explicitly reset before a sparse segment")
        x = self.layout.encode_validated(record)
        x = np.clip((x - self.mean) / self.std, -self.clip, self.clip)
        threshold = None if self.schedule is None else float(self.schedule.threshold(cycle))
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("calibration threshold must be finite")
        probability = self.scorer(x)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("classifier emitted invalid probability")
        self.count = self.count + 1 if threshold is not None and probability > threshold else 0
        alarm = self.count >= self.config["persistence"]
        if alarm and self.first_alarm_cycle is None:
            self.first_alarm_cycle = cycle
        self.output = MonitorOutput(
            cycle,
            {self.config["score_name"]: probability},
            alarm,
            threshold,
            self.count,
            {
                "first_alarm_cycle": self.first_alarm_cycle,
                "calibration_status": "score_only" if threshold is None else "caller_supplied",
                "checkpoint_sha256": self.checkpoint_hash,
                "config_sha256": self.config_hash,
            },
        )
        self.last_cycle = cycle

    def observe_record(self, record: Mapping[str, Any]) -> None:
        self.observe(record, {}, record["policy_state"])

    def score(self) -> Mapping[str, float]:
        if self.output is None:
            raise RuntimeError("no observed cycle")
        return dict(self.output.scores)

    def alarm(self) -> bool:
        return False if self.output is None else bool(self.output.alarm)

    def cycle_output(self) -> dict[str, Any]:
        if self.output is None:
            raise RuntimeError("no observed cycle")
        return asdict(self.output)

    @property
    def threshold(self) -> float | None:
        return None if self.output is None else self.output.threshold

    @property
    def persistence_count(self) -> int:
        return 0 if self.output is None else int(self.output.persistence_count)

    @property
    def output_metadata(self) -> Mapping[str, Any]:
        return {} if self.output is None else self.output.metadata
