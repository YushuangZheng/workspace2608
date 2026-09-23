"""Frozen feature adapter for M3, independent of score-backend storage.

``from_scorer`` binds an existing frozen logpZO scorer without requiring a new
checkpoint or an adapter-specific on-disk weight format. The legacy weight
loader remains optional. Neither entrypoint supplies a missing score backend
or makes unrelated official Square features compatible with DynaMAC.
"""

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

from .adapter import TorchLogpZOScorer
from .preprocessing import ENCODER_SCHEMA, FeatureLayout, runtime_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalFailDetectMonitor(RuntimeMonitor):
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> CanonicalFailDetectMonitor:
        """Build the frozen task scorer selected by A's method registry."""

        from .model import build_official_velocity_model

        task = mapping.get("_task_id")
        method_hash = mapping.get("_method_config_sha256")
        if not isinstance(task, str) or not task:
            raise ValueError("M3 monitor factory requires an injected task id")
        if not isinstance(method_hash, str) or len(method_hash) != 64:
            raise ValueError("M3 monitor factory requires the method-config identity")
        root = Path(mapping["checkpoint_root"])
        config = Path(mapping["backend_config"])
        schedule = mapping.get("_threshold_schedule")
        return cls(
            build_official_velocity_model(),
            root / task / "seed_1103" / "model.pt",
            config,
            threshold_schedule=schedule,
            method_config_hash=method_hash,
        )

    @classmethod
    def from_scorer(
        cls,
        score_model: Any,
        config: str | Path,
        *,
        binding: Mapping[str, Any],
        threshold_schedule: Any = None,
        allow_development_scorer: bool = False,
    ) -> CanonicalFailDetectMonitor:
        """Adapt an existing scorer (including a method-isolated inference service).

        The method-internal binding describes the scorer's *actual* input layout
        and normalizer, frozen config and scorer identities, and task. It is not
        an additional evaluator interface or a request to create training data.
        A matching feature representation is still required; dimensional padding
        alone is not evidence that an unrelated model is a valid backend.

        Synthetic test doubles must set ``development_only`` and require the
        explicit opt-in. Their outputs are not production development golden.
        """
        if not callable(score_model):
            raise TypeError("score_model must be callable")
        self = cls.__new__(cls)
        self.config = json.loads(Path(config).read_text())
        self.config_hash = _sha256(Path(config))
        self.method_config_hash = self.config_hash
        # Detach nested metadata from mutable caller state.
        self.metadata = json.loads(json.dumps(dict(binding), allow_nan=False))
        if self.metadata.get("development_only", False) and not allow_development_scorer:
            raise ValueError("development-only scorer cannot be used as a Main-10 monitor")
        if (
            self.metadata.get("feature_schema") != FEATURE_SCHEMA
            or self.metadata.get("encoder_schema") != ENCODER_SCHEMA
            or self.metadata.get("config_sha256") != self.config_hash
            or self.metadata.get("score_definition") != self.config["score_definition"]
            or not isinstance(self.metadata.get("task"), str)
            or not self.metadata["task"]
        ):
            raise ValueError("M3 scorer/config/feature identity mismatch")
        self.scorer_hash = self.metadata.get("scorer_sha256")
        self.checkpoint_hash = self.metadata.get("checkpoint_sha256")
        for key in ("scorer_sha256", "checkpoint_sha256"):
            digest = self.metadata.get(key)
            if key == "checkpoint_sha256" and digest is None:
                continue
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("scorer/checkpoint identity must be a SHA256 digest")
        if self.config.get("observation_window") != 1:
            raise ValueError("canonical adapter supports the frozen single-cycle window")
        persistence = self.config.get("persistence")
        if isinstance(persistence, bool) or not isinstance(persistence, int) or persistence < 1:
            raise ValueError("persistence must be a positive integer")
        self.layout = FeatureLayout.from_dict(self.metadata["layout"])
        self._load_normalizer()
        self.scorer = score_model
        self.schedule = threshold_schedule
        self.context = None
        self.output = None
        return self

    def __init__(
        self,
        velocity_model: Any,
        checkpoint: str | Path,
        config: str | Path,
        *,
        threshold_schedule: Any = None,
        allow_development_checkpoint: bool = False,
        method_config_hash: str | None = None,
    ) -> None:
        import torch

        self.config = json.loads(Path(config).read_text())
        self.config_hash = _sha256(Path(config))
        self.method_config_hash = (
            self.config_hash if method_config_hash is None else str(method_config_hash)
        )
        self.checkpoint_hash = _sha256(Path(checkpoint))
        self.scorer_hash = self.checkpoint_hash
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema") != "essay2608.iclr2027.m3-logpzo.v1":
            raise ValueError(
                "requires DynaMAC-feature logpZO checkpoint, not official Square weights"
            )
        self.metadata = payload["metadata"]
        if self.metadata.get("development_only", False) and not allow_development_checkpoint:
            raise ValueError("development-only weights cannot be used as a trained Main-10 monitor")
        if (
            self.metadata["feature_schema"] != FEATURE_SCHEMA
            or self.metadata["encoder_schema"] != ENCODER_SCHEMA
            or self.metadata["config_sha256"] != self.config_hash
            or self.metadata["fit_split"] != "normal_demonstrations"
            or self.metadata["unet_input_channels"] != self.config["unet_input_channels"]
        ):
            raise ValueError("M3 checkpoint/config/training-source identity mismatch")
        self.layout = FeatureLayout.from_dict(self.metadata["layout"])
        self._load_normalizer()
        velocity_model.load_state_dict(payload["model_state_dict"], strict=True)
        self.scorer = TorchLogpZOScorer(
            velocity_model, input_dim=self.config["unet_input_channels"], device="cpu"
        )
        self.schedule = threshold_schedule
        self.context = None
        self.output = None

    def _load_normalizer(self) -> None:
        self.mean = np.asarray(self.metadata["normalizer"]["mean"], dtype=np.float32)
        self.std = np.asarray(self.metadata["normalizer"]["std"], dtype=np.float32)
        clip = self.config["normalizer"]["clip"]
        if (
            self.mean.shape != (self.layout.input_dim,)
            or self.std.shape != self.mean.shape
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or not (self.std > 0).all()
            or isinstance(clip, bool)
            or not isinstance(clip, (int, float))
            or not np.isfinite(clip)
            or clip <= 0
        ):
            raise ValueError("invalid frozen feature normalizer")

    def reset(self, episode_context: EpisodeContext) -> None:
        if not isinstance(episode_context, EpisodeContext):
            raise TypeError("reset requires A's EpisodeContext")
        if (
            episode_context.task_id != self.metadata["task"]
            or episode_context.method_id != "M3"
            or episode_context.bimanual != (len(self.layout.arms) == 2)
            or episode_context.feature_schema != FEATURE_SCHEMA
            or episode_context.method_config_hash != self.method_config_hash
            or episode_context.checkpoint_hash != self.checkpoint_hash
        ):
            raise ValueError("episode context disagrees with M3 checkpoint identity")
        self.context = episode_context
        self.last_cycle = None
        self.count = 0
        self.first_alarm_cycle = None
        self.output = None

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
        vector = self.layout.encode_validated(record)
        clip = self.config["normalizer"]["clip"]
        vector = np.clip((vector - self.mean) / self.std, -clip, clip)
        policy_step = record["policy_state"].get("policy_step")
        if isinstance(policy_step, bool) or not isinstance(policy_step, int):
            raise ValueError("M3 time-varying threshold requires an integer policy_step")
        schedule_step = int(policy_step)
        if self.schedule is not None:
            horizon = getattr(self.schedule, "horizon", None)
            if horizon is not None:
                if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                    raise ValueError("conformal schedule horizon must be a positive integer")
                schedule_step = min(schedule_step, horizon - 1)
        threshold = (
            None
            if self.schedule is None
            else float(self.schedule.threshold(schedule_step))
        )
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("conformal threshold must be finite")
        score = float(self.scorer(vector[None, :]))
        if not np.isfinite(score):
            raise ValueError("logpZO emitted a nonfinite score")
        self.count = self.count + 1 if threshold is not None and score > threshold else 0
        alarm = self.count >= self.config["persistence"]
        if alarm and self.first_alarm_cycle is None:
            self.first_alarm_cycle = cycle
        self.output = MonitorOutput(
            cycle,
            {self.config["score_name"]: score},
            alarm,
            threshold,
            self.count,
            {
                "first_alarm_cycle": self.first_alarm_cycle,
                "checkpoint_sha256": self.checkpoint_hash,
                "scorer_sha256": self.scorer_hash,
                "config_sha256": self.config_hash,
                "calibration_status": "score_only" if threshold is None else "caller_supplied",
                "development_only": bool(self.metadata.get("development_only", False)),
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
