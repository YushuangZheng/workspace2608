import hashlib
import json
from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch
from integrations.rlbench.rlbench_dynamac.report import evaluation_videos, failure_videos
from integrations.rlbench.rlbench_dynamac.report import v4_post_evaluation_replays as replay


def _cell(tmp_path, *, paper_success_rate=0.8):
    return v4_formal_launch.FormalCell(
        name="wipe_desk_teleport",
        task="wipe_desk",
        scenario="teleport",
        evaluator_module="fixture",
        evaluator_arguments=(),
        models_dir=tmp_path / "models" / "v4",
        result=tmp_path / "result.json",
        paper_success_rate=paper_success_rate,
    )


def _payload(successes, failures, *, success_rate=None):
    rows = [
        {"episode": episode, "success": episode < successes}
        for episode in range(successes + failures)
    ]
    return {
        "release": "v4",
        "task": "wipe_desk",
        "scenario": "teleport",
        "success_rate": (
            successes / (successes + failures) if success_rate is None else success_rate
        ),
        "results": rows,
    }


def _independent_rank(cell_id, outcome, episode, seed):
    payload = (
        f"{evaluation_videos.SELECTION_PROTOCOL_ID}\n{seed}\n{cell_id}\n{outcome}\n{episode}\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_plan_cell_uses_exact_quota_and_sha_rank_with_finite_backups(tmp_path):
    cell = _cell(tmp_path)
    payload = _payload(8, 22)
    selection, jobs = replay._plan_cell(
        cell,
        payload,
        source_sha256="a" * 64,
        selection_seed=19,
    )

    assert selection.quota_tier == "sr_lt_50"
    assert (selection.required_successes, selection.required_failures) == (5, 20)
    assert [job.outcome for job in jobs] == ["success", "failure"]
    for job in jobs:
        expected = sorted(
            (
                row["episode"]
                for row in payload["results"]
                if bool(row["success"]) is (job.outcome == "success")
            ),
            key=lambda episode: (
                _independent_rank(cell.cell_id, job.outcome, episode, 19),
                episode,
            ),
        )
        assert list(job.candidate_episodes) == expected
        assert job.required_confirmed < len(job.candidate_episodes)


def test_plan_cell_keeps_class_shortfall_without_borrowing_quota(tmp_path):
    selection, jobs = replay._plan_cell(
        _cell(tmp_path, paper_success_rate=0.5),
        _payload(198, 2),
        source_sha256="b" * 64,
        selection_seed=0,
    )

    assert selection.quota_tier == "sr_ge_80"
    assert (selection.requested_successes, selection.requested_failures) == (3, 3)
    assert (selection.required_successes, selection.required_failures) == (3, 2)
    assert [job.required_confirmed for job in jobs] == [3, 2]


def test_plan_cell_omits_replays_when_paper_result_is_close(tmp_path):
    selection, jobs = replay._plan_cell(
        _cell(tmp_path, paper_success_rate=0.99),
        _payload(99, 1),
        source_sha256="c" * 64,
        selection_seed=0,
    )

    assert selection.quota_tier == "sr_ge_80_paper_diff_lt_2pp"
    assert jobs == ()


def test_recorder_command_is_accepted_by_failure_video_cli(tmp_path):
    _selection, jobs = replay._plan_cell(
        _cell(tmp_path),
        _payload(8, 22),
        source_sha256="d" * 64,
        selection_seed=1,
    )
    job = jobs[0]
    command = job.recorder_command(
        sim_python=Path("/sim/python3.8"),
        policy_python=Path("/policy/python3.10"),
        ffmpeg=Path("/usr/bin/ffmpeg"),
        fps=12,
        resolution=(640, 360),
        maximum_attempts=1,
        overwrite=False,
    )
    parsed = failure_videos.build_parser().parse_args(command[3:])

    assert parsed.episode == list(job.candidate_episodes)
    assert parsed.expected_outcome == "success"
    assert parsed.minimum_confirmed == job.required_confirmed
    assert parsed.cameras == ["front", "overhead"]
    assert parsed.models_dir == job.models_dir
    assert parsed.output_root == replay.REPLAY_ROOT
    assert parsed.overwrite is False


def test_every_release_v4_source_routes_to_v4_models_and_metadata():
    generic = {"release": "v4", "task": "wipe_desk"}
    coordination = {
        "release": "v4",
        "schema": "dynamac-table-iii-coordination-local-v4",
        "task": "bimanual_handover_item_dynamic",
        "policy_task_alias": "bimanual_handover_item",
    }

    assert failure_videos._is_v4_source(generic, "wipe_desk")
    assert failure_videos._v4_models_dir(generic, "wipe_desk", "unimanual") == (
        failure_videos.INTEGRATION_ROOT / "models" / "v4"
    )
    assert failure_videos._v4_replay_manifest_metadata(generic, "wipe_desk") == {"release": "v4"}
    assert (
        failure_videos._v4_models_dir(
            coordination,
            "bimanual_handover_item",
            "coordination",
        )
        == failure_videos.INTEGRATION_ROOT / "models" / "v4" / "table_iii"
    )
    assert (
        failure_videos._v4_replay_manifest_metadata(
            coordination,
            "bimanual_handover_item",
        )["coordination_protocol_id"]
        == failure_videos.V4_COORDINATION_PROTOCOL_ID
    )


def test_canonical_pretrigger_candidate_is_admitted_and_must_replay_same_state(
    tmp_path,
):
    original = {
        "episode": 3,
        "success": False,
        "invalid_actions": 0,
        "scenario_events": [],
        "intervention_eligible": False,
        "intervention_reached": False,
        "pre_intervention_terminal": True,
        "dynamic_condition_exercised": False,
        "dynamic_condition_unexercised": True,
        "intervention_effective": None,
        "intervention_complete": None,
    }
    source = {
        "task": "bimanual_handover_item",
        "scenario": "teleport",
        "seed": 100,
        "horizon": 1000,
        "model_identity": {"manifest_authenticated": True},
        "results": [original],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    _payload, selected, _sha256, _path = failure_videos._load_source(
        source_path,
        "bimanual_handover_item",
        (3,),
    )
    assert selected[3] == original
    assert (
        failure_videos._assess_replay_attempt(
            original,
            dict(original),
            source,
            expected_success=False,
        )["confirmed_for_publication"]
        is True
    )

    intervention_replay = dict(
        original,
        scenario_events=[
            {
                "kind": "teleport_task",
                "applied": True,
                "protocol_effective": True,
                "complete": True,
            }
        ],
        intervention_eligible=True,
        intervention_reached=True,
        pre_intervention_terminal=False,
        dynamic_condition_exercised=True,
        dynamic_condition_unexercised=False,
        intervention_effective=True,
        intervention_complete=True,
    )
    assessment = failure_videos._assess_replay_attempt(
        original,
        intervention_replay,
        source,
        expected_success=False,
    )
    assert assessment["confirmed_for_publication"] is False
    assert assessment["disposition"] == "discarded_source_condition_mismatch"


@pytest.mark.parametrize(
    ("task", "protocol_key", "protocol"),
    (
        (
            "wipe_desk",
            "protocol",
            {
                "intervention_max_attempts": 100,
                "smooth_motion_calls": 10,
            },
        ),
        (
            "bimanual_handover_item",
            "scenario_protocol",
            {
                "max_sampling_attempts": 100,
                "smooth_interpolation_calls": None,
            },
        ),
    ),
)
def test_current_formal_controller_budget_is_authenticated_for_both_families(
    task,
    protocol_key,
    protocol,
):
    controller = {
        **failure_videos.global_ik_controller_metadata(),
        "worker_clock_handshake_id": (
            failure_videos.FORMAL_POLICY_CLOCK_SEMANTICS_ID
        ),
    }
    source = {
        "release": "v4",
        "task": task,
        "scenario": "static",
        "controller": controller,
        "final_settling_protocol": {"maximum_physics_steps": 10},
        protocol_key: protocol,
    }

    budgets = failure_videos._source_protocol_budgets(source)
    assert budgets == {
        "smooth_steps": 10,
        "motion_attempts": 100,
        "final_settling_steps": 10,
        "primary_action_attempts": 1,
    }

    source["controller"] = {
        **controller,
        "profile": "tampered",
        "primary_action_retry": {
            "max_primary_action_attempts_per_policy_tick": 3
        },
    }
    with pytest.raises(RuntimeError, match="noncanonical formal controller"):
        failure_videos._source_protocol_budgets(source)


def test_legacy_retry_budget_remains_available_for_archived_sources():
    source = {
        "task": "wipe_desk",
        "scenario": "static",
        "controller": {
            "primary_action_retry": {
                "max_primary_action_attempts_per_policy_tick": 3
            }
        },
        "final_settling_protocol": {"maximum_physics_steps": 10},
        "protocol": {
            "intervention_max_attempts": 100,
            "smooth_motion_calls": 10,
        },
    }

    assert failure_videos._source_protocol_budgets(source)[
        "primary_action_attempts"
    ] == 3


def test_gpu_parser_caps_distinct_reusable_lanes_at_eight():
    assert replay._parse_gpus("0,1,2,3,4,5,6,7") == tuple(range(8))
    with pytest.raises(Exception):
        replay._parse_gpus("0,1,2,3,4,5,6,7,8")
    with pytest.raises(Exception):
        replay._parse_gpus("0,1,1")


def test_published_job_validator_authenticates_two_view_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(replay, "REPLAY_ROOT", tmp_path / "replay_video")
    source_result = (tmp_path / "result.json").resolve()
    job = replay.ReplayJob(
        cell_name="wipe_desk_static",
        cell_id="wipe_desk/static",
        source_task="wipe_desk",
        scenario="static",
        policy_task="wipe_desk",
        outcome="success",
        quota_tier="fixture",
        requested_quota=1,
        required_confirmed=1,
        candidate_episodes=(7, 3),
        source_result=source_result,
        source_result_sha256="f" * 64,
        models_dir=tmp_path / "models",
        output_key="wipe_desk_static_success",
    )
    job.target.mkdir(parents=True)
    video_path = job.target / "episode.mp4"
    video_path.write_bytes(b"fixture-video")
    sidecar_path = job.target / "episode.json"
    sidecar = {
        "release": "v4",
        "task": job.policy_task,
        "source_task": job.source_task,
        "scenario": job.scenario,
        "episode": 7,
        "source_result": str(job.source_result),
        "source_result_sha256": job.source_result_sha256,
        "expected_outcome": job.outcome,
        "confirmed_for_publication": True,
        "video": {
            "file": video_path.name,
            "sha256": replay._sha256(video_path),
            "requested_cameras": ["front", "overhead"],
            "used_cameras": ["front", "overhead"],
        },
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    manifest = {
        "schema": failure_videos.SCHEMA,
        "release": "v4",
        "task": job.policy_task,
        "source_task": job.source_task,
        "scenario": job.scenario,
        "output_key": job.output_key,
        "source_result": str(job.source_result),
        "source_result_sha256": job.source_result_sha256,
        "expected_outcome": job.outcome,
        "required_confirmed_trajectories": 1,
        "candidate_episode_order": [7, 3],
        "episodes": [
            {
                "episode": 7,
                "expected_outcome": job.outcome,
                "confirmed_for_publication": True,
                "replay_confirmed_outcome": True,
                "release": "v4",
                "metadata": sidecar_path.name,
                "metadata_sha256": replay._sha256(sidecar_path),
                "video": video_path.name,
                "video_sha256": replay._sha256(video_path),
            }
        ],
    }
    manifest_path = job.target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert replay._validate_job_output(job)["release"] == "v4"
    manifest["episodes"][0]["metadata_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="two-view V4 video"):
        replay._validate_job_output(job)


def test_queue_continuously_refills_without_exceeding_lane_count(
    tmp_path,
    monkeypatch,
):
    jobs = []
    for index in range(5):
        jobs.append(
            replay.ReplayJob(
                cell_name=f"cell_{index}",
                cell_id=f"task_{index}/static",
                source_task="wipe_desk",
                scenario="static",
                policy_task="wipe_desk",
                outcome="success",
                quota_tier="fixture",
                requested_quota=1,
                required_confirmed=1,
                candidate_episodes=(index,),
                source_result=tmp_path / f"result_{index}.json",
                source_result_sha256="e" * 64,
                models_dir=tmp_path / "models",
                output_key=f"cell_{index}_success",
            )
        )

    active = 0
    maximum_active = 0

    class Process:
        next_pid = 100

        def __init__(self, *_args, **_kwargs):
            nonlocal active, maximum_active
            self.pid = Process.next_pid
            Process.next_pid += 1
            self.done = False
            active += 1
            maximum_active = max(maximum_active, active)

        def poll(self):
            nonlocal active
            if not self.done:
                self.done = True
                active -= 1
            return 0

    monkeypatch.setattr(replay.subprocess, "Popen", Process)
    monkeypatch.setattr(replay, "_validate_job_output", lambda _job: {})
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        replay.v4_formal_launch,
        "_launch_environment",
        lambda _python, gpu: {"CUDA_VISIBLE_DEVICES": str(gpu)},
    )
    run_root = tmp_path / "run"
    run_root.mkdir()

    assignments = replay._run_queue(
        jobs,
        run_root=run_root,
        sim_python=Path("/sim/python"),
        policy_python=Path("/policy/python"),
        xvfb_run=Path("/usr/bin/xvfb-run"),
        ffmpeg=Path("/usr/bin/ffmpeg"),
        gpus=(3, 7),
        fps=12,
        resolution=(640, 360),
        maximum_attempts=1,
        overwrite=False,
    )

    assert len(assignments) == 5
    assert maximum_active == 2
    assert {row["gpu"] for row in assignments} == {3, 7}


def test_execute_skips_only_valid_existing_targets_and_queues_pending(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(replay, "REPLAY_ROOT", tmp_path / "replay_video")
    monkeypatch.setattr(replay, "LAUNCH_ROOT", tmp_path / "launch")

    def job(index):
        return replay.ReplayJob(
            cell_name=f"cell_{index}",
            cell_id=f"wipe_desk/scenario_{index}",
            source_task="wipe_desk",
            scenario="static",
            policy_task="wipe_desk",
            outcome="success",
            quota_tier="fixture",
            requested_quota=1,
            required_confirmed=1,
            candidate_episodes=(index,),
            source_result=(tmp_path / f"result_{index}.json").resolve(),
            source_result_sha256=str(index) * 64,
            models_dir=tmp_path / "models",
            output_key=f"cell_{index}_success",
        )

    existing, pending = job(1), job(2)
    existing.target.mkdir(parents=True)
    (existing.target / "manifest.json").write_text("{}", encoding="utf-8")
    plan = replay.ReplayPlan(
        evaluation_set={"evaluation_set_id": "fixture"},
        selections=(),
        jobs=(existing, pending),
        selection_seed=0,
    )
    monkeypatch.setattr(replay, "build_validated_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        replay,
        "_validate_runtime",
        lambda **kwargs: (
            kwargs["sim_python"],
            kwargs["policy_python"],
            kwargs["xvfb_run"],
            kwargs["ffmpeg"],
            tuple(kwargs["gpus"]),
            {"manifest_sha256": "model"},
        ),
    )
    validated = []

    def validate(value):
        validated.append(value.output_key)
        return {"episodes": [{}]}

    monkeypatch.setattr(replay, "_validate_job_output", validate)
    queued = []

    def run_queue(jobs, **_kwargs):
        queued.extend(job.output_key for job in jobs)
        return ({"output_key": jobs[0].output_key},)

    monkeypatch.setattr(replay, "_run_queue", run_queue)
    summary = replay.execute(
        selection_seed=0,
        sim_python=Path("/sim/python"),
        policy_python=Path("/policy/python"),
        xvfb_run=Path("/usr/bin/xvfb-run"),
        ffmpeg=Path("/usr/bin/ffmpeg"),
        gpus=(0,),
        fps=12,
        resolution=(640, 360),
        maximum_attempts=1,
        overwrite=False,
    )

    assert validated == [existing.output_key]
    assert queued == [pending.output_key]
    assert [row["output_key"] for row in summary["skipped_valid"]] == [
        existing.output_key
    ]
    assert summary["assignments"] == [{"output_key": pending.output_key}]

    monkeypatch.setattr(
        replay,
        "_validate_job_output",
        lambda _job: (_ for _ in ()).throw(RuntimeError("invalid existing target")),
    )
    with pytest.raises(RuntimeError, match="invalid existing target"):
        replay.execute(
            selection_seed=0,
            sim_python=Path("/sim/python"),
            policy_python=Path("/policy/python"),
            xvfb_run=Path("/usr/bin/xvfb-run"),
            ffmpeg=Path("/usr/bin/ffmpeg"),
            gpus=(0,),
            fps=12,
            resolution=(640, 360),
            maximum_attempts=1,
            overwrite=False,
        )
    assert queued == [pending.output_key]
