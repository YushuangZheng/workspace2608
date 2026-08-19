from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac import v4_formal_launch as launch
from integrations.rlbench.rlbench_dynamac import direct_evaluate
from integrations.rlbench.rlbench_dynamac import table_iii_coordination


def _option_values(command, option):
    index = command.index(option)
    return command[index + 1]


def _inside(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def test_six_cell_plan_has_exact_formal_identity_and_paths():
    assert len(launch.FORMAL_CELLS) == 6
    assert len({cell.cell_id for cell in launch.FORMAL_CELLS}) == 6

    for cell in launch.FORMAL_CELLS:
        command = cell.command(Path("/sim/python3.8"), Path("/policy/python3.10"))
        assert _option_values(command, "--episodes") == "200"
        assert _option_values(command, "--seed") == "2608000000"
        assert _option_values(command, "--horizon") == "1000"
        assert _option_values(command, "--eval-set-id") == "rlbench_eval_v2"
        assert _option_values(command, "--release") == "v4"
        assert _option_values(command, "--max-primary-action-attempts") == "3"
        assert "--record-v4-evaluation-videos" in command
        assert "--headless" in command
        assert _inside(cell.result, launch.RESULTS_ROOT)
        assert _inside(cell.video_cell, launch.VIDEO_ROOT)
        assert not _inside(cell.result, launch.INTEGRATION_ROOT / "results" / "v3")


def test_direct_and_coordination_routes_use_the_correct_v4_model_roots():
    direct = launch.FORMAL_CELLS[:4]
    coordination = launch.FORMAL_CELLS[4:]

    assert all(
        cell.evaluator_module.endswith("direct_evaluate") for cell in direct
    )
    assert all(cell.models_dir == launch.MODELS_ROOT for cell in direct)
    assert all(
        cell.evaluator_module.endswith("table_iii_coordination")
        for cell in coordination
    )
    assert all(
        cell.models_dir == launch.MODELS_ROOT / "table_iii"
        for cell in coordination
    )
    assert coordination[0].result.name.startswith(
        "coordination_hand_left_v4_cartesian_tick235_"
    )
    assert coordination[1].result.name.startswith(
        "coordination_hand_right_v4_cartesian_tick235_"
    )


def test_every_generated_evaluator_command_is_accepted_by_its_real_cli():
    sim_python = Path("/sim/python3.8")
    policy_python = Path("/policy/python3.10")
    for cell in launch.FORMAL_CELLS:
        command_arguments = list(cell.command(sim_python, policy_python)[3:])
        if cell.evaluator_module.endswith("direct_evaluate"):
            parsed = direct_evaluate.build_parser().parse_args(command_arguments)
        else:
            parsed = table_iii_coordination.build_parser().parse_args(
                command_arguments
            )
        assert parsed.release == "v4"
        assert parsed.eval_set_id == "rlbench_eval_v2"
        assert parsed.episodes == 200
        assert parsed.seed == 2_608_000_000
        assert parsed.horizon == 1_000
        assert parsed.record_v4_evaluation_videos is True
        assert parsed.output == cell.result


def test_default_plan_is_read_only_and_contains_six_isolated_launches(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan mode must not start subprocesses")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    assert launch.main([]) == 0
    output = capsys.readouterr().out
    assert "PLAN ONLY; no simulator is started" in output
    assert output.count("CUDA_VISIBLE_DEVICES=") == 6
    assert "--record-v4-evaluation-videos" in output
    assert "rlbench_eval_v2" in output


def test_regular_executable_discovers_bare_command_on_path(tmp_path, monkeypatch):
    executable = tmp_path / "python3.10"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(
        launch.shutil,
        "which",
        lambda command: str(executable) if command == "python3.10" else None,
    )

    assert (
        launch._regular_executable(Path("python3.10"), "policy Python")
        == executable
    )


def test_abort_or_partial_video_state_is_never_treated_as_completed(
    tmp_path,
    monkeypatch,
):
    result = tmp_path / "result.json"
    video_cell = tmp_path / "videos"
    cell = launch.FormalCell(
        name="fixture",
        task="fixture",
        scenario="static",
        evaluator_module="fixture.module",
        evaluator_arguments=(),
        models_dir=tmp_path / "models",
        result=result,
        video_cell=video_cell,
    )
    assert launch.cell_state(cell) == "PENDING"

    video_cell.mkdir()
    (video_cell / "episode_000.mp4").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="video artifacts exist without"):
        launch.cell_state(cell)

    (video_cell / "episode_000.mp4").unlink()
    result.write_text("{}", encoding="utf-8")
    admitted = []
    monkeypatch.setattr(
        launch,
        "_validate_completed_result",
        lambda value, identity: admitted.append((value.name, identity)),
    )
    canonical = {"seal": "fixture"}
    assert launch.cell_state(cell, canonical) == "COMPLETED_VALIDATED"
    assert admitted == [("fixture", canonical)]


def test_completed_result_validator_receives_the_preflight_seal_identity(
    monkeypatch,
):
    from integrations.rlbench.rlbench_dynamac import v4_partial_report

    cell = launch.FORMAL_CELLS[0]
    canonical = {"manifest_sha256": "a" * 64}
    seen = []

    def validate(path, specification, results_root, *, canonical_eval_identity):
        seen.append(
            (path, specification.cell_id, results_root, canonical_eval_identity)
        )

    monkeypatch.setattr(v4_partial_report, "_validate_v4_result", validate)
    launch._validate_completed_result(cell, canonical)

    assert seen == [(cell.result, cell.cell_id, launch.RESULTS_ROOT, canonical)]


def test_nothing_to_run_summary_records_exact_eval_and_model_release_identity(
    monkeypatch,
):
    states = {cell.name: "COMPLETED_VALIDATED" for cell in launch.FORMAL_CELLS}
    evaluation_set = {
        "evaluation_set_id": "rlbench_eval_v2",
        "manifest_sha256": "a" * 64,
        "manifest_fingerprint": "b" * 64,
        "spec_sha256": "c" * 64,
        "spec_fingerprint": "d" * 64,
        "selected_batches": {},
    }
    model_release = {
        "manifest_sha256": "e" * 64,
        "manifest_fingerprint": "f" * 64,
    }
    monkeypatch.setattr(
        launch,
        "_preflight_with_identity",
        lambda **_kwargs: (states, evaluation_set, model_release),
    )

    summary = launch.execute(
        sim_python=Path("/sim/python3.8"),
        policy_python=Path("/policy/python3.10"),
        xvfb_run=Path("/usr/bin/xvfb-run"),
        ffmpeg=Path("/usr/bin/ffmpeg"),
        gpus=(0, 1, 2, 3, 4, 5),
    )

    assert summary["status"] == "nothing_to_run"
    assert summary["evaluation_set"] is evaluation_set
    assert summary["model_release"] is model_release


def test_gpu_parser_requires_six_unique_indices():
    assert launch._parse_gpus("0,1,2,3,4,5") == (0, 1, 2, 3, 4, 5)
    with pytest.raises(Exception):
        launch._parse_gpus("0,1,2")
    with pytest.raises(Exception):
        launch._parse_gpus("0,1,2,3,4,4")
