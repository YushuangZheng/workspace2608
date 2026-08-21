from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac.eval import direct_evaluate
from integrations.rlbench.rlbench_dynamac.eval import table_iii_coordination
from integrations.rlbench.rlbench_dynamac.eval import unimanual_evaluate
from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch as launch


def _option_values(command, option):
    index = command.index(option)
    return command[index + 1]


def _inside(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def test_22_cell_plan_has_exact_formal_identity_and_paths():
    assert len(launch.FORMAL_CELLS) == 22
    assert len({cell.cell_id for cell in launch.FORMAL_CELLS}) == 22

    for cell in launch.FORMAL_CELLS:
        command = cell.command(Path("/sim/python3.8"), Path("/policy/python3.10"))
        assert _option_values(command, "--episodes") == "200"
        assert _option_values(command, "--seed") == "2608000000"
        assert _option_values(command, "--horizon") == "1000"
        assert _option_values(command, "--eval-set-id") == "rlbench_eval_v2"
        assert _option_values(command, "--release") == "v4"
        assert _option_values(command, "--max-primary-action-attempts") == "1"
        assert "--record-v4-evaluation-videos" not in command
        assert "--headless" in command
        assert _inside(cell.result, launch.RESULTS_ROOT)
        assert not _inside(cell.result, launch.INTEGRATION_ROOT / "results" / "v3")


def test_formal_launch_and_current_report_share_the_exact_22_cell_matrix():
    from integrations.rlbench.rlbench_dynamac.report import v4_partial_report

    launched = {
        cell.cell_id: cell.paper_success_rate for cell in launch.FORMAL_CELLS
    }
    reported = {
        cell.cell_id: cell.paper_target
        for cell in v4_partial_report.TARGET_CELLS
    }

    assert len(launched) == 22
    assert reported == launched


def test_direct_and_coordination_routes_use_the_correct_v4_model_roots():
    direct = [
        cell for cell in launch.FORMAL_CELLS
        if cell.evaluator_module.endswith("direct_evaluate")
    ]
    unimanual = [
        cell for cell in launch.FORMAL_CELLS
        if cell.evaluator_module.endswith("unimanual_evaluate")
    ]
    coordination = [
        cell for cell in launch.FORMAL_CELLS
        if cell.evaluator_module.endswith("table_iii_coordination")
    ]

    assert len(direct) == 8
    assert len(unimanual) == 12
    assert len(coordination) == 2
    assert all(cell.models_dir == launch.MODELS_ROOT for cell in direct)
    assert all(cell.models_dir == launch.MODELS_ROOT for cell in unimanual)
    assert all(
        cell.models_dir == launch.MODELS_ROOT / "table_iii"
        for cell in coordination
    )
    assert coordination[0].result.name.startswith(
        "coordination_hand_left_v4_smooth_clock_tick235_"
    )
    assert coordination[1].result.name.startswith(
        "coordination_hand_right_v4_smooth_clock_tick235_"
    )


def test_every_generated_evaluator_command_is_accepted_by_its_real_cli():
    sim_python = Path("/sim/python3.8")
    policy_python = Path("/policy/python3.10")
    for cell in launch.FORMAL_CELLS:
        command_arguments = list(cell.command(sim_python, policy_python)[3:])
        if cell.evaluator_module.endswith("direct_evaluate"):
            parsed = direct_evaluate.build_parser().parse_args(command_arguments)
        elif cell.evaluator_module.endswith("unimanual_evaluate"):
            parsed = unimanual_evaluate.build_parser().parse_args(command_arguments)
        else:
            parsed = table_iii_coordination.build_parser().parse_args(
                command_arguments
            )
        assert parsed.release == "v4"
        assert parsed.eval_set_id == "rlbench_eval_v2"
        assert parsed.episodes == 200
        assert parsed.seed == 2_608_000_000
        assert parsed.horizon == 1_000
        assert parsed.record_v4_evaluation_videos is False
        assert parsed.output == cell.result


def test_default_plan_is_read_only_and_contains_22_queued_launches(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan mode must not start subprocesses")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    assert launch.main([]) == 0
    output = capsys.readouterr().out
    assert "PLAN ONLY; no simulator is started" in output
    assert output.count("CUDA_VISIBLE_DEVICES=") == 22
    assert "--record-v4-evaluation-videos" not in output
    assert "GPU-lane=7 GPU=7" in output
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


def test_replay_artifacts_do_not_change_formal_result_state(
    tmp_path,
    monkeypatch,
):
    result = tmp_path / "result.json"
    cell = launch.FormalCell(
        name="fixture",
        task="fixture",
        scenario="static",
        evaluator_module="fixture.module",
        evaluator_arguments=(),
        models_dir=tmp_path / "models",
        result=result,
        paper_success_rate=0.5,
    )
    assert launch.cell_state(cell) == "PENDING"
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


def test_cell_local_batch_change_is_pending_without_invalidating_other_cells(
    tmp_path,
    monkeypatch,
):
    from integrations.rlbench.rlbench_dynamac.report.v4_partial_report import (
        StaleEvaluationBatchError,
    )

    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    cell = launch.FormalCell(
        name="fixture",
        task="fixture",
        scenario="static",
        evaluator_module="fixture.module",
        evaluator_arguments=(),
        models_dir=tmp_path / "models",
        result=result,
        paper_success_rate=0.5,
    )

    def stale(*_args, **_kwargs):
        raise StaleEvaluationBatchError("fixture selected batch changed")

    monkeypatch.setattr(launch, "_validate_completed_result", stale)
    assert launch.cell_state(cell, {"seal": "fixture"}) == "PENDING"


def test_cell_state_still_fails_closed_on_malformed_present_result(
    tmp_path,
    monkeypatch,
):
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    cell = launch.FormalCell(
        name="fixture",
        task="fixture",
        scenario="static",
        evaluator_module="fixture.module",
        evaluator_arguments=(),
        models_dir=tmp_path / "models",
        result=result,
        paper_success_rate=0.5,
    )

    def malformed(*_args, **_kwargs):
        raise ValueError("malformed result")

    monkeypatch.setattr(launch, "_validate_completed_result", malformed)
    with pytest.raises(ValueError, match="malformed result"):
        launch.cell_state(cell, {"seal": "fixture"})


def test_stale_result_refresh_preserves_old_artifact_and_publishes_new_one(
    tmp_path,
    monkeypatch,
):
    results_root = tmp_path / "results" / "v4"
    monkeypatch.setattr(launch, "RESULTS_ROOT", results_root)
    target = results_root / "table_i" / "result.json"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")
    staged = tmp_path / "run" / "replacement_results" / "result.json"
    staged.parent.mkdir(parents=True)
    staged.write_text("new\n", encoding="utf-8")
    cell = launch.FormalCell(
        name="fixture",
        task="fixture",
        scenario="static",
        evaluator_module="fixture.module",
        evaluator_arguments=(),
        models_dir=tmp_path / "models",
        result=target,
        paper_success_rate=0.5,
    )

    archive = launch._publish_refreshed_result(
        cell=cell,
        staged_result=staged,
        run_root=tmp_path / "run",
    )

    assert target.read_text(encoding="utf-8") == "new\n"
    assert archive.read_text(encoding="utf-8") == "old\n"
    assert not staged.exists()


def test_completed_result_validator_receives_the_preflight_seal_identity(
    monkeypatch,
):
    from integrations.rlbench.rlbench_dynamac.report import v4_partial_report

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
        gpus=tuple(range(8)),
    )

    assert summary["status"] == "nothing_to_run"
    assert summary["evaluation_set"] is evaluation_set
    assert summary["model_release"] is model_release


def test_gpu_parser_accepts_reusable_unique_lanes():
    assert launch._parse_gpus("0,1,2,3,4,5,6,7") == tuple(range(8))
    assert launch._parse_gpus("0,1,2") == (0, 1, 2)
    with pytest.raises(Exception):
        launch._parse_gpus("0,1,2,3,4,4")
