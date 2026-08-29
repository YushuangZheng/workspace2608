from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac.eval import (
    direct_evaluate,
    unimanual_evaluate,
)
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DIAGNOSTIC_SCRIPT = (
    _REPOSITORY_ROOT
    / "evaluations"
    / "phase6_rlbench_integration"
    / "run_normal_diagnostic_subset.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_normal_diagnostic_subset", _DIAGNOSTIC_SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("stack_wine", unimanual_evaluate),
        ("bimanual_lift_tray", direct_evaluate),
    ],
)
def test_evaluator_dispatch_covers_uni_and_bimanual_tasks(task, expected):
    assert diagnostic._evaluator_for_task(task) is expected


def test_store_bottle_diagnostic_uses_frozen_v4_budgets():
    assert diagnostic._task_protocol_args("bimanual_put_bottle_in_fridge") == [
        "--scenario-max-attempts",
        "1000",
        "--final-settling-steps",
        "10",
    ]
    assert diagnostic._task_protocol_args("bimanual_sweep_to_dustpan") == []


def test_diagnostic_loader_slices_multiple_existing_sealed_plans(monkeypatch):
    seen = {}
    plans = [{"episode_index": index} for index in range(FIXED_EVAL_EPISODES)]

    def fake_loader(args):
        seen["seed"] = args.seed
        seen["episodes"] = args.episodes
        return {"schema": "sealed"}, {"plans": plans, "other": "preserved"}

    monkeypatch.setattr(unimanual_evaluate, "_load_fixed_motion_plans", fake_loader)
    args = Namespace(seed=GLOBAL_EVAL_SEED_START + 3, episodes=3)

    manifest, selected = diagnostic._diagnostic_loader(
        unimanual_evaluate,
        (3, 7, 11),
    )(args)

    assert seen == {
        "seed": GLOBAL_EVAL_SEED_START,
        "episodes": FIXED_EVAL_EPISODES,
    }
    assert args.seed == GLOBAL_EVAL_SEED_START + 3
    assert args.episodes == 3
    assert manifest == {"schema": "sealed"}
    assert selected == {
        "plans": [
            {"episode_index": 3},
            {"episode_index": 7},
            {"episode_index": 11},
        ],
        "other": "preserved",
    }


def test_episode_indices_reject_duplicates_and_out_of_range():
    with pytest.raises(ValueError, match="unique"):
        diagnostic._normalized_episode_indices((1, 1))
    with pytest.raises(ValueError, match="outside"):
        diagnostic._normalized_episode_indices((FIXED_EVAL_EPISODES,))


@pytest.mark.parametrize("protocol_key", ["scenario_protocol", "protocol"])
def test_mark_diagnostic_disables_formal_and_paper_claims(
    tmp_path,
    protocol_key,
):
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps(
            {
                "evaluation_protocol_id": "sealed-v2",
                "fixed_eval_set": {"formal_access": "complete"},
                protocol_key: {"paper_comparable": True},
                "paper_comparable": True,
            }
        ),
        encoding="utf-8",
    )

    diagnostic._mark_diagnostic(
        output,
        task="wipe_desk",
        episode_indices=(2, 5),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnostic_subset"] == {
        "schema": "rlbench-fixed-eval-read-only-normal-subset-v1",
        "task": "wipe_desk",
        "episode_indices": [2, 5],
        "episode_seeds": [GLOBAL_EVAL_SEED_START + 2, GLOBAL_EVAL_SEED_START + 5],
        "formal_result": False,
        "paper_comparable": False,
        "plan_regenerated": False,
    }
    assert payload["evaluation_protocol_id"] == "sealed-v2+normal-diagnostic-subset-v1"
    assert (
        payload["fixed_eval_set"]["formal_access"]
        == "canonical_id_read_only_normal_diagnostic_subset"
    )
    assert payload[protocol_key]["paper_comparable"] is False
    assert payload["paper_comparable"] is False
