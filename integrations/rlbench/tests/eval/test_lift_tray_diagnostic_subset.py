from argparse import Namespace
import importlib.util
import json
from pathlib import Path

from integrations.rlbench.rlbench_dynamac.eval import direct_evaluate
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DIAGNOSTIC_SCRIPT = (
    _REPOSITORY_ROOT
    / "evaluations"
    / "phase6_rlbench_integration"
    / "run_lift_tray_diagnostic_subset.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_lift_tray_diagnostic_subset", _DIAGNOSTIC_SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


def test_diagnostic_loader_slices_existing_sealed_plan(monkeypatch):
    seen = {}
    plans = [{"episode_index": index} for index in range(FIXED_EVAL_EPISODES)]

    def fake_loader(args):
        seen["seed"] = args.seed
        seen["episodes"] = args.episodes
        return {"schema": "sealed"}, {"plans": plans, "other": "preserved"}

    monkeypatch.setattr(direct_evaluate, "_load_fixed_motion_plans", fake_loader)
    args = Namespace(seed=GLOBAL_EVAL_SEED_START + 35, episodes=1)

    manifest, selected = diagnostic._diagnostic_loader(35)(args)

    assert seen == {
        "seed": GLOBAL_EVAL_SEED_START,
        "episodes": FIXED_EVAL_EPISODES,
    }
    assert args.seed == GLOBAL_EVAL_SEED_START + 35
    assert args.episodes == 1
    assert manifest == {"schema": "sealed"}
    assert selected == {
        "plans": [{"episode_index": 35}],
        "other": "preserved",
    }


def test_mark_diagnostic_disables_formal_and_paper_claims(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps(
            {
                "evaluation_protocol_id": "sealed-v2",
                "fixed_eval_set": {"formal_access": "complete"},
                "scenario_protocol": {"paper_comparable": True},
            }
        ),
        encoding="utf-8",
    )

    diagnostic._mark_diagnostic(output, 42)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnostic_subset"] == {
        "schema": "rlbench-fixed-eval-read-only-diagnostic-subset-v1",
        "episode_index": 42,
        "episode_seed": GLOBAL_EVAL_SEED_START + 42,
        "formal_result": False,
        "paper_comparable": False,
        "plan_regenerated": False,
    }
    assert payload["evaluation_protocol_id"] == "sealed-v2+diagnostic-subset-v1"
    assert (
        payload["fixed_eval_set"]["formal_access"]
        == "canonical_id_read_only_diagnostic_subset"
    )
    assert payload["scenario_protocol"]["paper_comparable"] is False
