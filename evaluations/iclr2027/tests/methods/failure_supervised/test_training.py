from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evaluations.iclr2027.methods.failure_supervised.model import (  # noqa: E402
    CausalGRUClassifier,
)
from evaluations.iclr2027.methods.failure_supervised.training import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
    train_step,
)


def test_training_step_and_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(2608)
    model = CausalGRUClassifier(input_dim=4, hidden_dim=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    features = torch.randn(2, 5, 4)
    targets = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]])
    valid_mask = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])

    metrics = train_step(
        model,
        optimizer,
        features,
        targets,
        valid_mask,
        max_gradient_norm=1.0,
    )
    assert metrics.loss > 0.0
    assert metrics.valid_cycles == 7
    assert metrics.gradient_norm >= 0.0

    checkpoint = save_training_checkpoint(
        tmp_path / "m4.pt",
        model,
        optimizer=optimizer,
        training_step=1,
        metadata={"seed": 2608, "feature_schema": "not-yet-frozen"},
    )
    restored, payload = load_training_checkpoint(checkpoint)

    assert restored.checkpoint_metadata() == model.checkpoint_metadata()
    assert payload["training_step"] == 1
    assert payload["metadata"]["seed"] == 2608
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[name])
