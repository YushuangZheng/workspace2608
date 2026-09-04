from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evaluations.iclr2027.training.causal_gru import (  # noqa: E402
    CausalGRUClassifier,
    masked_binary_cross_entropy,
)


def test_gru_outputs_are_prefix_causal() -> None:
    torch.manual_seed(2608)
    model = CausalGRUClassifier(input_dim=5, hidden_dim=7).eval()
    features = torch.randn(2, 6, 5)

    full_logits, _ = model(features)
    prefix_logits, _ = model(features[:, :3])

    assert full_logits.shape == (2, 6)
    assert torch.equal(full_logits[:, :3], prefix_logits)
    assert model.checkpoint_metadata()["bidirectional"] is False


def test_masked_loss_excludes_right_padding() -> None:
    logits = torch.tensor([[0.0, 1.0, 99.0], [-1.0, 2.0, -99.0]])
    targets = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])

    loss = masked_binary_cross_entropy(logits, targets, mask)
    reference = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2],
        targets[:, :2],
    )

    assert torch.equal(loss, reference)
