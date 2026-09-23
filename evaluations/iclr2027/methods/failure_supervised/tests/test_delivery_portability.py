"""Scoped-copy and byte-identity checks for B's delivery tooling."""

import hashlib
from pathlib import Path

import pytest

from evaluations.iclr2027.methods.failure_supervised.tools import (
    check_portable_delivery as portable,
)
from evaluations.iclr2027.methods.failure_supervised.tools import prepare_handoff as handoff


def test_documented_repository_relative_manifest_argument(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(handoff, "ROOT", root)
    relative = Path("evaluations/iclr2027/results/b_delivery/B_TO_A_DELIVERY.json")
    assert handoff.resolve_manifest_path(relative) == root / relative
    assert handoff.resolve_manifest_path(root / relative) == root / relative
    with pytest.raises(ValueError, match="parents"):
        handoff.resolve_manifest_path(Path("../outside.json"))
    with pytest.raises(ValueError, match="inside"):
        handoff.resolve_manifest_path(tmp_path / "outside.json")


def test_payload_identity_ignores_only_its_own_report():
    file = {
        "path": "evaluations/iclr2027/methods/fail_detect/runtime.py",
        "bytes": 1,
        "sha256": "a" * 64,
    }
    report = {
        "path": str(handoff.PORTABILITY.relative_to(handoff.ROOT)),
        "bytes": 2,
        "sha256": "b" * 64,
    }
    assert handoff.payload_identity([file]) == handoff.payload_identity([report, file])
    assert handoff.payload_identity([file]) != handoff.payload_identity(
        [{**file, "sha256": "c" * 64}]
    )


def test_scoped_copy_rejects_tampering_and_path_escape(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("x")
    monkeypatch.setattr(portable, "ROOT", source)
    entry = {"path": "file.txt", "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
    target = tmp_path / "target"
    portable.copy_entry(entry, target)
    assert (target / "file.txt").read_text() == "x"
    portable.copy_entry(entry, target)
    with pytest.raises(ValueError, match="noncanonical"):
        portable.copy_entry({**entry, "path": "../file.txt"}, target)
    with pytest.raises(ValueError, match="identity"):
        portable.copy_entry({**entry, "sha256": "0" * 64}, target)
    (target / "file.txt").write_text("y")
    with pytest.raises(ValueError, match="disagree"):
        portable.copy_entry(entry, target)
