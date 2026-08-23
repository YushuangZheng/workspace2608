#!/usr/bin/env python3
"""Report live SPR/Guardian tmux workloads that block AHA launch."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


BLOCKING_PREFIXES = ("dynamac_spr", "dynamac_guardian")
NO_SERVER_ERRORS = ("no server running", "failed to connect to server")


def live_blocking_sessions(
    rows: Iterable[str], proc_root: Path = Path("/proc")
) -> tuple[str, ...]:
    active: set[str] = set()
    for row in rows:
        fields = row.rstrip("\n").split("\t")
        if len(fields) != 3:
            raise ValueError(f"unexpected tmux pane row: {row!r}")
        session, pane_dead, pane_pid_text = fields
        if not session.startswith(BLOCKING_PREFIXES) or pane_dead != "0":
            continue
        try:
            pane_pid = int(pane_pid_text)
        except ValueError as exc:
            raise ValueError(f"invalid tmux pane PID: {pane_pid_text!r}") from exc
        if (proc_root / str(pane_pid)).exists():
            active.add(session)
    return tuple(sorted(active))


def query_tmux() -> tuple[str, ...]:
    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-a",
            "-F",
            "#{session_name}\t#{pane_dead}\t#{pane_pid}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip().lower()
        if any(fragment in error for fragment in NO_SERVER_ERRORS):
            return ()
        raise RuntimeError(
            f"tmux list-panes failed with rc={result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return live_blocking_sessions(result.stdout.splitlines())


def main() -> int:
    for session in query_tmux():
        print(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
