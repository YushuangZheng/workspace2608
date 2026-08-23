#!/usr/bin/env python3
"""Import and strict-load the real released logpZO module from a pinned checkout."""

import argparse
import json
import subprocess
from pathlib import Path

from logpzo_network import build_logpzo_network, load_logpzo_module


EXPECTED_COMMIT = "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.expected_commit:
        raise RuntimeError("upstream commit mismatch")
    for command in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        if subprocess.run(["git", "-C", str(upstream)] + command, check=False).returncode != 0:
            raise RuntimeError("pinned upstream has tracked changes")

    module = load_logpzo_module(upstream)
    source = build_logpzo_network(upstream, 20)
    target = build_logpzo_network(upstream, 20)
    result = target.load_state_dict(source.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("real logpZO module strict-load smoke failed")
    output = {
        "schema": "dynamac-fail-detect-logpzo-module-smoke-v1",
        "upstream_commit": actual_commit,
        "module": str(Path(module.__file__).resolve().relative_to(upstream)),
        "input_dimension": 20,
        "parameters": int(sum(parameter.numel() for parameter in source.parameters())),
        "strict_load": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
