#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline_dir="$(cd "$script_dir/.." && pwd)"
upstream_dir="$baseline_dir/upstream"

if [[ ! -d "$upstream_dir/embodichain" ]]; then
  echo "AgentChord checkout not found: $upstream_dir" >&2
  exit 2
fi

case "${1-}" in
  ""|--llm) ;;
  *)
    echo "Usage: $0 [--llm]" >&2
    exit 2
    ;;
esac

cd "$upstream_dir"

python -m compileall -q embodichain

python - "$upstream_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
configs = sorted((root / "configs").rglob("*.json"))
if not configs:
    raise SystemExit("No AgentChord JSON configurations found")
for path in configs:
    with path.open(encoding="utf-8") as handle:
        json.load(handle)
print(f"Parsed {len(configs)} AgentChord JSON configurations.")
PY

python - <<'PY'
import embodichain

print(f"Imported embodichain {embodichain.__version__}.")
PY

python -m pytest -q \
  tests/sim/agent/test_graph_spec.py \
  tests/sim/agent/test_agent_graph.py

if [[ "${1-}" != "--llm" ]]; then
  missing=()
  [[ -n "${OPENAI_API_KEY-}" ]] || missing+=(OPENAI_API_KEY)
  [[ -n "${LLM_URL-}" ]] || missing+=(LLM_URL)
  if ((${#missing[@]})); then
    echo "LLM connectivity not attempted; missing: ${missing[*]}."
  else
    echo "LLM connectivity not attempted by default; pass --llm explicitly."
  fi
  exit 0
fi

missing=()
[[ -n "${OPENAI_API_KEY-}" ]] || missing+=(OPENAI_API_KEY)
[[ -n "${LLM_URL-}" ]] || missing+=(LLM_URL)
if ((${#missing[@]})); then
  echo "Cannot run LLM health check; missing: ${missing[*]}." >&2
  exit 2
fi

python embodichain/agents/hierarchy/llm.py
