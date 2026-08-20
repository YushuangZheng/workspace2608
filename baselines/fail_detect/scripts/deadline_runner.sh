#!/usr/bin/env bash
set -euo pipefail

status_script=""
status_file=""
deadline_marker=""
duration="24h"
kill_after="5m"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status-script) status_script="$2"; shift 2 ;;
    --status-file) status_file="$2"; shift 2 ;;
    --deadline-marker) deadline_marker="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --kill-after) kill_after="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$status_script" || -z "$status_file" || -z "$deadline_marker" || $# -eq 0 ]]; then
  echo "deadline runner requires status paths, marker, and a command" >&2
  exit 2
fi

mkdir -p "$(dirname "$deadline_marker")"
rm -f "$deadline_marker"
set +e
timeout --signal=TERM --kill-after="$kill_after" "$duration" "$@"
rc=$?
set -e

if [[ "$rc" -eq 124 || "$rc" -eq 137 || "$rc" -eq 143 || -e "$deadline_marker" ]]; then
  : >"$deadline_marker"
  python3 "$status_script" "$status_file" update \
    --state stopped --stage deadline --detail "post-gate deadline reached; timeout exit=$rc"
  exit 124
fi
if [[ "$rc" -eq 0 ]]; then
  python3 "$status_script" "$status_file" update \
    --state complete --stage complete --detail "bounded 50+50 evaluation completed"
  exit 0
fi
python3 "$status_script" "$status_file" update \
  --state stopped --stage failed --detail "pipeline exited with code $rc"
exit "$rc"
