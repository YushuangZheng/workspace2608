#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 1)) || [[ -z "$1" ]]; then
  echo 'usage: tmux_session_has_live_pane.sh SESSION' >&2
  exit 2
fi

session=$1
tmux has-session -t "$session" 2>/dev/null || exit 1

while read -r actual_session pane_dead pane_pid; do
  if [[ "$actual_session" == "$session" && "$pane_dead" == '0' \
      && "$pane_pid" =~ ^[1-9][0-9]*$ ]] && \
      kill -0 "$pane_pid" 2>/dev/null; then
    exit 0
  fi
done < <(
  tmux list-panes -t "$session" \
    -F '#{session_name} #{pane_dead} #{pane_pid}' 2>/dev/null
)

exit 1
