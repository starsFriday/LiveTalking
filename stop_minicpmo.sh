#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/logs/minicpmo"
SUPERVISOR_PIDFILE="$RUNTIME_DIR/supervisor.pid"

if [[ -s "$SUPERVISOR_PIDFILE" ]]; then
  read -r supervisor_pid <"$SUPERVISOR_PIDFILE"
  supervisor_args="$(ps -p "$supervisor_pid" -o args= 2>/dev/null || true)"
  if [[ "$supervisor_pid" =~ ^[0-9]+$ ]] && [[ "$supervisor_args" == *start_minicpmo.sh* ]]; then
    kill "$supervisor_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$supervisor_pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -9 "$supervisor_pid" 2>/dev/null || true
  fi
fi

pids=()

for name in gateway worker backend; do
  pidfile="$RUNTIME_DIR/$name.pid"
  if [[ -s "$pidfile" ]]; then
    read -r pid <"$pidfile"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && pids+=("$pid")
  fi
done

while IFS= read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done \
  < <(pgrep -f '[p]ython.*-m py_backend[.]server.*--port 22500' || true)
while IFS= read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done \
  < <(pgrep -f '[p]ython.*worker[.]py.*--port 22400' || true)
while IFS= read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done \
  < <(pgrep -f '[p]ython.*gateway[.]py.*--port 8006' || true)

if [[ ${#pids[@]} -gt 0 ]]; then
  kill "${pids[@]}" 2>/dev/null || true
fi
if command -v fuser >/dev/null; then
  for port in 8006 22400 22500; do
    fuser -k -TERM "$port/tcp" >/dev/null 2>&1 || true
  done
fi

for _ in $(seq 1 50); do
  alive=0
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 0.1
done

if [[ ${#pids[@]} -gt 0 ]]; then
  kill -9 "${pids[@]}" 2>/dev/null || true
fi
if command -v fuser >/dev/null; then
  for port in 8006 22400 22500; do
    fuser -k -KILL "$port/tcp" >/dev/null 2>&1 || true
  done
fi

rm -f "$RUNTIME_DIR/backend.pid" "$RUNTIME_DIR/worker.pid" \
  "$RUNTIME_DIR/gateway.pid" "$SUPERVISOR_PIDFILE"

echo "MiniCPM-o API 已停止"
