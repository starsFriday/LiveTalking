#!/usr/bin/env bash
set -euo pipefail

PIDS="$(pgrep -f '[p]ython.*app.py.*--transport webrtc' || true)"
if [[ -z "$PIDS" ]]; then
  echo "LiveTalking 当前没有运行"
  exit 0
fi

kill $PIDS
echo "LiveTalking 已停止 (PID: $PIDS)"
echo "TURN 服务保持运行"
