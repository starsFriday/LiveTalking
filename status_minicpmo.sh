#!/usr/bin/env bash
set -euo pipefail

if curl -fsS --max-time 3 http://127.0.0.1:8006/health; then
  echo
  echo "MiniCPM-o API 正常"
  exit 0
fi

echo "MiniCPM-o API 未运行或尚未就绪" >&2
exit 1
