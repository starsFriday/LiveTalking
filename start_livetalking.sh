#!/usr/bin/env bash
set -euo pipefail

# ==================== 参数区 ====================
MODEL="musetalk"       # 可选：musetalk / wav2lip
AVATAR_ID="musetalk_222wave"
BATCH_SIZE="4"
# ========================================================
# MODEL="wav2lip"
# AVATAR_ID="wav2lip256_avatar1"
# BATCH_SIZE="8"
# ========================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURN_CONFIG="$ROOT_DIR/turnserver.conf"
MINICPM_HEALTH_URL="${MINICPM_HEALTH_URL:-http://127.0.0.1:8006/health}"

cd "$ROOT_DIR"
mkdir -p logs
LIVETALKING_PIDFILE="$ROOT_DIR/logs/livetalking.pid"

if ! command -v python >/dev/null 2>&1; then
  echo "找不到 Python，请先执行: conda activate livetalking" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "缺少 setsid，无法可靠管理前台服务进程。" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "livetalking" ]]; then
  echo "当前 Conda 环境: ${CONDA_DEFAULT_ENV:-未激活}；请先执行: conda activate livetalking" >&2
  exit 1
fi

stop_previous_livetalking() {
  local pid args
  local -a pids=()

  if [[ -s "$LIVETALKING_PIDFILE" ]]; then
    read -r pid <"$LIVETALKING_PIDFILE"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      [[ "$args" == *app.py* && "$args" == *"--transport webrtc"* ]] && pids+=("$pid")
    fi
  fi
  while IFS= read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] && pids+=("$pid")
  done < <(pgrep -f '[p]ython.*app[.]py.*--transport webrtc' || true)

  if [[ ${#pids[@]} -eq 0 ]]; then
    rm -f "$LIVETALKING_PIDFILE"
    return
  fi

  echo "发现旧的 LiveTalking 进程，正在停止后前台重启……"
  kill "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 100); do
    local alive=0
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    [[ "$alive" -eq 0 ]] && break
    sleep 0.1
  done
  kill -9 "${pids[@]}" 2>/dev/null || true
  rm -f "$LIVETALKING_PIDFILE"
}

stop_previous_livetalking

APP_MODE_ARGS=()
if curl -fsS --max-time 3 "$MINICPM_HEALTH_URL" >/dev/null; then
  echo "启动模式: MiniCPM-o 实时音视频对话"
else
  echo "MiniCPM-o API 不可用，自动降级为传统 LLM + EdgeTTS 模式" >&2
  APP_MODE_ARGS+=(--no-minicpmo_enabled --tts edgetts --REF_FILE zh-CN-YunxiaNeural)
fi

if command -v turnserver >/dev/null && ! pgrep -f "[t]urnserver.*$TURN_CONFIG" >/dev/null; then
  turnserver -c "$TURN_CONFIG" --daemon
fi

echo "LiveTalking 将在当前终端前台运行；按 Ctrl+C 停止。"
echo "访问地址: http://127.0.0.1:8010/index.html?v=minicpm-turn-tcp-v1"
echo "通过 VS Code/SSH 转发访问时，还需转发 3478/TCP（WebRTC TURN）。"

APP_PID=""
cleanup_livetalking() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    echo "正在停止 LiveTalking……"
    kill -TERM -- "-$APP_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$APP_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$APP_PID" 2>/dev/null || true
  fi
  if [[ -f "$LIVETALKING_PIDFILE" ]] \
    && [[ "$(cat "$LIVETALKING_PIDFILE" 2>/dev/null || true)" == "$APP_PID" ]]; then
    rm -f "$LIVETALKING_PIDFILE"
  fi
  [[ -n "$APP_PID" ]] && echo "LiveTalking 已停止"
  return "$exit_code"
}

handle_livetalking_interrupt() {
  echo
  echo "收到 Ctrl+C，正在关闭 LiveTalking……"
  exit 130
}

handle_livetalking_terminate() {
  echo
  echo "收到停止信号，正在关闭 LiveTalking……"
  exit 143
}

trap cleanup_livetalking EXIT
trap handle_livetalking_interrupt INT
trap handle_livetalking_terminate TERM

setsid python app.py --transport webrtc \
  --model "$MODEL" \
  --avatar_id "$AVATAR_ID" \
  --batch_size "$BATCH_SIZE" \
  "${APP_MODE_ARGS[@]}" \
  "$@" &
APP_PID=$!
echo "$APP_PID" >"$LIVETALKING_PIDFILE"
wait "$APP_PID"
