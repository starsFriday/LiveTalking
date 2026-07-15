#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$ROOT_DIR/MiniCPM-o-Demo"
MODEL_DIR="$ROOT_DIR/models/MiniCPM-o-4_5"
RUNTIME_DIR="$ROOT_DIR/logs/minicpmo"
SUPERVISOR_PIDFILE="$RUNTIME_DIR/supervisor.pid"

if ! command -v python >/dev/null 2>&1; then
  echo "找不到 Python，请先执行: conda activate livetalking-minicpm" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "缺少 setsid，无法可靠管理 MiniCPM 子进程。" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "livetalking-minicpm" ]]; then
  echo "当前 Conda 环境: ${CONDA_DEFAULT_ENV:-未激活}；请先执行: conda activate livetalking-minicpm" >&2
  exit 1
fi
if [[ ! -f "$DEMO_DIR/gateway.py" ]]; then
  echo "缺少官方推理仓库: $DEMO_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "缺少模型权重: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR" "$DEMO_DIR/data" "$DEMO_DIR/tmp"
if [[ ! -f "$DEMO_DIR/config.json" ]]; then
  cp "$DEMO_DIR/config.example.json" "$DEMO_DIR/config.json"
fi

is_running() {
  local pid
  [[ -s "$1" ]] || return 1
  read -r pid <"$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

stop_managed_processes() {
  local name pidfile pid port
  local -a pids=()

  for name in gateway worker backend; do
    pidfile="$RUNTIME_DIR/$name.pid"
    if is_running "$pidfile"; then
      read -r pid <"$pidfile"
      pids+=("$pid")
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
    local alive=0
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    if command -v fuser >/dev/null; then
      for port in 8006 22400 22500; do
        fuser "$port/tcp" >/dev/null 2>&1 && alive=1
      done
    fi
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
  rm -f "$RUNTIME_DIR/backend.pid" "$RUNTIME_DIR/worker.pid" "$RUNTIME_DIR/gateway.pid"
}

stop_previous_supervisor() {
  local pid args
  if is_running "$SUPERVISOR_PIDFILE"; then
    read -r pid <"$SUPERVISOR_PIDFILE"
    if [[ "$pid" != "$$" ]]; then
      args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if [[ "$args" == *start_minicpmo.sh* ]]; then
        echo "发现旧的 MiniCPM 启动进程 (PID: $pid)，正在停止……"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 100); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.1
        done
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "$SUPERVISOR_PIDFILE"
}

TAIL_PID=""
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
  echo
  echo "正在停止 MiniCPM-o API……"
  stop_managed_processes
  if [[ -f "$SUPERVISOR_PIDFILE" ]] && [[ "$(cat "$SUPERVISOR_PIDFILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$SUPERVISOR_PIDFILE"
  fi
  echo "MiniCPM-o API 已停止"
  return "$exit_code"
}

handle_interrupt() {
  echo
  echo "收到 Ctrl+C，正在关闭 MiniCPM-o API……"
  exit 130
}

handle_terminate() {
  echo
  echo "收到停止信号，正在关闭 MiniCPM-o API……"
  exit 143
}

trap cleanup EXIT
trap handle_interrupt INT
trap handle_terminate TERM

stop_previous_supervisor
echo "$$" >"$SUPERVISOR_PIDFILE"
echo "启动前清理旧的 MiniCPM backend / worker / gateway……"
stop_managed_processes

: >"$RUNTIME_DIR/backend.log"
: >"$RUNTIME_DIR/worker.log"
: >"$RUNTIME_DIR/gateway.log"

cd "$DEMO_DIR"

setsid python -m py_backend.server \
  --host 127.0.0.1 --port 22500 --gpu-id 0 \
  --model-path "$MODEL_DIR" \
  >"$RUNTIME_DIR/backend.log" 2>&1 &
echo $! >"$RUNTIME_DIR/backend.pid"

tail --pid=$$ -n +1 -F \
  "$RUNTIME_DIR/backend.log" \
  "$RUNTIME_DIR/worker.log" \
  "$RUNTIME_DIR/gateway.log" &
TAIL_PID=$!

echo "等待 MiniCPM backend 加载模型……"
for _ in $(seq 1 300); do
  if curl -sf http://127.0.0.1:22500/health >/dev/null; then
    break
  fi
  if ! is_running "$RUNTIME_DIR/backend.pid"; then
    echo "MiniCPM backend 启动失败，请查看 $RUNTIME_DIR/backend.log" >&2
    exit 1
  fi
  sleep 2
done
curl -sf http://127.0.0.1:22500/health >/dev/null || {
  echo "MiniCPM backend 600 秒内未就绪" >&2
  exit 1
}

setsid python worker.py \
  --host 127.0.0.1 --port 22400 --gpu-id 0 \
  --backend-server-url http://127.0.0.1:22500 \
  >"$RUNTIME_DIR/worker.log" 2>&1 &
echo $! >"$RUNTIME_DIR/worker.pid"

for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:22400/health >/dev/null && break
  if ! is_running "$RUNTIME_DIR/worker.pid"; then
    echo "MiniCPM worker 启动失败，请查看 $RUNTIME_DIR/worker.log" >&2
    exit 1
  fi
  sleep 1
done
curl -sf http://127.0.0.1:22400/health >/dev/null || {
  echo "MiniCPM worker 30 秒内未就绪，请查看 $RUNTIME_DIR/worker.log" >&2
  exit 1
}

setsid python gateway.py \
  --host 127.0.0.1 --port 8006 --http \
  --workers 127.0.0.1:22400 \
  >"$RUNTIME_DIR/gateway.log" 2>&1 &
echo $! >"$RUNTIME_DIR/gateway.pid"

for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:8006/health >/dev/null && break
  sleep 1
done

curl -sf http://127.0.0.1:8006/health >/dev/null || {
  echo "MiniCPM Gateway 启动失败，请查看 $RUNTIME_DIR/gateway.log" >&2
  exit 1
}

echo "MiniCPM-o Realtime API 已就绪: ws://127.0.0.1:8006/v1/realtime?mode=video"
echo "健康检查: http://127.0.0.1:8006/health"
echo "MiniCPM 三路日志正在当前终端显示；按 Ctrl+C 全部停止。"

while true; do
  for name in backend worker gateway; do
    if ! is_running "$RUNTIME_DIR/$name.pid"; then
      echo "MiniCPM $name 进程异常退出" >&2
      exit 1
    fi
  done
  sleep 2
done
