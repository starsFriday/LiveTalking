#!/usr/bin/env sh
set -eu

# ==================== 配置区域 ====================
MODEL="musetalk"  # 可选：musetalk / wav2lip
SOURCE="./LTX-2_wave.mp4"
AVATAR_ID="musetalk_222wave"
# ========================================================

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "找不到 Python，请先执行: conda activate livetalking" >&2
  exit 1
fi

if [ "$MODEL" = "musetalk" ]; then
  python -m avatars.musetalk.genavatar \
    --file "$SOURCE" \
    --avatar_id "$AVATAR_ID" \
    --version v15 \
    --bbox_shift 0 \
    --extra_margin 10 \
    --parsing_mode jaw
elif [ "$MODEL" = "wav2lip" ]; then
  python -m avatars.wav2lip.genavatar \
    --video_path "$SOURCE" \
    --avatar_id "$AVATAR_ID" \
    --img_size 256 \
    --face_det_batch_size 4 \
    --pads 0 20 0 0
else
  echo "MODEL 只能填写 musetalk 或 wav2lip" >&2
  exit 2
fi
