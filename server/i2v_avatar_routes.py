"""HTTP routes for the optional Joyfox image-to-Avatar workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from aiohttp import web
from PIL import Image, ImageOps, UnidentifiedImageError

from server.grok_i2v import (
    MAX_CONCURRENCY,
    MAX_SUBMISSIONS_PER_SECOND,
    STANDBY_PROMPTS,
    api_base_url,
)
from server.i2v_avatar_manager import i2v_avatar_manager
from server.session_manager import session_manager
from utils.logger import logger


ROOT_DIR = Path(__file__).resolve().parents[1]
JOB_ROOT = ROOT_DIR / "data" / "i2v_jobs"
AVATAR_ROOT = ROOT_DIR / "data" / "avatars"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
STD_PROMPT_COUNT = 7
LITE_PROMPT_COUNT = 1
GENERATION_MODES = {"lite", "std", "pro"}


def _avatar_action_name(label: str) -> str:
    # Keep Chinese/Unicode letters and numbers, while preventing path separators
    # or punctuation from becoming part of the Avatar directory name.
    return re.sub(r"[^\w-]+", "_", str(label), flags=re.UNICODE).strip("_") or "待机"


def _build_avatar_specs(model: str, timestamp: str) -> list[dict]:
    return [
        {
            "index": index,
            "label": label,
            "prompt": prompt,
            "avatar_id": f"{model}_joyfox_{_avatar_action_name(label)}_{timestamp}",
            "batch_timestamp": timestamp,
        }
        for index, (prompt, label) in enumerate(STANDBY_PROMPTS, start=1)
    ]


def json_ok(data=None, *, status: int = 200) -> web.Response:
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps(body, ensure_ascii=False),
    )


def json_error(message: str, *, status: int = 400) -> web.Response:
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps({"code": -1, "msg": str(message)}, ensure_ascii=False),
    )


async def _save_uploaded_image(request: web.Request, destination: Path) -> str:
    if not request.content_type.startswith("multipart/"):
        raise ValueError("请使用 multipart/form-data 上传图片。")
    reader = await request.multipart()
    found = False
    total = 0
    generation_mode = "lite"
    with destination.open("wb") as output:
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "mode":
                generation_mode = (await part.text()).strip().lower()
                continue
            if part.name != "image":
                await part.release()
                continue
            if found:
                await part.release()
                continue
            found = True
            while True:
                chunk = await part.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError("图片不能超过 25 MB。")
                output.write(chunk)
    if not found or total == 0:
        raise ValueError("没有收到有效图片。")
    if generation_mode not in GENERATION_MODES:
        raise ValueError("生成模式必须是 lite、std 或 pro。")
    return generation_mode


def _prepare_720p_image(source: Path, destination: Path) -> tuple[int, int]:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("无法识别上传的图片，请使用 JPG、PNG 或 WebP。") from exc

    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError("图片尺寸过小，宽高至少需要 64 像素。")

    if width > height:
        max_width, max_height = 1280, 720
    elif height > width:
        max_width, max_height = 720, 1280
    else:
        max_width = max_height = 720
    scale = min(1.0, max_width / width, max_height / height)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    if (target_width, target_height) != image.size:
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=92, optimize=True, progressive=True)
    return image.size


async def create_i2v_avatar_task(request: web.Request) -> web.Response:
    if i2v_avatar_manager.has_active_task():
        return json_error("已有数字人正在生成，请等待当前任务完成。", status=409)
    if session_manager.sessions:
        return json_error("请先断开当前数字人连接，再开始生成新形象。", status=409)

    opt = request.app.get("opt")
    model = str(getattr(opt, "model", "") or "")
    if model not in {"wav2lip", "musetalk"}:
        return json_error(f"当前启动模型 {model or 'unknown'} 暂不支持自动生成。")
    if not os.getenv("XAI_API_KEY", "").strip():
        return json_error("项目 .env 中未配置 XAI_API_KEY，请配置并重启服务。")
    task_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = JOB_ROOT / task_id
    upload_path = job_dir / "upload.bin"
    source_path = job_dir / "source_720p.jpg"
    job_dir.mkdir(parents=True, exist_ok=False)

    try:
        generation_mode = await _save_uploaded_image(request, upload_path)
        if generation_mode == "lite":
            prompt_limit = min(LITE_PROMPT_COUNT, len(STANDBY_PROMPTS))
        elif generation_mode == "std":
            prompt_limit = min(STD_PROMPT_COUNT, len(STANDBY_PROMPTS))
        else:
            prompt_limit = len(STANDBY_PROMPTS)
        avatar_specs = _build_avatar_specs(model, timestamp)[:prompt_limit]
        conflicting_ids = [
            spec["avatar_id"]
            for spec in avatar_specs
            if (AVATAR_ROOT / spec["avatar_id"]).exists()
        ]
        if conflicting_ids:
            raise FileExistsError("当前秒已存在同名数字人，请稍后一秒再试。")
        image_size = await asyncio.to_thread(_prepare_720p_image, upload_path, source_path)
        upload_path.unlink(missing_ok=True)
        task = i2v_avatar_manager.add_task(
            task_id=task_id,
            model=model,
            avatar_specs=avatar_specs,
            source_image=source_path,
            job_dir=job_dir,
            image_size=image_size,
        )
        return json_ok(task, status=202)
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.exception("Failed to create Joyfox image-to-Avatar task")
        status = 409 if "正在生成" in str(exc) or isinstance(exc, FileExistsError) else 400
        return json_error(str(exc), status=status)


async def get_i2v_avatar_task(request: web.Request) -> web.Response:
    task = i2v_avatar_manager.get_task(request.match_info.get("task_id", ""))
    if task is None:
        return json_error("生成任务不存在。", status=404)
    return json_ok(task)


async def get_active_i2v_avatar_task(request: web.Request) -> web.Response:
    return json_ok({"task": i2v_avatar_manager.get_active_task()})


async def get_i2v_avatar_config(request: web.Request) -> web.Response:
    opt = request.app.get("opt")
    model = str(getattr(opt, "model", "") or "")
    return json_ok(
        {
            "model": model,
            "supported": model in {"wav2lip", "musetalk"},
            "prompt_count": len(STANDBY_PROMPTS),
            "lite_prompt_count": min(LITE_PROMPT_COUNT, len(STANDBY_PROMPTS)),
            "std_prompt_count": min(STD_PROMPT_COUNT, len(STANDBY_PROMPTS)),
            "pro_prompt_count": len(STANDBY_PROMPTS),
            "max_concurrency": MAX_CONCURRENCY,
            "max_submissions_per_second": MAX_SUBMISSIONS_PER_SECOND,
            "api_configured": bool(os.getenv("XAI_API_KEY", "").strip()),
            "api_base_url": api_base_url(),
        }
    )


async def shutdown_i2v_avatar_manager(app: web.Application) -> None:
    await asyncio.to_thread(i2v_avatar_manager.shutdown)


def setup_i2v_avatar_routes(app: web.Application) -> None:
    app.router.add_post("/api/i2v-avatar/task", create_i2v_avatar_task)
    app.router.add_get("/api/i2v-avatar/task/{task_id}", get_i2v_avatar_task)
    app.router.add_get("/api/i2v-avatar/active", get_active_i2v_avatar_task)
    app.router.add_get("/api/i2v-avatar/config", get_i2v_avatar_config)
    app.on_shutdown.append(shutdown_i2v_avatar_manager)
