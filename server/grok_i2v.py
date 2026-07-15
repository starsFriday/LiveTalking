"""Grok image-to-video batch generation used by the Joyfox avatar workflow."""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import mimetypes
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


DEFAULT_API_BASE_URL = "https://api.x.ai/v1"
MODEL = "grok-imagine-video-1.5"
DURATION = 5
RESOLUTION = "720p"
ASPECT_RATIO = "auto"
AVATAR_FPS = 25
MAX_CONCURRENCY = 20
MAX_SUBMISSIONS_PER_SECOND = 10
PRIMARY_HEAD_START_SECONDS = 2.0
PRIMARY_RACE_COPIES = 3
MAX_SUBMIT_RETRIES = 6
POLL_INTERVAL_SECONDS = 5.0
GENERATION_TIMEOUT_SECONDS = 20 * 60
HTTP_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 5 * 60

# Built-in standalone Avatar generation priority order.
# Concurrent jobs may finish out of order, but results always follow this index.
STANDBY_PROMPTS: tuple[tuple[str, str], ...] = (
    ("人物面向镜头自然站立，保持自然微笑，嘴巴自然闭合，双手保持原位，身体随呼吸轻微自然晃动，偶尔自然眨眼，人物和背景保持稳定，固定镜头，没有背景音乐，没有人说话", "自然待机"),
    ("人物面带微笑，微微点了点头，固定镜头，没有背景音乐，没有人说话", "点头"),
    ("人物面对镜头微微侧过头，一只手放在耳后做出倾听的姿势，表情好奇，眼睛亮晶晶的，固定镜头，没有背景音乐，没有人说话", "倾听好奇"),
    ("人物面带微笑，挥了挥手，在向大家打招呼，固定镜头，没有背景音乐，没有人说话", "打招呼"),
    ("人物站在原地向左看了看，又向右看了看，像是在等人，固定镜头，没有背景音乐，没有人说话", "左右观察"),
    ("人物微微歪了一下头，眼睛转了转，像是在想什么事情，然后恢复正脸，固定镜头，没有背景音乐，没有人说话", "歪头思考"),
    ("人物低头看了一眼自己的衣领，伸手轻轻整理了一下，然后抬起头来，固定镜头，没有背景音乐，没有人说话", "整理衣领"),
    ("人物抬手轻轻把额头前的头发撩到耳后，然后自然地放下手，固定镜头，没有背景音乐，没有人说话", "撩头发"),
    ("人物双手交叉抱在胸前，面带微笑地看着镜头，身体微微晃了一下，固定镜头，没有背景音乐，没有人说话", "抱臂微笑"),
    ("人物面对镜头耸了耸肩，撇着嘴露出一副无奈的表情，双手朝外摊开了一下，固定镜头，没有背景音乐，没有人说话", "耸肩无奈"),
    ("人物面对镜头双手合十放在胸前，微微歪着头，表情带着恳求和期待，眼神真诚地看着前方，固定镜头，没有背景音乐，没有人说话", "双手合十"),
    ("人物面对镜头低着头，重重叹了一口气，肩膀微微下沉，表情失落沮丧，固定镜头，没有背景音乐，没有人说话", "沮丧叹气"),
    ("人物面对镜头抬手拍了一下自己的额头，闭上眼睛露出懊恼的表情，嘴巴微微张开，固定镜头，没有背景音乐，没有人说话", "扶额懊恼"),
    ("人物面对镜头瞪大了眼睛，双手捂住嘴巴，露出一副震惊不敢相信的表情，固定镜头，没有背景音乐，没有人说话", "捂嘴惊讶"),
)

EventCallback = Callable[[dict[str, Any]], None]
ClipReadyCallback = Callable[[dict[str, Any]], None]


class SubmissionRateLimiter:
    """Keep API submissions below the account's request-per-second limit."""

    def __init__(self, requests_per_second: int = MAX_SUBMISSIONS_PER_SECOND) -> None:
        self.requests_per_second = max(1, requests_per_second)
        self._timestamps: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 1.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_second:
                    self._timestamps.append(now)
                    return
                wait_seconds = max(0.01, 1.0 - (now - self._timestamps[0]))
            time.sleep(wait_seconds)


def _emit(callback: EventCallback | None, event: str, **payload: Any) -> None:
    if callback:
        callback({"event": event, **payload})


def require_api_key() -> str:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 XAI_API_KEY，请在项目 .env 文件中设置后重启 LiveTalking。")
    return api_key


def api_base_url() -> str:
    return os.getenv("XAI_API_BASE_URL", DEFAULT_API_BASE_URL).strip().rstrip("/")


def image_to_data_uri(image_path: Path) -> str:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在：{image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    if not mime_type.startswith("image/"):
        raise ValueError(f"无法识别图片格式：{image_path.name}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_error(response: requests.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text.strip()
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(body.get("message") or body)
    return str(body or f"HTTP {response.status_code}")


def _checked_json(response: requests.Response) -> dict[str, Any]:
    if not response.ok:
        raise RuntimeError(
            f"xAI API 请求失败（HTTP {response.status_code}）：{_response_error(response)}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("xAI API 返回了非 JSON 响应。") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"xAI API 返回格式异常：{type(data).__name__}")
    return data


def _create_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "joyfox-grok-i2v/1.0",
        }
    )
    return session


def _submit_generation(
    session: requests.Session,
    *,
    image_data_uri: str,
    prompt: str,
    model: str,
    duration: int,
    resolution: str,
    aspect_ratio: str,
    rate_limiter: SubmissionRateLimiter,
    callback: EventCallback | None,
    index: int,
    label: str,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image": {"url": image_data_uri},
        "duration": duration,
        "resolution": resolution,
    }
    if aspect_ratio != "auto":
        payload["aspect_ratio"] = aspect_ratio
    response = None
    for attempt in range(MAX_SUBMIT_RETRIES):
        rate_limiter.acquire()
        response = session.post(
            f"{api_base_url()}/videos/generations",
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After", "").strip()
        try:
            delay = max(1.0, float(retry_after))
        except ValueError:
            delay = min(12.0, 1.5 * (2 ** attempt))
        _emit(
            callback,
            "clip_retry",
            index=index,
            label=label,
            attempt=attempt + 1,
            delay=delay,
        )
        response.close()
        time.sleep(delay)
    assert response is not None
    data = _checked_json(response)
    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError(f"xAI API 未返回 request_id：{data}")
    return str(request_id)


def _wait_for_video(
    session: requests.Session,
    request_id: str,
    *,
    timeout_seconds: int,
    status_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise concurrent.futures.CancelledError()
        response = session.get(
            f"{api_base_url()}/videos/{request_id}", timeout=HTTP_TIMEOUT_SECONDS
        )
        data = _checked_json(response)
        if cancel_event is not None and cancel_event.is_set():
            raise concurrent.futures.CancelledError()
        status = str(data.get("status", "unknown")).lower()
        if status != last_status:
            if status_callback:
                status_callback(status)
            last_status = status
        if status == "done":
            video = data.get("video")
            video_url = video.get("url") if isinstance(video, dict) else None
            if not video_url:
                raise RuntimeError("任务已完成，但响应中没有视频 URL。")
            return str(video_url)
        if status in {"failed", "expired"}:
            detail = data.get("error") or data.get("message") or data
            raise RuntimeError(f"视频生成任务 {status}：{detail}")
        if cancel_event is not None:
            if cancel_event.wait(POLL_INTERVAL_SECONDS):
                raise concurrent.futures.CancelledError()
        else:
            time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"等待视频生成超时（{timeout_seconds} 秒）")


def _download_video(
    session: requests.Session,
    video_url: str,
    output_path: Path,
    *,
    cancel_event: threading.Event | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".part.mp4")
    try:
        with session.get(video_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            if not response.ok:
                raise RuntimeError(
                    f"视频下载失败（HTTP {response.status_code}）：{_response_error(response)}"
                )
            with temporary_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if cancel_event is not None and cancel_event.is_set():
                        raise concurrent.futures.CancelledError()
                    if chunk:
                        output_file.write(chunk)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("下载得到的 MP4 文件为空。")
    return output_path


def _generate_one(
    *,
    index: int,
    prompt: str,
    label: str,
    output_path: Path,
    api_key: str,
    image_data_uri: str,
    model: str,
    duration: int,
    resolution: str,
    aspect_ratio: str,
    timeout_seconds: int,
    callback: EventCallback | None,
    rate_limiter: SubmissionRateLimiter,
    submitted_event: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    _emit(callback, "clip_start", index=index, label=label)
    with _create_session(api_key) as session:
        request_id = _submit_generation(
            session,
            image_data_uri=image_data_uri,
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            rate_limiter=rate_limiter,
            callback=callback,
            index=index,
            label=label,
        )
        if submitted_event is not None:
            submitted_event.set()
        _emit(callback, "clip_submitted", index=index, label=label)
        video_url = _wait_for_video(
            session,
            request_id,
            timeout_seconds=timeout_seconds,
            status_callback=lambda status: _emit(
                callback, "clip_status", index=index, label=label, status=status
            ),
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise concurrent.futures.CancelledError()
        saved_path = _download_video(
            session,
            video_url,
            output_path,
            cancel_event=cancel_event,
        )
    _emit(callback, "clip_done", index=index, label=label)
    return saved_path


def _concatenate_videos(video_paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".part.mp4")
    manifest_path = output_path.with_suffix(".concat.txt")
    manifest_path.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in video_paths),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-an",
                "-vf",
                f"fps={AVATAR_FPS}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary_path),
            ],
            capture_output=True,
            text=True,
            timeout=max(300, len(video_paths) * 60),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg 合并视频失败")
    finally:
        manifest_path.unlink(missing_ok=True)
    temporary_path.replace(output_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("合并后的视频为空。")
    return output_path


def generate_standby_clips(
    image_path: Path,
    clips_dir: Path,
    *,
    prompts: Iterable[tuple[str, str]] = STANDBY_PROMPTS,
    concurrency: int = MAX_CONCURRENCY,
    timeout_seconds: int = GENERATION_TIMEOUT_SECONDS,
    callback: EventCallback | None = None,
    clip_ready_callback: ClipReadyCallback | None = None,
) -> dict[str, Any]:
    """Generate independent action clips and return them in prompt priority order."""
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("预设提示词为空。")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"并发数必须在 1 到 {MAX_CONCURRENCY} 之间。")

    api_key = require_api_key()
    image_data_uri = image_to_data_uri(Path(image_path))
    clips_dir = Path(clips_dir).expanduser().resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, Path] = {}
    failures: list[dict[str, Any]] = []
    total = len(prompt_list)
    request_count = total + PRIMARY_RACE_COPIES - 1
    workers = min(
        MAX_CONCURRENCY,
        request_count,
        max(concurrency, PRIMARY_RACE_COPIES),
    )
    rate_limiter = SubmissionRateLimiter()
    _emit(
        callback,
        "batch_start",
        total=total,
        concurrency=workers,
        max_submissions_per_second=MAX_SUBMISSIONS_PER_SECOND,
    )

    state_lock = threading.Lock()
    completed_actions = 0

    def report_action_finished() -> None:
        nonlocal completed_actions
        with state_lock:
            completed_actions += 1
            completed = completed_actions
            succeeded = len(results)
            failed = len(failures)
        _emit(
            callback,
            "batch_progress",
            completed=completed,
            total=total,
            succeeded=succeeded,
            failed=failed,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[concurrent.futures.Future[Path], tuple[int, str]] = {}

        def submit_clip(
            index: int,
            prompt: str,
            label: str,
            *,
            output_path: Path | None = None,
            submitted_event: threading.Event | None = None,
            cancel_event: threading.Event | None = None,
            track_regular: bool = True,
        ) -> concurrent.futures.Future[Path]:
            future = executor.submit(
                _generate_one,
                index=index,
                prompt=prompt,
                label=label,
                output_path=output_path or clips_dir / f"clip_{index:02d}.mp4",
                api_key=api_key,
                image_data_uri=image_data_uri,
                model=MODEL,
                duration=DURATION,
                resolution=RESOLUTION,
                aspect_ratio=ASPECT_RATIO,
                timeout_seconds=timeout_seconds,
                callback=callback,
                rate_limiter=rate_limiter,
                submitted_event=submitted_event,
                cancel_event=cancel_event,
            )
            if track_regular:
                futures[future] = (index, label)
            return future

        def consume_ready_clip(index: int, label: str, path: Path) -> None:
            if not clip_ready_callback:
                return
            try:
                clip_ready_callback({
                    "index": index,
                    "label": label,
                    "prompt": prompt_list[index - 1][0],
                    "path": str(path),
                })
            except Exception as exc:
                _emit(
                    callback,
                    "clip_consumer_failed",
                    index=index,
                    label=label,
                    error=str(exc),
                )

        # Race three independent xAI requests for the top-priority action. Once
        # all three have either been accepted or failed during submission, keep
        # a two-second head start before releasing actions #2..N.
        primary_prompt, primary_label = prompt_list[0]
        primary_cancel = threading.Event()
        primary_submissions: list[
            tuple[int, threading.Event, concurrent.futures.Future[Path]]
        ] = []
        for candidate in range(1, PRIMARY_RACE_COPIES + 1):
            submitted_event = threading.Event()
            candidate_path = clips_dir / f"clip_01_candidate_{candidate}.mp4"
            candidate_path.unlink(missing_ok=True)
            candidate_path.with_suffix(".part.mp4").unlink(missing_ok=True)
            future = submit_clip(
                1,
                primary_prompt,
                primary_label,
                output_path=candidate_path,
                submitted_event=submitted_event,
                cancel_event=primary_cancel,
                track_regular=False,
            )
            primary_submissions.append((candidate, submitted_event, future))

        # Monitor the race immediately instead of waiting for the head-start
        # delay to finish. This preserves the real completion order even if one
        # candidate returns in under two seconds.
        def resolve_primary_race() -> None:
            primary_futures = {
                future: candidate
                for candidate, _, future in primary_submissions
            }
            errors: list[str] = []
            for future in concurrent.futures.as_completed(primary_futures):
                candidate = primary_futures[future]
                try:
                    candidate_path = future.result()
                    canonical_path = clips_dir / "clip_01.mp4"
                    canonical_path.unlink(missing_ok=True)
                    candidate_path.replace(canonical_path)
                except concurrent.futures.CancelledError:
                    continue
                except Exception as exc:
                    errors.append(str(exc))
                    continue

                with state_lock:
                    results[1] = canonical_path
                primary_cancel.set()
                _emit(
                    callback,
                    "primary_race_won",
                    index=1,
                    label=primary_label,
                    candidate=candidate,
                )
                report_action_finished()
                consume_ready_clip(1, primary_label, canonical_path)
                return

            failure = {
                "index": 1,
                "label": primary_label,
                "error": "；".join(errors) or "首动作的三路请求均未返回可用视频",
            }
            with state_lock:
                failures.append(failure)
            _emit(callback, "clip_failed", **failure)
            report_action_finished()

        primary_monitor = threading.Thread(
            target=resolve_primary_race,
            name="joyfox-primary-race",
            daemon=True,
        )
        primary_monitor.start()

        while any(
            not submitted.is_set() and not future.done()
            for _, submitted, future in primary_submissions
        ):
            time.sleep(0.05)

        if total > 1:
            submitted_count = sum(
                submitted.is_set() for _, submitted, _ in primary_submissions
            )
            _emit(
                callback,
                "primary_head_start",
                index=1,
                label=primary_label,
                delay=PRIMARY_HEAD_START_SECONDS,
                copies=PRIMARY_RACE_COPIES,
                submitted=submitted_count,
            )
            time.sleep(PRIMARY_HEAD_START_SECONDS)
            for index, (prompt, label) in enumerate(prompt_list[1:], start=2):
                submit_clip(index, prompt, label)

        for future in concurrent.futures.as_completed(futures):
            index, label = futures[future]
            try:
                path = future.result()
            except Exception as exc:
                failure = {"index": index, "label": label, "error": str(exc)}
                with state_lock:
                    failures.append(failure)
                _emit(callback, "clip_failed", **failure)
            else:
                with state_lock:
                    results[index] = path
                consume_ready_clip(index, label, path)
            report_action_finished()

        primary_monitor.join()

    for candidate_path in clips_dir.glob("clip_01_candidate_*"):
        candidate_path.unlink(missing_ok=True)

    ordered_indices = sorted(results)
    ordered_paths = [results[index] for index in ordered_indices]
    if not ordered_paths:
        details = "; ".join(item["error"] for item in failures[:3])
        raise RuntimeError(f"全部 Grok 视频任务均失败：{details}")

    clips = [
        {
            "index": index,
            "label": prompt_list[index - 1][1],
            "prompt": prompt_list[index - 1][0],
            "path": str(results[index]),
        }
        for index in ordered_indices
    ]
    result = {
        "total": total,
        "succeeded": len(ordered_paths),
        "failed": len(failures),
        "failures": failures,
        "clips": clips,
        "action_order": [prompt_list[index - 1][1] for index in ordered_indices],
    }
    _emit(callback, "clips_ready", clips=len(clips), action_order=result["action_order"])
    return result


def generate_standby_video(
    image_path: Path,
    output_path: Path,
    *,
    clips_dir: Path | None = None,
    prompts: Iterable[tuple[str, str]] = STANDBY_PROMPTS,
    concurrency: int = MAX_CONCURRENCY,
    keep_clips: bool = False,
    timeout_seconds: int = GENERATION_TIMEOUT_SECONDS,
    callback: EventCallback | None = None,
) -> dict[str, Any]:
    """Compatibility helper: generate clips, then concatenate them into one video."""
    output_path = Path(output_path).expanduser().resolve()
    resolved_clips_dir = (
        Path(clips_dir).expanduser().resolve()
        if clips_dir
        else output_path.parent / f"{output_path.stem}_clips"
    )
    result = generate_standby_clips(
        image_path,
        resolved_clips_dir,
        prompts=prompts,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        callback=callback,
    )
    ordered_paths = [Path(clip["path"]) for clip in result["clips"]]
    _emit(callback, "combining", clips=len(ordered_paths))
    combined_path = _concatenate_videos(ordered_paths, output_path)
    if not keep_clips:
        for video_path in ordered_paths:
            video_path.unlink(missing_ok=True)
        try:
            resolved_clips_dir.rmdir()
        except OSError:
            pass
    result["output"] = str(combined_path)
    _emit(callback, "batch_done", **result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joyfox Grok 20 路并发图生视频")
    parser.add_argument("image", type=Path, nargs="?", default=Path("222.png"))
    parser.add_argument("-o", "--output", type=Path, default=Path("grok_video.mp4"))
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--keep-clips", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    args = _parse_args()
    started_at = time.time()

    def print_event(event: dict[str, Any]) -> None:
        name = event.get("event")
        if name == "batch_progress":
            print(
                f"进度：{event['completed']}/{event['total']}，"
                f"成功 {event['succeeded']}，失败 {event['failed']}",
                flush=True,
            )

    result = generate_standby_video(
        args.image,
        args.output,
        concurrency=args.concurrency,
        keep_clips=args.keep_clips,
        callback=print_event,
    )
    print(
        f"完成：{result['output']}，成功 {result['succeeded']}/{result['total']}，"
        f"耗时 {time.time() - started_at:.1f} 秒",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
