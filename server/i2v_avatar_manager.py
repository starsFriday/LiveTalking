"""Background image-to-video-to-Avatar pipeline for the main dashboard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from server.grok_i2v import MAX_CONCURRENCY, generate_standby_clips
from utils.logger import logger


ROOT_DIR = Path(__file__).resolve().parents[1]
AVATAR_ROOT = ROOT_DIR / "data" / "avatars"


class I2VAvatarTask:
    def __init__(
        self,
        *,
        task_id: str,
        model: str,
        avatar_specs: list[dict[str, Any]],
        source_image: Path,
        job_dir: Path,
        image_size: tuple[int, int],
    ) -> None:
        self.task_id = task_id
        self.model = model
        if not avatar_specs:
            raise ValueError("数字人动作列表不能为空。")
        self.avatar_specs = [dict(spec) for spec in avatar_specs]
        # Keep avatar_id for backward-compatible clients; it is the top-priority Avatar.
        self.avatar_id = str(self.avatar_specs[0]["avatar_id"])
        self.avatar_ids: list[str] = []
        self.source_image = source_image
        self.job_dir = job_dir
        self.image_size = image_size
        self.status = "pending"
        self.stage = "pending"
        self.progress = 2
        self.message = "任务已创建，正在准备生成"
        self.error = ""
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.prompt_count = len(self.avatar_specs)
        self.completed_clips = 0
        self.successful_clips = 0
        self.failed_clips = 0
        self.action_order: list[str] = []
        self.generated_avatar_count = 0
        self.current_action = ""
        self.primary_ready = False
        self.avatar_failures: list[dict[str, Any]] = []
        self.worker_pid: int | None = None
        self.worker_paused = False
        self.worker_state = "idle"
        self.avatar_build_lock = threading.Lock()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "avatar_id": self.avatar_id,
            "avatar_ids": list(self.avatar_ids),
            "planned_avatar_ids": [spec["avatar_id"] for spec in self.avatar_specs],
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration": (self.finished_at or time.time()) - self.created_at,
            "prompt_count": self.prompt_count,
            "completed_clips": self.completed_clips,
            "successful_clips": self.successful_clips,
            "failed_clips": self.failed_clips,
            "action_order": list(self.action_order),
            "generated_avatar_count": self.generated_avatar_count,
            "current_action": self.current_action,
            "primary_ready": self.primary_ready,
            "avatar_failures": list(self.avatar_failures),
            "worker_pid": self.worker_pid,
            "worker_paused": self.worker_paused,
            "worker_state": self.worker_state,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "preview": f"/api/avatar-preview/{self.avatar_id}"
            if self.primary_ready
            else None,
            "avatars": [
                {
                    "index": spec["index"],
                    "action_name": spec["label"],
                    "avatar_id": spec["avatar_id"],
                    "preview": f"/api/avatar-preview/{spec['avatar_id']}",
                }
                for spec in self.avatar_specs
                if spec["avatar_id"] in self.avatar_ids
            ],
        }


class I2VAvatarManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="joyfox-avatar")
        self._tasks: dict[str, I2VAvatarTask] = {}
        self._active_task_id: str | None = None
        self._lock = threading.RLock()
        self._runtime_model_name = ""
        self._runtime_model = None
        self._shutting_down = False
        self._worker_processes: dict[str, subprocess.Popen] = {}

    def configure_runtime(self, model_name: str, model_payload: Any) -> None:
        """Reuse the renderer model already loaded by the LiveTalking process."""
        with self._lock:
            self._runtime_model_name = model_name
            self._runtime_model = model_payload

    def has_active_task(self) -> bool:
        with self._lock:
            return self._active_task_id is not None

    def blocks_realtime(self) -> bool:
        """Block WebRTC only until the first, highest-priority Avatar is usable."""
        with self._lock:
            task = self._tasks.get(self._active_task_id or "")
            return bool(task and not task.primary_ready)

    def add_task(
        self,
        *,
        task_id: str,
        model: str,
        avatar_specs: list[dict[str, Any]],
        source_image: Path,
        job_dir: Path,
        image_size: tuple[int, int],
    ) -> dict[str, Any]:
        if model not in {"wav2lip", "musetalk"}:
            raise ValueError(f"当前启动模型 {model or 'unknown'} 暂不支持自动生成数字人。")
        with self._lock:
            if self._runtime_model_name and model != self._runtime_model_name:
                raise RuntimeError(
                    f"生成模型 {model} 与当前启动模型 {self._runtime_model_name} 不一致。"
                )
            if self._active_task_id is not None:
                raise RuntimeError("已有数字人正在生成，请等待当前任务完成。")
            if self._shutting_down:
                raise RuntimeError("服务正在停止，暂时无法创建生成任务。")
            task = I2VAvatarTask(
                task_id=task_id,
                model=model,
                avatar_specs=avatar_specs,
                source_image=source_image,
                job_dir=job_dir,
                image_size=image_size,
            )
            self._tasks[task_id] = task
            self._active_task_id = task_id
            self._executor.submit(self._run_task, task_id)
            return task.to_dict()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def get_active_task(self) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(self._active_task_id or "")
            return task.to_dict() if task else None

    def _set_task(self, task: I2VAvatarTask, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(task, key, value)

    def _handle_grok_event(self, task: I2VAvatarTask, event: dict[str, Any]) -> None:
        name = event.get("event")
        with self._lock:
            if name == "batch_start":
                task.prompt_count = int(event.get("total", task.prompt_count))
                task.message = (
                    f"正在并行生成 {task.prompt_count} 个待机动作"
                    f"（最大 {MAX_CONCURRENCY} 路）"
                )
            elif name == "batch_progress":
                total = max(1, int(event.get("total", task.prompt_count)))
                task.completed_clips = int(event.get("completed", 0))
                task.successful_clips = int(event.get("succeeded", 0))
                task.failed_clips = int(event.get("failed", 0))
                grok_progress = 8 + round(54 * task.completed_clips / total)
                if task.primary_ready:
                    task.stage = "grok_background"
                    task.progress = max(task.progress, grok_progress)
                    task.message = (
                        f"首个数字人已就绪，后台视频生成 "
                        f"{task.completed_clips}/{total}"
                    )
                else:
                    task.progress = grok_progress
                    task.message = (
                        f"视频生成 {task.completed_clips}/{total}，"
                        f"成功 {task.successful_clips}，失败 {task.failed_clips}"
                    )
            elif name == "clips_ready":
                task.stage = "avatar_background" if task.primary_ready else "avatar"
                task.progress = max(task.progress, 64)
                task.message = (
                    f"{int(event.get('clips', 0))} 个动作视频已就绪，"
                    + (
                        "正在后台继续制作其余数字人"
                        if task.primary_ready
                        else "即将按优先级制作独立数字人"
                    )
                )

    def _build_avatar_as_soon_as_ready(
        self, task: I2VAvatarTask, clip: dict[str, Any]
    ) -> None:
        """Build every action as soon as its Grok clip becomes available."""
        action_index = int(clip.get("index", -1))
        specs_by_index = {
            int(spec["index"]): spec for spec in task.avatar_specs
        }
        spec = specs_by_index.get(action_index)
        if spec is None or str(spec["avatar_id"]) in task.avatar_ids:
            return
        primary_index = int(task.avatar_specs[0]["index"])
        with task.avatar_build_lock:
            if str(spec["avatar_id"]) in task.avatar_ids:
                return
            is_primary = action_index == primary_index
            try:
                self._set_task(
                    task,
                    stage="avatar" if is_primary else "avatar_background",
                    progress=max(task.progress, 65),
                    current_action=str(clip.get("label", "")),
                    message=(
                        "首个动作视频已完成，正在立即构建数字人"
                        if is_primary
                        else f"动作 {clip.get('label', action_index)} 已完成，正在继续构建数字人"
                    ),
                )
                self._generate_avatars(task, [clip])
            except Exception as exc:
                logger.exception(
                    "Failed to build streaming Avatar: task=%s action=%s",
                    task.task_id,
                    action_index,
                )
                self._set_task(
                    task,
                    stage="grok_background",
                    message=(
                        "首个数字人提前构建失败，将在视频批次完成后重试"
                        if is_primary
                        else f"动作 {clip.get('label', action_index)} 构建失败，批次结束后重试"
                    ),
                    error=str(exc) if is_primary else task.error,
                )
            else:
                self._set_task(
                    task,
                    stage="grok_background",
                    message=(
                        "首个数字人已就绪，其余动作正在边生成边构建"
                        if is_primary
                        else f"已生成 {task.generated_avatar_count}/{len(task.avatar_specs)} 个数字人"
                    ),
                    error="" if is_primary else task.error,
                )

    def _run_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return

        try:
            self._set_task(
                task,
                status="running",
                stage="grok",
                progress=6,
                message="正在向 Grok 提交预设待机动作",
            )
            result = generate_standby_clips(
                task.source_image,
                task.job_dir / "clips",
                prompts=[
                    (str(spec["prompt"]), str(spec["label"]))
                    for spec in task.avatar_specs
                ],
                concurrency=MAX_CONCURRENCY,
                callback=lambda event: self._handle_grok_event(task, event),
                clip_ready_callback=lambda clip: self._build_avatar_as_soon_as_ready(
                    task, clip
                ),
            )
            self._set_task(
                task,
                successful_clips=int(result["succeeded"]),
                failed_clips=int(result["failed"]),
                action_order=list(result["action_order"]),
                stage="avatar_background" if task.primary_ready else "avatar",
                progress=max(task.progress, 65),
                message=(
                    f"视频生成完成，正在使用 {task.model} 按顺序制作 "
                    f"{int(result['succeeded'])} 个数字人"
                ),
            )
            clips = list(result["clips"])
            if not task.primary_ready:
                # The primary clip may have failed its early callback once. Retry the
                # highest-priority available action before allowing realtime sessions.
                self._generate_avatars(task, [clips[0]])
            built_avatars = self._run_background_worker(task, clips)
            avatar_ids = [item["avatar_id"] for item in built_avatars]
            partial_errors = [
                str(item.get("error", "")) for item in result.get("failures", [])
            ] + [
                str(item.get("error", "")) for item in task.avatar_failures
            ]
            is_partial = bool(partial_errors)
            self._set_task(
                task,
                avatar_ids=avatar_ids,
                avatar_id=avatar_ids[0],
                status="partial" if is_partial else "completed",
                stage="partial" if is_partial else "completed",
                progress=100,
                message=(
                    f"已生成 {len(avatar_ids)} 个数字人，部分动作生成失败"
                    if is_partial
                    else f"已按动作优先级生成 {len(avatar_ids)} 个数字人"
                ),
                error="；".join(error for error in partial_errors if error)[:2000],
                finished_at=time.time(),
            )
            logger.info(
                "Joyfox Avatar task %s completed: model=%s avatars=%s",
                task.task_id,
                task.model,
                ",".join(avatar_ids),
            )
            shutil.rmtree(task.job_dir / "clips", ignore_errors=True)
        except Exception as exc:
            logger.exception("Joyfox Avatar task %s failed", task.task_id)
            if task.primary_ready:
                self._set_task(
                    task,
                    status="partial",
                    stage="partial",
                    message=(
                        f"后台生成中断，已成功生成 {task.generated_avatar_count} 个数字人"
                    ),
                    error=str(exc),
                    finished_at=time.time(),
                )
            else:
                self._set_task(
                    task,
                    status="failed",
                    stage="failed",
                    message="数字人生成失败",
                    error=str(exc),
                    finished_at=time.time(),
                )
        finally:
            with self._lock:
                if self._active_task_id == task_id:
                    self._active_task_id = None

    def _run_background_worker(
        self, task: I2VAvatarTask, clips: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build the remaining Avatars continuously in a low-priority subprocess."""

        specs_by_index = {
            int(spec["index"]): spec for spec in task.avatar_specs
        }
        all_items = []
        for clip in sorted(clips, key=lambda item: int(item["index"])):
            spec = specs_by_index.get(int(clip["index"]))
            if spec is None:
                continue
            all_items.append({
                **spec,
                "prompt": clip["prompt"],
                "video_path": clip["path"],
            })
        remaining_items = [
            item for item in all_items if item["avatar_id"] not in task.avatar_ids
        ]
        if not remaining_items:
            return all_items

        manifest_path = task.job_dir / "avatar_worker_manifest.json"
        status_path = task.job_dir / "avatar_worker_status.json"
        build_root = task.job_dir / "avatar_worker_build"
        worker_log_path = task.job_dir / "avatar_worker.log"
        task.job_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "task_id": task.task_id,
            "model": task.model,
            "avatar_count": len(task.avatar_specs),
            "items": remaining_items,
            "build_root": str(build_root.resolve()),
            "avatar_root": str(AVATAR_ROOT.resolve()),
            "status_path": str(status_path.resolve()),
        }
        temporary_manifest = manifest_path.with_suffix(".tmp.json")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
        status_path.unlink(missing_ok=True)

        command = self._background_worker_command(manifest_path.resolve())
        worker_log = worker_log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with self._lock:
            self._worker_processes[task.task_id] = process
        self._set_task(
            task,
            stage="avatar_worker",
            worker_pid=process.pid,
            worker_paused=False,
            worker_state="starting",
            message="其余数字人正在独立 Worker 中连续生成",
        )

        latest_status: dict[str, Any] = {}

        def read_status() -> dict[str, Any]:
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (OSError, ValueError, TypeError):
                return {}

        def apply_status(status: dict[str, Any]) -> None:
            if not status:
                return
            completed_ids = set(task.avatar_ids)
            completed_ids.update(
                str(item["avatar_id"])
                for item in status.get("completed", [])
                if isinstance(item, dict) and item.get("avatar_id")
            )
            ordered_ids = [
                str(spec["avatar_id"])
                for spec in task.avatar_specs
                if str(spec["avatar_id"]) in completed_ids
            ]
            current = status.get("current")
            current_action = (
                str(current.get("label", "")) if isinstance(current, dict) else ""
            )
            failures = [
                dict(item)
                for item in status.get("failures", [])
                if isinstance(item, dict)
            ]
            progress = 65 + round(34 * len(ordered_ids) / len(task.avatar_specs))
            self._set_task(
                task,
                avatar_ids=ordered_ids,
                avatar_id=ordered_ids[0] if ordered_ids else task.avatar_id,
                generated_avatar_count=len(ordered_ids),
                current_action=current_action,
                avatar_failures=failures,
                worker_state=str(status.get("state", task.worker_state)),
                progress=max(task.progress, progress),
                message=(
                    f"独立 Worker 后台生成 {len(ordered_ids)}/{len(task.avatar_specs)}"
                    + (f"：{current_action}" if current_action else "")
                ),
            )

        try:
            while process.poll() is None:
                if self._shutting_down:
                    process.terminate()
                    raise RuntimeError("服务正在停止，后台数字人 Worker 已终止。")

                latest_status = read_status() or latest_status
                apply_status(latest_status)
                time.sleep(0.25)

            latest_status = read_status() or latest_status
            apply_status(latest_status)
            if process.returncode != 0:
                raise RuntimeError(
                    f"后台数字人 Worker 异常退出（code={process.returncode}），"
                    f"请查看 {worker_log_path}"
                )
        finally:
            worker_log.close()
            with self._lock:
                self._worker_processes.pop(task.task_id, None)
            self._set_task(
                task,
                worker_pid=None,
                worker_paused=False,
                worker_state=str(latest_status.get("state", "stopped")),
            )

        completed_ids = set(task.avatar_ids)
        built_items = [
            item
            for item in all_items
            if item["avatar_id"] in completed_ids
            and self._is_avatar_ready(AVATAR_ROOT / item["avatar_id"], task.model)
        ]
        if not built_items:
            raise RuntimeError("后台 Worker 未生成任何可用数字人。")
        return built_items

    @staticmethod
    def _background_worker_command(manifest_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "server.i2v_avatar_worker",
            "--manifest",
            str(manifest_path),
        ]

    def _generate_avatars(
        self, task: I2VAvatarTask, clips: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        build_root = task.job_dir / "avatar_build"
        shutil.rmtree(build_root, ignore_errors=True)
        build_root.mkdir(parents=True, exist_ok=True)
        ordered_clips = sorted(clips, key=lambda item: int(item["index"]))
        if not ordered_clips:
            raise RuntimeError("没有可用于制作数字人的动作视频。")
        specs_by_index = {
            int(spec["index"]): spec for spec in task.avatar_specs
        }
        built_avatars: list[dict[str, Any]] = []
        total = len(task.avatar_specs)

        for clip in ordered_clips:
            action_index = int(clip["index"])
            spec = specs_by_index.get(action_index)
            if spec is None:
                raise RuntimeError(f"动作 {action_index} 缺少 Avatar ID 配置。")
            item = {
                **spec,
                "prompt": clip["prompt"],
                "video_path": clip["path"],
            }
            if str(spec["avatar_id"]) in task.avatar_ids:
                built_avatars.append(item)
                continue
            completed_before = len(task.avatar_ids)
            build_number = completed_before + 1
            self._set_task(
                task,
                current_action=str(spec["label"]),
                stage="avatar" if action_index == int(task.avatar_specs[0]["index"]) else "avatar_background",
                message=f"正在制作数字人 {build_number}/{total}：{spec['label']}",
            )

            def progress_callback(
                progress: int,
                *,
                _completed_before: int = completed_before,
                _build_number: int = build_number,
            ) -> None:
                generator_progress = max(0, min(100, int(progress)))
                overall = 65 + round(
                    34 * (_completed_before + generator_progress / 100) / total
                )
                self._set_task(
                    task,
                    progress=max(task.progress, overall),
                    message=(f"正在制作数字人 {_build_number}/{total}："
                             f"{spec['label']} {generator_progress}%"),
                )

            generated_path = build_root / str(spec["avatar_id"])
            try:
                self._invoke_avatar_generator(
                    task,
                    video_path=Path(str(clip["path"])),
                    avatar_id=str(spec["avatar_id"]),
                    build_root=build_root,
                    progress_callback=progress_callback,
                )
                if not self._is_avatar_ready(generated_path, task.model):
                    raise RuntimeError(
                        f"动作 {spec['label']} 已处理，但数字人文件不完整。"
                    )
            except Exception as exc:
                logger.exception(
                    "Failed to build action Avatar: task=%s action=%s",
                    task.task_id,
                    spec["label"],
                )
                shutil.rmtree(generated_path, ignore_errors=True)
                failure = {
                    "index": action_index,
                    "label": str(spec["label"]),
                    "error": str(exc),
                }
                self._set_task(
                    task,
                    avatar_failures=[*task.avatar_failures, failure],
                    message=f"动作 {spec['label']} 制作失败，继续下一项",
                )
                continue
            self._publish_avatar(task, item, generated_path, total)
            built_avatars.append(item)
            remaining_failures = [
                failure
                for failure in task.avatar_failures
                if int(failure.get("index", -1)) != action_index
            ]
            completed_ids = set(task.avatar_ids)
            completed_ids.add(str(spec["avatar_id"]))
            avatar_ids = [
                str(candidate["avatar_id"])
                for candidate in task.avatar_specs
                if str(candidate["avatar_id"]) in completed_ids
            ]
            primary_id = str(task.avatar_specs[0]["avatar_id"])
            self._set_task(
                task,
                avatar_ids=avatar_ids,
                avatar_id=avatar_ids[0],
                generated_avatar_count=len(avatar_ids),
                primary_ready=primary_id in avatar_ids,
                avatar_failures=remaining_failures,
                stage="avatar_background" if len(avatar_ids) < total else "avatar",
                message=(
                    f"首个数字人已就绪，其余正在边生成边构建（{len(avatar_ids)}/{total}）"
                    if primary_id in avatar_ids and len(avatar_ids) < total
                    else f"已生成 {len(avatar_ids)}/{total} 个数字人"
                ),
            )
        if not built_avatars:
            raise RuntimeError("所有动作的数字人均制作失败。")
        return built_avatars

    def _invoke_avatar_generator(
        self,
        task: I2VAvatarTask,
        *,
        video_path: Path,
        avatar_id: str,
        build_root: Path,
        progress_callback,
    ) -> None:
        if task.model == "musetalk":
            from avatars.musetalk.genavatar import generate_avatar

            with self._lock:
                model_bundle = self._runtime_model
            if model_bundle is None:
                raise RuntimeError("当前 MuseTalk 运行模型尚未加载完成。")
            generate_avatar(
                video_path=str(video_path),
                avatar_id=avatar_id,
                save_path=str(build_root),
                bbox_shift=0,
                extra_margin=10,
                parsing_mode="jaw",
                version="v15",
                progress_callback=progress_callback,
                model_bundle=model_bundle,
            )
        else:
            from avatars.wav2lip.genavatar import generate_avatar

            generate_avatar(
                video_path=str(video_path),
                avatar_id=avatar_id,
                save_path=str(build_root),
                img_size=256,
                pads=[0, 20, 0, 0],
                nosmooth=False,
                face_det_batch_size=4,
                progress_callback=progress_callback,
            )

    def _publish_avatar(
        self,
        task: I2VAvatarTask,
        item: dict[str, Any],
        generated_path: Path,
        avatar_count: int,
    ) -> None:
        avatar_id = str(item["avatar_id"])
        destination = AVATAR_ROOT / avatar_id
        AVATAR_ROOT.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Avatar ID 已存在：{avatar_id}")
        metadata = {
            "source": "joyfox_grok_i2v",
            "task_id": task.task_id,
            "model": task.model,
            "action_index": int(item["index"]),
            "action_name": str(item["label"]),
            "prompt": str(item["prompt"]),
            "batch_timestamp": str(item["batch_timestamp"]),
            "batch_avatar_count": avatar_count,
            "created_at": time.time(),
        }
        (generated_path / "joyfox_generation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(generated_path, destination)

    @staticmethod
    def _is_avatar_ready(path: Path, model: str) -> bool:
        common = (
            (path / "coords.pkl").is_file()
            and (path / "full_imgs").is_dir()
            and any((path / "full_imgs").iterdir())
        )
        if not common:
            return False
        if model == "musetalk":
            return (
                (path / "latents.pt").is_file()
                and (path / "mask_coords.pkl").is_file()
                and (path / "mask").is_dir()
            )
        return (path / "face_imgs").is_dir() and any((path / "face_imgs").iterdir())

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            worker_processes = list(self._worker_processes.values())
        for process in worker_processes:
            if process.poll() is not None:
                continue
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)


i2v_avatar_manager = I2VAvatarManager()
