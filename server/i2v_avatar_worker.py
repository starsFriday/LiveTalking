"""Low-priority subprocess for building non-primary Joyfox Avatars."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _avatar_ready(path: Path, model: str) -> bool:
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


def _publish(
    *,
    generated_path: Path,
    avatar_root: Path,
    item: dict[str, Any],
    model: str,
    task_id: str,
    avatar_count: int,
) -> None:
    avatar_id = str(item["avatar_id"])
    destination = avatar_root / avatar_id
    if destination.exists():
        if _avatar_ready(destination, model):
            shutil.rmtree(generated_path, ignore_errors=True)
            return
        raise RuntimeError(f"Avatar ID 已存在但文件不完整：{avatar_id}")
    metadata = {
        "source": "joyfox_grok_i2v",
        "task_id": task_id,
        "model": model,
        "action_index": int(item["index"]),
        "action_name": str(item["label"]),
        "prompt": str(item["prompt"]),
        "batch_timestamp": str(item["batch_timestamp"]),
        "batch_avatar_count": avatar_count,
        "created_at": time.time(),
        "worker_pid": os.getpid(),
    }
    (generated_path / "joyfox_generation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(generated_path, destination)


def _generate_one(
    *,
    model: str,
    item: dict[str, Any],
    build_root: Path,
    progress_callback,
    musetalk_bundle=None,
) -> None:
    if model == "musetalk":
        from avatars.musetalk.genavatar import generate_avatar

        generate_avatar(
            video_path=str(item["video_path"]),
            avatar_id=str(item["avatar_id"]),
            save_path=str(build_root),
            bbox_shift=0,
            extra_margin=10,
            parsing_mode="jaw",
            version="v15",
            progress_callback=progress_callback,
            model_bundle=musetalk_bundle,
        )
    else:
        from avatars.wav2lip.genavatar import generate_avatar

        generate_avatar(
            video_path=str(item["video_path"]),
            avatar_id=str(item["avatar_id"]),
            save_path=str(build_root),
            img_size=256,
            pads=[0, 20, 0, 0],
            nosmooth=False,
            face_det_batch_size=4,
            progress_callback=progress_callback,
        )


def run(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = str(manifest["model"])
    task_id = str(manifest["task_id"])
    items = sorted(manifest["items"], key=lambda item: int(item["index"]))
    avatar_count = int(manifest["avatar_count"])
    build_root = Path(manifest["build_root"]).resolve()
    avatar_root = Path(manifest["avatar_root"]).resolve()
    status_path = Path(manifest["status_path"]).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    avatar_root.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "state": "starting",
        "pid": os.getpid(),
        "model": model,
        "total": len(items),
        "completed": [],
        "failures": [],
        "current": None,
        "item_progress": 0,
        "updated_at": time.time(),
    }

    def write_status(**changes: Any) -> None:
        status.update(changes)
        status["updated_at"] = time.time()
        _atomic_json(status_path, status)

    write_status(state="loading")
    musetalk_bundle = None
    if model == "musetalk" and items:
        import torch
        from avatars.musetalk.utils.utils import load_all_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        musetalk_bundle = load_all_model(device=device)

    for position, item in enumerate(items, start=1):
        avatar_id = str(item["avatar_id"])
        destination = avatar_root / avatar_id
        if _avatar_ready(destination, model):
            status["completed"].append(item)
            write_status(state="running", current=item, item_progress=100)
            continue

        generated_path = build_root / avatar_id
        shutil.rmtree(generated_path, ignore_errors=True)
        last_status_time = 0.0
        last_progress = -1

        def progress_callback(progress: int) -> None:
            nonlocal last_status_time, last_progress
            value = max(0, min(100, int(progress)))
            now = time.monotonic()
            if value == last_progress or (value < 100 and now - last_status_time < 0.35):
                return
            last_progress = value
            last_status_time = now
            write_status(
                state="running",
                current=item,
                position=position,
                item_progress=value,
            )

        try:
            write_status(
                state="running", current=item, position=position, item_progress=0
            )
            _generate_one(
                model=model,
                item=item,
                build_root=build_root,
                progress_callback=progress_callback,
                musetalk_bundle=musetalk_bundle,
            )
            if not _avatar_ready(generated_path, model):
                raise RuntimeError("生成结束但数字人文件不完整。")
            _publish(
                generated_path=generated_path,
                avatar_root=avatar_root,
                item=item,
                model=model,
                task_id=task_id,
                avatar_count=avatar_count,
            )
            status["completed"].append(item)
            write_status(item_progress=100)
        except Exception as exc:
            shutil.rmtree(generated_path, ignore_errors=True)
            status["failures"].append(
                {
                    "index": int(item["index"]),
                    "label": str(item["label"]),
                    "error": str(exc),
                }
            )
            write_status(state="running", item_progress=0)

    write_status(state="completed", current=None, item_progress=100)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joyfox background Avatar worker")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        os.nice(10)
    except OSError:
        pass
    return run(_parse_args().manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
