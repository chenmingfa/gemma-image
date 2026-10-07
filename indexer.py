"""扫描文件夹，让 Gemma 3 逐张写标签。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from gemma import ModelMissing, OllamaUnavailable, image_hint, missing_style_tags, recognize, require_model
from images import model_jpeg
from store import get_library

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
MAX_BYTES = 40 * 1024 * 1024

_job_lock = threading.Lock()
_job = {
    "running": False,
    "folder": "",
    "total": 0,
    "processed": 0,
    "recognized": 0,
    "skipped": 0,
    "errors": 0,
    "current": "",
    "message": "",
    "last_error": "",
}
_cancel = False
_worker: threading.Thread | None = None


def job_snapshot() -> dict:
    with _job_lock:
        return dict(_job)


def _update(**kwargs) -> None:
    with _job_lock:
        _job.update(kwargs)


def cancel_index() -> None:
    global _cancel
    with _job_lock:
        if _job["running"]:
            _cancel = True
            _job["message"] = "正在停止"


def cancel_requested() -> bool:
    with _job_lock:
        return _cancel


def start_index(folder: str, force: bool = False) -> tuple[bool, str]:
    global _cancel, _worker
    path = Path(folder.strip())
    if not folder.strip():
        return False, "先选择一个图片文件夹"
    if not path.exists() or not path.is_dir():
        return False, "文件夹不存在"
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        return False, "请选一个具体的文件夹，不要选整个磁盘"
    with _job_lock:
        if _job["running"]:
            return False, "已经有识别任务在进行"
        _cancel = False
        _job.update(
            {
                "running": True,
                "folder": str(resolved),
                "total": 0,
                "processed": 0,
                "recognized": 0,
                "skipped": 0,
                "errors": 0,
                "current": "",
                "message": "正在清点图片",
                "last_error": "",
            }
        )
    _worker = threading.Thread(target=_run, args=(resolved, force), daemon=True, name="gemma-index")
    _worker.start()
    return True, "开始识别"


def index_file(path: Path, model: str, force: bool = False, respect_user: bool = True) -> str:
    library = get_library()
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(f"无法读取文件：{exc}") from exc
    if stat.st_size > MAX_BYTES:
        library.save_error(path, mtime=stat.st_mtime, size=stat.st_size, message="文件超过 40MB")
        return "error"
    existing = library.get_by_path(path)
    if _unchanged(library, existing, stat):
        if not force:
            return "skipped"
        if existing["user_edited"] and respect_user:
            return "skipped"
    vocabulary = [item["name"] for item in library.top_tags(60)]
    try:
        jpeg, width, height = model_jpeg(path)
        caption, tags = recognize(jpeg, vocabulary, image_hint(str(path)), model)
    except (OllamaUnavailable, ModelMissing):
        raise
    except Exception as exc:
        library.save_error(path, mtime=stat.st_mtime, size=stat.st_size, message=str(exc))
        _update(last_error=f"{path.name}：{exc}")
        print(f"失败 {path.name}：{exc}", flush=True)
        return "error"
    library.save_success(
        path,
        mtime=stat.st_mtime,
        size=stat.st_size,
        width=width,
        height=height,
        caption=caption,
        tags=tags,
        model=model,
        source="model",
    )
    print(f"已识别 {path.name} → {'、'.join(tags)}", flush=True)
    return "ok"


def list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        relative = path.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(path)
    files.sort(key=lambda item: str(item).casefold())
    return files


def _unchanged(library, existing: dict | None, stat: os.stat_result) -> bool:
    if not existing or existing.get("error") or not existing.get("caption"):
        return False
    if abs(float(existing["mtime"]) - stat.st_mtime) >= 0.001 or int(existing["size"]) != stat.st_size:
        return False
    item = library.get(existing["id"])
    if item and missing_style_tags(item["tags"]):
        return False
    return True


def _run(folder: Path, force: bool) -> None:
    recognized = skipped = errors = 0
    try:
        files = list_images(folder)
        _update(total=len(files))
        if not files:
            _update(running=False, message="这个文件夹里没有支持的图片")
            return
        model = require_model()
        for index, path in enumerate(files, start=1):
            if cancel_requested():
                _update(
                    running=False,
                    current="",
                    message=f"已停止。新识别 {recognized}，跳过 {skipped}，失败 {errors}",
                )
                return
            _update(current=path.name, message=f"正在识别 {path.name}")
            try:
                result = index_file(path, model, force=force)
            except (OllamaUnavailable, ModelMissing) as exc:
                _update(running=False, current="", message=str(exc), last_error=str(exc))
                return
            if result == "ok":
                recognized += 1
            elif result == "skipped":
                skipped += 1
            else:
                errors += 1
            _update(
                processed=index,
                recognized=recognized,
                skipped=skipped,
                errors=errors,
            )
        _update(
            running=False,
            current="",
            message=f"完成。新识别 {recognized}，跳过 {skipped}，失败 {errors}",
        )
    except Exception as exc:
        _update(running=False, current="", message=f"识别中断：{exc}", last_error=str(exc))
