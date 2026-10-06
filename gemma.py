"""通过本机 Ollama 调用 Gemma 3 看图，并要回标签。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from store import folder_hint, normalize_tag

HOST = os.environ.get("GEMMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = "gemma3:4b"
VISION_CANDIDATES = ("gemma3:4b", "gemma3:12b", "gemma3:27b")

PROMPT = """请识别这张图片，给本地图片库写检索信息。只输出 JSON，不要 Markdown。
格式：{"caption":"一句话","tags":["标签"]}

caption：一句简体中文，不超过 40 字，只写画面里看得到的内容。
tags：8 到 12 个简体中文短标签，每个 2 到 8 个字。
覆盖确实可见的主体、场景、颜色、材质、风格、动作。
标签要适合以后搜索，不要用句子，不要重复，不要编造看不清的细节。
"""


class GemmaError(Exception):
    pass


class OllamaUnavailable(GemmaError):
    pass


class ModelMissing(GemmaError):
    pass


class RecognizeError(GemmaError):
    pass


def expected_model() -> str:
    return os.environ.get("GEMMA_MODEL", "").strip() or DEFAULT_MODEL


def choose_model(installed_names: list[str], wanted: str = "") -> str | None:
    names = []
    for name in installed_names:
        head = name.split()[0].strip()
        if head:
            names.append(head)
    if wanted:
        return _match(names, wanted)
    for candidate in VISION_CANDIDATES:
        found = _match(names, candidate)
        if found:
            return found
    return None


def _match(names: list[str], wanted: str) -> str | None:
    for name in names:
        if name == wanted or name.startswith(wanted + "-"):
            return name
    return None


def ollama_status() -> dict:
    expected = expected_model()
    try:
        data = _get_json("/api/tags", timeout=3)
    except OllamaUnavailable:
        return {
            "online": False,
            "ready": False,
            "model": None,
            "expected": expected,
            "models": [],
            "hint": "Ollama 没在运行。安装并打开后，刷新这个页面。",
        }
    names = [item.get("name", "") for item in data.get("models", [])]
    wanted = os.environ.get("GEMMA_MODEL", "").strip()
    chosen = choose_model(names, wanted)
    if chosen:
        hint = f"{chosen} 已就绪"
    else:
        hint = f"Ollama 已连接，还没有 {expected}"
    return {
        "online": True,
        "ready": bool(chosen),
        "model": chosen,
        "expected": expected,
        "models": names,
        "hint": hint,
    }


def require_model() -> str:
    status = ollama_status()
    if not status["online"]:
        raise OllamaUnavailable(status["hint"])
    if not status["model"]:
        raise ModelMissing(status["hint"] + f"。可运行：ollama pull {status['expected']}")
    return status["model"]


def build_prompt(vocabulary: list[str], hint: str) -> str:
    parts = [PROMPT]
    if vocabulary:
        parts.append("尽量复用这些已有标签，避免同义词：" + "、".join(vocabulary[:60]))
    if hint:
        parts.append(f"文件放在名为「{hint}」的文件夹里。这个名字只作参考，和画面不符就不要写进标签。")
    return "\n".join(parts)


def recognize(image_jpeg: bytes, vocabulary: list[str], hint: str, model: str) -> tuple[str, list[str]]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "messages": [
            {
                "role": "user",
                "content": build_prompt(vocabulary, hint),
                "images": [_b64(image_jpeg)],
            }
        ],
        "options": {"temperature": 0.2, "num_predict": 280},
    }
    last_error: Exception | None = None
    for _ in range(2):
        try:
            data = _post_json("/api/chat", payload, timeout=300)
            content = ((data.get("message") or {}).get("content")) or ""
            return parse_recognition(content)
        except RecognizeError as exc:
            last_error = exc
        except GemmaError:
            raise
    raise RecognizeError(str(last_error) if last_error else "没有识别出标签")


def parse_recognition(text: str) -> tuple[str, list[str]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        found = re.search(r"\{.*\}", raw, re.S)
        if not found:
            raise RecognizeError("模型没有返回 JSON")
        try:
            data = json.loads(found.group(0))
        except json.JSONDecodeError as exc:
            raise RecognizeError("模型返回的 JSON 无法解析") from exc
    if not isinstance(data, dict):
        raise RecognizeError("模型返回的不是对象")
    caption = re.sub(r"\s+", " ", str(data.get("caption") or "")).strip()[:80]
    raw_tags = data.get("tags")
    if not isinstance(raw_tags, list):
        raise RecognizeError("tags 不是数组")
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw_tags:
        tag = normalize_tag(str(item))
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
        if len(tags) >= 12:
            break
    if not tags:
        raise RecognizeError("没有得到标签")
    return caption, tags


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def pull_model(model_name: str, on_line) -> int:
    exe = find_ollama()
    if not exe:
        raise OllamaUnavailable("没有找到 ollama。请先安装 Ollama。")
    proc = subprocess.Popen(
        [exe, "pull", model_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    buf = ""
    while True:
        chunk = proc.stdout.read(256)
        if not chunk:
            break
        buf += chunk.decode("utf-8", "replace")
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for line in parts:
            text = line.strip()
            if text:
                on_line(text[:180])
    tail = buf.strip()
    if tail:
        on_line(tail[:180])
    return proc.wait()


def image_hint(path: str) -> str:
    return folder_hint(path)


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _get_json(path: str, timeout: float) -> dict:
    return _request(path, None, timeout)


def _post_json(path: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    return _request(path, body, timeout)


def _request(path: str, body: bytes | None, timeout: float) -> dict:
    req = urllib.request.Request(
        HOST + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        if exc.code == 404:
            raise ModelMissing(detail[:240] or "模型不存在")
        raise GemmaError(detail[:240] or f"Ollama 返回 {exc.code}")
    except urllib.error.URLError as exc:
        raise OllamaUnavailable("Ollama 没在运行。安装并打开后，刷新这个页面。") from exc
    except TimeoutError as exc:
        raise GemmaError("模型响应超时") from exc
