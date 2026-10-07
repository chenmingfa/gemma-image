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

from store import folder_hint, normalize_tag, tokenize

HOST = os.environ.get("GEMMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = "gemma3:4b"
VISION_CANDIDATES = ("gemma3:4b", "gemma3:12b", "gemma3:27b")

# 标签主线：品类、金属、主石在前，形状、工艺、风格在后。
JEWELRY_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("品类", ("戒指", "项链", "耳环", "耳钉", "手链", "手镯", "吊坠", "胸针", "脚链")),
    ("金属", ("黄金", "足金", "K金", "玫瑰金", "白金", "铂金", "银")),
    ("主石", ("钻石", "翡翠", "珍珠", "红宝石", "蓝宝石", "祖母绿", "玉石", "水晶", "玛瑙")),
    ("形状", ("圆形", "水滴", "心形", "方形", "椭圆", "梨形")),
    ("工艺", ("爪镶", "六爪", "包镶", "密镶", "雕刻", "镂空", "珐琅", "素圈")),
    ("风格", ("经典", "复古", "极简", "华丽", "婚庆", "日常")),
)
JEWELRY_ALIASES = {
    "指环": ("戒指",),
    "钻戒": ("戒指", "钻石"),
    "对戒": ("戒指",),
    "求婚戒": ("戒指", "婚庆"),
    "颈链": ("项链",),
    "吊坠项链": ("项链", "吊坠"),
    "耳坠": ("耳环",),
    "耳饰": ("耳环",),
    "耳圈": ("耳环",),
    "手环": ("手链",),
    "手串": ("手链",),
    "脚镯": ("脚链",),
    "圆钻": ("钻石", "圆形"),
    "水滴形": ("水滴",),
    "18k": ("K金",),
    "14k": ("K金",),
    "pt950": ("铂金",),
}
_ALIAS_BY_KEY = {key.casefold(): values for key, values in JEWELRY_ALIASES.items()}
_AXIS_OF = {
    word.casefold(): index
    for index, (_, words) in enumerate(JEWELRY_AXES)
    for word in words
}


def _jewelry_prompt() -> str:
    lines = [
        "请看这张图片，按珠宝检索来写标签。只输出下面两行，不要 Markdown，不要解释。",
        "说明：一句简体中文，不超过40个字，先写珠宝本身",
        "标签：戒指、玫瑰金、钻石、圆形、爪镶、婚庆",
        "",
        "标签用顿号「、」分开，6 到 12 个，每个 2 到 6 个字。",
        "顺序固定：品类、金属、主石、形状、工艺、风格。背景和佩戴放在最后。",
        "看得到的珠宝必须先写品类。看不清的金属、主石、克拉数不要写。",
    ]
    for name, words in JEWELRY_AXES:
        verb = "只用" if name in {"品类", "金属", "主石"} else "尽量用"
        lines.append(f"{name}{verb}：{'、'.join(words)}。")
    lines.append("没有珠宝时，标签只写：无珠宝。")
    lines.append("上面的戒指、玫瑰金只是格式例子，必须改成这张图的内容，不要照抄。")
    lines.append("不要写句子，不要重复。")
    return "\n".join(lines)


PROMPT = _jewelry_prompt()


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
        parts.append("库里已有这些叫法，能对上就复用，不要另造同义词：" + "、".join(vocabulary[:60]))
    if hint:
        parts.append(f"文件放在名为「{hint}」的文件夹里。这个名字只作参考，和画面不符就不要写进标签。")
    return "\n".join(parts)


def recognize(image_jpeg: bytes, vocabulary: list[str], hint: str, model: str) -> tuple[str, list[str]]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {
                "role": "user",
                "content": build_prompt(vocabulary, hint),
                "images": [_b64(image_jpeg)],
            }
        ],
        "options": {"temperature": 0.2, "num_predict": 400},
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
    raw = _strip_fence(text)
    data = _load_object(raw)
    if data is not None:
        caption = _clean_caption(data.get("caption") or data.get("说明") or data.get("画面") or "")
        tags = _collect_tags(data.get("tags") if "tags" in data else data.get("标签"))
        if tags:
            return caption, arrange_jewelry_tags(tags)
    caption, tags = _parse_labeled_lines(raw)
    if tags:
        return caption, arrange_jewelry_tags(tags)
    raise RecognizeError("没有得到标签")


def arrange_jewelry_tags(tags: list[str]) -> list[str]:
    """同义词收成主线叫法，并按品类、金属、主石、形状、工艺、风格排列。"""
    expanded: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        for item in _ALIAS_BY_KEY.get(tag.casefold(), (tag,)):
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            expanded.append(item)
    if "无珠宝" in seen and len(expanded) > 1:
        expanded = [item for item in expanded if item != "无珠宝"]
    buckets: list[list[str]] = [[] for _ in JEWELRY_AXES]
    rest: list[str] = []
    for tag in expanded:
        index = _AXIS_OF.get(tag.casefold())
        if index is None:
            rest.append(tag)
        else:
            buckets[index].append(tag)
    ordered = [tag for bucket in buckets for tag in bucket]
    return (ordered + rest)[:12]


def _strip_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def _load_object(raw: str) -> dict | None:
    candidates = [raw]
    found = re.search(r"\{.*\}", raw, re.S)
    if found:
        candidates.append(found.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _clean_caption(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:80]


def _collect_tags(value: object) -> list[str]:
    pieces: list[str] = []
    if isinstance(value, str):
        pieces = tokenize(value)
    elif isinstance(value, list):
        for item in value:
            pieces.extend(tokenize(str(item)))
    tags: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        tag = normalize_tag(piece)
        if not tag or tag in {"标签", "说明"}:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
        if len(tags) >= 12:
            break
    return tags


def _parse_labeled_lines(raw: str) -> tuple[str, list[str]]:
    caption = ""
    tags: list[str] = []
    for line in raw.splitlines():
        text = line.strip().lstrip("-*• ").strip()
        if not text:
            continue
        parts = re.split(r"[:：]", text, maxsplit=1)
        if len(parts) != 2:
            continue
        name = parts[0].strip().casefold()
        body = parts[1].strip()
        if name in {"说明", "caption", "画面", "描述"} and body:
            caption = _clean_caption(body)
        elif name in {"标签", "tags", "tag"} and body:
            tags = _collect_tags(body)
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
