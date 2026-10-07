"""从模型原文里抽出 JSON，并修掉常见的格式损坏。"""

from __future__ import annotations

import json
import re


def loads_repaired(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    blob = _extract_object(raw)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 无法解析：{exc.msg}") from exc
    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError("JSON 必须是一个对象")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("JSON 必须是一个对象")
    return data


def _extract_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("模型没有返回 JSON 对象")
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("JSON 没有闭合")
