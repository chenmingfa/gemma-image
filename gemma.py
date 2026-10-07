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

# 标签主线：品类和款式形状在前，然后是物体、成色、金属、主石、切工形状、工艺、风格。
DESIGN_BY_CATEGORY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("戒指", ("单石戒", "光环戒", "群镶戒", "三石戒", "排戒", "素圈戒", "开口戒", "扭纹戒", "分叉戒", "宽面戒", "对戒")),
    ("项链", ("锁骨链", "吊坠链", "Y字链", "多层链", "毛衣链", "珠串链")),
    ("耳环", ("耳钉款", "耳圈", "耳坠", "耳爬", "耳扣", "流苏坠")),
    ("手链", ("链节手链", "珠串手链", "网球手链", "吊饰手链")),
    ("手镯", ("实心镯", "开口镯", "活口镯", "宽面镯")),
    ("胸针", ("花卉胸针", "动物胸针", "蝴蝶结", "几何胸针")),
    ("吊坠", ("水滴吊坠", "圆形吊坠", "心形吊坠", "十字吊坠", "锁形吊坠", "生肖吊坠")),
    ("脚链", ("链节脚链", "珠串脚链", "吊饰脚链")),
)
OBJECT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("花卉", ("花朵", "玫瑰", "牡丹", "莲花", "梅花", "樱花", "四叶草", "三叶草", "叶子", "藤蔓", "麦穗")),
    ("动物", ("蝴蝶", "蜜蜂", "蜻蜓", "鸟", "天鹅", "孔雀", "鹤", "鱼", "龙", "凤", "蛇", "猫", "兔", "鹿", "蝙蝠", "貔貅", "鼠", "牛", "虎", "马", "羊", "猴", "鸡", "狗", "猪", "生肖")),
    ("符号", ("星星", "月亮", "太阳", "爱心", "十字", "皇冠", "蝴蝶结", "如意", "葫芦", "平安扣", "中国结", "云纹", "福字", "字母", "眼睛")),
    ("构件", ("珠子", "链条", "圆环", "麻花", "编织", "锁", "钥匙", "流苏", "羽毛", "铃铛", "几何", "素面")),
)
_DESIGN_TAGS = tuple(word for _, words in DESIGN_BY_CATEGORY for word in words)
_OBJECT_TAGS = tuple(word for _, words in OBJECT_GROUPS for word in words)
JEWELRY_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("品类", ("戒指", "项链", "耳环", "耳钉", "手链", "手镯", "吊坠", "胸针", "脚链")),
    ("款式", _DESIGN_TAGS),
    ("物体", _OBJECT_TAGS),
    ("成色", ("足金", "千足金", "万足金", "24K", "22K", "18K", "14K", "10K", "9K", "足银", "990银", "925银", "PT990", "PT950", "PT900", "成色不清")),
    ("金属", ("黄金", "K金", "玫瑰金", "白金", "铂金", "银")),
    ("主石", ("钻石", "翡翠", "珍珠", "红宝石", "蓝宝石", "祖母绿", "玉石", "水晶", "玛瑙")),
    ("形状", ("圆形", "水滴", "心形", "方形", "椭圆", "梨形")),
    ("工艺", ("爪镶", "六爪", "包镶", "密镶", "雕刻", "镂空", "珐琅", "素圈")),
    ("风格", ("经典", "复古", "极简", "华丽", "婚庆", "日常")),
)
JEWELRY_ALIASES = {
    "指环": ("戒指",),
    "钻戒": ("戒指", "钻石"),
    "对戒": ("戒指", "对戒"),
    "求婚戒": ("戒指", "婚庆"),
    "颈链": ("项链",),
    "吊坠项链": ("项链", "吊坠"),
    "耳坠": ("耳环", "耳坠"),
    "耳饰": ("耳环",),
    "耳圈": ("耳环", "耳圈"),
    "光环": ("光环戒",),
    "单石": ("单石戒",),
    "手环": ("手链",),
    "手串": ("手链", "珠子"),
    "脚镯": ("脚链",),
    "圆钻": ("钻石", "圆形"),
    "水滴形": ("水滴",),
    "花卉胸针": ("花卉胸针", "花朵"),
    "动物胸针": ("动物胸针", "动物"),
    "几何胸针": ("几何胸针", "几何"),
    "心形吊坠": ("心形吊坠", "爱心", "心形"),
    "十字吊坠": ("十字吊坠", "十字"),
    "锁形吊坠": ("锁形吊坠", "锁"),
    "水滴吊坠": ("水滴吊坠", "水滴"),
    "圆形吊坠": ("圆形吊坠", "圆形"),
    "生肖吊坠": ("生肖吊坠", "生肖"),
    "珠串链": ("珠串链", "珠子"),
    "珠串手链": ("珠串手链", "珠子"),
    "珠串脚链": ("珠串脚链", "珠子"),
    "流苏坠": ("流苏坠", "流苏"),
    "花": ("花朵",),
    "花卉": ("花朵",),
    "小花": ("花朵",),
    "花瓣": ("花朵",),
    "玫瑰花": ("玫瑰",),
    "牡丹花": ("牡丹",),
    "荷花": ("莲花",),
    "莲": ("莲花",),
    "四叶": ("四叶草",),
    "幸运草": ("四叶草",),
    "树叶": ("叶子",),
    "叶片": ("叶子",),
    "叶": ("叶子",),
    "藤": ("藤蔓",),
    "藤枝": ("藤蔓",),
    "小麦": ("麦穗",),
    "五角星": ("星星",),
    "星形": ("星星",),
    "星": ("星星",),
    "星月": ("星星", "月亮"),
    "月牙": ("月亮",),
    "弯月": ("月亮",),
    "新月": ("月亮",),
    "心": ("爱心",),
    "爱心形": ("爱心",),
    "十字架": ("十字",),
    "十字形": ("十字",),
    "王冠": ("皇冠",),
    "弓结": ("蝴蝶结",),
    "如意纹": ("如意",),
    "金葫芦": ("葫芦",),
    "盘长": ("中国结",),
    "盘长结": ("中国结",),
    "祥云": ("云纹",),
    "福": ("福字",),
    "转运珠": ("珠子",),
    "珠串": ("珠子",),
    "珠": ("珠子",),
    "链子": ("链条",),
    "链": ("链条",),
    "麻花纹": ("麻花",),
    "麻花辫": ("麻花",),
    "编织纹": ("编织",),
    "圆圈": ("圆环",),
    "小铃铛": ("铃铛",),
    "恶魔之眼": ("眼睛",),
    "字母纹": ("字母",),
    "凤凰": ("凤",),
    "龙纹": ("龙",),
    "老虎": ("虎",),
    "兔子": ("兔",),
    "玉兔": ("兔",),
    "小猫": ("猫",),
    "猫咪": ("猫",),
    "金鱼": ("鱼",),
    "锦鲤": ("鱼",),
    "小鸟": ("鸟",),
    "蝶": ("蝴蝶",),
    "仙鹤": ("鹤",),
    "梅花鹿": ("鹿",),
}
_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ｋＫｐＰｔＴａＡｕＵｇＧｓＳ",
    "0123456789kKpPtTaAuUgGsS",
)
_KARAT_OUT = {
    "24": ("24K", "黄金"),
    "22": ("22K", "K金"),
    "18": ("18K", "K金"),
    "14": ("14K", "K金"),
    "10": ("10K", "K金"),
    "9": ("9K", "K金"),
}
_AU_OUT = {
    "9999": ("万足金", "黄金"),
    "999": ("千足金", "黄金"),
    "990": ("足金", "黄金"),
    "916": ("22K", "K金"),
    "750": ("18K", "K金"),
    "585": ("14K", "K金"),
    "417": ("10K", "K金"),
    "375": ("9K", "K金"),
}
_PT_OUT = {
    "990": ("PT990", "铂金"),
    "950": ("PT950", "铂金"),
    "900": ("PT900", "铂金"),
}
_EXACT_MARKS = {
    "9999": ("万足金", "黄金"),
    "999": ("千足金", "黄金"),
    "990": ("足金", "黄金"),
    "925": ("925银", "银"),
    "916": ("22K", "K金"),
    "750": ("18K", "K金"),
    "585": ("14K", "K金"),
    "417": ("10K", "K金"),
    "375": ("9K", "K金"),
}
_AU_RE = re.compile(r"(?<![a-z0-9])(?:au|g)(9999|999|990|916|750|585|417|375)(?![0-9])")
_KARAT_RE = re.compile(r"(?<![a-z0-9])(24|22|18|14|10|9)k(?![a-z0-9])")
_PT_RE = re.compile(r"(?<![a-z0-9])pt(990|950|900)(?![0-9])")
_PT_ZH_RE = re.compile(r"铂(990|950|900)(?![0-9])")
_SILVER_RE = re.compile(r"(?<![a-z0-9])(?:s925|ag925|925银|银925)(?![0-9])")
_SILVER990_RE = re.compile(r"(?<![a-z0-9])(?:990银|银990|ag990)(?![0-9])")
_FINE_SILVER_RE = re.compile(r"足银|千足银|(?<![a-z0-9])(?:999银|银999|ag999)(?![0-9])")
_WAN_RE = re.compile(r"万足金|足金9999")
_QIAN_RE = re.compile(r"千足金|足金999(?!9)")
_ZU_RE = re.compile(r"(?<![千万])足金(?!999)")
_CODE_JIN_RE = re.compile(r"(?<![0-9])(9999|999|990|916|750|585|417|375)金")
_STAMP_RE = re.compile(
    r"(?:印记|印有|刻有|刻着|钢印|纯度)[^0-9a-z]{0,6}(9999|999|990|925|916|750|585|417|375)(?![0-9])"
)
_UNKNOWN_RE = re.compile(r"成色不清|成色不明|成色未知|纯度不清|纯度未知|印记不清|印记不明")


def _norm_mark(text: str) -> str:
    cleaned = text.translate(_FULLWIDTH)
    return re.sub(r"[\s·•・.．]+", "", cleaned)


_ALIAS_BY_KEY = {_norm_mark(key).casefold(): values for key, values in JEWELRY_ALIASES.items()}
_AXIS_OF = {
    word.casefold(): index
    for index, (_, words) in enumerate(JEWELRY_AXES)
    for word in words
}
_CATEGORY_KEYS = {word.casefold() for word in JEWELRY_AXES[0][1]}
_DESIGN_KEYS = {word.casefold() for word in _DESIGN_TAGS}
_OBJECT_KEYS = {word.casefold() for word in _OBJECT_TAGS}
_FINENESS_KEYS = {word.casefold() for word in JEWELRY_AXES[3][1]}
_REAL_FINENESS = _FINENESS_KEYS - {"成色不清"}
_AUTO_METAL_KEYS = frozenset({"k金", "黄金", "银", "铂金"})
_BROAD_OBJECTS = {
    "花朵": frozenset({"玫瑰", "牡丹", "莲花", "梅花", "樱花"}),
    "动物": frozenset(_OBJECT_KEYS & {
        "蝴蝶", "蜜蜂", "蜻蜓", "鸟", "天鹅", "孔雀", "鹤", "鱼", "龙", "凤", "蛇", "猫", "兔", "鹿",
        "蝙蝠", "貔貅", "鼠", "牛", "虎", "马", "羊", "猴", "鸡", "狗", "猪", "生肖",
    }),
    "生肖": frozenset({"鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"}),
}
def _build_phrases() -> list[tuple[str, tuple[str, ...]]]:
    phrases: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for key, values in _ALIAS_BY_KEY.items():
        phrases.append((key, values))
        seen.add(key)
    for word in (item for _, words in JEWELRY_AXES for item in words):
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        phrases.append((key, (word,)))
    return phrases


_PHRASES = _build_phrases()


def _jewelry_prompt() -> str:
    lines = [
        "请看这张图片，按珠宝检索来写标签。只输出下面两行，不要 Markdown，不要解释。",
        "说明：一句简体中文，不超过40个字，先写珠宝本身。看见纯度印记时，把印记原文写进说明。",
        "标签：项链、吊坠链、四叶草、月亮、成色不清、黄金、珍珠、圆形、包镶、日常",
        "",
        "标签用顿号「、」分开，8 到 18 个，每个 1 到 6 个字。",
        "顺序固定：品类、款式、物体、成色、金属、主石、形状、工艺、风格。背景和佩戴放在最后。",
        "有珠宝时必须写齐四项：品类、款式形状、物体、成色。款式只能用该品类下面的词。",
        "物体看见几个写几个，写具体名字。没有纹样、没有造型时才写素面。",
        "成色只写印记或刻字上的纯度，不写颜色。看不到印记就写成色不清，不要猜。",
    ]
    for name, words in JEWELRY_AXES:
        if name == "款式":
            lines.append("款式必须写，按品类选用：")
            for category, shapes in DESIGN_BY_CATEGORY:
                lines.append(f"{category}：{'、'.join(shapes)}。")
            continue
        if name == "物体":
            lines.append("物体必须写具体项，可以多项：")
            for group, motifs in OBJECT_GROUPS:
                lines.append(f"{group}：{'、'.join(motifs)}。")
            lines.append("玫瑰不要写成花朵，龙不要写成动物或生肖。同一件上的纹样都要写上。")
            continue
        if name == "成色":
            lines.append("成色必须写，只能从这些纯度里选：" + "、".join(words) + "。")
            lines.append(
                "印记对照：足金999或Au999写成千足金，Au9999或足金9999写成万足金，Au990写成足金，"
                "Au750或18K金写成18K，Au585写成14K，Au375写成9K，S925写成925银，Pt950写成PT950。"
            )
            lines.append("黄金色不是足金，玫瑰金不是18K，白色金属不是铂金，也不能把K金当成成色。")
            continue
        if name == "品类":
            lines.append(f"品类必须写：{'、'.join(words)}。")
            continue
        verb = "只用" if name in {"金属", "主石"} else "尽量用"
        lines.append(f"{name}{verb}：{'、'.join(words)}。")
    lines.append("没有珠宝时，标签只写：无珠宝。")
    lines.append("例子里的每一个词都要换成这张图的内容，不要照抄四叶草、月亮或成色不清。")
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
        parts.append(
            f"文件放在名为「{hint}」的文件夹里。这个名字只作参考。"
            "若其中写了纯度，而且画面没有相反印记，成色可以用这个纯度；印记和文件夹不一致时以印记为准。"
            "和画面无关的文件夹名字不要写进标签。"
        )
    return "\n".join(parts)


def recognize(image_jpeg: bytes, vocabulary: list[str], hint: str, model: str) -> tuple[str, list[str]]:
    note = ""
    last_error: Exception | None = None
    image = _b64(image_jpeg)
    for _ in range(3):
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(vocabulary, hint) + note,
                    "images": [image],
                }
            ],
            "options": {"temperature": 0.2, "num_predict": 640},
        }
        try:
            data = _post_json("/api/chat", payload, timeout=300)
            content = ((data.get("message") or {}).get("content")) or ""
            return parse_recognition(content)
        except RecognizeError as exc:
            last_error = exc
            note = (
                f"\n上次没有通过：{exc}。"
                "补齐品类、款式形状、物体、成色后再输出两行。"
                "物体按看见的纹样逐个写具体名字，不要只写花朵或动物。"
                "成色只写印记，看不清写成色不清，不要根据颜色猜。"
            )
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
            return caption, _finish_tags(tags, caption)
    caption, tags = _parse_labeled_lines(raw)
    if tags:
        return caption, _finish_tags(tags, caption)
    raise RecognizeError("没有得到标签")


def missing_style_tags(tags: list[str]) -> list[str]:
    """有品类时，款式形状、物体、成色也必须有。"""
    keys = {tag.casefold() for tag in tags}
    if not keys or "无珠宝" in keys or not (keys & _CATEGORY_KEYS):
        return []
    missing = []
    if not (keys & _DESIGN_KEYS):
        missing.append("款式形状")
    if not (keys & _OBJECT_KEYS):
        missing.append("物体")
    if not (keys & _FINENESS_KEYS):
        missing.append("成色")
    return missing


def _finish_tags(tags: list[str], caption: str = "") -> list[str]:
    ordered = arrange_jewelry_tags(_override_fineness_from_caption(tags, caption))
    missing = missing_style_tags(ordered)
    if missing:
        raise RecognizeError("这张款式还缺标签：" + "、".join(missing))
    return ordered


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def _mark_hits(key: str) -> list[tuple[tuple[int, int], tuple[str, ...]]]:
    hits: list[tuple[tuple[int, int], tuple[str, ...]]] = []

    def take(match: re.Match[str], outputs: tuple[str, ...]) -> None:
        span = match.span()
        if _overlaps(span, [item[0] for item in hits]):
            return
        hits.append((span, outputs))

    for match in _WAN_RE.finditer(key):
        take(match, ("万足金", "黄金"))
    for match in _QIAN_RE.finditer(key):
        take(match, ("千足金", "黄金"))
    for match in _ZU_RE.finditer(key):
        take(match, ("足金", "黄金"))
    for match in _AU_RE.finditer(key):
        take(match, _AU_OUT[match.group(1)])
    for match in _KARAT_RE.finditer(key):
        take(match, _KARAT_OUT[match.group(1)])
    for match in _PT_RE.finditer(key):
        take(match, _PT_OUT[match.group(1)])
    for match in _PT_ZH_RE.finditer(key):
        take(match, _PT_OUT[match.group(1)])
    for match in _SILVER_RE.finditer(key):
        take(match, ("925银", "银"))
    for match in _SILVER990_RE.finditer(key):
        take(match, ("990银", "银"))
    for match in _FINE_SILVER_RE.finditer(key):
        take(match, ("足银", "银"))
    for match in _CODE_JIN_RE.finditer(key):
        take(match, _AU_OUT[match.group(1)])
    for match in _STAMP_RE.finditer(key):
        code = match.group(1)
        take(match, ("925银", "银") if code == "925" else _AU_OUT[code])
    for match in _UNKNOWN_RE.finditer(key):
        take(match, ("成色不清",))
    return hits


def _phrase_spans(
    key: str, occupied: list[tuple[int, int]]
) -> list[tuple[int, int, tuple[str, ...]]]:
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for phrase, outputs in _PHRASES:
        if len(phrase) == 1 and phrase != key:
            continue
        start = 0
        while True:
            index = key.find(phrase, start)
            if index < 0:
                break
            candidates.append((index, index + len(phrase), outputs))
            start = index + 1
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    chosen: list[tuple[int, int, tuple[str, ...]]] = []
    taken = list(occupied)
    for start, end, outputs in candidates:
        if _overlaps((start, end), taken):
            continue
        taken.append((start, end))
        chosen.append((start, end, outputs))
    chosen.sort(key=lambda item: item[0])
    return chosen


def _expand_token(tag: str) -> list[str]:
    key = _norm_mark(tag).casefold()
    if not key:
        return []
    if key in _EXACT_MARKS:
        return list(_EXACT_MARKS[key])
    found: list[str] = []
    seen: set[str] = set()
    occupied: list[tuple[int, int]] = []

    def add(item: str) -> None:
        item_key = item.casefold()
        if item_key in seen:
            return
        seen.add(item_key)
        found.append(item)

    for span, outputs in _mark_hits(key):
        occupied.append(span)
        for item in outputs:
            add(item)
    for _start, _end, outputs in _phrase_spans(key, occupied):
        for item in outputs:
            add(item)
    return found or [tag]


def _fineness_marks(text: str) -> list[str]:
    key = _norm_mark(text).casefold()
    if not key:
        return []
    if key in _EXACT_MARKS:
        hits = list(_EXACT_MARKS[key])
    else:
        hits = [item for _span, outputs in _mark_hits(key) for item in outputs]
    found: list[str] = []
    seen: set[str] = set()
    for item in hits:
        item_key = item.casefold()
        if item_key not in _REAL_FINENESS or item_key in seen:
            continue
        seen.add(item_key)
        found.append(item)
    return found


def _strip_fineness_tokens(tags: list[str]) -> list[str]:
    kept: list[str] = []
    for tag in tags:
        expanded = _expand_token(tag)
        if not any(item.casefold() in _REAL_FINENESS for item in expanded):
            kept.append(tag)
            continue
        for item in expanded:
            key = item.casefold()
            if key in _FINENESS_KEYS or key in _AUTO_METAL_KEYS:
                continue
            kept.append(item)
    return kept


def _override_fineness_from_caption(tags: list[str], caption: str) -> list[str]:
    """说明里写了印记、标签里的成色又对不上时，以说明中的印记为准。"""
    caption_marks = _fineness_marks(caption)
    if not caption_marks:
        return tags
    tag_marks = [
        item
        for tag in tags
        for item in _expand_token(tag)
        if item.casefold() in _REAL_FINENESS
    ]
    tag_set = {item.casefold() for item in tag_marks}
    caption_set = {item.casefold() for item in caption_marks}
    if caption_set <= tag_set:
        return tags
    if tag_set and tag_set.isdisjoint(caption_set):
        return _strip_fineness_tokens(tags) + caption_marks
    return tags + [mark for mark in caption_marks if mark.casefold() not in tag_set]


def _drop_redundant(tags: list[str]) -> list[str]:
    keys = {tag.casefold() for tag in tags}
    drop: set[str] = set()
    if "成色不清" in keys and (keys & _REAL_FINENESS):
        drop.add("成色不清")
    if len(keys & _OBJECT_KEYS) > 1:
        drop.add("素面")
    for broad, specifics in _BROAD_OBJECTS.items():
        if broad in keys and (specifics & keys):
            drop.add(broad)
    if not drop:
        return tags
    return [tag for tag in tags if tag.casefold() not in drop]


def arrange_jewelry_tags(tags: list[str]) -> list[str]:
    """同义词和印记收成主线叫法，并按品类、款式、物体、成色排列。"""
    expanded: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        for item in _expand_token(tag):
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            expanded.append(item)
    if any(item == "无珠宝" for item in expanded) and len(expanded) > 1:
        expanded = [item for item in expanded if item != "无珠宝"]
    expanded = _drop_redundant(expanded)
    buckets: list[list[str]] = [[] for _ in JEWELRY_AXES]
    rest: list[str] = []
    for tag in expanded:
        index = _AXIS_OF.get(tag.casefold())
        if index is None:
            rest.append(tag)
        else:
            buckets[index].append(tag)
    ordered = [tag for bucket in buckets for tag in bucket]
    return (ordered + rest)[:20]


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
        if len(tags) >= 24:
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
