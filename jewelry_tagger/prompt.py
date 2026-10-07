"""按配置文件生成给 Gemma 3 的看图提示。"""

from __future__ import annotations

from jewelry_tagger.taxonomy import Axis, Taxonomy

_SKELETON = """{
  "category": {"main": "ring", "confidence": 0.95},
  "material": [{"name": "gold", "confidence": 0.9}, {"name": "diamond", "confidence": 0.88}],
  "cut": {"name": "brilliant", "confidence": 0.85},
  "design": ["solitaire"],
  "style": ["luxury", "classic"],
  "occasion": ["wedding", "engagement"],
  "color": {"primary": "#FFD700", "secondary": ["#FFFFFF"]},
  "visual": {"background": "gradient", "angle": "45_degree", "lighting": "soft"},
  "caption": "一枚黄金镶圆形钻石的订婚戒指"
}"""


def build_prompt(taxonomy: Taxonomy) -> str:
    lines = [
        "你在给珠宝图片写检索标签。只输出 JSON，不要 Markdown，不要解释。",
        "看不清的金属、克拉数、印记不要编造。白金(white_gold)和铂金(platinum)分不清就不要写。",
        "没有宝石时 cut 必须是 null。有宝石时 cut 只写能确认的切工。",
        "design 是款式形状，至少写一个，而且必须属于当前品类。看得到几种形状就写几种。",
        "caption 用一句简体中文，不超过 40 字，只写画面里看得到的珠宝。",
        "颜色用 #RRGGBB。置信度是 0 到 1 的小数。",
        "id 必须从下面的列表里选，不要发明新词。",
        "",
        _axis_line(taxonomy.axis("category"), "珠宝类别，只选一个"),
        _axis_line(taxonomy.axis("material"), "材质，可多项，金属和宝石都写在这里"),
        _axis_line(taxonomy.axis("cut"), "宝石切割，只选一个；没有宝石则 null。这里的 emerald 是祖母绿式切割，不是宝石"),
        _axis_line(
            taxonomy.axis("design"),
            "款式形状，可多项，只选和品类相符的",
            {item.id: item.zh for item in taxonomy.axis("category").options},
        ),
        _axis_line(taxonomy.axis("style"), "风格，可多项"),
        _axis_line(taxonomy.axis("occasion"), "使用场景，可多项"),
        _axis_line(taxonomy.axis("background"), "背景"),
        _axis_line(taxonomy.axis("angle"), "拍摄角度"),
        _axis_line(taxonomy.axis("lighting"), "光影"),
    ]
    for axis in taxonomy.extra_axes.values():
        kind = "可多项" if axis.multiple else "只选一个"
        lines.append(_axis_line(axis, f"扩展标签 {axis.name}，{kind}，放进 extras.{axis.name}"))
    lines.extend(["", "按这个形状输出：", _SKELETON])
    return "\n".join(lines)


def repair_prompt(base: str, raw: str, error: str) -> str:
    clipped = raw.strip()[:1200]
    return (
        f"{base}\n\n"
        "上一次输出没有通过校验。请只重新输出修正后的 JSON。\n"
        f"错误：{error}\n"
        f"上次输出：\n{clipped}"
    )


def _axis_line(axis: Axis, title: str, group_names: dict[str, str] | None = None) -> str:
    if not any(item.categories for item in axis.options):
        choices = "、".join(f"{item.id}({item.zh})" for item in axis.options)
        return f"{title}：{choices}"
    grouped: dict[str, list] = {}
    for item in axis.options:
        for category in item.categories:
            grouped.setdefault(category, []).append(item)
    lines = [f"{title}："]
    names = group_names or {}
    for category, items in grouped.items():
        label = names.get(category, category)
        choices = "、".join(f"{item.id}({item.zh})" for item in items)
        lines.append(f"{label}：{choices}")
    return "\n".join(lines)
