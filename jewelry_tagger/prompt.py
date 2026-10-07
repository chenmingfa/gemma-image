"""按配置文件生成给 Gemma 3 的看图提示。"""

from __future__ import annotations

from jewelry_tagger.taxonomy import Axis, Taxonomy

_SKELETON = """{
  "category": "ring",
  "sub_category": "engagement_ring",
  "material": ["18K金", "铂金"],
  "gemstone": ["钻石", "蓝宝石"],
  "metal_color": "white_gold",
  "style": ["vintage", "art_deco"],
  "stone_shape": "emerald_cut",
  "setting": "pave",
  "occasion": ["wedding", "engagement"],
  "audience": "women",
  "brand_hint": null,
  "era": "art_deco",
  "confidence": 0.92,
  "tags": ["奢华", "复古", "婚戒", "群镶"]
}"""


def build_prompt(taxonomy: Taxonomy) -> str:
    lines = [
        "你在给珠宝图片写检索标签。只输出 JSON，不要 Markdown，不要解释。",
        "看不清的克拉数、印记、品牌不要编造。没有商标就让 brand_hint 为 null。",
        "没有宝石时 gemstone 用空数组，stone_shape 和 setting 为 null。",
        "sub_category 是款式形状，必须写一个，而且只能选当前品类下面的词。",
        "material、gemstone、tags 用中文。其余字段用英文 id。confidence 是 0 到 1 的小数。",
        "id 必须从下面的列表里选，不要发明新词。",
        "",
        _axis_line(taxonomy.axis("category"), "品类，只选一个"),
        _axis_line(
            taxonomy.axis("sub_category"),
            "款式形状，只选一个，必须属于品类",
            {item.id: item.zh for item in taxonomy.axis("category").options},
        ),
        _axis_line(taxonomy.axis("material"), "材质，可多项，输出中文"),
        _axis_line(taxonomy.axis("gemstone"), "宝石类型，可多项，没有就空数组，输出中文"),
        _axis_line(taxonomy.axis("metal_color"), "金属颜色，只选一个，看不清则 null"),
        _axis_line(taxonomy.axis("style"), "工艺风格，可多项"),
        _axis_line(taxonomy.axis("stone_shape"), "主石形状，只选一个，没有主石则 null"),
        _axis_line(taxonomy.axis("setting"), "镶嵌方式，只选一个，没有宝石则 null"),
        _axis_line(taxonomy.axis("occasion"), "使用场景，可多项"),
        _axis_line(taxonomy.axis("audience"), "目标人群，只选一个"),
        _axis_line(taxonomy.axis("brand_hint"), "品牌特征，只选一个，看不清商标则 null，不要猜品牌名"),
        _axis_line(taxonomy.axis("era"), "年代或流派，只选一个，看不出则 null"),
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
