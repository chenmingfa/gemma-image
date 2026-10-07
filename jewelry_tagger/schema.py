"""识别结果。材质和宝石用中文，其余枚举用配置里的 id。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jewelry_tagger.taxonomy import Axis, Taxonomy


class JewelryTags(BaseModel):
    category: str
    sub_category: str
    material: list[str] = Field(min_length=1)
    gemstone: list[str] = Field(default_factory=list)
    metal_color: str | None = None
    style: list[str] = Field(min_length=1)
    stone_shape: str | None = None
    setting: str | None = None
    occasion: list[str] = Field(min_length=1)
    audience: str
    brand_hint: str | None = None
    era: str | None = None
    confidence: float = Field(ge=0, le=1)
    tags: list[str] = Field(min_length=1)
    extras: dict[str, list[str]] = Field(default_factory=dict)

    def as_json(self) -> dict:
        data = self.model_dump()
        if not data.get("extras"):
            data.pop("extras", None)
        return data


def parse_tags(data: dict, taxonomy: Taxonomy) -> JewelryTags:
    return JewelryTags.model_validate(normalize_payload(data, taxonomy))


def normalize_payload(data: dict, taxonomy: Taxonomy) -> dict[str, Any]:
    category = taxonomy.axis("category").resolve(_token(data.get("category")))
    if category is None:
        raise ValueError("category 不能为空")
    sub_category = taxonomy.axis("sub_category").resolve(_token(data.get("sub_category")))
    if sub_category is None:
        raise ValueError("sub_category 不能为空，要写这张款式的形状")
    if not taxonomy.axis("sub_category").allows(sub_category, category):
        allowed = "、".join(
            option.id
            for option in taxonomy.axis("sub_category").options
            if not option.categories or category in option.categories
        )
        raise ValueError(f"sub_category {sub_category} 不属于 {category}。可以用：{allowed}")

    material_ids = taxonomy.resolve_many("material", _as_list(data.get("material")))
    if not material_ids:
        raise ValueError("material 至少要有一项")
    gemstone_ids = taxonomy.resolve_many("gemstone", _as_list(data.get("gemstone")))
    style = taxonomy.resolve_many("style", _as_list(data.get("style")))
    if not style:
        raise ValueError("style 至少要有一项")
    occasion = taxonomy.resolve_many("occasion", _as_list(data.get("occasion")))
    if not occasion:
        raise ValueError("occasion 至少要有一项")
    audience = taxonomy.axis("audience").resolve(_token(data.get("audience")))
    if audience is None:
        raise ValueError("audience 不能为空")

    payload = {
        "category": category,
        "sub_category": sub_category,
        "material": _labels(taxonomy.axis("material"), material_ids),
        "gemstone": _labels(taxonomy.axis("gemstone"), gemstone_ids),
        "metal_color": taxonomy.axis("metal_color").resolve(_token(data.get("metal_color"))),
        "style": style,
        "stone_shape": taxonomy.axis("stone_shape").resolve(_token(data.get("stone_shape"))),
        "setting": taxonomy.axis("setting").resolve(_token(data.get("setting"))),
        "occasion": occasion,
        "audience": audience,
        "brand_hint": taxonomy.axis("brand_hint").resolve(_token(data.get("brand_hint"))),
        "era": taxonomy.axis("era").resolve(_token(data.get("era"))),
        "confidence": _confidence(data.get("confidence"), data.get("category")),
        "extras": _extras(data, taxonomy),
        "tags": _tags(data.get("tags"), taxonomy, {
            "sub_category": [sub_category],
            "style": style,
            "occasion": occasion,
            "setting": [taxonomy.axis("setting").resolve(_token(data.get("setting")))],
        }),
    }
    return payload


def _token(value: object) -> object:
    if isinstance(value, dict):
        return value.get("main") or value.get("name") or value.get("id")
    return value


def _as_list(value: object) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError("标签项必须是字符串或数组")


def _extras(data: dict, taxonomy: Taxonomy) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    raw_extras = data.get("extras") if isinstance(data.get("extras"), dict) else {}
    for name in taxonomy.extra_axes:
        raw = raw_extras.get(name)
        if raw is None:
            raw = data.get(name)
        if raw is None:
            continue
        found[name] = taxonomy.resolve_many(name, raw)
    return found


def _labels(axis: Axis, ids: list[str]) -> list[str]:
    names = {option.id: option.zh for option in axis.options}
    return [names[item] for item in ids]


def _confidence(value: object, category: object) -> float:
    if value is None and isinstance(category, dict):
        value = category.get("confidence")
    if value is None or value == "":
        raise ValueError("confidence 缺失")
    number = float(value)
    if number > 1:
        number = number / 100
    if number < 0 or number > 1:
        raise ValueError(f"confidence 必须在 0 到 1 之间，收到的是 {value}")
    return number


def _tags(raw: object, taxonomy: Taxonomy, chosen: dict[str, list]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        label = text.strip()
        if not label or len(label) > 8:
            return
        key = label.casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(label)

    for item in _as_list(raw):
        token = str(_token(item) or "").strip()
        if not token:
            continue
        zh = _lookup_zh(taxonomy, token)
        add(zh or token)
    if found:
        return found[:12]
    for axis_name, ids in chosen.items():
        axis = taxonomy.axis(axis_name)
        names = {option.id: option.zh for option in axis.options}
        for item in ids:
            if item:
                add(names.get(item, item))
    if not found:
        raise ValueError("tags 不能为空")
    return found[:12]


def _lookup_zh(taxonomy: Taxonomy, token: str) -> str | None:
    key = token.casefold()
    for axis in taxonomy.axes.values():
        option_id = axis.lookup.get(key)
        if option_id is None:
            continue
        for option in axis.options:
            if option.id == option_id:
                return option.zh
    return None
