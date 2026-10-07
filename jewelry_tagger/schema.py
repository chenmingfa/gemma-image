"""识别结果的结构。字段和示例 JSON 对齐，取值由配置文件约束。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from jewelry_tagger.taxonomy import Taxonomy

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_COLOR_NAMES = {
    "gold": "#FFD700",
    "white": "#FFFFFF",
    "black": "#000000",
    "red": "#C41E3A",
    "blue": "#0F52BA",
    "green": "#046307",
    "pink": "#FFC0CB",
    "silver": "#C0C0C0",
    "金色": "#FFD700",
    "白色": "#FFFFFF",
    "黑色": "#000000",
    "红色": "#C41E3A",
    "蓝色": "#0F52BA",
    "绿色": "#046307",
    "粉色": "#FFC0CB",
    "银色": "#C0C0C0",
}


class CategoryTag(BaseModel):
    main: str
    confidence: float = Field(ge=0, le=1)


class MaterialTag(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)


class CutTag(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)


class ColorTag(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)

    @field_validator("primary")
    @classmethod
    def _primary_hex(cls, value: str) -> str:
        return _color_to_hex(value)

    @field_validator("secondary")
    @classmethod
    def _secondary_hex(cls, values: list[str]) -> list[str]:
        return [_color_to_hex(value) for value in values]


class VisualTag(BaseModel):
    background: str
    angle: str
    lighting: str


class JewelryTags(BaseModel):
    category: CategoryTag
    material: list[MaterialTag] = Field(min_length=1)
    cut: CutTag | None
    style: list[str] = Field(min_length=1)
    occasion: list[str] = Field(min_length=1)
    color: ColorTag
    visual: VisualTag
    caption: str = Field(min_length=1, max_length=80)
    extras: dict[str, list[str]] = Field(default_factory=dict)


def _color_to_hex(value: str) -> str:
    text = value.strip()
    if _HEX.match(text):
        return text.upper()
    named = _COLOR_NAMES.get(text.casefold()) or _COLOR_NAMES.get(text)
    if named:
        return named
    raise ValueError(f"颜色要写成 #RRGGBB，收到的是 {value}")


def _confidence(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    number = float(value)
    if number > 1:
        number = number / 100
    if number < 0 or number > 1:
        raise ValueError(f"置信度必须在 0 到 1 之间，收到的是 {value}")
    return number


def normalize_payload(data: dict, taxonomy: Taxonomy) -> dict[str, Any]:
    """把模型常见的变形收成示例里的 JSON 形状，再用词表核对 id。"""
    category = data.get("category")
    if isinstance(category, str):
        category = {"main": category}
    if not isinstance(category, dict):
        raise ValueError("category 必须是对象")
    main = taxonomy.axis("category").resolve(category.get("main"))
    if main is None:
        raise ValueError("category.main 不能为空")

    materials = data.get("material")
    if isinstance(materials, dict):
        materials = [materials]
    if isinstance(materials, str):
        materials = [materials]
    if not isinstance(materials, list) or not materials:
        raise ValueError("material 至少要有一项")
    material_items = []
    for item in materials:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            raise ValueError("material 里的每一项必须是对象或字符串")
        name = taxonomy.axis("material").resolve(item.get("name"))
        if name is None:
            continue
        confidence = _confidence(item.get("confidence"))
        if confidence is None:
            raise ValueError(f"material.{name} 缺少 confidence")
        material_items.append({"name": name, "confidence": confidence})
    if not material_items:
        raise ValueError("material 没有可用的材质")

    cut_raw = data.get("cut")
    cut: dict | None
    if cut_raw in (None, "", "null"):
        cut = None
    else:
        if isinstance(cut_raw, str):
            cut_raw = {"name": cut_raw}
        if not isinstance(cut_raw, dict):
            raise ValueError("cut 必须是对象或 null")
        cut_name = taxonomy.axis("cut").resolve(cut_raw.get("name"))
        if cut_name is None:
            cut = None
        else:
            cut_confidence = _confidence(cut_raw.get("confidence"))
            if cut_confidence is None:
                raise ValueError("cut.confidence 缺失")
            cut = {"name": cut_name, "confidence": cut_confidence}

    color = data.get("color") or {}
    if isinstance(color, str):
        color = {"primary": color}
    if not isinstance(color, dict) or not color.get("primary"):
        raise ValueError("color.primary 缺失")
    secondary = color.get("secondary") or []
    if isinstance(secondary, str):
        secondary = [secondary]

    visual = data.get("visual") or {}
    if not isinstance(visual, dict):
        raise ValueError("visual 必须是对象")

    extras: dict[str, list[str]] = {}
    for name in taxonomy.extra_axes:
        raw_extra = data.get("extras", {}).get(name) if isinstance(data.get("extras"), dict) else None
        if raw_extra is None:
            raw_extra = data.get(name)
        if raw_extra is None:
            continue
        extras[name] = taxonomy.resolve_many(name, raw_extra)

    category_confidence = _confidence(category.get("confidence"))
    if category_confidence is None:
        raise ValueError("category.confidence 缺失")

    return {
        "category": {"main": main, "confidence": category_confidence},
        "material": material_items,
        "cut": cut,
        "style": taxonomy.resolve_many("style", data.get("style")),
        "occasion": taxonomy.resolve_many("occasion", data.get("occasion")),
        "color": {"primary": color.get("primary"), "secondary": secondary},
        "visual": {
            "background": taxonomy.axis("background").resolve(visual.get("background")),
            "angle": taxonomy.axis("angle").resolve(visual.get("angle")),
            "lighting": taxonomy.axis("lighting").resolve(visual.get("lighting")),
        },
        "caption": str(data.get("caption") or "").strip(),
        "extras": extras,
    }


def parse_tags(data: dict, taxonomy: Taxonomy) -> JewelryTags:
    return JewelryTags.model_validate(normalize_payload(data, taxonomy))
