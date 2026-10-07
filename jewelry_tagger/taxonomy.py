"""从 YAML 读取珠宝标签体系，并把中文或英文别名收成 id。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "jewelry_tags.yaml"


@dataclass(frozen=True)
class TagOption:
    id: str
    zh: str
    en: str
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class Axis:
    name: str
    multiple: bool
    nullable: bool
    options: tuple[TagOption, ...]
    lookup: dict[str, str]

    def resolve(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "无"}:
            return None
        found = self.lookup.get(text.casefold())
        if found is None:
            allowed = "、".join(item.id for item in self.options)
            raise ValueError(f"{self.name} 不能是 {text}。只能用：{allowed}")
        return found

    def allows(self, option_id: str, category: str) -> bool:
        for option in self.options:
            if option.id != option_id:
                continue
            return not option.categories or category in option.categories
        return False


@dataclass
class Taxonomy:
    axes: dict[str, Axis]
    extra_axes: dict[str, Axis] = field(default_factory=dict)

    def axis(self, name: str) -> Axis:
        if name not in self.axes:
            raise KeyError(name)
        return self.axes[name]

    def resolve_many(self, name: str, values: object) -> list[str]:
        axis = self.axis(name) if name in self.axes else self.extra_axes[name]
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            raise ValueError(f"{name} 必须是数组")
        found: list[str] = []
        seen: set[str] = set()
        for item in values:
            token = item["name"] if isinstance(item, dict) and "name" in item else item
            resolved = axis.resolve(token)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
        return found


def _build_axis(name: str, spec: dict) -> Axis:
    options = tuple(
        TagOption(
            id=str(item["id"]),
            zh=str(item.get("zh") or item["id"]),
            en=str(item.get("en") or item["id"]),
            categories=_categories(item.get("categories")),
        )
        for item in spec.get("options") or []
    )
    if not options:
        raise ValueError(f"{name} 至少要有一个标签")
    lookup: dict[str, str] = {}
    for option in options:
        for alias in (option.id, option.zh, option.en):
            lookup[alias.casefold()] = option.id
    return Axis(
        name=name,
        multiple=bool(spec.get("multiple")),
        nullable=bool(spec.get("nullable")),
        options=options,
        lookup=lookup,
    )


def _categories(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    config_path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    axes = {name: _build_axis(name, spec) for name, spec in (data.get("axes") or {}).items()}
    required = {"category", "material", "cut", "design", "motif", "fineness", "style", "occasion", "background", "angle", "lighting"}
    missing = required - set(axes)
    if missing:
        raise ValueError("配置缺少标签轴：" + "、".join(sorted(missing)))
    extras = {name: _build_axis(name, spec) for name, spec in (data.get("extra_axes") or {}).items()}
    return Taxonomy(axes=axes, extra_axes=extras)
