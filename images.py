"""把原图缩小后再送给模型，并生成页面上用的缩略图。"""

from __future__ import annotations

import io
from pathlib import Path


def pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("需要安装 Pillow。在项目目录运行：.venv\\Scripts\\python -m pip install pillow") from exc
    return Image, ImageOps


def open_rgb(path: str | Path):
    Image, ImageOps = pillow()
    with Image.open(path) as im:
        fixed = ImageOps.exif_transpose(im) or im
        if getattr(fixed, "is_animated", False):
            fixed.seek(0)
        rgb = fixed.convert("RGB")
        rgb.load()
        return rgb


def model_jpeg(path: str | Path, max_side: int = 768) -> tuple[bytes, int, int]:
    image = open_rgb(path)
    width, height = image.size
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), width, height


def write_thumb(src: str | Path, dest: str | Path, max_side: int = 520) -> None:
    image = open_rgb(src)
    image.thumbnail((max_side, max_side))
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=82)
