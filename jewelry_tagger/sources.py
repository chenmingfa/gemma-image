"""读取本地或 URL 上的 JPG、PNG、WEBP。"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

MAX_BYTES = 30 * 1024 * 1024
ALLOWED = {"JPEG", "PNG", "WEBP"}


class ImageSourceError(ValueError):
    pass


def load_image(source: str, max_side: int = 2048) -> Image.Image:
    if source.startswith(("http://", "https://")):
        data = _download(source)
        name = source
    else:
        path = Path(source)
        if not path.is_file():
            raise ImageSourceError(f"找不到图片：{source}")
        if path.stat().st_size > MAX_BYTES:
            raise ImageSourceError(f"图片超过 {MAX_BYTES // (1024 * 1024)}MB：{source}")
        data = path.read_bytes()
        name = path.name
    return _decode(data, name, max_side)


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "jewelry-tagger"})
    try:
        with urlopen(request, timeout=20) as response:
            chunks = []
            total = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ImageSourceError(f"图片超过 {MAX_BYTES // (1024 * 1024)}MB：{url}")
                chunks.append(chunk)
    except URLError as exc:
        raise ImageSourceError(f"图片地址打不开：{url}") from exc
    return b"".join(chunks)


def _resize(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    try:
        import cv2
        import numpy as np
    except ImportError:
        image.thumbnail((max_side, max_side))
        return image
    array = np.array(image)
    height, width = array.shape[:2]
    scale = max_side / max(height, width)
    resized = cv2.resize(
        array,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return Image.fromarray(resized)


def _decode(data: bytes, name: str, max_side: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if getattr(image, "is_animated", False):
                image.seek(0)
            fmt = (image.format or "").upper()
            if fmt == "JPG":
                fmt = "JPEG"
            if fmt not in ALLOWED:
                raise ImageSourceError(f"只支持 JPG、PNG、WEBP，这张是 {fmt or '未知格式'}：{name}")
            rgb = image.convert("RGB")
            rgb.load()
            return _resize(rgb, max_side).copy()
    except ImageSourceError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageSourceError(f"这不是可读的图片：{name}") from exc
