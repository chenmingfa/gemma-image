"""本地 HTTP 接口：上传珠宝图片，返回多标签 JSON。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from jewelry_tagger.sources import ImageSourceError
from jewelry_tagger.tagger import JewelryTagger, TaggingFailed

ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp"}


class UrlRequest(BaseModel):
    url: str


def create_app(tagger: JewelryTagger | None = None) -> FastAPI:
    app = FastAPI(title="珠宝图像标签", version="1")
    holder: dict[str, JewelryTagger | None] = {"tagger": tagger}

    def get_tagger() -> JewelryTagger:
        if holder["tagger"] is None:
            holder["tagger"] = JewelryTagger()
        return holder["tagger"]

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/v1/tag")
    async def tag_file(file: UploadFile = File(...)) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIX:
            raise HTTPException(status_code=400, detail="只支持 JPG、PNG、WEBP")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="文件是空的")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            path = handle.name
        try:
            tags = get_tagger().tag(path)
        except ImageSourceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TaggingFailed as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            Path(path).unlink(missing_ok=True)
        return tags.as_json()

    @app.post("/v1/tag/url")
    def tag_url(body: UrlRequest) -> dict:
        try:
            tags = get_tagger().tag(body.url)
        except ImageSourceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TaggingFailed as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return tags.as_json()

    return app


app = create_app()
