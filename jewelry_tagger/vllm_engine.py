"""把图片发给本机 vLLM 的 OpenAI 兼容接口。模型进程单独启动。"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image


class VllmEngine:
    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model_id: str = "google/gemma-3-4b-it"):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    def complete(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        return self.complete_batch([(image, prompt)], max_new_tokens)[0]

    def complete_batch(self, pairs: list[tuple[Image.Image, str]], max_new_tokens: int) -> list[str]:
        return [self._one(image, prompt, max_new_tokens) for image, prompt in pairs]

    def _one(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM 没有连上：{self.base_url}") from exc
        return data["choices"][0]["message"]["content"]


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded
