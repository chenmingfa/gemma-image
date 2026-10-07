"""用 Transformers 加载 Gemma 3，并在 GPU / CPU 之间自动切换。"""

from __future__ import annotations

import threading

from PIL import Image


class GemmaEngine:
    def __init__(self, model_id: str = "google/gemma-3-4b-it", device: str = "auto", pan_and_scan: bool = False):
        self.model_id = model_id
        self.device_pref = device
        self.pan_and_scan = pan_and_scan
        self.device = "cpu"
        self.dtype = None
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.device, self.dtype = resolve_device(self.device_pref)
        device_map: str | dict = "auto" if self.device == "cuda" else {"": self.device}
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=device_map,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left")

    def complete(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        return self.complete_batch([(image, prompt)], max_new_tokens)[0]

    def complete_batch(self, pairs: list[tuple[Image.Image, str]], max_new_tokens: int) -> list[str]:
        if not pairs:
            return []
        self.load()
        with self._lock:
            try:
                return self._generate(pairs, max_new_tokens)
            except (RuntimeError, TypeError, ValueError):
                if len(pairs) == 1:
                    raise
                texts = []
                for pair in pairs:
                    texts.extend(self._generate([pair], max_new_tokens))
                return texts

    def _generate(self, pairs: list[tuple[Image.Image, str]], max_new_tokens: int) -> list[str]:
        import torch

        conversations = [_messages(image, prompt) for image, prompt in pairs]
        processor = self._processor
        model = self._model
        assert processor is not None and model is not None
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": len(conversations) > 1,
        }
        if self.pan_and_scan:
            template_kwargs["do_pan_and_scan"] = True
        inputs = processor.apply_chat_template(conversations if len(conversations) > 1 else conversations[0], **template_kwargs)
        inputs = inputs.to(model.device)
        if "pixel_values" in inputs and self.dtype is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.batch_decode(output[:, input_len:], skip_special_tokens=True)


def resolve_device(prefer: str):
    import torch

    choice = (prefer or "auto").lower()
    if choice == "auto":
        if torch.cuda.is_available():
            choice = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            choice = "mps"
        else:
            choice = "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但这台机器没有可用的 GPU。")
        return "cuda", torch.bfloat16
    if choice == "mps":
        return "mps", torch.float16
    if choice == "cpu":
        return "cpu", torch.float32
    raise RuntimeError(f"不认识的设备：{prefer}。可以用 auto、cuda、mps、cpu。")


def _messages(image: Image.Image, prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你只输出珠宝标签 JSON。"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
