"""单张与批量识别。图片读取并发，模型按批生成，失败最多重试 3 次。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import ValidationError

from jewelry_tagger.prompt import build_prompt, repair_prompt
from jewelry_tagger.repair import loads_repaired
from jewelry_tagger.schema import JewelryTags, parse_tags
from jewelry_tagger.sources import ImageSourceError, load_image
from jewelry_tagger.taxonomy import Taxonomy, load_taxonomy


class TaggingFailed(RuntimeError):
    pass


@dataclass
class TagResult:
    source: str
    tags: JewelryTags | None
    error: str | None
    attempts: int


class JewelryTagger:
    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device: str = "auto",
        config_path: str | None = None,
        retries: int = 3,
        max_new_tokens: int = 700,
        pan_and_scan: bool = False,
        backend: str = "transformers",
        vllm_url: str = "http://127.0.0.1:8000/v1",
        engine=None,
        taxonomy: Taxonomy | None = None,
    ):
        if retries < 1:
            raise ValueError("retries 至少为 1")
        self.retries = retries
        self.max_new_tokens = max_new_tokens
        self.taxonomy = taxonomy or load_taxonomy(config_path)
        self.prompt = build_prompt(self.taxonomy)
        if engine is None:
            if backend == "vllm":
                from jewelry_tagger.vllm_engine import VllmEngine

                engine = VllmEngine(base_url=vllm_url, model_id=model_id)
            elif backend == "transformers":
                from jewelry_tagger.engine import GemmaEngine

                engine = GemmaEngine(model_id=model_id, device=device, pan_and_scan=pan_and_scan)
            else:
                raise ValueError("backend 只能是 transformers 或 vllm")
        self.engine = engine

    def tag(self, source: str) -> JewelryTags:
        result = self.tag_many([source])[0]
        if result.tags is None:
            raise TaggingFailed(result.error or "没有得到标签")
        return result.tags

    def tag_many(self, sources: list[str], batch_size: int = 1, workers: int = 4) -> list[TagResult]:
        if batch_size < 1:
            raise ValueError("batch_size 至少为 1")
        loaded: dict[int, object] = {}
        failed: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(load_image, source): index for index, source in enumerate(sources)}
            for future, index in futures.items():
                try:
                    loaded[index] = future.result()
                except ImageSourceError as exc:
                    failed[index] = str(exc)

        pending = [index for index in range(len(sources)) if index in loaded]
        attempts = {index: 0 for index in pending}
        prompts = {index: self.prompt for index in pending}
        done: dict[int, TagResult] = {}

        while pending:
            wave = pending[:batch_size]
            pending = pending[batch_size:]
            pairs = [(loaded[index], prompts[index]) for index in wave]
            texts = self.engine.complete_batch(pairs, self.max_new_tokens)
            retry: list[int] = []
            for index, text in zip(wave, texts):
                attempts[index] += 1
                try:
                    tags = parse_tags(loads_repaired(text), self.taxonomy)
                except (ValueError, ValidationError) as exc:
                    if attempts[index] >= self.retries:
                        done[index] = TagResult(sources[index], None, str(exc), attempts[index])
                    else:
                        prompts[index] = repair_prompt(self.prompt, text, str(exc))
                        retry.append(index)
                    continue
                done[index] = TagResult(sources[index], tags, None, attempts[index])
            pending = retry + pending

        results = []
        for index, source in enumerate(sources):
            if index in failed:
                results.append(TagResult(source, None, failed[index], 0))
            else:
                results.append(done[index])
        return results
