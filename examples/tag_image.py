"""Python API 示例。需要已登录 Hugging Face 并接受 Gemma 许可。

  .venv\\Scripts\\python examples\\tag_image.py path\\to\\ring.jpg
"""

from __future__ import annotations

import json
import sys

from jewelry_tagger import JewelryTagger


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python examples/tag_image.py 图片.jpg [更多图片...]", file=sys.stderr)
        return 2
    tagger = JewelryTagger(model_id="google/gemma-3-4b-it", device="auto")
    if len(sys.argv) == 2:
        tags = tagger.tag(sys.argv[1])
        print(json.dumps(tags.model_dump(), ensure_ascii=False, indent=2))
        return 0
    results = tagger.tag_many(sys.argv[1:], batch_size=2, workers=4)
    payload = [
        {"source": item.source, "error": item.error, "tags": None if item.tags is None else item.tags.model_dump()}
        for item in results
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item.tags is not None for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
