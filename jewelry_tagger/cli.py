"""命令行：python -m jewelry_tagger photo.jpg"""

from __future__ import annotations

import argparse
import json
import sys

from jewelry_tagger.tagger import JewelryTagger


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="用 Gemma 3 给珠宝图片写结构化标签")
    parser.add_argument("images", nargs="+", help="本地路径或 http(s) URL，支持 JPG、PNG、WEBP")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="google/gemma-3-4b-it 或 google/gemma-3-27b-it")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--config", default=None, help="标签体系 YAML，默认使用 configs/jewelry_tags.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="同时读取图片的线程数")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--pan-and-scan", action="store_true", help="高分辨率珠宝图切开再看，更慢、更清楚")
    parser.add_argument("-o", "--output", help="把 JSON 写入文件")
    args = parser.parse_args(argv)

    tagger = JewelryTagger(
        model_id=args.model,
        device=args.device,
        config_path=args.config,
        retries=args.retries,
        max_new_tokens=args.max_new_tokens,
        pan_and_scan=args.pan_and_scan,
    )
    results = tagger.tag_many(args.images, batch_size=args.batch_size, workers=args.workers)
    payload = [
        {
            "source": item.source,
            "attempts": item.attempts,
            "error": item.error,
            "tags": None if item.tags is None else item.tags.model_dump(),
        }
        for item in results
    ]
    body = payload[0] if len(payload) == 1 else payload
    text = json.dumps(body, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if all(item.tags is not None for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
