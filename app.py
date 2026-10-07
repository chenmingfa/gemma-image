"""本地图片库。

Gemma 3 在这台电脑上看图、写标签。之后按标签取回原图路径。

  python app.py
  python app.py index D:\\照片
  python app.py search 海边 猫
"""

from __future__ import annotations

import argparse
import sys

from gemma import ModelMissing, OllamaUnavailable, require_model
from indexer import index_file, list_images
from server import serve
from store import get_library


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="用 Gemma 3 把本地图片识别成可按标签检索的图片库")
    sub = parser.add_subparsers(dest="cmd")

    serve_parser = sub.add_parser("serve", help="打开本地网页")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--no-browser", action="store_true")

    index_parser = sub.add_parser("index", help="识别一个文件夹里的图片")
    index_parser.add_argument("folder")
    index_parser.add_argument("--force", action="store_true", help="重新识别未改过标签的图片")

    search_parser = sub.add_parser("search", help="按标签打印本地图片路径")
    search_parser.add_argument("tags", nargs="+")
    search_parser.add_argument("--match", choices=["all", "any"], default="all")

    args = parser.parse_args(argv)
    if args.cmd == "index":
        return _index_cli(args.folder, args.force)
    if args.cmd == "search":
        return _search_cli(args.tags, args.match)
    port = args.port if args.cmd == "serve" else 8765
    open_browser = not (args.cmd == "serve" and args.no_browser)
    serve(port, open_browser=open_browser)
    return 0


def _index_cli(folder: str, force: bool) -> int:
    from pathlib import Path

    root = Path(folder)
    if not root.is_dir():
        print("文件夹不存在", file=sys.stderr)
        return 1
    files = list_images(root)
    if not files:
        print("这个文件夹里没有支持的图片")
        return 0
    try:
        model = require_model()
    except (OllamaUnavailable, ModelMissing) as exc:
        print(exc, file=sys.stderr)
        return 1
    ok = skipped = errors = 0
    for path in files:
        try:
            result = index_file(path, model, force=force)
        except (OllamaUnavailable, ModelMissing) as exc:
            print(exc, file=sys.stderr)
            return 1
        if result == "ok":
            ok += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1
            print(f"失败 {path}", file=sys.stderr)
    print(f"完成。新识别 {ok}，跳过 {skipped}，失败 {errors}")
    return 0 if errors == 0 else 1


def _search_cli(tags: list[str], match: str) -> int:
    found = get_library().search(" ".join(tags), match, limit=None)
    images = found["images"]
    print(f"{found['total']} 张")
    for item in images:
        print(item["path"])
        print("  " + "、".join(item["tags"]))
        if item["caption"]:
            print("  " + item["caption"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
