"""本地网页：选择文件夹、看识别进度、按标签取图。"""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from gemma import find_ollama, ollama_status, pull_model
from images import write_thumb
from indexer import cancel_index, job_snapshot, start_index
from store import ROOT, get_library

WEB = ROOT / "web"
_pull_lock = threading.Lock()
_pull = {"running": False, "message": "", "ok": None}


def pull_snapshot() -> dict:
    with _pull_lock:
        return dict(_pull)


def start_pull() -> tuple[bool, str]:
    status = ollama_status()
    if not status["online"] and find_ollama() is None:
        return False, status["hint"]
    model = status["expected"]
    with _pull_lock:
        if _pull["running"]:
            return False, "已经在下载"
        _pull.update({"running": True, "message": f"开始下载 {model}", "ok": None})
    threading.Thread(target=_pull_worker, args=(model,), daemon=True, name="gemma-pull").start()
    return True, f"开始下载 {model}"


def _pull_worker(model: str) -> None:
    def on_line(line: str) -> None:
        with _pull_lock:
            _pull["message"] = line

    try:
        code = pull_model(model, on_line)
    except Exception as exc:
        with _pull_lock:
            _pull.update({"running": False, "message": str(exc), "ok": False})
        return
    with _pull_lock:
        if code == 0:
            _pull.update({"running": False, "message": f"{model} 已下载", "ok": True})
        else:
            _pull.update({"running": False, "message": f"下载没有完成（{code}）", "ok": False})


def pick_folder() -> str:
    if os.name != "nt":
        return ""
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "选择图片文件夹"
$dialog.ShowNewFolderButton = $false
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Show()
$result = $dialog.ShowDialog($owner)
$owner.Close()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  [Console]::Out.Write($dialog.SelectedPath)
}
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return _decode_output(proc.stdout).strip()


def reveal_path(path: str) -> None:
    full = os.path.normpath(path)
    subprocess.Popen(["explorer", f"/select,{full}"])


def _decode_output(raw: bytes) -> str:
    if not raw:
        return ""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff") or b"\x00" in raw[:80]:
        return raw.decode("utf-16", "ignore")
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


class Handler(BaseHTTPRequestHandler):
    server_version = "GemmaImageLib/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(WEB / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.css":
            self._send_file(WEB / "app.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file(WEB / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path == "/api/status":
            library = get_library()
            self._json(200, {"model": ollama_status(), "job": job_snapshot(), "pull": pull_snapshot(), "counts": library.counts()})
            return
        if path == "/api/tags":
            self._json(200, {"tags": get_library().top_tags(48)})
            return
        if path == "/api/search":
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get("q", [""])[0]
            match = qs.get("match", ["all"])[0]
            self._json(200, get_library().search(query, match))
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "images" and parts[2].isdigit():
            self._serve_image(int(parts[2]), parts[3])
            return
        self._json(404, {"error": "没有这个地址"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            data = self._read_json()
        except json.JSONDecodeError:
            self._json(400, {"error": "请求不是 JSON"})
            return
        if path == "/api/pick-folder":
            self._json(200, {"path": pick_folder()})
            return
        if path == "/api/index":
            ok, message = start_index(str(data.get("folder") or ""), bool(data.get("force")))
            self._json(200 if ok else 400, {"ok": ok, "message": message, "job": job_snapshot()})
            return
        if path == "/api/index/cancel":
            cancel_index()
            self._json(200, {"ok": True, "job": job_snapshot()})
            return
        if path == "/api/model/pull":
            ok, message = start_pull()
            self._json(200 if ok else 400, {"ok": ok, "message": message, "pull": pull_snapshot()})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "images" and parts[2].isdigit():
            image_id = int(parts[2])
            action = parts[3]
            if action == "tags":
                tags = data.get("tags") or []
                if isinstance(tags, str):
                    from store import tokenize

                    tags = tokenize(tags)
                item = get_library().set_user_tags(image_id, list(tags))
                if item is None:
                    self._json(404, {"error": "图片不在库里"})
                    return
                self._json(200, {"ok": True, "image": item})
                return
            if action == "recognize":
                self._recognize_one(image_id)
                return
            if action == "reveal":
                item = get_library().get(image_id)
                if item is None:
                    self._json(404, {"error": "图片不在库里"})
                    return
                reveal_path(item["path"])
                self._json(200, {"ok": True})
                return
        self._json(404, {"error": "没有这个地址"})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "images" and parts[2].isdigit():
            removed = get_library().delete(int(parts[2]))
            self._json(200 if removed else 404, {"ok": removed})
            return
        self._json(404, {"error": "没有这个地址"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[图片库] {self.address_string()} {fmt % args}", flush=True)

    def _recognize_one(self, image_id: int) -> None:
        from gemma import ModelMissing, OllamaUnavailable, require_model
        from indexer import index_file

        item = get_library().get(image_id)
        if item is None:
            self._json(404, {"error": "图片不在库里"})
            return
        path = Path(item["path"])
        if not path.is_file():
            self._json(400, {"error": "原图文件不在了"})
            return
        if job_snapshot()["running"]:
            self._json(400, {"ok": False, "error": "文件夹识别还在进行，稍后再看这一张"})
            return
        try:
            model = require_model()
            result = index_file(path, model, force=True, respect_user=False)
        except (OllamaUnavailable, ModelMissing) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc)})
            return
        fresh = get_library().get(image_id)
        if result != "ok" or fresh is None:
            self._json(500, {"ok": False, "error": (fresh or {}).get("error") or "没有写出标签"})
            return
        self._json(200, {"ok": True, "image": fresh})

    def _serve_image(self, image_id: int, kind: str) -> None:
        item = get_library().get(image_id)
        if item is None or item["missing"]:
            self._json(404, {"error": "找不到原图"})
            return
        src = Path(item["path"])
        if kind == "file":
            mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
            self._send_file(src, mime)
            return
        if kind == "thumb":
            dest = ROOT / "data" / "thumbs" / f"{image_id}.jpg"
            try:
                src_mtime = src.stat().st_mtime
                if not dest.exists() or dest.stat().st_mtime < src_mtime:
                    write_thumb(src, dest)
                self._send_file(dest, "image/jpeg")
            except Exception:
                mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
                self._send_file(src, mime)
            return
        self._json(404, {"error": "没有这个地址"})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._json(404, {"error": "文件不在"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise SystemExit(f"端口 {port} 无法打开：{exc}") from exc
    url = f"http://127.0.0.1:{port}"
    print(f"本地图片库：{url}", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭", flush=True)
    finally:
        httpd.server_close()
