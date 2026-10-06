"""本地图片库的存储。标签和原图路径都在这台电脑上。"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GENERIC_FOLDERS = {
    "dcim",
    "camera",
    "cameras",
    "pictures",
    "photos",
    "photo",
    "images",
    "image",
    "img",
    "imgs",
    "screenshots",
    "screenshot",
    "download",
    "downloads",
    "desktop",
    "camera roll",
    "saved pictures",
    "桌面",
    "图片",
    "照片",
    "新建文件夹",
}


def default_db_path() -> Path:
    env = os.environ.get("LIBRARY_DB")
    if env:
        return Path(env)
    return ROOT / "data" / "library.db"


def canon_path(path: str | Path) -> str:
    return os.path.normpath(os.path.abspath(path))


def path_key(path: str | Path) -> str:
    return os.path.normcase(canon_path(path))


def tokenize(query: str) -> list[str]:
    return [part for part in re.split(r"[\s,，、;；]+", query.strip()) if part]


def normalize_tag(value: str) -> str:
    tag = value.strip().strip("#").strip("，,。；;、")
    tag = re.sub(r"\s+", " ", tag).strip()
    if not tag or len(tag) > 16:
        return ""
    if any(ch in tag for ch in "{}[]"):
        return ""
    if not re.search(r"[\w\u4e00-\u9fff]", tag):
        return ""
    return tag


def folder_hint(path: str | Path) -> str:
    name = Path(path).parent.name.strip()
    if not name or name.casefold() in GENERIC_FOLDERS:
        return ""
    if re.fullmatch(r"[\d_\-\s.年月日]+", name):
        return ""
    if len(name) > 20:
        return ""
    return name


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Library:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path) if db_path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                path_key TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                folder TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                caption TEXT,
                model TEXT,
                indexed_at TEXT,
                user_edited INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS image_tags (
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                source TEXT NOT NULL,
                position INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (image_id, tag_id)
            );

            CREATE INDEX IF NOT EXISTS idx_image_tags_image ON image_tags(image_id);
            """
        )
        self.conn.commit()

    def counts(self) -> dict[str, int]:
        with self._lock:
            images = self.conn.execute(
                """
                SELECT COUNT(*) AS n FROM images
                WHERE caption IS NOT NULL OR user_edited = 1
                """
            ).fetchone()["n"]
            tags = self.conn.execute("SELECT COUNT(*) AS n FROM tags").fetchone()["n"]
        return {"images": images, "tags": tags}

    def get_by_path(self, path: str | Path) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM images WHERE path_key = ?",
                (path_key(path),),
            ).fetchone()
        return dict(row) if row else None

    def get(self, image_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
            if row is None:
                return None
            item = self._public(row, self._tags_for(image_id))
        item["missing"] = not os.path.exists(item["path"])
        return item

    def top_tags(self, limit: int = 40) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT t.name AS name,
                       COUNT(*) AS count,
                       SUM(CASE WHEN it.source = 'user' THEN 2 ELSE 1 END) AS weight
                FROM tags t
                JOIN image_tags it ON it.tag_id = t.id
                GROUP BY t.id
                ORDER BY weight DESC, count DESC, t.name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [{"name": row["name"], "count": row["count"]} for row in rows]

    def search(self, query: str, match: str = "all", limit: int = 300) -> dict:
        tokens = tokenize(query)
        mode = "any" if match == "any" else "all"
        with self._lock:
            images = self._load_unlocked()
        matched = []
        for item in images:
            score = _score(item, tokens)
            if tokens and mode == "all" and score < len(tokens):
                continue
            if tokens and mode == "any" and score <= 0:
                continue
            item["score"] = score
            item["missing"] = not os.path.exists(item["path"])
            matched.append(item)
        if tokens:
            matched.sort(key=lambda item: (item["score"], item["indexed_at"] or ""), reverse=True)
        total = len(matched)
        return {
            "images": matched[:limit],
            "total": total,
            "truncated": total > limit,
            "query": tokens,
            "match": mode,
        }

    def save_success(
        self,
        path: str | Path,
        *,
        mtime: float,
        size: int,
        width: int | None,
        height: int | None,
        caption: str,
        tags: list[str],
        model: str,
        source: str = "model",
    ) -> dict:
        with self._lock:
            image_id = self._upsert_image(
                path,
                mtime=mtime,
                size=size,
                width=width,
                height=height,
                caption=caption,
                model=model,
                user_edited=1 if source == "user" else 0,
                error=None,
            )
            self._replace_tags(image_id, tags, source)
            self.conn.commit()
            row = self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            item = self._public(row, self._tags_for(image_id))
        item["missing"] = not os.path.exists(item["path"])
        return item

    def save_error(self, path: str | Path, *, mtime: float, size: int, message: str) -> None:
        with self._lock:
            key = path_key(path)
            existing = self.conn.execute(
                "SELECT id, caption FROM images WHERE path_key = ?",
                (key,),
            ).fetchone()
            text = message.strip()[:300]
            if existing and existing["caption"]:
                self.conn.execute(
                    "UPDATE images SET error = ? WHERE id = ?",
                    (text, existing["id"]),
                )
            else:
                self._upsert_image(
                    path,
                    mtime=mtime,
                    size=size,
                    width=None,
                    height=None,
                    caption=None,
                    model=None,
                    user_edited=0,
                    error=text,
                )
            self.conn.commit()

    def set_user_tags(self, image_id: int, tags: list[str]) -> dict | None:
        clean = _dedupe_tags(tags)
        with self._lock:
            row = self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is None:
                return None
            self.conn.execute(
                """
                UPDATE images
                SET user_edited = 1, error = NULL, indexed_at = ?
                WHERE id = ?
                """,
                (_now(), image_id),
            )
            self._replace_tags(image_id, clean, "user")
            self.conn.commit()
            row = self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            item = self._public(row, self._tags_for(image_id))
        item["missing"] = not os.path.exists(item["path"])
        return item

    def delete(self, image_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            self._drop_orphan_tags()
            self.conn.commit()
            return cur.rowcount > 0

    def _upsert_image(
        self,
        path: str | Path,
        *,
        mtime: float,
        size: int,
        width: int | None,
        height: int | None,
        caption: str | None,
        model: str | None,
        user_edited: int,
        error: str | None,
    ) -> int:
        full = canon_path(path)
        key = path_key(path)
        folder = str(Path(full).parent)
        filename = Path(full).name
        existing = self.conn.execute(
            "SELECT id FROM images WHERE path_key = ?",
            (key,),
        ).fetchone()
        fields = (
            full,
            filename,
            folder,
            mtime,
            size,
            width,
            height,
            caption,
            model,
            _now() if caption or user_edited else None,
            user_edited,
            error,
        )
        if existing:
            self.conn.execute(
                """
                UPDATE images
                SET path = ?, filename = ?, folder = ?, mtime = ?, size = ?,
                    width = ?, height = ?, caption = ?, model = ?, indexed_at = ?,
                    user_edited = ?, error = ?
                WHERE id = ?
                """,
                (*fields, existing["id"]),
            )
            return existing["id"]
        cur = self.conn.execute(
            """
            INSERT INTO images (
                path, path_key, filename, folder, mtime, size, width, height,
                caption, model, indexed_at, user_edited, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (full, key, filename, folder, mtime, size, width, height, caption, model, fields[9], user_edited, error),
        )
        return int(cur.lastrowid)

    def _replace_tags(self, image_id: int, tags: list[str], source: str) -> None:
        self.conn.execute("DELETE FROM image_tags WHERE image_id = ?", (image_id,))
        for position, tag in enumerate(_dedupe_tags(tags)):
            key = tag.casefold()
            row = self.conn.execute(
                "SELECT id FROM tags WHERE name_key = ?",
                (key,),
            ).fetchone()
            if row:
                tag_id = row["id"]
            else:
                cur = self.conn.execute(
                    "INSERT INTO tags (name, name_key) VALUES (?, ?)",
                    (tag, key),
                )
                tag_id = int(cur.lastrowid)
            self.conn.execute(
                """
                INSERT INTO image_tags (image_id, tag_id, source, position)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, tag_id, source, position),
            )
        self._drop_orphan_tags()

    def _drop_orphan_tags(self) -> None:
        self.conn.execute(
            "DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM image_tags)"
        )

    def _tags_for(self, image_id: int) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT t.name
            FROM image_tags it
            JOIN tags t ON t.id = it.tag_id
            WHERE it.image_id = ?
            ORDER BY it.position
            """,
            (image_id,),
        ).fetchall()
        return [row["name"] for row in rows]

    def _load_unlocked(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT i.*, t.name AS tag
            FROM images i
            LEFT JOIN image_tags it ON it.image_id = i.id
            LEFT JOIN tags t ON t.id = it.tag_id
            WHERE i.caption IS NOT NULL OR i.user_edited = 1
            ORDER BY i.indexed_at DESC, i.id DESC, it.position ASC
            """
        ).fetchall()
        grouped: list[dict] = []
        current: dict | None = None
        for row in rows:
            if current is None or current["id"] != row["id"]:
                current = self._public(row, [])
                grouped.append(current)
            if row["tag"]:
                current["tags"].append(row["tag"])
        return grouped

    def _public(self, row: sqlite3.Row, tags: list[str]) -> dict:
        return {
            "id": row["id"],
            "path": row["path"],
            "filename": row["filename"],
            "folder": row["folder"],
            "caption": row["caption"] or "",
            "tags": list(tags),
            "width": row["width"],
            "height": row["height"],
            "indexed_at": row["indexed_at"],
            "user_edited": bool(row["user_edited"]),
            "model": row["model"],
            "error": row["error"],
        }


def _dedupe_tags(tags: list[str]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = normalize_tag(str(raw))
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        clean.append(tag)
        if len(clean) >= 16:
            break
    return clean


def _token_hit(token: str, tags: list[str], caption: str) -> bool:
    needle = token.casefold()
    for tag in tags:
        hay = tag.casefold()
        if needle == hay or needle in hay or hay in needle:
            return True
    return needle in caption.casefold()


def _score(item: dict, tokens: list[str]) -> int:
    if not tokens:
        return 1
    return sum(1 for token in tokens if _token_hit(token, item["tags"], item["caption"]))


_library: Library | None = None
_library_lock = threading.Lock()


def get_library() -> Library:
    global _library
    with _library_lock:
        if _library is None:
            _library = Library()
        return _library
