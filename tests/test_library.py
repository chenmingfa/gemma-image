import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma import choose_model, parse_recognition
from store import Library, folder_hint, normalize_tag


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Library(Path(self.tmp.name) / "library.db")
        self.photo = Path(self.tmp.name) / "cat.jpg"
        self.photo.write_bytes(b"fake")
        self.sea = Path(self.tmp.name) / "sea.jpg"
        self.sea.write_bytes(b"fake")

    def tearDown(self):
        self.db.conn.close()
        self.tmp.cleanup()

    def test_search_all_and_any(self):
        self.db.save_success(
            self.photo, mtime=1, size=4, width=10, height=10,
            caption="一只猫坐在窗台", tags=["猫", "窗台", "室内"], model="gemma3:4b",
        )
        self.db.save_success(
            self.sea, mtime=1, size=4, width=10, height=10,
            caption="傍晚的海边", tags=["海边", "日落"], model="gemma3:4b",
        )
        both = self.db.search("猫 窗台", "all")
        self.assertEqual([item["filename"] for item in both["images"]], ["cat.jpg"])
        either = self.db.search("猫 海边", "any")
        self.assertEqual(either["total"], 2)
        none = self.db.search("海边 窗台", "all")
        self.assertEqual(none["total"], 0)
        by_caption = self.db.search("傍晚")
        self.assertEqual(by_caption["images"][0]["filename"], "sea.jpg")

    def test_user_tags_replace_and_rank(self):
        saved = self.db.save_success(
            self.photo, mtime=1, size=4, width=None, height=None,
            caption="猫", tags=["猫咪"], model="gemma3:4b",
        )
        edited = self.db.set_user_tags(saved["id"], ["猫", "猫咪", "窗边"])
        self.assertEqual(edited["tags"], ["猫", "猫咪", "窗边"])
        self.assertTrue(edited["user_edited"])
        self.assertEqual(self.db.top_tags(5)[0]["name"], "猫")

    def test_delete_removes_unused_tags(self):
        saved = self.db.save_success(
            self.photo, mtime=1, size=4, width=1, height=1,
            caption="猫", tags=["猫"], model="gemma3:4b",
        )
        self.assertTrue(self.db.delete(saved["id"]))
        self.assertEqual(self.db.counts()["tags"], 0)
        self.assertEqual(self.db.search("")["total"], 0)

    def test_normalize_and_folder_hint(self):
        self.assertEqual(normalize_tag("  猫  "), "猫")
        self.assertEqual(normalize_tag("这是一句长得不能当标签的话啊真的太长"), "")
        self.assertEqual(folder_hint(r"D:\照片\旅行\a.jpg"), "旅行")
        self.assertEqual(folder_hint(r"D:\Pictures\a.jpg"), "")
        self.assertEqual(folder_hint(r"D:\DCIM\2024\a.jpg"), "")

    def test_parse_and_choose_model(self):
        caption, tags = parse_recognition('{"caption":"一只白猫","tags":["猫","窗台","猫"]}')
        self.assertEqual(caption, "一只白猫")
        self.assertEqual(tags, ["猫", "窗台"])
        self.assertEqual(choose_model(["llama3", "gemma3:12b"]), "gemma3:12b")
        self.assertEqual(choose_model(["gemma3:4b", "gemma3:27b"]), "gemma3:4b")
        self.assertIsNone(choose_model(["gemma3:1b"]))
        self.assertEqual(choose_model(["gemma3:12b"], "gemma3:4b"), None)
        self.assertEqual(choose_model(["gemma3:4b-it-qat"], "gemma3:4b"), "gemma3:4b-it-qat")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    unittest.main()
