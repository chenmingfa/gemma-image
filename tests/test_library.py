import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma import RecognizeError, arrange_jewelry_tags, choose_model, missing_style_tags, parse_recognition
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

    def test_search_pages_without_checking_every_file(self):
        paths = []
        for index in range(3):
            path = Path(self.tmp.name) / f"p{index}.jpg"
            path.write_bytes(b"x")
            paths.append(path)
            self.db.save_success(
                path, mtime=1, size=1, width=1, height=1,
                caption=str(index), tags=["图"], model="gemma3:4b",
            )
        os.remove(paths[0])
        first = self.db.search("", limit=2, offset=0)
        self.assertEqual(first["total"], 3)
        self.assertEqual(len(first["images"]), 2)
        self.assertTrue(first["truncated"])
        self.assertNotIn("missing", first["images"][0])
        rest = self.db.search("", limit=2, offset=2)
        self.assertEqual(len(rest["images"]), 1)
        self.assertFalse(rest["truncated"])
        seen = {item["id"] for item in first["images"] + rest["images"]}
        self.assertEqual(len(seen), 3)
        tagged = self.db.search("图", "all", limit=2, offset=0)
        self.assertEqual(tagged["total"], 3)
        self.assertEqual(len(tagged["images"]), 2)

    def test_thumb_stays_within_card_size(self):
        from PIL import Image

        from images import write_thumb

        src = Path(self.tmp.name) / "big.jpg"
        Image.new("RGB", (2400, 1600), (20, 80, 140)).save(src, quality=90)
        dest = Path(self.tmp.name) / "thumb.jpg"
        write_thumb(src, dest)
        with Image.open(dest) as im:
            self.assertLessEqual(max(im.size), 384)
            self.assertEqual(im.format, "JPEG")

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
        caption, tags = parse_recognition('{"说明":"一只白猫","标签":"猫、窗台、猫"}')
        self.assertEqual(caption, "一只白猫")
        self.assertEqual(tags, ["猫", "窗台"])
        caption, tags = parse_recognition("说明：一只白猫坐在窗台\n标签：猫、窗台、暖光")
        self.assertEqual(caption, "一只白猫坐在窗台")
        self.assertEqual(tags, ["猫", "窗台", "暖光"])
        with self.assertRaises(RecognizeError):
            parse_recognition("说明：玫瑰金钻戒\n标签：钻戒、玫瑰金、圆形、爪镶、暖光")
        _, shaped = parse_recognition("说明：玫瑰金光环钻戒\n标签：钻戒、光环戒、花朵、18K、玫瑰金、圆形")
        self.assertEqual(shaped[:4], ["戒指", "光环戒", "花朵", "18K"])
        self.assertEqual(missing_style_tags(["戒指", "光环戒"]), ["物体", "成色"])
        self.assertEqual(missing_style_tags(["猫", "窗台"]), [])
        with self.assertRaises(Exception):
            parse_recognition("这张图很好看，但没有按格式写。")
        self.assertEqual(choose_model(["llama3", "gemma3:12b"]), "gemma3:12b")
        self.assertEqual(choose_model(["gemma3:4b", "gemma3:27b"]), "gemma3:4b")
        self.assertIsNone(choose_model(["gemma3:1b"]))
        self.assertEqual(choose_model(["gemma3:12b"], "gemma3:4b"), None)
        self.assertEqual(choose_model(["gemma3:4b-it-qat"], "gemma3:4b"), "gemma3:4b-it-qat")

    def test_fineness_marks_normalize_without_guessing_from_color(self):
        self.assertEqual(arrange_jewelry_tags(["足金9999"]), ["万足金", "黄金"])
        self.assertEqual(arrange_jewelry_tags(["足金999"]), ["千足金", "黄金"])
        self.assertEqual(arrange_jewelry_tags(["足金"]), ["足金", "黄金"])
        self.assertEqual(arrange_jewelry_tags(["Au750"]), ["18K", "K金"])
        self.assertEqual(arrange_jewelry_tags(["18K金"]), ["18K", "K金"])
        self.assertEqual(arrange_jewelry_tags(["１８Ｋ"]), ["18K", "K金"])
        self.assertEqual(arrange_jewelry_tags(["S925"]), ["925银", "银"])
        self.assertEqual(arrange_jewelry_tags(["Pt950"]), ["PT950", "铂金"])
        self.assertEqual(arrange_jewelry_tags(["成色不明"]), ["成色不清"])
        self.assertEqual(arrange_jewelry_tags(["750"]), ["18K", "K金"])
        self.assertEqual(arrange_jewelry_tags(["编号750"]), ["编号750"])
        self.assertNotIn("成色不清", arrange_jewelry_tags(["18K", "成色不清"]))
        self.assertEqual(
            arrange_jewelry_tags(["18K", "PT950"]),
            ["18K", "PT950", "K金", "铂金"],
        )
        self.assertEqual(missing_style_tags(["戒指", "素圈戒", "素面", "K金"]), ["成色"])

        _, stamped = parse_recognition("说明：戒臂印有Au999\n标签：戒指、素圈戒、素面、18K、玫瑰金")
        self.assertIn("千足金", stamped)
        self.assertIn("黄金", stamped)
        self.assertIn("玫瑰金", stamped)
        self.assertNotIn("18K", stamped)
        self.assertNotIn("K金", stamped)

        _, silver = parse_recognition("说明：扣件刻着S925\n标签：手链、链节手链、链条、成色不清")
        self.assertIn("925银", silver)
        self.assertIn("银", silver)
        self.assertNotIn("成色不清", silver)

        _, inner = parse_recognition("说明：内圈印有750\n标签：戒指、素圈戒、素面、成色不清")
        self.assertIn("18K", inner)
        self.assertNotIn("成色不清", inner)

        _, color_only = parse_recognition("说明：玫瑰金素圈\n标签：戒指、素圈戒、素面、成色不清")
        self.assertIn("成色不清", color_only)
        self.assertNotIn("18K", color_only)

        _, priced = parse_recognition("说明：售价750元的素圈\n标签：戒指、素圈戒、素面、成色不清")
        self.assertIn("成色不清", priced)
        self.assertNotIn("18K", priced)

    def test_object_tags_keep_specific_motifs(self):
        motifs = arrange_jewelry_tags(["花朵", "玫瑰", "四叶草", "叶子", "素面"])
        self.assertEqual(motifs, ["玫瑰", "四叶草", "叶子"])
        self.assertEqual(arrange_jewelry_tags(["素面", "蝴蝶"]), ["蝴蝶"])
        animals = arrange_jewelry_tags(["动物", "生肖", "龙"])
        self.assertEqual(animals, ["龙"])
        self.assertEqual(arrange_jewelry_tags(["花卉胸针"]), ["花卉胸针", "花朵"])
        self.assertEqual(arrange_jewelry_tags(["珠串手链"]), ["珠串手链", "珠子"])
        self.assertEqual(arrange_jewelry_tags(["心形吊坠"]), ["心形吊坠", "爱心", "心形"])
        self.assertEqual(arrange_jewelry_tags(["手链"]), ["手链"])
        self.assertEqual(arrange_jewelry_tags(["光环戒"]), ["光环戒"])
        self.assertEqual(arrange_jewelry_tags(["对戒"]), ["戒指", "对戒"])
        self.assertEqual(arrange_jewelry_tags(["梅花鹿"]), ["鹿"])
        self.assertEqual(arrange_jewelry_tags(["星月"]), ["星星", "月亮"])
        self.assertIn("四叶草", arrange_jewelry_tags(["幸运草"]))
        self.assertEqual(missing_style_tags(["戒指", "光环戒", "四叶草"]), ["成色"])
        self.assertEqual(missing_style_tags(["戒指", "光环戒", "18K"]), ["物体"])


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    unittest.main()
