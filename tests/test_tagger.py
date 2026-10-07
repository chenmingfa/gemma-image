import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jewelry_tagger.repair import loads_repaired
from jewelry_tagger.schema import parse_tags
from jewelry_tagger.sources import ImageSourceError, load_image
from jewelry_tagger.tagger import JewelryTagger, TaggingFailed
from jewelry_tagger.taxonomy import load_taxonomy

SAMPLE = {
    "category": "ring",
    "sub_category": "engagement_ring",
    "material": ["18K金", "铂金"],
    "gemstone": ["钻石", "蓝宝石"],
    "metal_color": "white_gold",
    "style": ["vintage", "art_deco"],
    "stone_shape": "emerald_cut",
    "setting": "pave",
    "occasion": ["wedding", "engagement"],
    "audience": "women",
    "brand_hint": None,
    "era": "art_deco",
    "confidence": 0.92,
    "tags": ["奢华", "复古", "婚戒", "群镶"],
}


class FakeEngine:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.batches = []

    def complete_batch(self, pairs, max_new_tokens):
        self.batches.append(len(pairs))
        texts = []
        for _ in pairs:
            texts.append(self.outputs.pop(0))
        return texts


def _png(folder: Path, name: str) -> str:
    path = folder / name
    Image.new("RGB", (16, 16), (200, 160, 40)).save(path, format="PNG")
    return str(path)


class TaggerTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = load_taxonomy()

    def test_sample_json_and_chinese_aliases(self):
        tags = parse_tags(SAMPLE, self.taxonomy)
        self.assertEqual(tags.category, "ring")
        self.assertEqual(tags.sub_category, "engagement_ring")
        self.assertEqual(tags.material, ["18K金", "铂金"])
        self.assertEqual(tags.gemstone, ["钻石", "蓝宝石"])
        self.assertEqual(tags.tags, ["奢华", "复古", "婚戒", "群镶"])
        self.assertNotIn("extras", tags.as_json())
        aliased = json.loads(json.dumps(SAMPLE))
        aliased["category"] = "戒指"
        aliased["confidence"] = 92
        aliased["material"] = ["黄金", "铂金"]
        tags = parse_tags(aliased, self.taxonomy)
        self.assertEqual(tags.category, "ring")
        self.assertEqual(tags.material[0], "黄金")
        self.assertEqual(tags.confidence, 0.92)
        shaped = json.loads(json.dumps(SAMPLE))
        shaped["sub_category"] = "耳钉"
        with self.assertRaises(ValueError):
            parse_tags(shaped, self.taxonomy)

    def test_repair_trailing_comma_and_fence(self):
        raw = "```json\n" + json.dumps(SAMPLE, ensure_ascii=False).replace("}", ",}") + "\n```"
        self.assertEqual(loads_repaired(raw)["category"], "ring")

    def test_unknown_category_rejected(self):
        bad = json.loads(json.dumps(SAMPLE))
        bad["category"] = "watch"
        with self.assertRaises(ValueError):
            parse_tags(bad, self.taxonomy)

    def test_extra_axis_from_config(self):
        text = """
version: 1
axes:
  category: {multiple: false, options: [{id: ring, zh: 戒指, en: ring}]}
  sub_category:
    multiple: false
    options: [{id: solitaire, zh: 单石戒, en: solitaire, categories: [ring]}]
  material: {multiple: true, options: [{id: gold, zh: 黄金, en: gold}]}
  gemstone: {multiple: true, options: [{id: diamond, zh: 钻石, en: diamond}]}
  metal_color: {multiple: false, nullable: true, options: [{id: yellow, zh: 黄金色, en: yellow}]}
  style: {multiple: true, options: [{id: classic, zh: 经典, en: classic}]}
  stone_shape: {multiple: false, nullable: true, options: [{id: brilliant, zh: 圆形, en: brilliant}]}
  setting: {multiple: false, nullable: true, options: [{id: prong, zh: 爪镶, en: prong}]}
  occasion: {multiple: true, options: [{id: daily, zh: 日常, en: daily}]}
  audience: {multiple: false, options: [{id: women, zh: 女款, en: women}]}
  brand_hint: {multiple: false, nullable: true, options: [{id: hallmark, zh: 印记, en: hallmark}]}
  era: {multiple: false, nullable: true, options: [{id: contemporary, zh: 当代, en: contemporary}]}
extra_axes:
  pattern: {multiple: true, options: [{id: floral, zh: 花卉, en: floral}]}
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tags.yaml"
            path.write_text(text, encoding="utf-8")
            taxonomy = load_taxonomy(path)
        payload = {
            "category": "戒指",
            "sub_category": "单石戒",
            "material": ["黄金"],
            "gemstone": [],
            "style": ["经典"],
            "occasion": ["日常"],
            "audience": "女款",
            "confidence": 1,
            "extras": {"pattern": ["花卉"]},
        }
        tags = parse_tags(payload, taxonomy)
        self.assertEqual(tags.sub_category, "solitaire")
        self.assertEqual(tags.material, ["黄金"])
        self.assertEqual(tags.extras["pattern"], ["floral"])

    def test_retry_then_accept(self):
        good = json.dumps(SAMPLE, ensure_ascii=False)
        engine = FakeEngine(["不是 JSON", good])
        tagger = JewelryTagger(engine=engine, taxonomy=self.taxonomy, retries=3)
        with tempfile.TemporaryDirectory() as folder:
            path = _png(Path(folder), "ring.png")
            tags = tagger.tag(path)
        self.assertEqual(tags.category, "ring")
        self.assertEqual(engine.batches, [1, 1])

    def test_three_failures_stop(self):
        engine = FakeEngine(["坏", "还是坏", "仍然坏"])
        tagger = JewelryTagger(engine=engine, taxonomy=self.taxonomy, retries=3)
        with tempfile.TemporaryDirectory() as folder:
            path = _png(Path(folder), "ring.png")
            with self.assertRaises(TaggingFailed):
                tagger.tag(path)
        self.assertEqual(len(engine.batches), 3)

    def test_batch_keeps_order_and_rejects_gif(self):
        good = json.dumps(SAMPLE, ensure_ascii=False)
        engine = FakeEngine([good, good])
        tagger = JewelryTagger(engine=engine, taxonomy=self.taxonomy)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _png(root, "a.png")
            second = _png(root, "b.png")
            gif = root / "c.gif"
            Image.new("RGB", (8, 8)).save(gif, format="GIF")
            with self.assertRaises(ImageSourceError):
                load_image(str(gif))
            opened = load_image(first)
            self.assertEqual(opened.mode, "RGB")
            self.assertEqual(opened.size, (16, 16))
            results = tagger.tag_many([second, first], batch_size=2)
        self.assertEqual([item.source for item in results], [second, first])
        self.assertEqual(engine.batches, [2])
        self.assertTrue(all(item.tags is not None for item in results))

    def test_api_upload(self):
        from io import BytesIO

        from fastapi.testclient import TestClient

        from jewelry_tagger.api import create_app

        taxonomy = self.taxonomy

        class Stub:
            def tag(self, source):
                return parse_tags(SAMPLE, taxonomy)

        client = TestClient(create_app(Stub()))
        buffer = BytesIO()
        Image.new("RGB", (8, 8), (180, 140, 40)).save(buffer, format="PNG")
        response = client.post("/v1/tag", files={"file": ("ring.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category"], "ring")
        self.assertEqual(body["material"], ["18K金", "铂金"])
        self.assertEqual(body["gemstone"], ["钻石", "蓝宝石"])
        self.assertNotIn("extras", body)
        self.assertTrue(client.get("/health").json()["ok"])


if __name__ == "__main__":
    unittest.main()
