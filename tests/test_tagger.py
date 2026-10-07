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
    "category": {"main": "ring", "confidence": 0.95},
    "material": [{"name": "gold", "confidence": 0.9}, {"name": "diamond", "confidence": 0.88}],
    "cut": {"name": "brilliant", "confidence": 0.85},
    "design": ["solitaire"],
    "style": ["luxury", "classic"],
    "occasion": ["wedding", "engagement"],
    "color": {"primary": "#FFD700", "secondary": ["#FFFFFF"]},
    "visual": {"background": "gradient", "angle": "45_degree", "lighting": "soft"},
    "caption": "一枚黄金镶圆形钻石的订婚戒指",
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
        self.assertEqual(tags.category.main, "ring")
        self.assertEqual(tags.material[0].name, "gold")
        self.assertEqual(tags.color.primary, "#FFD700")
        aliased = json.loads(json.dumps(SAMPLE))
        aliased["category"] = {"main": "戒指", "confidence": 0.95}
        aliased["material"] = [
            {"name": "黄金", "confidence": 90},
            {"name": "钻石", "confidence": 0.8},
        ]
        aliased["color"] = {"primary": "金色", "secondary": ["白色"]}
        tags = parse_tags(aliased, self.taxonomy)
        self.assertEqual(tags.category.main, "ring")
        self.assertEqual(tags.material[0].name, "gold")
        self.assertEqual(tags.material[0].confidence, 0.9)
        self.assertEqual(tags.color.secondary, ["#FFFFFF"])
        self.assertEqual(tags.design, ["solitaire"])
        shaped = json.loads(json.dumps(SAMPLE))
        shaped["design"] = ["光环戒"]
        self.assertEqual(parse_tags(shaped, self.taxonomy).design, ["halo"])
        shaped["design"] = ["耳钉"]
        with self.assertRaises(ValueError):
            parse_tags(shaped, self.taxonomy)

    def test_repair_trailing_comma_and_fence(self):
        raw = "```json\n" + json.dumps(SAMPLE, ensure_ascii=False).replace("}", ",}") + "\n```"
        self.assertEqual(loads_repaired(raw)["category"]["main"], "ring")

    def test_unknown_category_rejected(self):
        bad = json.loads(json.dumps(SAMPLE))
        bad["category"] = {"main": "watch", "confidence": 0.5}
        with self.assertRaises(ValueError):
            parse_tags(bad, self.taxonomy)

    def test_extra_axis_from_config(self):
        text = """
version: 1
axes:
  category: {multiple: false, options: [{id: ring, zh: 戒指, en: ring}]}
  material: {multiple: true, options: [{id: gold, zh: 黄金, en: gold}]}
  cut: {multiple: false, nullable: true, options: [{id: brilliant, zh: 圆形, en: brilliant}]}
  style: {multiple: true, options: [{id: classic, zh: 经典, en: classic}]}
  occasion: {multiple: true, options: [{id: daily, zh: 日常, en: daily}]}
  background: {multiple: false, options: [{id: solid, zh: 纯色, en: solid}]}
  angle: {multiple: false, options: [{id: front, zh: 正面, en: front}]}
  lighting: {multiple: false, options: [{id: soft, zh: 柔光, en: soft}]}
  design:
    multiple: true
    options:
      - {id: solitaire, zh: 单石戒, en: solitaire, categories: [ring]}
extra_axes:
  motif: {multiple: true, options: [{id: floral, zh: 花卉, en: floral}]}
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tags.yaml"
            path.write_text(text, encoding="utf-8")
            taxonomy = load_taxonomy(path)
        payload = json.loads(json.dumps(SAMPLE))
        payload["category"] = {"main": "ring", "confidence": 1}
        payload["material"] = [{"name": "gold", "confidence": 1}]
        payload["style"] = ["classic"]
        payload["occasion"] = ["daily"]
        payload["visual"] = {"background": "solid", "angle": "front", "lighting": "soft"}
        payload["design"] = ["solitaire"]
        payload["extras"] = {"motif": ["花卉"]}
        tags = parse_tags(payload, taxonomy)
        self.assertEqual(tags.extras["motif"], ["floral"])

    def test_retry_then_accept(self):
        good = json.dumps(SAMPLE, ensure_ascii=False)
        engine = FakeEngine(["不是 JSON", good])
        tagger = JewelryTagger(engine=engine, taxonomy=self.taxonomy, retries=3)
        with tempfile.TemporaryDirectory() as folder:
            path = _png(Path(folder), "ring.png")
            tags = tagger.tag(path)
        self.assertEqual(tags.category.main, "ring")
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


if __name__ == "__main__":
    unittest.main()
