"""用 Gemma 3 识别珠宝图片标签。"""

from jewelry_tagger.schema import JewelryTags
from jewelry_tagger.tagger import JewelryTagger, TagResult, TaggingFailed

__all__ = ["JewelryTagger", "JewelryTags", "TagResult", "TaggingFailed"]
