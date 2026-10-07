# 珠宝图片标签

用 Gemma 3 看珠宝图片，输出可校验的 JSON 标签。标签词表在 `configs/jewelry_tags.yaml`，改文件即可扩展。

页面图片库（`python app.py`）仍用本机 Ollama。这一套是独立的 Transformers 接口。

## 目录

```
configs/jewelry_tags.yaml    标签体系
jewelry_tagger/              识别程序
  taxonomy.py                读取配置、把中文收成 id
  schema.py                  Pydantic 校验
  repair.py                  抽出并修复 JSON
  prompt.py                  按词表生成提示
  sources.py                 本地路径或 URL
  engine.py                  加载 Gemma 3，GPU / CPU 自动切换
  tagger.py                  批量、重试
  cli.py                     命令行
examples/tag_image.py        Python API 示例
```

## 安装

先在 Hugging Face 接受 Gemma 许可并登录：`huggingface-cli login`。

```
.venv\Scripts\python -m pip install -r requirements.txt
```

有 NVIDIA 显卡时，安装与 CUDA 匹配的 PyTorch：https://pytorch.org

## 使用

命令行，默认 `google/gemma-3-4b-it`。27B 换成 `--model google/gemma-3-27b-it`。

```
.venv\Scripts\python -m jewelry_tagger ring.jpg
.venv\Scripts\python -m jewelry_tagger a.jpg b.webp https://example.com/ring.png --batch-size 2 -o tags.json
```

Python：

```python
from jewelry_tagger import JewelryTagger

tagger = JewelryTagger(model_id="google/gemma-3-4b-it", device="auto")
tags = tagger.tag("ring.jpg")
print(tags.category.main, tags.caption)

results = tagger.tag_many(["a.jpg", "b.png"], batch_size=2, workers=4)
```

`tag` 在 3 次都失败后抛出异常。`tag_many` 单张失败不会打断其余图片，结果里带 `error`。

输出形状：

```json
{
  "category": {"main": "ring", "confidence": 0.95},
  "material": [{"name": "gold", "confidence": 0.9}, {"name": "diamond", "confidence": 0.88}],
  "cut": {"name": "brilliant", "confidence": 0.85},
  "design": ["solitaire"],
  "motif": ["floral"],
  "fineness": ["k18"],
  "style": ["luxury", "classic"],
  "occasion": ["wedding", "engagement"],
  "color": {"primary": "#FFD700", "secondary": ["#FFFFFF"]},
  "visual": {"background": "gradient", "angle": "45_degree", "lighting": "soft"},
  "caption": "一枚黄金镶圆形钻石的订婚戒指"
}
```

## 扩展标签

在 `configs/jewelry_tags.yaml` 的对应轴下加一项即可，例如风格：

```yaml
- {id: art_deco, zh: 装饰艺术, en: art deco}
```

`id` 是 JSON 里的值，`zh` 和 `en` 是模型或人工可能写出的别名，解析时都会收成 `id`。

新增一整类时，写到 `extra_axes`。结果出现在 `extras` 里，不用改 Pydantic 字段：

```yaml
extra_axes:
  motif:
    multiple: true
    options:
      - {id: floral, zh: 花卉, en: floral}
```

提示词会自动带上这些词。调用时加上 `--config 你的文件.yaml`。

## 性能

- 默认用 4B。27B 在 bfloat16 下大约要 50GB 显存，单卡放不下时再考虑 int4（torchao），不要一上来就加载 27B。
- 推理用 bfloat16 和 SDPA。GPU 上 `device_map="auto"`，没有 GPU 时自动改用 CPU（会慢很多）。
- 同一份权重不要并发调用 `generate`。读取图片用线程（`--workers`），看图按批（`--batch-size`，建议 1 到 4）。显存不够时程序会退回逐张。
- 批量时分词器左侧补齐，避免补齐符被当成正文。
- 珠宝特写要看清切工时加 `--pan-and-scan`。平时关掉，因为会把一张图切成多块，更慢。
- 原图先缩到长边 2048 再送进模型。视觉编码器本身是 896。
- 进程内只加载一次模型。`do_sample=False`，输出更稳定，也少走采样开销。
- JSON 在本地先修复（去掉 Markdown、截取对象、去掉尾逗号），修不好再让模型重答，最多 3 次。
