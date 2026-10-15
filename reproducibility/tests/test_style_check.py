"""reprobench.style_check の単体テスト(記事本体から独立した契約の固定)。"""
from reprobench.style_check import (
    chapters_without_summary,
    fat_paragraphs,
    long_sentences,
    paragraphs,
    sentences,
)

_SAMPLE = """---
title: "x"
---

## 見出し

> 要約です。

地の文の段落です。2文目もあります。

- 箇条書き1
- 箇条書き2

| a | b |
|---|---|
| 1 | 2 |

```python
code_is_not_body = True
```
"""


def test_paragraphs_excludes_non_body_blocks():
    assert paragraphs(_SAMPLE) == ["地の文の段落です。2文目もあります。"]


def test_sentences_splits_on_terminators_and_keeps_closers():
    assert sentences("これは文1です。これは文2ですか？はい。") == [
        "これは文1です。",
        "これは文2ですか？",
        "はい。",
    ]
    assert sentences("「引用です。」のあとに続く文。") == [
        "「引用です。」",
        "のあとに続く文。",
    ]


def test_long_sentences_detects_over_limit():
    short = "短い文です。"
    long_sent = "あ" * 121 + "。"
    md = f"{short}{long_sent}\n"
    assert long_sentences(md, limit=120) == [long_sent]


def test_fat_paragraphs_detects_over_limit():
    six = "".join(f"文{i}です。" for i in range(6))
    md = f"{six}\n"
    assert fat_paragraphs(md, limit=5) == [six]


def test_chapters_without_summary_flags_missing_quote():
    md = "## 章1\n\n> 要約。\n\n本文。\n\n## 章2\n\n本文のみ。\n"
    assert chapters_without_summary(md) == ["章2"]
