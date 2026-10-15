"""記事(articles/reproducibility-of-randomness.md)が文体規約を満たすかを検査する。

記事本体と画像はこのパッケージ(reproducibility/)の外、リポジトリルート直下の
articles/・images/ に置かれる(spec `成果物パス` 節のとおり)。コンテナには
reproducibility/ だけでなく、これら2つのディレクトリも
`/articles`・`/images/reproducibility-of-randomness` としてマウントする
(reprobench.figures.py が既に `/images/reproducibility-of-randomness` を
デフォルトパスとして使っている規約と合わせた。Makefileの`figures`ターゲット
が同じ理由でこのディレクトリをreproducibility/の外側にマウントしている)。
環境変数 REPRO_ARTICLE_PATH / REPRO_IMAGES_DIR で上書きできる。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from reprobench.style_check import (
    chapters_without_summary,
    fat_paragraphs,
    long_sentences,
)

ARTICLE_PATH = Path(
    os.environ.get("REPRO_ARTICLE_PATH", "/articles/reproducibility-of-randomness.md")
)
IMAGES_DIR = Path(
    os.environ.get("REPRO_IMAGES_DIR", "/images/reproducibility-of-randomness")
)

if not ARTICLE_PATH.is_file():
    pytest.skip(
        f"article not found at {ARTICLE_PATH} "
        "(mount it, or set REPRO_ARTICLE_PATH)",
        allow_module_level=True,
    )

ARTICLE_TEXT = ARTICLE_PATH.read_text(encoding="utf-8")


def _frontmatter_and_body(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[1:idx]), "\n".join(lines[idx + 1 :])
    return "", text


FRONTMATTER, BODY = _frontmatter_and_body(ARTICLE_TEXT)


def _chapter_section(text: str, index: int) -> str:
    """index番目(1始まり)の`## `見出しから、次の`## `見出し直前までを返す。"""
    _, body = _frontmatter_and_body(text)
    lines = body.splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(r"^##\s+\S", line.strip())]
    assert len(starts) >= index, f"章が{index}個未満しか無い(見つかった数: {len(starts)})"
    start = starts[index - 1]
    end = starts[index] if index < len(starts) else len(lines)
    return "\n".join(lines[start:end])


def test_frontmatter_has_required_keys():
    required = ("title", "emoji", "type", "topics", "published")
    for key in required:
        assert re.search(rf'^{key}:\s*\S', FRONTMATTER, re.MULTILINE), (
            f"frontmatterに{key!r}が無い"
        )


def test_all_referenced_images_exist():
    names = re.findall(r"/images/reproducibility-of-randomness/([\w.\-]+\.png)", BODY)
    assert names, "記事内に画像参照が1つも無い"
    for name in names:
        assert (IMAGES_DIR / name).is_file(), f"参照されている画像が実在しない: {name}"


def test_no_placeholder_numbers():
    for marker in ("TODO", "TBD", "???"):
        assert marker not in BODY, f"placeholderが残っている: {marker}"
    assert not re.search(r"(?<![A-Za-z0-9])XX(?![A-Za-z0-9])", BODY), (
        "placeholderの'XX'が残っている"
    )


def test_mentions_all_five_levels():
    for level in ("L0", "L1", "L2", "L3", "L4"):
        assert level in BODY, f"{level}への言及が無い"


def test_has_decision_flow_and_wording_template():
    # spec §2: 第8章が判定フローと納品文言テンプレート
    chapter8 = _chapter_section(ARTICLE_TEXT, 8)
    assert "f9_flowchart.png" in chapter8, "第8章に判定フロー図(f9_flowchart.png)への参照が無い"
    assert "```" in chapter8, "第8章に納品文言テンプレートのコードブロックが無い"
    fence_blocks = re.findall(r"```[^\n]*\n(.*?)```", chapter8, re.DOTALL)
    assert any(
        all(level in block for level in ("L0", "L1", "L2", "L3", "L4"))
        for block in fence_blocks
    ), "第8章のコードブロックにL0〜L4すべてを含む納品文言テンプレートが無い"


def test_no_sentence_longer_than_two_lines():
    offenders = long_sentences(ARTICLE_TEXT, limit=120)
    assert offenders == [], f"120字を超える文がある: {offenders}"


def test_no_paragraph_over_five_sentences():
    offenders = fat_paragraphs(ARTICLE_TEXT, limit=5)
    assert offenders == [], f"6文以上の段落がある: {offenders}"


def test_every_chapter_starts_with_summary():
    offenders = chapters_without_summary(ARTICLE_TEXT)
    assert offenders == [], f"冒頭に要約の引用ブロックが無い章がある: {offenders}"


def test_body_has_no_emoji():
    emoji_re = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "\U00002190-\U000021FF"
        "\U00002B00-\U00002BFF"
        "]"
    )
    found = emoji_re.findall(BODY)
    assert found == [], f"本文に絵文字が残っている: {found}"


def test_no_decorative_dashes():
    for mark in ("—", "─", "━"):
        assert mark not in BODY, f"装飾記号が残っている: {mark!r}"
