"""記事(Markdown)のスタイル規約を機械的に検査するための補助関数。

依頼者の文体規約(1文は2行まで、1段落は3〜5文、各章冒頭に要約の引用ブロックを
置く)を検査するため、Markdown本文から「地の文」だけを取り出し、文単位・
段落単位・章単位で規約違反を検出する。コードブロック・表・箇条書き・見出し・
引用・frontmatterはすべて除外対象であり、地の文の対象にしない。
"""
from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^#{1,6}\s")
_CHAPTER_RE = re.compile(r"^##\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?")
_TABLE_RE = re.compile(r"^\|")
_BULLET_RE = re.compile(r"^[-*+]\s")
_ORDERED_RE = re.compile(r"^\d+[.)]\s")
_CODE_FENCE_RE = re.compile(r"^```")

_SENTENCE_TERMINATORS = "。！？!?"
_SENTENCE_CLOSERS = "」』\"')"


def _strip_frontmatter(lines: list[str]) -> list[str]:
    """先頭のYAML frontmatter(`---`で挟まれた部分)を取り除く。"""
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return lines[idx + 1 :]
    return lines


def _is_structural(stripped: str) -> bool:
    """見出し・引用・表・箇条書きのいずれかに該当する行かどうかを返す。"""
    if not stripped:
        return False
    return bool(
        _HEADING_RE.match(stripped)
        or _BLOCKQUOTE_RE.match(stripped)
        or _TABLE_RE.match(stripped)
        or _BULLET_RE.match(stripped)
        or _ORDERED_RE.match(stripped)
    )


def paragraphs(md: str) -> list[str]:
    """frontmatter・コードブロック・表・箇条書き・見出し・引用を除いた、地の文の段落一覧を返す。

    段落は空行または上記の除外対象行で区切られる、連続した地の文行をまとめたもの。
    """
    lines = _strip_frontmatter(md.splitlines())

    result: list[str] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        if buf:
            text = "".join(buf)
            if text.strip():
                result.append(text)
            buf.clear()

    for raw in lines:
        stripped = raw.strip()
        if _CODE_FENCE_RE.match(stripped):
            in_code = not in_code
            flush()
            continue
        if in_code:
            continue
        if not stripped:
            flush()
            continue
        if _is_structural(stripped):
            flush()
            continue
        buf.append(stripped)
    flush()
    return result


def sentences(para: str) -> list[str]:
    """段落内の文を句点・疑問符・感嘆符(直後の閉じ括弧類を含む)で分割する。"""
    text = para.strip()
    if not text:
        return []

    result: list[str] = []
    buf = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf += ch
        if ch in _SENTENCE_TERMINATORS:
            j = i + 1
            while j < n and text[j] in _SENTENCE_CLOSERS:
                buf += text[j]
                j += 1
            result.append(buf.strip())
            buf = ""
            i = j
        else:
            i += 1
    if buf.strip():
        result.append(buf.strip())
    return [s for s in result if s]


def long_sentences(md: str, limit: int = 120) -> list[str]:
    """地の文のうち、文字数がlimitを超える文の一覧を返す(規約を満たせば空)。"""
    out: list[str] = []
    for para in paragraphs(md):
        for sent in sentences(para):
            if len(sent) > limit:
                out.append(sent)
    return out


def fat_paragraphs(md: str, limit: int = 5) -> list[str]:
    """地の文の段落のうち、文数がlimitを超える段落の一覧を返す(規約を満たせば空)。"""
    out: list[str] = []
    for para in paragraphs(md):
        if len(sentences(para)) > limit:
            out.append(para)
    return out


def chapters_without_summary(md: str) -> list[str]:
    """`##`見出し直後に要約の引用ブロックが無い章の見出し文字列の一覧を返す(規約を満たせば空)。"""
    lines = _strip_frontmatter(md.splitlines())
    missing: list[str] = []
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        m = _CHAPTER_RE.match(stripped)
        if m:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j >= n or not _BLOCKQUOTE_RE.match(lines[j].strip()):
                missing.append(m.group(1).strip())
        i += 1
    return missing
