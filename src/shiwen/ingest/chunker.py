"""结构感知切分：规范化白文 Markdown -> 带层级元数据的 Chunk 列表。

层级约定：
  - `#`   = part/卷（可选，如史记的本纪/世家/列传）
  - `##`  = chapter/篇（引据原子单位）
  - chapter 名若含 `·`（如「本纪·项羽本纪」）拆为 part/chapter

切分规则：chunk 永不跨章；整章 <= max_chars 则整章一个 chunk；
超长章按句打包，相邻 chunk 以 overlap 字符重叠（滑动窗口）。
"""

from __future__ import annotations

import re

from .models import BookMeta, Chunk, Section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MIDDLE_DOT_RE = re.compile(r"[·・]")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;:：])")


def split_sentences(text: str) -> list[str]:
    """按句末标点 + 换行切分，返回非空句子列表。"""
    parts: list[str] = []
    for seg in re.split(r"\n+", text):
        for s in _SENT_SPLIT_RE.split(seg):
            s = s.strip()
            if s:
                parts.append(s)
    return parts


def parse_structure(md_text: str) -> list[Section]:
    """识别 `#`/`##` 标题，返回层级结构；`##` 携带全局 chapter_index。"""
    sections: list[Section] = []
    current: Section | None = None
    chapter_idx = 0

    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2)
            if level <= 2:
                if current is not None:
                    sections.append(current)
                current = Section(level=level, title=title)
                if level == 2:
                    current.chapter_index = chapter_idx
                    chapter_idx += 1
                continue
            # level > 2 的标题当正文，落到下方 append
        if current is not None:
            current.paragraphs.append(line)
        # 首个标题前的零散文本（如书名行）忽略

    if current is not None:
        sections.append(current)
    return sections


def _pack_chapter(text: str, max_chars: int, overlap: int) -> list[str]:
    """单章 -> 若干 chunk 文本（不跨章）。"""
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = split_sentences(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for s in sentences:
        if len(s) >= max_chars:  # 单句超长，硬作为一个 chunk，避免死循环
            if buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            chunks.append(s)
            continue
        if buf_len + len(s) > max_chars:
            chunks.append("".join(buf))
            buf, buf_len = _tail_overlap(buf, overlap)
        buf.append(s)
        buf_len += len(s)

    if buf:
        chunks.append("".join(buf))
    return chunks


def _tail_overlap(buf: list[str], overlap: int) -> tuple[list[str], int]:
    """从 buf 尾部保留累计 >= overlap 字符的句子，作为下一个 chunk 的开头。"""
    kept: list[str] = []
    total = 0
    for s in reversed(buf):
        kept.append(s)
        total += len(s)
        if total >= overlap:
            break
    kept.reverse()
    return kept, total


def _split_part_chapter(title: str, current_part: str | None) -> tuple[str | None, str]:
    if _MIDDLE_DOT_RE.search(title):
        part, chapter = _MIDDLE_DOT_RE.split(title, maxsplit=1)
        return (part.strip() or None, chapter.strip() or title)
    return (current_part, title)


def chunk_book(md_text: str, book: BookMeta, max_chars: int | None = None,
               overlap: int | None = None) -> list[Chunk]:
    """一本书的白文 -> Chunk 列表（chunk id = {book_id}:{chapter_idx}:{chunk_idx}）。"""
    max_chars = max_chars or book.max_chars
    overlap = overlap if overlap is not None else book.overlap
    sections = parse_structure(md_text)

    current_part: str | None = None
    chunks: list[Chunk] = []
    for sec in sections:
        if sec.level == 1:
            current_part = sec.title
            continue
        # level == 2：一个 chapter
        part, chapter = _split_part_chapter(sec.title, current_part)
        text = "\n".join(sec.paragraphs).strip()
        for i, chunk_text in enumerate(_pack_chapter(text, max_chars, overlap)):
            chunks.append(Chunk(
                id=f"{book.id}:{sec.chapter_index}:{i}",
                text=chunk_text,
                book_id=book.id,
                book=book.name,
                author=book.author,
                dynasty=book.dynasty,
                category=book.category,
                version=book.version,
                part=part,
                chapter=chapter,
                chapter_index=sec.chapter_index,
                chunk_index=i,
            ))
    return chunks
