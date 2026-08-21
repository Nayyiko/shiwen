"""数据模型：语料元数据（BookMeta/Manifest）+ 切分结构（Section/Chunk）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel


class BookMeta(BaseModel):
    """单部书的权威元数据（author/dynasty 覆盖 raw JSON 中的错误标注）。"""

    id: str
    name: str
    author: str
    dynasty: str
    category: str
    version: str = "通行本"
    file: str
    license: str = "公有领域"
    max_chars: int = 500
    overlap: int = 80


class Manifest(BaseModel):
    source_base: str
    license_note: str = ""
    books: list[BookMeta]


def load_manifest(path: str | Path = "data/corpus/manifest.yaml") -> Manifest:
    """加载语料清单；defaults 里的 max_chars/overlap 合并到每本书。"""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    books = [BookMeta(**{**defaults, **b}) for b in raw["books"]]
    return Manifest(
        source_base=raw["source_base"],
        license_note=raw.get("license_note", ""),
        books=books,
    )


@dataclass
class Section:
    """结构层级：level 1 = part/卷（可选），level 2 = chapter/篇（引据原子单位）。"""

    level: int
    title: str
    paragraphs: list[str] = field(default_factory=list)
    chapter_index: int = -1  # 全局 chapter 序号（level == 2 时有意义）


@dataclass
class Chunk:
    """一个可入库的文本块。id 与 PostgreSQL 的 source-of-truth 对齐。"""

    id: str  # {book_id}:{chapter_idx}:{chunk_idx}
    text: str
    book_id: str
    book: str
    author: str
    dynasty: str
    category: str
    version: str
    part: str | None
    chapter: str
    chapter_index: int
    chunk_index: int
