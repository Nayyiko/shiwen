"""语料规范化：raw JSON → 规范化 Markdown（data/corpus/{id}.md）。

输出为纯白文：每个 chapter 一个 `## {name}` 标题 + 段落；type==1 的标题段跳过
（已由 `## name` 表达，避免重复）。书名/作者等元数据不写入 md，统一走 manifest。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import BookMeta, load_manifest

RAW_DIR = Path("data/corpus/raw")
OUT_DIR = Path("data/corpus")


def normalize_book(book: BookMeta, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> tuple[Path, int]:
    raw_path = raw_dir / book.file
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} 不存在，请先运行 download（如 python -m src.shiwen.ingest download）"
        )
    data = json.loads(raw_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    n_chapters = 0
    for ch in data.get("chapters", []):
        name = (ch.get("name") or "").strip()
        if name:
            lines.append(f"## {name}")
            n_chapters += 1
        for p in ch.get("paragraphs", []):
            if p.get("type") == 1:  # 章节标题段，跳过
                continue
            text = (p.get("paragraph") or "").strip()
            if text:
                lines.append(text)

    out_path = out_dir / f"{book.id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(lines), encoding="utf-8")
    return out_path, n_chapters


def normalize(book_ids: list[str] | None = None) -> None:
    manifest = load_manifest()
    books = [b for b in manifest.books if book_ids is None or b.id in book_ids]
    for book in books:
        path, n_chapters = normalize_book(book)
        print(f"[normalize] {book.id}\t-> {path}（{n_chapters} 章）")
    print(f"[done] 共规范化 {len(books)} 部")


def main() -> None:
    parser = argparse.ArgumentParser(description="raw JSON -> 规范化 Markdown")
    parser.add_argument("--books", help="逗号分隔的 book id，缺省=全部", default=None)
    args = parser.parse_args()
    book_ids = args.books.split(",") if args.books else None
    normalize(book_ids=book_ids)


if __name__ == "__main__":
    main()
