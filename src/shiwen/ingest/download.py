"""语料下载：按 manifest.yaml 拉取 raw JSON 到 data/corpus/raw/。

国内直连 GitHub 常超时，可用 `--mirror ghfast`（或 `--base` 自定义）绕开。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from src.shiwen.config import get_settings

from .models import BookMeta, load_manifest

RAW_DIR = Path("data/corpus/raw")

# 国内可用的 GitHub raw 镜像前缀（按需 --mirror 选择，也可直接传完整前缀）
MIRRORS = {
    "ghfast": "https://ghfast.top/",
    "ghproxy": "https://mirror.ghproxy.com/",
}


def build_url(source_base: str, file_rel: str, mirror: str | None, base: str | None) -> str:
    if base:
        return f"{base.rstrip('/')}/{file_rel.lstrip('/')}"
    url = f"{source_base.rstrip('/')}/{file_rel.lstrip('/')}"
    if mirror:
        prefix = MIRRORS.get(mirror, mirror)
        url = prefix + url
    return url


def download_book(book: BookMeta, source_base: str, mirror: str | None, base: str | None,
                  force: bool, timeout: int) -> Path:
    dest = RAW_DIR / book.file
    if dest.exists() and not force:
        print(f"[skip] {book.id} 已存在 {dest}")
        return dest

    url = build_url(source_base, book.file, mirror, base)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {book.id} <- {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[ok]   {book.id} -> {dest} ({len(resp.content)} bytes)")
    return dest


def download(book_ids: list[str] | None = None, mirror: str | None = None,
             base: str | None = None, force: bool = False, timeout: int = 60) -> None:
    manifest = load_manifest()
    if base is None:
        base = get_settings().corpus_raw_base or None
    books = [b for b in manifest.books if book_ids is None or b.id in book_ids]
    if not books:
        print("[!] 没有匹配的书籍，请检查 --books 参数")
        return
    for book in books:
        download_book(book, manifest.source_base, mirror, base, force, timeout)
    print(f"[done] 共 {len(books)} 部")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载语料 raw JSON")
    parser.add_argument("--books", help="逗号分隔的 book id，缺省=全部", default=None)
    parser.add_argument("--mirror", help="镜像预设：ghfast/ghproxy，或完整前缀", default=None)
    parser.add_argument("--base", help="直接覆盖 base URL", default=None)
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    book_ids = args.books.split(",") if args.books else None
    download(book_ids=book_ids, mirror=args.mirror, base=args.base,
             force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    main()
