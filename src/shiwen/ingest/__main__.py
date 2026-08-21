"""识文新裁 Ingestion CLI 入口。

用法（在 backend 容器内，CWD=/app）：
  docker compose run --rm backend python -m src.shiwen.ingest download --mirror ghfast
  docker compose run --rm backend python -m src.shiwen.ingest normalize
  docker compose run --rm backend python -m src.shiwen.ingest reindex [--books lunyu,mengzi]
  docker compose run --rm backend python -m src.shiwen.ingest seed-people
  docker compose run --rm backend python -m src.shiwen.ingest verify
  docker compose run --rm backend python -m src.shiwen.ingest inspect --book lunyu
"""

from __future__ import annotations

import argparse

from . import corpus, download, golden, milvus_store, people, pipeline, pg_store


def _split(s: str | None) -> list[str] | None:
    return s.split(",") if s else None


def _cmd_download(args: argparse.Namespace) -> None:
    download.download(book_ids=_split(args.books), mirror=args.mirror,
                      base=args.base, force=args.force, timeout=args.timeout)


def _cmd_normalize(args: argparse.Namespace) -> None:
    corpus.normalize(book_ids=_split(args.books))


def _cmd_reindex(args: argparse.Namespace) -> None:
    pipeline.run_reindex(book_ids=_split(args.books), batch_size=args.batch_size)


def _cmd_seed_people(args: argparse.Namespace) -> None:
    n = people.seed()
    print(f"[seed-people] 已灌 {n} 位人物")


def _cmd_verify(args: argparse.Namespace) -> None:
    ok = golden.verify()
    raise SystemExit(0 if ok else 1)


def _cmd_inspect(args: argparse.Namespace) -> None:
    client = milvus_store._client()
    book_filter = f'book_id == "{args.book}"'
    rows = milvus_store.query(book_filter, limit=args.limit, client=client)
    n_milvus = milvus_store.count_where(book_filter, client=client)
    print(f"[inspect] {args.book} 共 {n_milvus} chunks，展示前 {len(rows)} 条：")
    for r in rows:
        text = (r["text"] or "").replace("\n", " ")[:48]
        print(f"  {r['id']}  [{r['part'] or '-'}/{r['chapter']}]  {text}...")
    engine = pg_store.get_engine()
    n_pg = pg_store.count_book(args.book, engine)
    print(f"[inspect] PG 侧 {args.book} 共 {n_pg} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingest", description="识文新裁语料 Ingestion 管线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="下载 raw JSON")
    p.add_argument("--books", default=None, help="逗号分隔 book id")
    p.add_argument("--mirror", default=None, help="ghfast/ghproxy 或完整前缀")
    p.add_argument("--base", default=None, help="覆盖 base URL")
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=int, default=60)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("normalize", help="raw -> 规范化 markdown")
    p.add_argument("--books", default=None)
    p.set_defaults(func=_cmd_normalize)

    p = sub.add_parser("reindex", help="切分 + 向量化 + 入库（清空重灌）")
    p.add_argument("--books", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=_cmd_reindex)

    p = sub.add_parser("seed-people", help="灌人物关系表")
    p.set_defaults(func=_cmd_seed_people)

    p = sub.add_parser("verify", help="golden 引据自检")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("inspect", help="查看某本书入库情况")
    p.add_argument("--book", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=_cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
