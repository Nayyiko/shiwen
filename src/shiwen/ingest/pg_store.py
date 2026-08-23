"""PostgreSQL 层：chunk 表（id 为 source-of-truth，与 Milvus 对齐）。

用 SQLAlchemy 2.0 + psycopg v3。人物关系三表定义在 people.py，共用此处的 Base。
"""

from __future__ import annotations

from sqlalchemy import Integer, String, Text, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.shiwen.config import get_settings
from .models import Chunk


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "chunk"

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    book_id: Mapped[str] = mapped_column(String(128), index=True)
    book: Mapped[str] = mapped_column(String(128))
    author: Mapped[str] = mapped_column(String(128))
    dynasty: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64), default="通行本")
    part: Mapped[str | None] = mapped_column(String(256), nullable=True)
    chapter: Mapped[str | None] = mapped_column(String(256), nullable=True)
    chapter_index: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)


def _url() -> str:
    s = get_settings()
    return (
        f"postgresql+psycopg://{s.postgres_user}:{s.postgres_password}"
        f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"
    )


def get_engine() -> Engine:
    return create_engine(_url(), pool_pre_ping=True)


def init_db(engine: Engine | None = None) -> None:
    """建所有已注册的表（含 people.py 的人物表，前提是其模块已被 import）。"""
    Base.metadata.create_all(engine or get_engine())


def _to_row(chunk: Chunk) -> dict:
    return {c: getattr(chunk, c) for c in (
        "id", "text", "book_id", "book", "author", "dynasty",
        "category", "version", "part", "chapter", "chapter_index", "chunk_index",
    )}


def insert_chunks(chunks: list[Chunk], engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    if not chunks:
        return 0
    rows = [_to_row(c) for c in chunks]
    with engine.begin() as conn:
        conn.execute(ChunkRow.__table__.insert(), rows)
    return len(rows)


def clear_chunks(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(ChunkRow.__table__.delete())


def count(engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(ChunkRow)).scalar_one()


def count_book(book_id: str, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.connect() as conn:
        return conn.execute(
            select(func.count()).where(ChunkRow.book_id == book_id)
        ).scalar_one()


def query_book(book_id: str, limit: int = 10, engine: Engine | None = None) -> list[dict]:
    """按 book_id 查询 chunk 记录（用 Session 以正确返回 ORM 实体）。

    注意：Connection.execute(select(ChunkRow)) 不重建实体，需用 Session
    才能拿到 ChunkRow 对象而非逐列原始值。
    """
    engine = engine or get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(ChunkRow).where(ChunkRow.book_id == book_id).limit(limit)
        ).all()
    return [_to_row(r) for r in rows]
