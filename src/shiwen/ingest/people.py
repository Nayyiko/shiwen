"""人物关系表：person / person_work / person_relation 三表 + 从 people.yaml 灌数据。

满足验收：`孔子 --著--> 论语`。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column

from .pg_store import Base, get_engine


class Person(Base):
    __tablename__ = "person"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    courtesy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dynasty: Mapped[str] = mapped_column(String(64))
    school: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class PersonWork(Base):
    __tablename__ = "person_work"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("person.id"), index=True)
    work_title: Mapped[str] = mapped_column(String(128))
    relation: Mapped[str] = mapped_column(String(32))  # 著/述/编/修/传/注
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class PersonRelation(Base):
    __tablename__ = "person_relation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("person.id"), index=True)
    target_person_id: Mapped[str] = mapped_column(ForeignKey("person.id"), index=True)
    relation: Mapped[str] = mapped_column(String(64))  # 师从/私淑/祖孙/...
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


def seed(path: str = "data/corpus/people.yaml", engine: Engine | None = None) -> int:
    """清空并重灌三张人物表（幂等）。返回人物数量。"""
    engine = engine or get_engine()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    people = data["people"]

    person_rows = [
        {k: p.get(k) for k in ("id", "name", "courtesy", "dynasty", "school", "notes")}
        for p in people
    ]
    work_rows = [
        {
            "person_id": p["id"],
            "work_title": w["title"],
            "relation": w.get("relation", "著"),
            "note": w.get("note"),
        }
        for p in people for w in p.get("works", [])
    ]
    relation_rows = [
        {
            "person_id": p["id"],
            "target_person_id": r["target"],
            "relation": r["relation"],
            "note": r.get("note"),
        }
        for p in people for r in p.get("relations", [])
    ]

    with engine.begin() as conn:
        conn.execute(PersonRelation.__table__.delete())
        conn.execute(PersonWork.__table__.delete())
        conn.execute(Person.__table__.delete())
        if person_rows:
            conn.execute(Person.__table__.insert(), person_rows)
        if work_rows:
            conn.execute(PersonWork.__table__.insert(), work_rows)
        if relation_rows:
            conn.execute(PersonRelation.__table__.insert(), relation_rows)

    return len(people)
