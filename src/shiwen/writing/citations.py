"""引据抽取与校验：确定性指标，无需 LLM。

核心评测指标：引据可回溯率 = 文章中的引据标签 ∈ 本次检索池 citation 集合的比例。
面试可逐项拆解："几个正则 + 集合交集，不依赖 LLM judge"。
"""

from __future__ import annotations

import re

# 匹配多种引据格式，按优先级排列（更具体的在前）
# 组：book, chapter, [version]
# 关键约束：book/chapter 组内不允许含《》（）「」，避免跨多组匹配
_CITATION_PATTERNS = [
    # 标准格式：「书名·篇名（版本）」
    (re.compile(r"「([^「」]+?)·([^「」]+?)（([^「」（）]+?)）」"), 3),
    # 括号+版本：（书名·篇名（版本））
    (re.compile(r"（([^《》（）]+?)·([^《》（）]+?)（([^《》（）]+?)））"), 3),
    # 括号无版本：（书名·篇名）
    (re.compile(r"（([^《》（）]+?)·([^《》（）]+?)）"), 2),
    # 书名号+括号：（《书名·篇名》...）
    (re.compile(r"（《([^《》]+?)·([^《》]+?)》[^）]*）"), 2),
    # 纯书名号：《书名·篇名》
    (re.compile(r"《([^《》]+?)·([^《》]+?[^）》])(?:》|[）\)])"), 2),
    # 无版本：「书名·篇名」
    (re.compile(r"「([^「」]+?)·([^「」]+?)」"), 2),
]


def _canonicalize(book: str, chapter: str, version: str = "") -> dict:
    """标准化：清理残余符号，提取版本，补默认版本。"""
    book = book.strip()
    chapter = chapter.strip()
    version = (version or "").strip()

    # 清理残余符号
    for ch in ["》", ")", "）", "「", "」", "《", "（"]:
        book = book.replace(ch, "")
        chapter = chapter.replace(ch, "")

    # 若 LLM 把"通行本"写进了 chapter（如"学而篇第一，通行本"），剥离出来
    # 先处理带逗号的："学而篇第一，通行本" → chapter="学而篇第一"
    m = re.match(r"^(.+?)[，,]\s*通行本$", chapter)
    if m:
        chapter = m.group(1).strip()
        if not version:
            version = "通行本"
    # 再处理不带逗号的："颜渊篇第十二通行本" → chapter="颜渊篇第十二"
    if chapter.endswith("通行本") and len(chapter) > 3:
        m = re.match(r"^(.+?)通行本$", chapter)
        if m:
            chapter = m.group(1).strip()
            if not version:
                version = "通行本"

    # 若 chapter 末尾含（...）且 version 为空，提取为 version
    if not version:
        m = re.match(r"^(.+?)（(.+?)）$", chapter)
        if m:
            chapter = m.group(1).strip()
            version = m.group(2).strip()

    if not version:
        version = "通行本"

    raw = f"「{book}·{chapter}（{version}）」"
    return {"book": book, "chapter": chapter, "version": version, "raw": raw}


def extract_citations(text: str) -> list[dict]:
    """从文本中抽取所有引据标签，自动识别多种格式。

    返回: [{"book": "论语", "chapter": "学而篇", "version": "通行本", "raw": "..."}]
    """
    # 记录已匹配的文本区间，按优先级匹配，后匹配的区间若与已匹配区间重叠则跳过
    covered: list[tuple[int, int]] = []
    citations: list[dict] = []
    seen_raw: set[str] = set()

    for pat, n_groups in _CITATION_PATTERNS:
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            # 检查是否与已有匹配区间重叠
            if any(cs <= span[0] < ce or cs < span[1] <= ce or
                   span[0] <= cs < span[1] for cs, ce in covered):
                continue

            groups = m.groups()
            if n_groups == 3:
                c = _canonicalize(groups[0], groups[1], groups[2])
            else:
                c = _canonicalize(groups[0], groups[1])

            covered.append(span)
            if c["raw"] not in seen_raw:
                seen_raw.add(c["raw"])
                citations.append(c)

    return citations


def verify_citations(
    article_text: str,
    chunk_pool: list[dict],
) -> dict:
    """验证文章引据是否可回溯到检索 chunk。

    Returns:
        {"total": N, "matched": M, "rate": M/N, "unmatched": [...], "matched_list": [...]}
    """
    extracted = extract_citations(article_text)
    if not extracted:
        return {"total": 0, "matched": 0, "rate": 1.0, "unmatched": [], "matched_list": []}

    # 构建检索池：book·chapter key 集合
    pool_keys: set[str] = set()
    pool_citations: set[str] = set()
    # 同时建 (book, chapter) 列表，用于前缀模糊匹配
    pool_chapters: list[tuple[str, str]] = []
    for c in chunk_pool:
        book = c.get("book", "")
        chapter = c.get("chapter", "")
        version = c.get("version", "")
        if book and chapter:
            pool_keys.add(f"{book}·{chapter}")
            pool_citations.add(f"「{book}·{chapter}（{version}）」")
            pool_chapters.append((book, chapter))

    def _fuzzy_match(cit_book: str, cit_chapter: str) -> bool:
        """前缀模糊匹配：citation chapter 是 pool chapter 的前缀（或反之），且 ≥2 字符。"""
        for pb, pc in pool_chapters:
            if pb != cit_book:
                continue
            # 双向前缀：citation 是 pool 的前缀，或 pool 是 citation 的前缀
            if (pc.startswith(cit_chapter) and len(cit_chapter) >= 2) or \
               (cit_chapter.startswith(pc) and len(pc) >= 2):
                return True
        return False

    matched: list[dict] = []
    unmatched: list[dict] = []
    for c in extracted:
        key = f"{c['book']}·{c['chapter']}"
        if c["raw"] in pool_citations or key in pool_keys:
            matched.append(c)
        elif _fuzzy_match(c["book"], c["chapter"]):
            matched.append(c)
        else:
            unmatched.append(c)

    return {
        "total": len(extracted),
        "matched": len(matched),
        "rate": len(matched) / len(extracted) if extracted else 1.0,
        "unmatched": unmatched,
        "matched_list": matched,
    }


def chunk_citation_label(c: dict) -> str:
    """从 chunk dict 生成标准引据标签。"""
    return f"「{c.get('book', '')}·{c.get('chapter', '')}（{c.get('version', '')}）」"