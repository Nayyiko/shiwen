"""S8 MCP 工具层：检索 / 引据校验 / 人物关系查询 / 写作引擎统一封装为 MCP 工具。

对外暴露 4 个工具（见 server.py）：
  - retrieve: 三路混合检索（向量 + BM25 + RRF）
  - verify: 引据合规校验
  - query_person: 人物关系查询
  - write: 研究写作引擎

运行：
  python -m src.shiwen.mcp.server
"""

from .server import mcp

__all__ = ["mcp"]