# MCP 工具层（S8）

将检索（retrieve）、引据校验（verify）、人物关系查询（query_person）、写作引擎（write）
统一封装为 MCP 工具，供三张业务图（检索图 / 辩论图 / 写作图）与外部客户端复用。

## 工具

| 工具 | 说明 | 输入 | 输出 |
|---|---|---|---|
| `retrieve` | 三路混合检索（元数据前置过滤 + 向量 + BM25）→ RRF 融合 | `query`, `top_k`, `book_id` | chunk 列表 JSON |
| `verify` | 引据合规校验：引用「书·篇·版本」与检索 chunk 元数据一致性 | `text`, `chunks_json` | 命中/未命中统计 JSON |
| `query_person` | 人物关系表查询（SQL join，关系类问题直答） | `name_or_id` | person/works/relations JSON |
| `write` | 研究写作引擎（大纲 → 逐节检索 → 逐节写作 → 综合润色） | `topic`, `max_sections` | article/sections/citations JSON |

## 技术

`fastmcp`（Python SDK）。实现见 `src/shiwen/mcp/server.py`。

## 运行

stdio 模式（供 Claude Code / 支持 MCP 的客户端连接）：

```bash
pip install -r requirements.txt
python -m src.shiwen.mcp.server
```

## Claude Code 接入

在 `.mcp.json` 中注册：

```json
{
  "mcpServers": {
    "shiwen": {
      "command": "python",
      "args": ["-m", "src.shiwen.mcp.server"],
      "cwd": "/path/to/shiwen"
    }
  }
}
```

> 工具内部会惰性加载检索 / 写作依赖，连接时无需数据库在线；
> 但 `retrieve` / `query_person` / `write` 首次调用时需 Milvus / PostgreSQL 已入库。

## 程序内调用

```python
from src.shiwen.mcp.server import retrieve, verify, query_person, write

# 各工具返回 JSON 字符串，可直接 json.loads 解析
print(retrieve("仁是什么", top_k=3, book_id="lunyu"))
print(query_person("孔子"))
```
