# MCP 工具层（S8）

将检索（retrieve）、引据校验（verify）、人物关系查询（query_person）、写作引擎（write）
统一封装为 MCP 工具，供三张业务图（检索图 / 辩论图 / 写作图）与外部客户端复用。

## 规划工具

| 工具 | 说明 |
|---|---|
| `retrieve` | 三路混合检索（元数据前置过滤 + 向量 + BM25）→ RRF 融合 |
| `verify` | 引据合规校验：引用「书·篇·版本」与检索 chunk 元数据一致性 |
| `query_person` | 轻量人物关系表查询（SQL join，关系类问题直答） |
| `write` | 研究写作引擎（大纲 → 逐节检索 → 逐节写作 → 综合润色） |

## 技术

`fastmcp` / `mcp`（Python SDK）。实现见 `src/shiwen/mcp/`。

外部客户端调用示例将在 S8 落地时补充。
