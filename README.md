# 识文新裁

> 贯穿古籍「数字化整理 → 学术化研究 → 大众化传播」全生命周期的 AI 平台，以 **LangGraph 工作流图**编排研微（学术研究助手）/ 新裁（沉浸式交互文创）/ 识文（智能整理）三大模块；**检索图一处构建，问答 / 辩论 / 写作三处复用**。

## 三大模块

| 模块 | 能力 | 对应图 |
|---|---|---|
| **研微** | 学术 RAG 问答（带引据）+ 研究写作 | Retrieval Graph / Writing Graph |
| **新裁** | 历史人物角色扮演 + 沉浸式叙事 | Persona Graph |
| **识文** | 古籍 OCR / 整理入口（可后置） | Ingestion Graph |

另有 **Multi-Agent 先贤辩论**：紧急度评分驱动孔子 / 孟子 / 老子 / 韩非子等先贤智能体发言权仲裁，人设漂移软监测。

## 技术栈

Python 3.11 · LangGraph · FastAPI · PostgreSQL 16 · Milvus（Lite）· BGE-M3 · DeepSeek API · MCP · React + Vite · Docker Compose · Nginx

## 目录结构

```
shiwen/
├── src/shiwen/            # 后端包
│   ├── api/               # FastAPI 入口（/health、/api/chat）
│   ├── ingest/            # S1 语料 + Ingestion
│   ├── rag/               # S2/S3 检索图
│   ├── agents/            # S4 辩论 / S6 角色扮演
│   ├── mcp/               # S8 MCP 工具层
│   └── eval/              # S7 评测
├── frontend/              # React + Vite SPA（+ 多阶段 Dockerfile）
├── mcp/                   # MCP 工具定义 + 外部客户端示例
├── eval/                  # 评测集 + report.md
├── data/corpus/           # 古籍语料（不进 git）
├── scripts/               # reindex.sh / deploy.sh
├── docker-compose.yml     # backend + postgres + nginx
├── nginx.conf             # 前端静态 + 反代 /api
└── .github/workflows/     # CI/CD 部署
```

完整架构与实施规划见 `识文新裁-完整交付文档.md`。

## 本地启动

```bash
# 1. 准备环境变量
cp .env.example .env        # 填入 DeepSeek API Key 等真实值

# 2. 启动（首次会构建镜像）
docker compose up -d --build

# 3. 验证（nginx 对外端口为 8090）
curl http://localhost:8090/health          # {"status":"ok"}
curl -X POST http://localhost:8090/api/chat -H "Content-Type: application/json" -d '{"query":"学而时习之"}'
```

后端开发（不经过 Docker）：

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.shiwen.api.main:app --reload            # 访问 http://localhost:8000/health
```

前端开发：

```bash
cd frontend && npm install && npm run dev
```

## 语料与 Ingestion（S1）

语料为公版古籍白文，转写 JSON 取自 [Guopop/chinese-philosophy](https://github.com/Guopop/chinese-philosophy)（CText 镜像，**许可未标注，仅研究/演示用，切勿商用**）。清单集中在 `data/corpus/manifest.yaml` —— 扩展古籍只需增删一行，代码无需改动。默认收录 **20 部**（经史子集：论语/孟子/道德经/庄子/荀子/韩非子/孙子兵法/史记/楚辞等）。

入库全流程在 **Docker backend 容器**内执行（绕开 milvus-lite 的 Windows 兼容问题、不占服务器资源）：

```bash
# 首次需构建镜像（拉 BGE-M3 权重约 2GB，缓存于 hf_cache 卷）
docker compose build

# 一键全流程：下载 → 规范化 → 切分 → 向量化 → 入库 → 灌人物表 → golden 自检
bash scripts/reindex.sh

# 或分步执行（国内下载可加 --mirror ghfast；HF_ENDPOINT 默认走 hf-mirror）
docker compose run --rm backend python -m src.shiwen.ingest download --mirror ghfast
docker compose run --rm backend python -m src.shiwen.ingest normalize
docker compose run --rm backend python -m src.shiwen.ingest reindex      # 全量入库 Milvus + PG
docker compose run --rm backend python -m src.shiwen.ingest seed-people  # 灌人物关系表
docker compose run --rm backend python -m src.shiwen.ingest verify       # 3/3 golden 通过
docker compose run --rm backend python -m src.shiwen.ingest inspect --book lunyu
```

数据落点：Milvus Lite（`data/milvus.db`）+ PostgreSQL（`chunk` 表 + `person/person_work/person_relation` 人物关系三表），人物关系满足「孔子-著-论语」等查询。切分为结构感知（卷/篇/章），篇是引据原子单位；chunk 永不跨章。

## 阿里云部署（ECS + Docker Compose）

### 首次部署

```bash
# 在服务器上（需已装 Docker + git）
git clone https://github.com/<user>/shiwen.git /opt/shiwen
cd /opt/shiwen
cp .env.example .env        # 填入真实密钥
docker compose up -d --build
```

> 识文新裁对外端口为 **8090**（服务器 80/8080 已被其他正式服务占用，独立端口共存）。
> 安全组仅开放 **8090**；backend 不映射公网端口，由 nginx 反代。
> 访问地址：`http://<ECS公网IP>:8090`（前端页 + `/api/chat`）

### CI/CD 自动部署

推送 `main` 分支后，GitHub Actions 通过 SSH 到 ECS 执行 `git pull + docker compose up -d --build`。

配置 secrets：

```bash
gh secret set ECS_HOST    -b"<ECS 公网 IP>"
gh secret set ECS_USER    -b"<root 或 ubuntu>"
gh secret set ECS_SSH_KEY -b"$(cat ~/.ssh/id_rsa)"
```

## 阶段进度

- [x] S0 环境 / 建仓 / docker-compose / CI/CD
- [x] S1 语料 + Ingestion（含人物关系表）
- [ ] S2 检索单跳
- [ ] S3 多跳闭环 + 自我反思
- [ ] S4 先贤辩论
- [ ] S5 研究写作
- [ ] S6 新裁角色扮演
- [ ] S7 评测与可观测
- [ ] S8 MCP 工具层
- [ ] S9 阿里云部署（线上试用版）

## License

语料许可见 `识文新裁-完整交付文档.md` §3；代码暂未定许可证。
