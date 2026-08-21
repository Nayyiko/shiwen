#!/usr/bin/env bash
# 一键重建索引：下载 → 规范化 → 切分/向量化/入库 → 灌人物表 → golden 自检
# 在宿主机运行（内部通过 docker compose run 进 backend 容器执行，避免 Windows 原生跑 milvus-lite）
#
# 用法：
#   bash scripts/reindex.sh                 # 全量
#   BOOKS=lunyu,mengzi bash scripts/reindex.sh   # 只跑部分书
#   MIRROR= bash scripts/reindex.sh         # 直连 GitHub（已配 CORPUS_RAW_BASE 时）
#
# 前置：已 docker compose build（首次会拉 BGE-M3 权重约 2GB）
set -euo pipefail

MIRROR="${MIRROR:-ghfast}"
BOOKS="${BOOKS:-}"

BOOKS_ARG=()
if [ -n "$BOOKS" ]; then
  BOOKS_ARG=(--books "$BOOKS")
fi

echo "==> 1. 下载语料（mirror=${MIRROR:-直连}）"
if [ -n "$MIRROR" ]; then
  docker compose run --rm backend python -m src.shiwen.ingest download --mirror "$MIRROR" "${BOOKS_ARG[@]}"
else
  docker compose run --rm backend python -m src.shiwen.ingest download "${BOOKS_ARG[@]}"
fi

echo "==> 2. 规范化 raw -> markdown"
docker compose run --rm backend python -m src.shiwen.ingest normalize "${BOOKS_ARG[@]}"

echo "==> 3. 重建索引（结构感知切分 + BGE-M3 向量化 + 入库 Milvus/PG）"
docker compose run --rm backend python -m src.shiwen.ingest reindex "${BOOKS_ARG[@]}"

echo "==> 4. 灌人物关系表"
docker compose run --rm backend python -m src.shiwen.ingest seed-people

echo "==> 5. golden 引据自检"
docker compose run --rm backend python -m src.shiwen.ingest verify
