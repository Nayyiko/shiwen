#!/usr/bin/env bash
# 阿里云 ECS 首次部署 / 更新脚本（在服务器上执行）
# 用法：bash deploy.sh
set -euo pipefail

APP_DIR="/opt/shiwen"
# 公开仓库地址（建仓后替换为实际地址）
REPO_URL="https://github.com/Nayyiko/shiwen.git"

echo "==> 1. 确保代码目录存在"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> 2. 拉取最新代码"
git pull origin main

echo "==> 3. 准备 .env"
if [ ! -f .env ]; then
  echo "!! 未找到 .env：请先 cp .env.example .env 并填入真实密钥（DeepSeek Key、DB 密码）"
  exit 1
fi

echo "==> 4. 构建并启动（前端由多阶段 Dockerfile 构建，服务器无需装 Node）"
docker compose up -d --build

echo "==> 5. 状态"
docker compose ps
