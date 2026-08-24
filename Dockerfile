FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# 服务器走 Cloudflare API embedding，不装本地模型（torch/sentence-transformers 已移入 requirements-local.txt），
# 避免 2G 内存小服务器 build 时 OOM。国内 PyPI 镜像加速其余依赖。
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY src ./src
# golden 引据数据（verify 用）与集成测试（容器内 pytest 用）
COPY eval ./eval
COPY tests ./tests
COPY pytest.ini .

EXPOSE 8000

CMD ["uvicorn", "src.shiwen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
