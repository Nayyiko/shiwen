FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# 先装 CPU 版 torch：PyPI 默认 Linux torch 会连带拉取 CUDA 全家桶(~10GB)，本机/服务器都用不上 GPU
RUN pip install --no-cache-dir "torch==2.13.0+cpu" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://mirrors.aliyun.com/pypi/simple/
# 国内 PyPI 镜像加速其余依赖（sentence-transformers 复用已装的 CPU torch，跳过 CUDA 版）
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY src ./src
# golden 引据数据（verify 用）与集成测试（容器内 pytest 用）
COPY eval ./eval
COPY tests ./tests
COPY pytest.ini .

EXPOSE 8000

CMD ["uvicorn", "src.shiwen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
