# 多阶段构建：builder 装依赖，runtime 只带必要产物
# 镜像只含基础依赖（API 模式 + 离线 stub 可跑）；RAG 重依赖（torch/FlagEmbedding）
# 体积巨大且通常在宿主机建索引，用 volume 挂 data/ 进来即可。
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
ENV PYTHONUNBUFFERED=1
# 默认生成一份茅台研报（无 .env 时走离线 stub 也能出报告）；
# 实际用法：docker run --env-file .env -v ./data:/app/data -v ./reports:/app/reports <image> python main.py "标的"
CMD ["python", "main.py", "贵州茅台 600519"]
