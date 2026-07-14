# 多阶段构建：builder 装依赖，runtime 只带必要产物
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
ENV PYTHONUNBUFFERED=1
# 默认生成一份茅台研报；实际用 docker run ... python main.py "标的"
CMD ["python", "main.py", "贵州茅台 600519"]
