FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY bentofile.yaml ./

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV MODEL_DEVICE=cpu

EXPOSE 8000

CMD ["bentoml", "serve", "mena_mlops.serving.bento_service:SentimentService", "--host", "0.0.0.0", "--port", "8000"]
