FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "bentoml>=1.4" \
    "fastapi>=0.142.2" \
    "numpy>=1.26" \
    "pydantic>=2" \
    "pandas>=2.2" \
    "prometheus-client>=0.26.0" \
    "redis>=5.0" \
    "transformers>=4.45" \
    "onnxruntime>=1.18"

COPY src ./src
COPY scripts ./scripts

ENV PYTHONPATH=/app/src
ENV MODEL_BACKEND=onnx
ENV MODEL_DEVICE=cpu

EXPOSE 8000

CMD ["bentoml", "serve", "mena_mlops.serving.bento_service:SentimentService", "--host", "0.0.0.0", "--port", "8000"]
