FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN python3 -m pip install --break-system-packages --no-cache-dir uv \
    && uv sync --frozen --no-dev \
    && uv pip uninstall --python /app/.venv/bin/python torch \
    && uv pip install \
        --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu124 \
        --no-deps \
        "torch==2.5.1+cu124"

CMD ["python", "scripts/vertex_train.py"]
