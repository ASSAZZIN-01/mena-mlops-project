FROM python:3.13-slim

WORKDIR /opt/mena-mlops-project

RUN pip install --no-cache-dir \
    "fastapi>=0.142.2" \
    "pandas>=2.2" \
    "pyarrow>=17.0" \
    "pydantic>=2" \
    "pyyaml>=6.0" \
    "uvicorn[standard]>=0.54.0"

COPY src ./src
COPY configs ./configs
COPY web ./web

ENV PYTHONPATH=/opt/mena-mlops-project/src

EXPOSE 8503

CMD ["uvicorn", "mena_mlops.control_plane:app", "--host=0.0.0.0", "--port=8503"]
