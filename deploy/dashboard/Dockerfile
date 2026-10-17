FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "pandas>=2.2" \
    "streamlit>=1.40"

COPY scripts/dashboard.py ./scripts/dashboard.py

EXPOSE 8501

CMD ["streamlit", "run", "scripts/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
