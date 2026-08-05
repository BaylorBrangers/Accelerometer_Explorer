FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ACCEL_DATA_DIR=/data \
    ACCEL_ARTIFACTS_DIR=/artifacts \
    HF_HOME=/artifacts/hf_cache \
    HF_HUB_CACHE=/artifacts/hf_cache/hub \
    TRANSFORMERS_CACHE=/artifacts/hf_cache

WORKDIR /app

# CPU PyTorch first for a smaller, portable image
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY data ./data
COPY .streamlit ./.streamlit

RUN pip install -e . \
    && mkdir -p /data /artifacts /artifacts/hf_cache

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
