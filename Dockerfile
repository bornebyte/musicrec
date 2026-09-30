FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY musicrec ./musicrec
ENV PYTHONUNBUFFERED=1 HF_HOME=/models NUMBA_CACHE_DIR=/tmp/numba
EXPOSE 8000
CMD ["uvicorn", "musicrec.api:app", "--host", "0.0.0.0", "--port", "8000"]
