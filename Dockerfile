# PyroCast Florida web demo — serves the precomputed risk map + live sklearn/LightGBM explanations.
# Dockerfile build (instead of Railpack) so we can guarantee LightGBM's OpenMP runtime (libgomp1),
# which the lightgbm wheel links against but does NOT bundle — without it `import lightgbm` fails and
# the model can't unpickle. Also sidesteps the mise Python-attestation issue entirely.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT at runtime; fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "cd web && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
