FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    GUNICORN_CMD_ARGS="--bind=0.0.0.0:10000 --workers=2 --access-logfile=- --error-logfile=- --timeout=30 --graceful-timeout=30 --worker-tmp-dir=/tmp"

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
# Wheels avoid compilers and their build dependencies in this runtime image.
RUN python -m pip install --no-cache-dir --only-binary=:all: -r backend/requirements.txt \
    && python -m pip check

RUN useradd --no-create-home --uid 10001 --user-group appuser

# Include application, validation, prediction, and shared ML schema modules.
COPY backend/ ./backend/
COPY frontend/templates/ ./frontend/templates/
COPY frontend/static/ ./frontend/static/

USER appuser

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import http.client; connection = http.client.HTTPConnection('127.0.0.1', 10000, timeout=2); connection.request('GET', '/health/ready'); raise SystemExit(0 if connection.getresponse().status == 200 else 1)"]

CMD ["gunicorn", "backend.app:app"]
