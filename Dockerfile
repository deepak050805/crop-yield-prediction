FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GUNICORN_CMD_ARGS="--bind=0.0.0.0:10000 --workers=2 --access-logfile=-"

WORKDIR /app

COPY backend/requirements.txt .
# Match the scikit-learn version used to serialize the bundled model.
RUN pip install --no-cache-dir -r requirements.txt scikit-learn==1.8.0 gunicorn

RUN useradd --create-home appuser

COPY backend/__init__.py backend/app.py ./backend/
COPY backend/data/real_final_dataset.csv ./backend/data/real_final_dataset.csv
COPY backend/models/ ./backend/models/
COPY frontend/templates/ ./frontend/templates/
COPY frontend/static/ ./frontend/static/

USER appuser

EXPOSE 10000

CMD ["gunicorn", "backend.app:app"]
