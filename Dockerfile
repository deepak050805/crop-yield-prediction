FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# Match the scikit-learn version used to serialize the bundled model.
RUN pip install --no-cache-dir -r requirements.txt scikit-learn==1.8.0 gunicorn

RUN useradd --create-home appuser

COPY app/ ./app/
COPY data/real_final_dataset.csv ./data/real_final_dataset.csv
COPY models/yield_model.pkl ./models/yield_model.pkl

USER appuser

EXPOSE 10000

CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--access-logfile", "-"]
