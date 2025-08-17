FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ /app/
# COPY alembic/ /app/alembic/
COPY .env .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONPYCACHEPREFIX=/tmp/pycache
RUN mkdir -p /tmp/pycache && chmod 777 /tmp/pycache

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "8"]
# CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:8000", "--reload", "--workers", "1"]
# CMD ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8000", "app.main:app"]


# granian --interface asgi main:app