FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHAINFORGE_ROOT=/app \
    CHAINFORGE_DATA=/app/data \
    PORT=7860

WORKDIR /app

COPY pyproject.toml README.md ./
COPY chainforge ./chainforge
COPY web ./web
COPY corpus ./corpus
COPY models ./models

RUN pip install --no-cache-dir torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .

RUN mkdir -p /app/data

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '7860'))" || exit 1

CMD ["sh", "-c", "uvicorn chainforge.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
