# Back end (FastAPI) for the Stock Market Predictor.
#
# Built two ways from the same file:
#   * locally:            docker compose up          (context = the repository root)
#   * Hugging Face Space: .github/workflows/deploy-backend.yml copies this file to the Space,
#                         next to the folders listed below.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Only what the API needs (no PyTorch, Streamlit or Jupyter): see backend/requirements-api.txt
COPY backend/requirements-api.txt ./backend/requirements-api.txt
RUN pip install --upgrade pip && pip install -r backend/requirements-api.txt

# Code, saved models and the data files the API reads (.dockerignore keeps the research data out)
COPY backend ./backend
COPY src ./src
COPY models ./models
COPY data ./data
COPY reports ./reports
COPY sentiment_study ./sentiment_study
COPY live ./live

# Run as a normal user (Hugging Face Spaces requires user id 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
