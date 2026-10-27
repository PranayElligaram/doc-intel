FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY docintel ./docintel
RUN pip install --no-cache-dir ".[app]"

FROM base AS test
COPY scripts ./scripts
COPY eval ./eval
COPY tests ./tests
RUN pip install --no-cache-dir ".[dev,app]"
CMD ["python", "-m", "pytest", "-q"]

FROM base AS app
RUN useradd --create-home appuser
COPY app ./app
USER appuser
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
