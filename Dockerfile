FROM python:3.11-slim AS base
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["python", "-m", "src.main"]
