# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ARG TARGETARCH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for build tools (install MS ODBC driver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    gnupg \
    unixodbc-dev \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && if [ "$TARGETARCH" = "amd64" ]; then \
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft-prod.list; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        echo "deb [arch=arm64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft-prod.list; \
    fi \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy slim prod requirements
COPY requirements-prod.txt ./
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

# Create non-root user
RUN useradd -m appuser
USER appuser

# Copy source code
COPY . .

EXPOSE 8080
ENV PORT=8080 \
    ENV=pilot

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

# Uvicorn entrypoint (FastAPI)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]
