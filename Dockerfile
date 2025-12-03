# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ARG TARGETARCH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for build tools (install MS ODBC driver when on amd64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    gnupg \
    unixodbc-dev \
    && if [ "$TARGETARCH" = "amd64" ]; then \
        curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft-prod.list && \
        apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
    else \
        echo "Skipping msodbcsql18 install on arch=$TARGETARCH"; \
    fi \
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

# Uvicorn entrypoint (FastAPI)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--log-level", "info"]
