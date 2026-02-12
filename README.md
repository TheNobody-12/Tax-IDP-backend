# Bookkeeper API (FastAPI)

FastAPI backend for the Bookkeeper document ingestion and review app. It handles uploads, runs the DI + LLM pipeline, exposes document endpoints, and optionally refreshes Gold DB tables.

## Features
- Async FastAPI with background upload processing.
- Azure Document Intelligence (DI) + Azure OpenAI LLM per-page extraction.
- Azure Blob Storage for bronze/silver artifacts.
- Optional Gold ETL writes to SQL Server (pyodbc).
- CSV export, PDF streaming, and review/patch endpoints.

## Prerequisites
- Python 3.11+
- Azure resources: Storage account, Document Intelligence, Azure OpenAI, SQL Server.
- For local SQL access: Microsoft ODBC Driver 18 for SQL Server (`msodbcsql18`) installed.
- Docker (build as `linux/amd64` so ODBC driver is present in the image).

## Environment Variables
Set via `.env` (no spaces around `=`). Key vars:
```
PORT=8080
AzureWebJobsStorage=...              # Blob connection
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
AZURE_OPENAI_API_KEY=...             # or AZURE_OPENAI_SUBSCRIPTION_KEY
AZURE_DI_ENDPOINT=...
AZURE_DI_KEY=...                     # or use AAD via AZURE_TENANT_ID/CLIENT_ID/CLIENT_SECRET
SQL_CONN_STR=Driver={ODBC Driver 18 for SQL Server};Server=...;Database=...;Uid=...;Pwd=...;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;
FORCE_PYODBC=true                    # prefer pyodbc
LLM_MAX_CONCURRENCY=3                # optional
```

## Local Development (venv)
```bash
cd bookkeeper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prod.txt
# ensure msodbcsql18 is installed on your OS
set -a; source .env; set +a
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
```
- Health: `GET /api/health`
- Readiness: `GET /api/readiness`

## Docker (local)
Build amd64 so the ODBC driver installs:
```bash
cd bookkeeper
docker build --platform=linux/amd64 -t bookkeeper-api:latest .
docker run --platform=linux/amd64 -p 8080:8080 --env-file .env bookkeeper-api:latest
```

## Azure Container Apps (ACA) Quick Deploy
```bash
# Push image to ACR
docker build --platform=linux/amd64 -t <ACR>.azurecr.io/bookkeeper-api:latest .
# Login to ACR
az acr login -n <ACR>
# Push image
docker push <ACR>.azurecr.io/bookkeeper-api:latest

# Create env (once)
az containerapp env create -g <RG> -n <ACA_ENV> --location <REGION>

# Deploy
az containerapp create \
  -g <RG> -n bookkeeper-api \
  --environment <ACA_ENV> \
  --image <ACR>.azurecr.io/bookkeeper-api:latest \
  --cpu 1 --memory 2Gi \
  --ingress external --target-port 8080 \
  --registry-server <ACR>.azurecr.io --registry-username <ACR_USER> --registry-password <ACR_PASS> \
  --secrets \
    openai-key=<AZURE_OPENAI_API_KEY> \
    di-key=<AZURE_DI_KEY> \
    sql-conn-str="<SQL_CONN_STR>" \
    storage-conn="<AzureWebJobsStorage>" \
  --env-vars \
    PORT=8080 \
    FORCE_PYODBC=true \
    AZURE_OPENAI_ENDPOINT=<endpoint> \
    AZURE_OPENAI_DEPLOYMENT=<deployment> \
    AZURE_OPENAI_API_KEY=secretref:openai-key \
    AZURE_DI_ENDPOINT=<di-endpoint> \
    AZURE_DI_KEY=secretref:di-key \
    SQL_CONN_STR=secretref:sql-conn-str \
    AzureWebJobsStorage=secretref:storage-conn
```
Probes: liveness `/api/health`, readiness `/api/readiness`.

## Key Endpoints
- `GET /api/health` / `GET /api/readiness`
- `POST /api/documents` (multipart: file, client_id?, tax_year?)
- `GET /api/documents` / `GET /api/documents/{doc_id}`
- `GET /api/documents/{doc_id}/pdf`
- `GET /api/documents/{doc_id}/silver` / `.../silver/csv`
- `PATCH /api/documents/{doc_id}/silver/page/{page_number}`
- `PATCH /api/documents/{doc_id}/status` / `PATCH /api/documents/{doc_id}`
- `GET /api/clients` / `POST /api/clients`
- `GET /api/dashboard`, `GET /api/metrics`

### Endpoint quick reference
| Method | Path | Purpose |
| --- | --- | --- |
| GET | /api/health | Liveness check |
| GET | /api/readiness | Dependency readiness (blob/SQL/LLM) |
| POST | /api/documents | Upload a PDF and start the pipeline (multipart: `file`, `client_id?`, `tax_year?`) |
| GET | /api/documents | List documents (filters: `client_id`, `tax_year`) |
| GET | /api/documents/{doc_id} | Document metadata + silver JSON |
| GET | /api/documents/{doc_id}/pdf | Stream original PDF |
| GET | /api/documents/{doc_id}/silver | Raw silver JSON |
| GET | /api/documents/{doc_id}/silver/csv | Silver as CSV (one row per page) |
| GET | /api/documents/other | Low-confidence/Other docs |
| GET | /api/dashboard | Aggregated counts/recent docs |
| PATCH | /api/documents/{doc_id}/silver/page/{page_number} | Patch a page’s extracted fields/category/confidence |
| PATCH | /api/documents/{doc_id}/status | Patch document status/category |
| PATCH | /api/documents/{doc_id} | Override document category (with rules) |
| GET | /api/clients | List clients |
| POST | /api/clients | Create/update a client |
| GET | /api/uploads | List upload jobs |
| GET | /api/uploads/{job_id} | Upload job status |
| GET | /api/metrics | Prometheus-style counters |

## Notes & Troubleshooting
- IM002 (ODBC driver not found): ensure the container is built as `linux/amd64` and `SQL_CONN_STR` uses `Driver={ODBC Driver 18 for SQL Server}` (no outer quotes), or point directly to the installed .so.
- LLM unavailable: check OpenAI env vars are present and valid inside the container; single worker (`--workers 1`) surfaces init errors.
- DI permission errors: use `AZURE_DI_KEY` or grant the AAD principal `Microsoft.CognitiveServices/accounts/FormRecognizer/documentmodels:analyze/action`.

## Updating the container image after changes
1) Rebuild the image (amd64):
```
cd bookkeeper
docker build --platform=linux/amd64 -t <ACR>.azurecr.io/bookkeeper-api:latest .
```
2) Push to ACR:
```
docker push <ACR>.azurecr.io/bookkeeper-api:latest
```
3) Update your Azure Container App to pull the new tag. Either:
   - Portal: Container App → Containers → pick `bookkeeper-api:latest` and Save/Restart.
   - CLI:
```
az containerapp update -g <RG> -n bookkeeper-api \
  --image <ACR>.azurecr.io/bookkeeper-api:latest \
  --registry-server <ACR>.azurecr.io \
  --registry-username <ACR_USER> --registry-password <ACR_PASS>
```
4) Verify health:
```
curl https://<FQDN>/api/health
```
