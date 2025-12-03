"""
main.py

FastAPI API for the Bookkeeper web frontend (no-SQL version).

Key endpoints:

- POST /api/documents                 -> upload & run pipeline
- GET  /api/documents                 -> list documents (from meta files)
- GET  /api/documents/<doc_id>        -> document + silver
- GET  /api/documents/<doc_id>/pdf    -> stream PDF
- GET  /api/documents/<doc_id>/silver -> raw silver JSON
- GET  /api/documents/<doc_id>/silver/csv -> page-wise CSV
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from threading import Lock
import uuid
import threading
import signal, time

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

# Import the local no-SQL pipeline
try:
    from src.pipeline.process_document import run as run_pipeline
    from src.pipeline.db import get_sql_conn
    from src.pipeline.azure_storage import download_blob_json, move_silver_category, upload_silver_json
    from src.pipeline.gold_etl import upsert_from_silver
    HAVE_GOLD_ETL = True
    HAVE_BLOB = True
except Exception:
    # Fallback relative import depending on how PYTHONPATH is set
    from .src.pipeline.process_document import run as run_pipeline  # type: ignore
    from .src.pipeline.db import get_sql_conn  # type: ignore
    HAVE_BLOB = False
    HAVE_GOLD_ETL = False

    async def download_blob_json(container: str, blob: str): return None
    async def move_silver_category(doc_id: str, old_category: str | None, new_category: str, client_id: str | None, tax_year: int | None): return {}
    async def upload_silver_json(blob_ref: str, silver_doc: Dict[str, Any]): return None
    def upsert_from_silver(meta: Dict[str, Any], silver_doc=None): return None

try:
    from azure.storage.blob import BlobServiceClient
    HAVE_AZURE_SDK = True
except Exception:
    HAVE_AZURE_SDK = False

from src.config.app_config import get_config
from src.config.log_context import set_log_context, get_log_context

cfg = get_config()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = get_log_context()
        if ctx.get("doc_id"):
            base["doc_id"] = ctx["doc_id"]
        if ctx.get("job_id"):
            base["job_id"] = ctx["job_id"]
        if hasattr(record, 'doc_id') and record.doc_id and 'doc_id' not in base:
            base['doc_id'] = record.doc_id
        return json.dumps(base)


# Reconfigure root handler to JSON
for h in logging.getLogger().handlers:
    h.setFormatter(JsonFormatter())

AZURE_SILVER_CONTAINER = os.getenv("AZURE_SILVER_CONTAINER", "silver")
AZURE_BRONZE_CONTAINER = os.getenv("AZURE_BRONZE_CONTAINER", "bronze")
SQL_COMMAND_TIMEOUT = int(os.getenv("SQL_COMMAND_TIMEOUT", "30"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Reduce verbosity for azure/httpx noisy logs in prod
for noisy in ("azure", "httpx", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# Helper to redact IDs in logs
def _redact(value: Any) -> str:
    s = str(value) if value is not None else ""
    if len(s) <= 4:
        return "***"
    return f"{s[:3]}***{s[-2:]}"


app = FastAPI(title="Bookkeeper API", version="1.0.0")

# CORS: allow specific origins when provided; default to permissive (legacy)
allowed_origins = os.getenv("ALLOWED_ORIGINS")
if allowed_origins:
    origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_edit_lock = Lock()
_job_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

# Metrics storage (define before use)
_metrics = {
    "documents_total": 0,
    "documents_failed": 0,
}


async def _run_blocking(func, *args, **kwargs):
    return await run_in_threadpool(func, *args, **kwargs)


def _resolve_anycase(root: Path, filename: str) -> Path | None:
    """
    Return a Path in root matching filename, attempting case-insensitive resolution.
    """
    exact = root / filename
    if exact.exists():
        return exact
    lower = root / filename.lower()
    if lower.exists():
        return lower
    # Fallback scan (rare)
    for candidate in root.glob("*"):
        if candidate.name.lower() == filename.lower():
            return candidate
    return None


def _set_job(job_id: str, **kwargs):
    with _job_lock:
        job = _jobs.get(job_id, {})
        job.update(kwargs)
        _jobs[job_id] = job
        return job


def _get_job(job_id: str) -> Dict[str, Any] | None:
    with _job_lock:
        return _jobs.get(job_id)


def _apply_cursor_timeout(cur):
    try:
        cur.timeout = SQL_COMMAND_TIMEOUT
    except Exception:
        pass


def _resolve_client_name(client_id: str | None) -> str | None:
    if not client_id:
        return None
    try:
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute("SELECT TOP 1 ClientName FROM gold.dimClient WHERE ClientID = ?", (client_id,))
        row = cur.fetchone()
        conn.close()
        return (row[0] if row and len(row) > 0 else None) or client_id
    except Exception as e:
        logger.warning(f"[API] Could not resolve client name for {client_id}: {e}")
        return client_id


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def _load_meta(doc_id: str) -> Dict[str, Any] | None:
    """Resolve meta from Gold DB first, then optional local cache."""
    meta = _load_meta_from_db(doc_id)
    if meta:
        return meta

    return None


def _load_silver(doc_id: str) -> Dict[str, Any] | None:
    """Build silver from Gold DB (DocumentPage) or optional local cache. No blob fallback."""
    silver_db = _load_silver_from_db(doc_id)
    if silver_db is not None:
        return silver_db
    return None


def _load_meta_from_db(doc_id: str) -> Dict[str, Any] | None:
    try:
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute("SELECT DocID, ClientID, ClientName, TaxYear, Category, Status, Confidence, LLMUsed, BronzeBlob, SilverJsonBlob, SilverPdfBlob FROM gold.dimDocument WHERE DocID = ?", (doc_id,))
        r = cur.fetchone()
        conn.close()
        if not r:
            return None
        return {
            "doc_id": getattr(r, 'DocID', None),
            "client_id": getattr(r, 'ClientID', None),
            "client_name": getattr(r, 'ClientName', None),
            "tax_year": getattr(r, 'TaxYear', None),
            "category": getattr(r, 'Category', None),
            "status": getattr(r, 'Status', None),
            "confidence": float(getattr(r, 'Confidence', 0) or 0),
            "llm_used": bool(getattr(r, 'LLMUsed', 0)),
            "llmUsed": bool(getattr(r, 'LLMUsed', 0)),
            "bronze_pdf_blob": getattr(r, 'BronzeBlob', None),
            "silver_json_blob": getattr(r, 'SilverJsonBlob', None),
            "silver_pdf_blob": getattr(r, 'SilverPdfBlob', None),
        }
    except Exception as e:
        logger.warning(f"[API] _load_meta_from_db failed: {e}")
        return None


def _load_silver_from_db(doc_id: str) -> Dict[str, Any] | None:
    try:
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute("SELECT DocID, ClientID, ClientName, TaxYear, Category, Status, Confidence, LLMUsed FROM gold.dimDocument WHERE DocID = ?", (doc_id,))
        meta_row = cur.fetchone()
        if not meta_row:
            conn.close()
            return None
        cur.execute(
            "SELECT PageNumber, Category, Confidence, Status, LLMUsed, ExtractedJson FROM gold.DocumentPage WHERE DocID = ? ORDER BY PageNumber ASC",
            (doc_id,)
        )
        pages_rows = cur.fetchall()
        conn.close()
        pages: List[Dict[str, Any]] = []
        for pr in pages_rows:
            try:
                fields = json.loads(getattr(pr, 'ExtractedJson', {}) or "{}")
            except Exception:
                fields = {}
            pages.append({
                "page_number": getattr(pr, 'PageNumber', None),
                "category": getattr(pr, 'Category', None),
                "confidence": float(getattr(pr, 'Confidence', 0) or 0),
                "status": getattr(pr, 'Status', None),
                "llm_used": bool(getattr(pr, 'LLMUsed', 0)),
                "extracted_fields": fields,
            })
        silver_doc = {
            "doc_id": getattr(meta_row, 'DocID', None),
            "category": getattr(meta_row, 'Category', None),
            "tax_year": getattr(meta_row, 'TaxYear', None),
            "confidence": float(getattr(meta_row, 'Confidence', 0) or 0),
            "llm_used": bool(getattr(meta_row, 'LLMUsed', 0)),
            "pages": pages,
        }
        return silver_doc
    except Exception as e:
        logger.warning(f"[API] _load_silver_from_db failed: {e}")
        return None


def _load_all_meta() -> List[Dict[str, Any]]:
    # Prefer Gold dimDocument when available
    try:
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute(
            """
            SELECT DocID, ClientID, ClientName, TaxYear, Category, Status, Confidence, LLMUsed, BronzeBlob, SilverJsonBlob, SilverPdfBlob, CreatedAt, UpdatedAt
            FROM gold.dimDocument
            ORDER BY CreatedAt DESC
            """
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "doc_id": getattr(r, 'DocID', None),
                "client_id": getattr(r, 'ClientID', None),
                "client_name": getattr(r, 'ClientName', None),
                "tax_year": getattr(r, 'TaxYear', None),
                "category": getattr(r, 'Category', None),
                "status": getattr(r, 'Status', None),
                "confidence": float(getattr(r, 'Confidence', 0) or 0),
                "llm_used": bool(getattr(r, 'LLMUsed', 0)),
                "llmUsed": bool(getattr(r, 'LLMUsed', 0)),
                "bronze_pdf_blob": getattr(r, 'BronzeBlob', None),
                "silver_json_blob": getattr(r, 'SilverJsonBlob', None),
                "silver_pdf_blob": getattr(r, 'SilverPdfBlob', None),
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"[API] _load_all_meta Gold fallback due to: {e}")
    return []


@app.post("/api/documents", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), client_id: str | None = Form(None), tax_year: int | None = Form(None)):
    """
    Upload a PDF and queue the pipeline.

    Form fields:
    - file: uploaded PDF
    - client_id: optional
    - tax_year: optional (int)
    """
    if file is None:
        raise HTTPException(status_code=400, detail="Missing file field")

    BASE_DIR = Path("/tmp/bookkeeper")
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / file.filename
    tmp_path.write_bytes(await file.read())

    job_id = str(uuid.uuid4())
    _set_job(job_id, status="queued", message="Waiting to start", doc_id=None, client_id=client_id, tax_year=tax_year, filename=file.filename)

    def _worker():
        set_log_context(job_id=job_id)
        _set_job(job_id, status="running", message="Processing", started_at=str(datetime.now()))
        try:
            def _progress(update: Dict[str, Any]):
                _set_job(job_id, **update)
            result = asyncio.run(
                run_pipeline(
                    input_pdf=tmp_path,
                    client_id=client_id,
                    tax_year=tax_year,
                    generate_gold=False,
                    progress_cb=_progress,
                )
            )
            _metrics['documents_total'] += 1
            _set_job(job_id, status="completed", message="Completed", doc_id=result.get("doc_id"), meta=result.get("meta"))
        except Exception as e:
            _metrics['documents_failed'] += 1
            logger.exception("Upload job failed", extra={"job_id": job_id})
            _set_job(job_id, status="failed", message=str(e))

    background_tasks.add_task(_worker)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/clients")
async def list_clients():
    """Return list of clients from SQL (gold.dimClient) or empty list if SQL unavailable."""
    def _fetch_clients():
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute(
            """
            SELECT ClientID, ClientName, Email, Phone, AddressLine1, AddressLine2, City, Province, PostalCode, Country, Notes
            FROM gold.dimClient
            ORDER BY COALESCE(ClientName, ClientID) ASC
            """
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "clientId": getattr(r, 'ClientID', None),
                "name": getattr(r, 'ClientName', None) or getattr(r, 'ClientID', None),
                "email": getattr(r, 'Email', None),
                "phone": getattr(r, 'Phone', None),
                "address": ", ".join([x for x in [getattr(r, 'AddressLine1', None), getattr(r, 'AddressLine2', None), getattr(r, 'City', None), getattr(r, 'Province', None), getattr(r, 'PostalCode', None), getattr(r, 'Country', None)] if x]),
                "notes": getattr(r, 'Notes', None),
            }
            for r in rows
        ]

    try:
        return await _run_blocking(_fetch_clients)
    except Exception as e:
        logger.warning(f"[API] /api/clients SQL fallback due to: {e}")
        return []  # graceful fallback


@app.get("/api/documents")
async def list_documents(client_id: str | None = None, tax_year: int | None = None):
    """
    List documents from Gold dimDocument. Optional filters: client_id, tax_year.
    """
    def _fetch_documents():
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        base = "SELECT DocID, ClientID, ClientName, TaxYear, Category, Status, Confidence, LLMUsed, BronzeBlob, SilverJsonBlob, SilverPdfBlob FROM gold.dimDocument"
        where = []
        params = []
        if client_id:
            where.append("ClientID = ?")
            params.append(client_id)
        if tax_year:
            where.append("TaxYear = ?")
            params.append(int(tax_year))
        if where:
            base += " WHERE " + " AND ".join(where)
        base += " ORDER BY CreatedAt DESC"
        cur.execute(base, params)
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "doc_id": getattr(r, 'DocID', None),
                "client_id": getattr(r, 'ClientID', None),
                "client_name": getattr(r, 'ClientName', None),
                "tax_year": getattr(r, 'TaxYear', None),
                "category": getattr(r, 'Category', None),
                "status": getattr(r, 'Status', None),
                "confidence": float(getattr(r, 'Confidence', 0) or 0),
                "llmUsed": bool(getattr(r, 'LLMUsed', 0)),
                "bronze_pdf_blob": getattr(r, 'BronzeBlob', None),
                "silver_json_blob": getattr(r, 'SilverJsonBlob', None),
                "silver_pdf_blob": getattr(r, 'SilverPdfBlob', None),
            }
            for r in rows
        ]

    try:
        return await _run_blocking(_fetch_documents)
    except Exception as e:
        logger.warning(f"/api/documents DB failed: {e}")
        return []


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document metadata (Gold dimDocument) + Silver json."""
    try:
        logger.info(f"[API] Fetching document from DB: doc_id={_redact(doc_id)}")
        meta = await _run_blocking(_load_meta_from_db, doc_id)
        if not meta:
            logger.info(f"[API] Document not found in DB: doc_id={_redact(doc_id)}")
            raise KeyError("not found")
    except Exception:
        logger.info(f"[API] Falling back to local meta for doc_id={_redact(doc_id)}")
        meta = await _run_blocking(_load_meta, doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    if meta and "llm_used" in meta and "llmUsed" not in meta:
        meta["llmUsed"] = bool(meta.get("llm_used"))
    logger.info(f"[API] Loading silver for doc_id={doc_id}")
    silver = await _run_blocking(_load_silver, doc_id)
    if silver is None:
        raise HTTPException(status_code=404, detail=f"Silver JSON for {doc_id} not found")
    return {"document": meta, "silver": silver}


@app.get("/api/documents/{doc_id}/pdf")
async def get_document_pdf(doc_id: str):
    """
    Stream the original PDF stored in Bronze.
    """
    def _download_pdf():
        meta = _load_meta(doc_id)
        if HAVE_BLOB and meta and meta.get("bronze_pdf_blob"):
            try:
                container, blob = meta["bronze_pdf_blob"].split(":", 1)
                conn = os.getenv("AzureWebJobsStorage") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if not conn:
                    raise HTTPException(status_code=500, detail="Storage connection not configured")
                bs = BlobServiceClient.from_connection_string(conn)
                bc = bs.get_blob_client(container, blob)
                if not bc.exists():
                    raise HTTPException(status_code=404, detail=f"PDF blob for {doc_id} not found")
                stream = bc.download_blob().readall()
                tmp = Path(f"/tmp/{doc_id}.pdf")
                tmp.write_bytes(stream)
                return tmp
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Blob retrieval failed: {e}")
        raise HTTPException(status_code=404, detail=f"PDF for {doc_id} not found")

    tmp = await _run_blocking(_download_pdf)
    # Use inline disposition so browsers embed instead of forcing download
    return FileResponse(
        tmp,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc_id}.pdf"'},
    )


@app.get("/api/documents/{doc_id}/silver")
async def get_document_silver(doc_id: str):
    """
    Return raw Silver JSON.
    """
    silver = await _run_blocking(_load_silver, doc_id)
    if silver is None:
        raise HTTPException(status_code=404, detail=f"Silver JSON for {doc_id} not found")
    return silver


@app.get("/api/documents/{doc_id}/silver/csv")
async def get_document_silver_csv(doc_id: str):
    """
    Return a page-wise CSV generated from Silver JSON.

    One row per page, with extracted_fields flattened.
    """
    silver = await _run_blocking(_load_silver, doc_id)
    if silver is None:
        raise HTTPException(status_code=404, detail=f"Silver JSON for {doc_id} not found")

    pages = silver.get("pages") or []
    output = io.StringIO()
    writer = None

    for p in pages:
        row: Dict[str, Any] = {
            "DocId": doc_id,
            "Page": p.get("page_number"),
            "Category": p.get("category"),
            "Confidence": p.get("confidence"),
        }
        fields = p.get("extracted_fields") or {}
        row.update(fields)

        if writer is None:
            writer = csv.DictWriter(output, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)

    csv_bytes = output.getvalue()

    return Response(
        csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="silver-{doc_id}.csv"'
        },
    )


@app.get("/api/documents/other")
async def list_other_documents():
    """Return only documents that are low-confidence or category matches 'other'."""
    metas = await _run_blocking(_load_all_meta)
    filtered = []
    for m in metas:
        category = (m.get("category") or "").lower()
        status = (m.get("status") or "").lower()
        confidence = float(m.get("confidence") or 0)
        is_other = "other" in category or status != "valid" or confidence < 0.6
        if is_other:
            filtered.append(m)
    return filtered


@app.get("/api/dashboard")
async def dashboard_summary():
    """Aggregate counts from Gold dimDocument and dimClient."""
    def _dashboard_from_db():
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute("SELECT COUNT(*) FROM gold.dimClient")
        total_clients = int(list(cur.fetchone())[0])
        cur.execute("SELECT COUNT(*) FROM gold.dimDocument")
        total_documents = int(list(cur.fetchone())[0])
        cur.execute("SELECT COUNT(*) FROM gold.dimDocument WHERE Status <> 'valid'")
        needs_review = int(list(cur.fetchone())[0])
        cur.execute("SELECT COUNT(*) FROM gold.dimDocument WHERE LOWER(Category) LIKE 'other%'")
        other_documents = int(list(cur.fetchone())[0])
        cur.execute("SELECT TOP 15 DocID, ClientID, ClientName, Category, TaxYear, Confidence, Status FROM gold.dimDocument ORDER BY UpdatedAt DESC")
        rows = cur.fetchall()
        conn.close()
        recent = [
            {
                "docId": getattr(r, 'DocID', None),
                "clientId": getattr(r, 'ClientID', None),
                "clientName": getattr(r, 'ClientName', None),
                "category": getattr(r, 'Category', None),
                "taxYear": getattr(r, 'TaxYear', None),
                "confidence": float(getattr(r, 'Confidence', 0) or 0),
                "status": getattr(r, 'Status', None),
            }
            for r in rows
        ]
        return {
            "totalClients": total_clients,
            "totalDocuments": total_documents,
            "needsReview": needs_review,
            "otherDocuments": other_documents,
            "recent": recent,
        }

    try:
        return await _run_blocking(_dashboard_from_db)
    except Exception as e:
        logger.warning(f"[API] /api/dashboard Gold fallback due to: {e}")
        metas = await _run_blocking(_load_all_meta)
        total_documents = len(metas)
        needs_review = sum(1 for m in metas if (m.get("status") or "") != "valid")
        other_documents = sum(1 for m in metas if "other" in (m.get("category") or "").lower())
        recent_sorted = sorted(metas, key=lambda m: m.get("confidence", 0), reverse=True)[:15]
        recent = [
            {
                "docId": m.get("doc_id"),
                "clientId": m.get("client_id"),
                "clientName": m.get("client_name"),
                "category": m.get("category"),
                "taxYear": m.get("tax_year"),
                "confidence": m.get("confidence"),
                "status": m.get("status"),
            }
            for m in recent_sorted
        ]
        return {
            "totalClients": len({m.get("client_id") for m in metas if m.get("client_id")}),
            "totalDocuments": total_documents,
            "needsReview": needs_review,
            "otherDocuments": other_documents,
            "recent": recent,
        }


async def _patch_silver_page(doc_id: str, page_number: int, payload: Dict[str, Any]):
    with _edit_lock:
        silver = _load_silver(doc_id)
        if silver is None:
            raise HTTPException(status_code=404, detail=f"Silver JSON for {doc_id} not found")
        pages = silver.get("pages") or []
        target = None
        for p in pages:
            pn = p.get("page_number") or p.get("pageNumber")
            if int(pn) == int(page_number):
                target = p
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Page {page_number} not found in silver")
        new_fields = payload.get("extracted_fields")
        if isinstance(new_fields, dict):
            target["extracted_fields"] = new_fields
        if "category" in payload:
            target["category"] = payload.get("category") or target.get("category")
        if "confidence" in payload:
            try:
                target["confidence"] = float(payload.get("confidence"))
            except Exception:
                pass
        meta = _load_meta(doc_id) or {}
        if HAVE_BLOB and meta.get("silver_json_blob"):
            try:
                await upload_silver_json(meta["silver_json_blob"], silver)
            except Exception as e:
                logger.warning(f"[API] Failed to update silver blob for {doc_id}: {e}")
        if HAVE_GOLD_ETL:
            try:
                await _run_blocking(upsert_from_silver, meta, silver)
            except Exception as e:
                logger.warning(f"[API] Gold ETL refresh failed for {doc_id}: {e}")
    return {"page_number": page_number, "page": target}


async def _patch_document_status(doc_id: str, payload: Dict[str, Any]):
    new_status = payload.get("status")
    new_category = payload.get("category")
    with _edit_lock:
        meta = _load_meta(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Meta for {doc_id} not found")
        if new_status:
            meta["status"] = new_status
        if new_category:
            meta["category"] = new_category
        if not meta.get("client_name"):
            meta["client_name"] = _resolve_client_name(meta.get("client_id"))
        silver = _load_silver(doc_id)
        if HAVE_BLOB and meta.get("silver_json_blob") and silver:
            try:
                await upload_silver_json(meta["silver_json_blob"], silver)
            except Exception as e:
                logger.warning(f"[API] Failed to update silver blob during status patch for {doc_id}: {e}")
        if HAVE_GOLD_ETL:
            try:
                await _run_blocking(upsert_from_silver, meta, silver)
            except Exception as e:
                logger.warning(f"[API] Gold ETL refresh failed for {doc_id} (status patch): {e}")
    return meta


VALID_CATEGORIES = {
    "Slips", "TSlips", "Medical expenses", "Charitable donations", "Political donations", "Child care expenses", "RRSP contribution", "Union and Professional Dues", "Property Tax receipt", "Rent receipt", "Other documents"
}


async def _patch_document_category(doc_id: str, payload: Dict[str, Any]):
    new_category = payload.get("category")
    if not new_category or new_category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    meta = _load_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    old_category = meta.get("category")
    allow_any = os.getenv("ALLOW_ANY_CATEGORY_CHANGE", "false").lower() == "true"
    if (old_category or "").lower() != "other documents" and not allow_any:
        raise HTTPException(status_code=409, detail="Category override allowed only when current category is 'Other documents'")
    silver = _load_silver(doc_id)
    if silver:
        silver["category"] = new_category
        pages = silver.get("pages") or []
        for p in pages:
            p_cat = (p.get("category") or "").lower()
            if not p_cat or p_cat == (old_category or "").lower():
                p["category"] = new_category

    if HAVE_BLOB and meta.get("silver_json_blob"):
        try:
            await move_silver_category(doc_id, old_category, new_category, meta.get("client_id"), meta.get("tax_year"))
        except Exception as e:
            logger.warning(f"Blob category move failed: {e}")
    meta["category"] = new_category
    if HAVE_BLOB and meta.get("silver_json_blob") and silver:
        try:
            await upload_silver_json(meta["silver_json_blob"], silver)
        except Exception as e:
            logger.warning(f"[API] Failed to update silver blob during category patch for {doc_id}: {e}")
    if HAVE_GOLD_ETL:
        try:
            await _run_blocking(upsert_from_silver, meta, silver)
        except Exception as e:
            logger.warning(f"[API] Gold ETL refresh failed for {doc_id} (category patch): {e}")
    return {"doc_id": doc_id, "category": new_category}


def _create_client_sync(payload: Dict[str, Any]):
    client_id = payload.get("clientId") or payload.get("ClientID")
    client_name = payload.get("clientName") or payload.get("ClientName")
    if not client_name:
        raise HTTPException(status_code=400, detail="clientName is required")
    if not client_id:
        client_id = f"C-{uuid.uuid4().hex[:8]}"
    try:
        logger.info(f"[API] Upserting client into DB: clientId={client_id}")
        conn = get_sql_conn()
        cur = conn.cursor()
        _apply_cursor_timeout(cur)
        cur.execute(
            f"""
            MERGE gold.dimClient AS target
            USING (SELECT ? AS ClientID) AS src
            ON target.ClientID = src.ClientID
            WHEN MATCHED THEN UPDATE SET
                ClientName = ?, Email = ?, Phone = ?, AddressLine1 = ?, AddressLine2 = ?, City = ?, Province = ?, PostalCode = ?, Country = ?, Notes = ?, UpdatedAt = SYSUTCDATETIME()
            WHEN NOT MATCHED THEN INSERT
                (ClientID, ClientName, Email, Phone, AddressLine1, AddressLine2, City, Province, PostalCode, Country, Notes, CreatedAt, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, SYSUTCDATETIME(), SYSUTCDATETIME());
            """,
            (
                client_id,
                client_name,
                payload.get("email"), payload.get("phone"), payload.get("addressLine1"), payload.get("addressLine2"),
                payload.get("city"), payload.get("province"), payload.get("postalCode"), payload.get("country"), payload.get("notes"),
                client_id, client_name, payload.get("email"), payload.get("phone"), payload.get("addressLine1"), payload.get("addressLine2"),
                payload.get("city"), payload.get("province"), payload.get("postalCode"), payload.get("country"), payload.get("notes"),
            )
        )
        conn.commit()
        conn.close()
        logger.info(f"[API] Upserted client successfully: clientId={client_id}")
        return {"clientId": client_id, "clientName": client_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[API] /api/clients create failed")
        raise HTTPException(status_code=500, detail=f"Create client failed: {e}")


@app.patch("/api/documents/{doc_id}/silver/page/{page_number}")
async def patch_silver_page(doc_id: str, page_number: int, payload: Dict[str, Any] = Body(default_factory=dict)):
    """Update extracted_fields (and optionally page category/confidence) for a single page in Silver JSON."""
    return await _patch_silver_page(doc_id, page_number, payload or {})


@app.patch("/api/documents/{doc_id}/status")
async def patch_document_status(doc_id: str, payload: Dict[str, Any] = Body(default_factory=dict)):
    """Update meta status or category after human review."""
    return await _patch_document_status(doc_id, payload or {})


@app.patch("/api/documents/{doc_id}")
async def patch_document_category(doc_id: str, payload: Dict[str, Any] = Body(default_factory=dict)):
    """Update overall document category in meta (and propagate to silver root),
    without changing per-page categories.

    Body: { "category": "<selected>" }
    Rule: Only allow when current meta.category is "Other documents".
    """
    return await _patch_document_category(doc_id, payload or {})


@app.post("/api/clients", status_code=201)
async def create_client(payload: Dict[str, Any] = Body(default_factory=dict)):
    """Create or update a client in gold.dimClient."""
    return await _run_blocking(_create_client_sync, payload or {})


@app.get("/api/uploads/{job_id}")
async def get_upload_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/uploads")
async def list_upload_jobs():
    with _job_lock:
        return list(_jobs.values())


@app.get("/api/readiness")
async def readiness():
    """Active dependency checks."""
    def _check():
        checks: Dict[str, Any] = {}
        try:
            if cfg.have_blob and HAVE_AZURE_SDK:
                bs = BlobServiceClient.from_connection_string(cfg.azure_storage_conn)
                _ = list(bs.list_containers(name_starts_with=cfg.bronze_container))
                checks["blob"] = "ok"
            else:
                checks["blob"] = "disabled"
        except Exception as e:
            checks["blob"] = f"error:{e}"
        try:
            conn = get_sql_conn()
            conn.close()
            checks["sql"] = "ok"
        except Exception as e:
            checks["sql"] = f"error:{e}"
        checks["llm"] = "ok" if os.getenv("HAVE_LLM", "true").lower() == "true" else "disabled"
        checks["env"] = cfg.environment
        status = "ok" if all(v == "ok" or v == "disabled" for v in checks.values()) else "degraded"
        return {"status": status, "checks": checks}

    return await _run_blocking(_check)


@app.get("/api/metrics")
async def metrics():
    lines = [
        f"bookkeeper_documents_total {_metrics['documents_total']}",
        f"bookkeeper_documents_failed {_metrics['documents_failed']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


# Graceful shutdown
_shutdown = {"requested": False}


def _handle_shutdown(signum, frame):
    if _shutdown["requested"]:
        return
    _shutdown["requested"] = True
    logger.info("Shutdown signal received", extra={"signum": signum})
    time.sleep(2)
    os._exit(0)


for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, _handle_shutdown)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
