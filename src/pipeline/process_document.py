"""
process_document.py (no-SQL simplified)

Pipeline steps:
  1. Copy uploaded PDF into data/bronze/<doc_id>/document.pdf
  2. Run Azure DI (optional) to get page texts (async wrapper)
  3. Write DI JSON => data/bronze/<doc_id>/di.json (minimal schema with pages)
  4. Build Silver JSON (page-wise) using silver_structured.build_silver_document
  5. Validate (if validator available) else mark valid
  6. Optionally build in-memory gold summary (no persistence)
  7. Write meta JSON => data/meta/<doc_id>.meta.json for listing endpoint

Exports: run(input_pdf, client_id?, tax_year?, generate_gold=False)
Returns dict with keys: doc_id, meta, silver_path, bronze_pdf_path, validation, gold_summary?
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from src.config.app_config import get_config
from src.config.log_context import set_log_context
import time

from src.pipeline.db import get_sql_conn

logger = logging.getLogger(__name__)

# Attempt to import real DI + silver modules; fallback to placeholders if absent.
try:
    from src.pipeline.di_extraction import extract_with_azure_di  # type: ignore
    HAVE_DI = True
    logger.info("DI client import succeeded")
except Exception:  # pragma: no cover
    HAVE_DI = False
    async def extract_with_azure_di(local_file_path: str):  # type: ignore
        raise RuntimeError("Azure DI client unavailable (import failed)")
    logger.exception("DI client import failed; DI will be unavailable.")

try:
    from src.pipeline.silver_structured import build_silver_document  # type: ignore
except Exception:  # pragma: no cover
    def build_silver_document(doc_id: str, bronze_dir: str | Path, output_dir: str | Path):  # type: ignore
        silver = {
            "doc_id": doc_id,
            "category": "Unknown",
            "tax_year": None,
            "pages": [
                {
                    "page_number": 1,
                    "status": "ok",
                    "category": "Unknown",
                    "confidence": 0.5,
                    "extracted_fields": {},
                    "error": None,
                }
            ],
        }
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{doc_id}.silver.json").write_text(json.dumps(silver, indent=2), encoding="utf-8")
        return silver

try:
    from src.pipeline.gold_writer import write_gold_from_silver  # type: ignore
except Exception:  # pragma: no cover
    def write_gold_from_silver(silver_doc: Dict[str, Any], validation: Dict[str, Any]):  # type: ignore
        return {
            "doc_id": silver_doc.get("doc_id"),
            "category": silver_doc.get("category"),
            "pages": [],
            "total_amount": 0.0,
            "validation": validation,
        }

try:
    from src.pipeline.silver_validator import validate_silver  # type: ignore
except Exception:  # pragma: no cover
    def validate_silver(silver_doc: Dict[str, Any]):  # type: ignore
        return {"is_valid": True, "errors": [], "warnings": []}

# NEW: optional LLM page processor (DI + OCR fusion)
try:
    from src.pipeline.llm_processing import process_page_with_llm  # type: ignore
    HAVE_LLM = True
except Exception as e:
    HAVE_LLM = False
    logger.exception("LLM processing unavailable; falling back to heuristic silver builder. Import failed", exc_info=e)

# NEW: configurable LLM concurrency
cfg = get_config()
LLM_MAX_CONCURRENCY = cfg.llm_max_concurrency

# -------------------- Helper functions for LLM silver --------------------
async def _run_llm_for_pages(
    page_texts: List[str],
    page_images_base64: Optional[List[Optional[str]]],
    tables: Optional[Any],
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Call LLM concurrently for each page (bounded + per-page timeout)."""
    sem = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

    async def _one(i: int) -> Dict[str, Any]:
        async with sem:
            try:
                return await asyncio.wait_for(
                    process_page_with_llm(
                        page_text=page_texts[i],
                        page_image_base64=(page_images_base64[i] if page_images_base64 and i < len(page_images_base64) else None),
                        page_number=i + 1,
                        tables=tables,
                    ),
                    timeout=cfg.llm_page_timeout,
                )
            except asyncio.TimeoutError:
                return {"page_number": i + 1, "status": "timeout", "category": None, "confidence": 0.0, "extracted_fields": {}, "error": "LLM timeout", "llm_used": True}
            finally:
                if progress_cb:
                    progress_cb({"pages_done": i + 1})
    tasks = [asyncio.create_task(_one(i)) for i in range(len(page_texts))]
    return await asyncio.gather(*tasks)

def _majority_category_and_conf(page_results: List[Dict[str, Any]]) -> tuple[str, float]:
    cats: List[str] = []
    confs: List[float] = []
    for pr in page_results:
        if pr.get("status") == "ok" and pr.get("category"):
            cats.append(str(pr["category"]))
            try:
                confs.append(float(pr.get("confidence") or 0.0))
            except Exception:
                confs.append(0.0)
    if not cats:
        return "Other documents", 0.0
    # majority vote; tie => first
    majority = max(set(cats), key=cats.count)
    max_conf = max(confs) if confs else 0.0
    return majority, max_conf

PRESCRIPTION_ID_PATTERNS = [r"\bN\d{5,6}\b", r"\bRX[:#]?\s?(\w+)", r"\b\d{6,7}\b"]

async def _build_silver_with_llm(
    doc_id: str,
    page_texts: List[str],
    page_images_base64: Optional[List[Optional[str]]],
    tables: Optional[Any],
    tax_year: Optional[int],
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    # Only per-page LLM calls (batch removed)
    page_results = await _run_llm_for_pages(page_texts, page_images_base64, tables, progress_cb=progress_cb)

    overall_category, overall_conf = _majority_category_and_conf(page_results)

    silver_pages: List[Dict[str, Any]] = []
    for idx, pr in enumerate(page_results):
        fields = pr.get("extracted_fields") or {}
        cat = (pr.get("category") or "").lower()
        # Inject PrescriptionId / IsPrescription for medical pages if missing
        if "medical" in cat and "PrescriptionId" not in fields:
            text = page_texts[idx] if idx < len(page_texts) else ""
            for pattern in PRESCRIPTION_ID_PATTERNS:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    pid = m.group(1) if m.lastindex else m.group(0)
                    fields["PrescriptionId"] = pid
                    fields.setdefault("IsPrescription", True)
                    break
        silver_pages.append(
            {
                "page_number": pr.get("page_number"),
                "status": pr.get("status"),
                "category": pr.get("category") or overall_category,
                "confidence": pr.get("confidence", 0.0),
                "extracted_fields": fields,
                "error": pr.get("error"),
                "llm_used": pr.get("llm_used", True),
            }
        )

    silver_doc: Dict[str, Any] = {
        "doc_id": doc_id,
        "category": overall_category,
        "tax_year": tax_year,
        "confidence": overall_conf,
        "pages": silver_pages,
        "llm_used": True,
    }

    return silver_doc

# -------------------- Existing bronze helpers --------------------

# Remove local filesystem containers (use only blob)
AZURE_SILVER_CONTAINER = cfg.silver_container
AZURE_BRONZE_CONTAINER = cfg.bronze_container
SQL_COMMAND_TIMEOUT = int(os.getenv("SQL_COMMAND_TIMEOUT", "30"))


def _write_bronze_pdf(src_pdf: Path, doc_id: str) -> Path:
    target_dir = AZURE_BRONZE_CONTAINER / doc_id
    # Ensure local folder exists before writing
    target_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = target_dir / f"{doc_id}.bronze.pdf"
    target_pdf.write_bytes(src_pdf.read_bytes())
    return target_pdf


def _write_di_json(doc_id: str, di_raw: Any, page_texts: list[str]) -> Path:
    """Create minimal DI JSON file used by silver builder."""
    di_dir = AZURE_BRONZE_CONTAINER / doc_id
    # Ensure local folder exists before writing
    di_dir.mkdir(parents=True, exist_ok=True)
    di_path = di_dir / f"{doc_id}.di.json"
    pages = []
    for idx, txt in enumerate(page_texts):
        pages.append({"page_number": idx + 1, "content": txt})
    di_doc = {"doc_id": doc_id, "pages": pages, "category": getattr(di_raw, "category", None)}
    di_path.write_text(json.dumps(di_doc, indent=2), encoding="utf-8")
    return di_path


try:
    from src.pipeline.azure_storage import upload_bronze_pdf, upload_silver_artifacts, upload_bronze_di
    HAVE_BLOB = True
except Exception:
    HAVE_BLOB = False
    async def upload_bronze_pdf(doc_id: str, local_pdf: str): return None
    async def upload_silver_artifacts(doc_id: str, silver_doc: Dict[str, Any], local_pdf: str, client_id: str | None, tax_year: int | None): return {}
    async def upload_bronze_di(doc_id: str, di_doc: Dict[str, Any]): return None

try:
    from src.pipeline.gold_etl import upsert_from_silver
    HAVE_GOLD_ETL = True
except Exception:
    HAVE_GOLD_ETL = False

async def run(
    input_pdf: str | Path,
    client_id: str | None = None,
    tax_year: int | None = None,
    generate_gold: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute pipeline (DI + optional LLM) for a single PDF (async)."""
    src_pdf = Path(input_pdf)
    if not src_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {src_pdf}")
    if not HAVE_DI:
        raise RuntimeError("Azure Document Intelligence unavailable; aborting upload")

    doc_id = str(uuid.uuid4())
    set_log_context(doc_id=doc_id)  # propagate correlation id
    logger.info("[PIPELINE] Starting", extra={"doc_id": doc_id})

    bronze_blob_ref = None
    if HAVE_BLOB:
        try:
            bronze_blob_ref = await upload_bronze_pdf(doc_id, str(src_pdf))
        except Exception as e:
            logger.warning(f"Bronze blob upload failed: {e}", extra={"doc_id": doc_id})

    di_attempts = 2
    for attempt in range(1, di_attempts + 1):
        try:
            page_texts, tables, page_images_base64, di_raw = await asyncio.wait_for(
                extract_with_azure_di(str(src_pdf)),
                timeout=cfg.di_timeout,
            )
            break
        except Exception as e:
            logger.warning(f"DI attempt {attempt} failed: {e}", extra={"doc_id": doc_id})
            if attempt == di_attempts:
                raise
            await asyncio.sleep(1.0 * attempt)
    if progress_cb:
        progress_cb({"status": "running", "message": "DI complete", "pages_total": len(page_texts), "pages_done": 0})

    bronze_di_blob = None
    try:
        di_doc = {
            "doc_id": doc_id,
            "pages": [{"page_number": i + 1, "content": txt} for i, txt in enumerate(page_texts)],
            "category": getattr(di_raw, "category", None),
        }
        if HAVE_BLOB:
            bronze_di_blob = await upload_bronze_di(doc_id, di_doc)
    except Exception as e:
        logger.warning(f"DI JSON upload failed: {e}", extra={"doc_id": doc_id})

    if HAVE_LLM:
        completed = {"count": 0}

        def _progress(update: Dict[str, Any]):
            completed["count"] = max(completed["count"], update.get("pages_done", 0))
            if progress_cb:
                progress_cb({"status": "running", "message": "LLM extracting", "pages_done": completed["count"], "pages_total": len(page_texts)})

        silver_doc = await _build_silver_with_llm(doc_id, page_texts, page_images_base64, tables, tax_year, progress_cb=_progress)  # type: ignore
    else:
        raise RuntimeError("LLM processing unavailable")

    validation = validate_silver(silver_doc)

    gold_summary = None
    if generate_gold:
        try:
            gold_summary = write_gold_from_silver(silver_doc, validation)
        except Exception as e:
            logger.warning(f"Gold summary build failed: {e}", extra={"doc_id": doc_id})

    silver_blob_refs = {}
    if HAVE_BLOB:
        try:
            silver_blob_paths = await upload_silver_artifacts(doc_id, silver_doc, str(src_pdf), client_id, tax_year)
            silver_blob_refs = {
                "silver_pdf_blob": f"silver:{silver_blob_paths['pdf']}",
                "silver_json_blob": f"silver:{silver_blob_paths['json']}",
            }
        except Exception as e:
            logger.warning(f"Silver blob upload failed: {e}", extra={"doc_id": doc_id})

    client_name = None
    if client_id:
        try:
            def _resolve():
                conn = get_sql_conn()
                with conn.cursor() as cur:
                    cur.execute("SELECT TOP 1 ClientName FROM gold.dimClient WHERE ClientID = ?", (client_id,))
                    row = cur.fetchone()
                    return (row[0] if row and len(row) > 0 else None) or client_id
            client_name = await asyncio.to_thread(_resolve)
        except Exception as e:
            logger.warning(f"Could not resolve client_name for {client_id}: {e}", extra={"doc_id": doc_id})

    meta = {
        "doc_id": doc_id,
        "client_id": client_id,
        "client_name": client_name,
        "tax_year": tax_year,
        "category": silver_doc.get("category"),
        "status": "valid" if validation.get("is_valid") else "needs_review",
        "confidence": max((p.get("confidence", 0) for p in silver_doc.get("pages", [])), default=0),
        "llmUsed": silver_doc.get("llm_used", HAVE_LLM),
        "llm_used": silver_doc.get("llm_used", HAVE_LLM),
        "bronze_pdf_blob": bronze_blob_ref,
        "bronze_di_blob": bronze_di_blob,
        **silver_blob_refs,
    }

    if HAVE_GOLD_ETL:
        try:
            await asyncio.to_thread(upsert_from_silver, meta, silver_doc)
        except Exception as e:
            logger.warning(f"Gold ETL failed: {e}", extra={"doc_id": doc_id})

    result: Dict[str, Any] = {
        "doc_id": doc_id,
        "meta": meta,
        "silver_path": silver_blob_refs.get("silver_json_blob"),
        "bronze_pdf_path": bronze_blob_ref,
        "validation": validation,
    }
    if gold_summary is not None:
        result["gold_summary"] = gold_summary

    logger.info("[PIPELINE] Complete", extra={"doc_id": doc_id})
    return result
