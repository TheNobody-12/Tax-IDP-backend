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
from src.extraction.base import DocumentContext, PROCESSOR_REGISTRY
from src.extraction.classifier import DocumentClassifier
import src.extraction.processors  # Trigger registration
from src.extraction.clients import get_llm_client

logger = logging.getLogger(__name__)

# Initialize components
classifier = DocumentClassifier()
HAVE_LLM = True 
cfg = get_config()

# Attempt to import real DI + silver modules; fallback to placeholders if absent.
try:
    from src.pipeline.di_extraction import extract_with_azure_di  # type: ignore
    HAVE_DI = True
    logger.info("DI client import succeeded")
except Exception:  # pragma: no cover
    HAVE_DI = False
    from src.pipeline.di_extraction import extract_with_azure_di  # type: ignore
    HAVE_DI = True
    logger.info("DI client import succeeded")
except Exception:  # pragma: no cover
    HAVE_DI = False
    async def extract_with_azure_di(local_file_path: str, file_url: str | None = None):  # type: ignore
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

# Legacy LLM helpers removed.
# The new system uses specialized processors for each document type.

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
    from src.pipeline.azure_storage import upload_bronze_pdf, upload_silver_artifacts, upload_bronze_di, generate_blob_sas_url
    HAVE_BLOB = True
except Exception:
    HAVE_BLOB = False
    async def upload_bronze_pdf(doc_id: str, local_pdf: str): return None
    async def upload_silver_artifacts(doc_id: str, silver_doc: Dict[str, Any], local_pdf: str, client_id: str | None, tax_year: int | None): return {}
    async def upload_bronze_di(doc_id: str, di_doc: Dict[str, Any]): return None
    async def generate_blob_sas_url(container: str, blob_name: str, expiry_minutes: int = 60): return None

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
    llm_provider: str = "azure",
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

    bronze_blob_sas = None
    if HAVE_BLOB and bronze_blob_ref:
        try:
             # bronze_blob_ref is convention "container:path" or just "container:path" returned by upload?
             # upload_bronze_pdf returns f"{_BRONZE_CONTAINER}:{blob_path}"
             # We need just the blob path.
             blob_name = f"{doc_id}/document.pdf"
             bronze_container = cfg.bronze_container # Or from azure_storage const if exposed, but config is safer
             bronze_blob_sas = await generate_blob_sas_url(bronze_container, blob_name)
             logger.info(f"Generated SAS URL for DI processing: {bronze_blob_sas.split('?')[0]}?...")
        except Exception as e:
            logger.warning(f"Failed to generate SAS URL: {e}")

    di_attempts = 2
    for attempt in range(1, di_attempts + 1):
        try:
            # Updated: extract_with_azure_di now accepts file_url
            # If we have a SAS URL, use it. Otherwise fall back to local file.
            page_texts, tables, page_images_base64, di_raw, markdown_content = await asyncio.wait_for(
                extract_with_azure_di(str(src_pdf), file_url=bronze_blob_sas),
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
        output_dir = str(Path(AZURE_BRONZE_CONTAINER) / doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 0. Initialize LLM Client
        ai_client = get_llm_client(provider=llm_provider)
        logger.info(f"Using LLM provider: {llm_provider} for document {doc_id}")

        # 1. Classify Directly
        classification = await classifier.classify(markdown_content, ai_client=ai_client)
        # classification.category is now a string (dynamic)
        logger.info(f"Document {doc_id} classified as {classification.category} ({classification.confidence:.2f})")
        
        # 2. Select Processor
        processor_name = classification.processor_name
        processor = PROCESSOR_REGISTRY.get(processor_name) or PROCESSOR_REGISTRY.get("slips")
        
        # 3. Create context for medallion architecture
        context = DocumentContext(
            doc_id=doc_id,
            pdf_path=str(src_pdf),
            markdown_path=None,
            markdown_content=markdown_content,
            total_pages=len(page_texts),
            metadata={
                "classification": {
                    "category": classification.category,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                },
                "client_id": client_id,
                "tax_year": tax_year
            }
        )
        
        # 4. Extract (Silver Generation)
        logger.info(f"Running processor {processor.name} for {doc_id}")
        extraction_result = await processor.process(context, output_dir, ai_client=ai_client)
        
        # 5. Format Silver Doc
        silver_pages = []
        for i, txt in enumerate(page_texts):
            # Extract items belonging to this page
            current_page = i + 1
            page_items = []
            for item in extraction_result.data:
                p_nums = item.get("page_numbers")
                if not p_nums:
                    if i == 0: 
                        page_items.append(item)
                    continue
                
                # Safe check
                if isinstance(p_nums, list):
                    if any(int(p) == current_page for p in p_nums if str(p).isdigit()):
                        page_items.append(item)
                elif int(p_nums) == current_page: # Fallback if single value
                     page_items.append(item)
            
            # Map to legacy silver page format for downstream compatibility
            silver_pages.append({
                "page_number": i + 1,
                "status": "ok",
                "category": classification.category,
                "confidence": classification.confidence,
                "extracted_fields": page_items[0] if page_items else {},
                "extracted_data": page_items,
                "error": None,
                "llm_used": True
            })

    # Calculate aggregate extraction confidence from results if available
    # If no confidence in items, fallback to classification confidence
    all_item_confidences = []
    for item in extraction_result.data:
        c = item.get("confidence")
        if c is not None:
            try:
                all_item_confidences.append(float(c))
            except (ValueError, TypeError):
                pass
    
    avg_extraction_conf = (sum(all_item_confidences) / len(all_item_confidences)) if all_item_confidences else classification.confidence

    silver_doc = {
        "doc_id": doc_id,
        "client_id": client_id,
        "category": classification.category,
        "tax_year": tax_year,
        "confidence": avg_extraction_conf,
        "classification_confidence": classification.confidence,
        "pages": silver_pages,
        "extraction_metadata": {
            "processor": extraction_result.processor_name,
            "items_count": extraction_result.items_extracted,
            "errors": extraction_result.errors
        },
        "llm_used": True,
    }
    
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
        "status": validation.get("status", "needs_review"),
        "confidence": avg_extraction_conf,
        "classification_confidence": classification.confidence,
        "validation_errors": validation.get("errors", []),
        "validation_warnings": validation.get("warnings", []),
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

    result = {
        "meta": meta,
        "silver": silver_doc
    }

    if progress_cb:
        progress_cb({"status": "completed", "message": "Completed", "pages_done": len(page_texts), "pages_total": len(page_texts)})

    logger.info("[PIPELINE] Complete", extra={"doc_id": doc_id})
    return result
