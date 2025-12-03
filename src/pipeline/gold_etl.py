from __future__ import annotations

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

from src.pipeline.db import get_sql_conn
from src.pipeline.azure_storage import download_silver_json

logger = logging.getLogger(__name__)

SCHEMA = 'gold'

def tq(name: str) -> str:
    return f"{SCHEMA}.{name}"

def _sha256_bytes(data: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(data)
    return h.digest()

def upsert_from_silver(meta: Dict[str, Any], silver_doc: Optional[Dict[str, Any]] = None) -> None:
    """
    Persist document and pages into Gold per schema.
    - Upsert gold.dimDocument
    - Upsert gold.DocumentPage rows
    - Insert generic fields into gold.factOtherDocuments
    """
    doc_id = meta.get('doc_id')
    if not doc_id:
        raise ValueError('meta.doc_id is required')

    client_id = meta.get('client_id')
    client_name = meta.get('client_name')
    tax_year = meta.get('tax_year')
    category = meta.get('category')
    status = meta.get('status')
    confidence = float(meta.get('confidence') or 0)
    llm_used = bool(meta.get('llm_used'))
    bronze_blob = meta.get('bronze_pdf_blob')
    silver_json_blob = meta.get('silver_json_blob')
    silver_pdf_blob = meta.get('silver_pdf_blob')

    silver: Optional[Dict[str, Any]] = silver_doc
    if silver is None:
        if not silver_json_blob:
            logger.warning('No silver_json_blob in meta and no in-memory silver provided; skipping Gold ETL')
            return
        logger.info(f"[GOLD ETL] Loading silver JSON for doc_id={doc_id} from blob={silver_json_blob}")
        silver = download_silver_json(silver_json_blob)
    raw_json_bytes = json.dumps(silver or {}, separators=(',', ':')).encode('utf-8')
    raw_hash = _sha256_bytes(raw_json_bytes)

    pages: List[Dict[str, Any]] = (silver or {}).get('pages') or []
    logger.info(f"[GOLD ETL] Silver payload pages={len(pages)} category={category} client_id={client_id} tax_year={tax_year}")

    conn = get_sql_conn()
    try:
        with conn.cursor() as cur:
            # Upsert dimDocument via MERGE
            logger.info(f"[GOLD ETL] Upserting dimDocument DocID={doc_id} ClientID={client_id} Category={category} Status={status}")
            cur.execute(
                f"""
                MERGE {tq('dimDocument')} AS target
                USING (SELECT CAST(? AS UNIQUEIDENTIFIER) AS DocID) AS src
                ON target.DocID = src.DocID
                WHEN MATCHED THEN UPDATE SET
                    ClientID = ?, ClientName = ?, TaxYear = ?, Category = ?, Status = ?, Confidence = ?, LLMUsed = ?,
                    BronzeBlob = ?, SilverJsonBlob = ?, SilverPdfBlob = ?, RawJsonHash = ?, UpdatedAt = SYSUTCDATETIME()
                WHEN NOT MATCHED THEN INSERT
                    (DocID, ClientID, ClientName, TaxYear, Category, Status, Confidence, LLMUsed,
                     BronzeBlob, SilverJsonBlob, SilverPdfBlob, RawJsonHash, CreatedAt, UpdatedAt)
                VALUES
                    (CAST(? AS UNIQUEIDENTIFIER), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, SYSUTCDATETIME(), SYSUTCDATETIME());
                """,
                (doc_id, client_id, client_name, tax_year, category, status, confidence, llm_used,
                 bronze_blob, silver_json_blob, silver_pdf_blob, raw_hash,
                 doc_id, client_id, client_name, tax_year, category, status, confidence, llm_used,
                 bronze_blob, silver_json_blob, silver_pdf_blob, raw_hash)
            )
            logger.info(f"[GOLD ETL] Upserted dimDocument DocID={doc_id}")

            # Upsert pages
            for p in pages:
                page_number = int(p.get('page_number') or p.get('pageNumber') or 0)
                p_cat = p.get('category')
                p_conf = float(p.get('confidence') or 0)
                p_llm = bool(p.get('llm_used'))
                p_status = p.get('status')
                p_json = json.dumps(p.get('extracted_fields') or p.get('extractedFields') or {}, ensure_ascii=False)

                logger.debug(f"[GOLD ETL] Upserting DocumentPage DocID={doc_id} PageNumber={page_number} Category={p_cat} Confidence={p_conf}")
                cur.execute(
                    f"""
                    MERGE {tq('DocumentPage')} AS target
                    USING (SELECT CAST(? AS UNIQUEIDENTIFIER) AS DocID, CAST(? AS INT) AS PageNumber) AS src
                    ON target.DocID = src.DocID AND target.PageNumber = src.PageNumber
                    WHEN MATCHED THEN UPDATE SET
                        Category = ?, Confidence = ?, LLMUsed = ?, Status = ?, ExtractedJson = ?, UpdatedAt = SYSUTCDATETIME()
                    WHEN NOT MATCHED THEN INSERT
                        (DocID, PageNumber, Category, Confidence, LLMUsed, Status, ExtractedJson, UpdatedAt)
                    VALUES (CAST(? AS UNIQUEIDENTIFIER), ?, ?, ?, ?, ?, ?, SYSUTCDATETIME());
                    """,
                    (doc_id, page_number, p_cat, p_conf, p_llm, p_status, p_json,
                     doc_id, page_number, p_cat, p_conf, p_llm, p_status, p_json)
                )

                # Generic fact insert for Other Documents
                # Flatten key-values if overall category is Other documents
                if (category or '').lower().startswith('other'):
                    kv_count = 0
                    for k, v in (p.get('extracted_fields') or {}).items():
                        cur.execute(
                            f"""
                            INSERT INTO {tq('factOtherDocuments')} (DocID, PageNumber, FieldKey, FieldValue)
                            VALUES (CAST(? AS UNIQUEIDENTIFIER), ?, ?, ?);
                            """,
                            (doc_id, page_number, str(k), str(v) if v is not None else None)
                        )
                        kv_count += 1
                    logger.debug(f"[GOLD ETL] Inserted {kv_count} OtherDocuments facts for DocID={doc_id} PageNumber={page_number}")
        conn.commit()
        logger.info(f"[GOLD ETL] Committed Gold ETL for DocID={doc_id} pages={len(pages)}")
    except Exception:
        conn.rollback()
        logger.exception('[GOLD ETL] Failed to upsert gold records')
        raise
    finally:
        conn.close()
        logger.info(f"[GOLD ETL] Connection closed for DocID={doc_id}")
