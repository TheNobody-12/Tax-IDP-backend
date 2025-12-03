"""
silver_structured.py

Build a Silver document from Bronze artifacts:

- Inputs: Bronze directory with PDF, DI JSON, OCR (optional).
- Output: Silver JSON (page-wise) ready for UI and CSV.

Silver format:

{
  "doc_id": "<uuid>",
  "category": "<overall_category>",
  "tax_year": 2024,
  "pages": [
    {
      "page_number": 1,
      "status": "ok" | "error",
      "category": "Medical expenses",
      "confidence": 0.90,
      "extracted_fields": {
        "ExpenseDate": "2024-08-06",
        "PatientName": "Joshua Burnett",
        "IsPrescription": true,
        "PrescriptionId": "N306331",
        "VendorName": "Aurora Compounding Pharmacy",
        "VendorCity": "Aurora, ON",
        "Description": "DHEA 75mg CAPSULE MIXTURE",
        "Amount": 166.96
      },
      "error": null
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
import re
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from azure.storage.blob.aio import BlobClient, ContainerClient

from src.pipeline.llm_processing import process_page_with_llm
from src.pipeline.bronze_store import BronzeArtifacts  # your existing dataclass
from src.pipeline.silver_validator import validate_silver_document, SilverValidationResult
from src.pipeline.classification_rules import rule_based_classification  # your existing rules

logger = logging.getLogger(__name__)

PRESCRIPTION_ID_PATTERNS = [
    r"\bN\d{5,6}\b",        # N306331, N304258...
    r"\b\d{6,7}\b",         # 1644574, 1644576...
    r"RX[:#]?\s?(\w+)",     # RX: 12345, RX#ABC
]


def _extract_prescription_id(text: str | None) -> str | None:
    if not text:
        return None
    for pattern in PRESCRIPTION_ID_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    return None


@dataclass
class SilverDocument:
    doc_id: str
    category: str
    structured: Dict[str, Any]
    blob_path: str
    confidence: float
    validation: SilverValidationResult
    _storage_uri: str = None   # private actual storage URI

    # --------------------------------------------------
    # Expected public properties
    # --------------------------------------------------
    @property
    def client_id(self) -> str:
        return self.structured.get("client_id")

    @property
    def tax_year(self) -> int:
        return self.structured.get("tax_year")

    @property
    def storage_uri(self) -> str:
        """
        Gold writer expects `.storage_uri`.
        It may come from structured payload OR from the pipeline wrapper.
        """
        return self.structured.get("storage_uri") or self._storage_uri

    @storage_uri.setter
    def storage_uri(self, value: str):
        self._storage_uri = value

    @property
    def extracted_fields(self):
        return self.structured.get("extracted_fields", {})

    @property
    def normalized_category(self):
        return (self.category or "").strip().lower()



async def _call_llm_for_all_pages(
    page_texts: List[str],
    page_images_base64: Optional[List[Optional[str]]],
    tables: Optional[Any],
) -> List[Dict[str, Any]]:
    """
    Call the LLM for all pages with limited concurrency.
    """
    sem = asyncio.Semaphore(3)

    async def _one(i: int) -> Dict[str, Any]:
        page_number = i + 1
        image_b64 = None
        if page_images_base64 and i < len(page_images_base64):
            image_b64 = page_images_base64[i]

        async with sem:
            return await process_page_with_llm(
                page_text=page_texts[i],
                page_image_base64=image_b64,
                page_number=page_number,
                tables=tables,
            )

    tasks = [asyncio.create_task(_one(i)) for i in range(len(page_texts))]
    return await asyncio.gather(*tasks)


def _majority_category_and_confidence(
    page_results: List[Dict[str, Any]]
) -> Tuple[str, float]:
    """
    Pick a document-level category and confidence from page-level results.
    """
    cats: List[str] = []
    confs: List[float] = []

    for pr in page_results:
        if pr.get("status") != "ok":
            continue
        cat = pr.get("category")
        if not cat:
            continue
        cats.append(str(cat))
        confs.append(float(pr.get("confidence") or 0.0))

    if not cats:
        return "Other documents", 0.0

    majority_cat, _ = Counter(cats).most_common(1)[0]
    max_conf = max(confs) if confs else 0.0
    return majority_cat, max_conf


def _build_page_results_payload(page_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Slim per-page payload persisted into Silver so we can debug/inspect later.
    """
    out: List[Dict[str, Any]] = []
    for pr in page_results:
        out.append(
            {
                "page_number": pr.get("page_number"),
                "status": pr.get("status"),
                "category": pr.get("category"),
                "confidence": pr.get("confidence"),
                "extracted_fields": pr.get("extracted_fields") or {},
                "error": pr.get("error"),
            }
        )
    return out


def _merge_extracted_fields(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple merged view: later pages override earlier keys.
    """
    merged: Dict[str, Any] = {}
    for pr in page_results:
        if pr.get("status") != "ok":
            continue
        fields = pr.get("extracted_fields") or {}
        if isinstance(fields, dict):
            merged.update(fields)
    return merged


def _stitch_medical_lines(page_results: List[Dict[str, Any]], page_texts: List[str]) -> List[Dict[str, Any]]:
    """
    For multi-page medical PDFs, collect each "Medical expenses" page's fields
    into a simple list. We don't try to be clever about multi-page single receipts:
    each page is one logical line; human review can merge if needed.

    Enhancements:
    - Automatically extracts PrescriptionId via regex from the page text and sets IsPrescription=True when found.
    """
    lines: List[Dict[str, Any]] = []
    for pr in page_results:
        if pr.get("status") != "ok":
            continue
        cat = (pr.get("category") or "").lower()
        if "medical" not in cat:
            continue

        fields = pr.get("extracted_fields") or {}
        if not isinstance(fields, dict):
            continue

        # Page number and text lookup
        page_number = int(pr.get("page_number") or 0)
        page_text = page_texts[page_number - 1] if page_number and page_number - 1 < len(page_texts) else ""

        # Extract prescription id from raw text if not already provided by LLM
        presc_id = fields.get("PrescriptionId") or _extract_prescription_id(page_text)
        if presc_id:
            fields["PrescriptionId"] = presc_id
            fields.setdefault("IsPrescription", True)

        # Copy the page's fields; validator will later clean amounts / dates.
        line = dict(fields)
        line.setdefault("PageNumber", page_number)
        lines.append(line)

    return lines


def _extract_page_fields(page_text: str) -> Dict[str, Any]:
    """
    *** PLACEHOLDER ***

    Your real extraction logic goes here.

    For now, this uses a super simple heuristic to show the shape.
    """
    fields: Dict[str, Any] = {}

    # TODO: Replace with your actual parsing (regexes, DI key-values, etc.)
    # Example placeholders:
    fields["ExpenseDate"] = None
    fields["PatientName"] = None
    fields["PrescriptionId"] = _extract_prescription_id(page_text)
    fields["IsPrescription"] = bool(fields["PrescriptionId"]) if fields["PrescriptionId"] else False
    fields["VendorName"] = None
    fields["VendorCity"] = None
    fields["Description"] = None
    fields["Amount"] = None

    return fields


def build_silver_document(
    doc_id: str,
    bronze_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Build a Silver document for a given doc_id, based on DI JSON in Bronze.

    Parameters
    ----------
    doc_id:
        Document identifier used across the pipeline.
    bronze_dir:
        Path to the Bronze root directory (where DI JSON lives).
    output_dir:
        Path where we should write silver JSON.

    Returns
    -------
    Dict[str, Any]
        The Silver document dict.
    """
    bronze_dir = Path(bronze_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    di_path = bronze_dir / doc_id / "di.json"
    if not di_path.exists():
        raise FileNotFoundError(f"DI JSON not found at {di_path}")

    with di_path.open("r", encoding="utf-8") as f:
        di = json.load(f)

    pages_raw = di.get("pages") or []
    overall_category = di.get("category") or None  # optional

    silver_pages: List[Dict[str, Any]] = []

    for page in pages_raw:
        page_num = page.get("pageNumber") or page.get("page_number")
        # DI's "content" often contains full text; some schemas have `lines` or `blocks`
        page_text = page.get("content") or ""

        try:
            fields = _extract_page_fields(page_text)
            page_entry = {
                "page_number": page_num,
                "status": "ok",
                "category": overall_category or "Unknown",
                "confidence": 0.9,  # placeholder; pull from LLM/DI if you have it
                "extracted_fields": fields,
                "error": None,
            }
        except Exception as e:
            logger.exception("Error extracting page %s fields: %s", page_num, e)
            page_entry = {
                "page_number": page_num,
                "status": "error",
                "category": overall_category or "Unknown",
                "confidence": 0.0,
                "extracted_fields": {},
                "error": str(e),
            }

        silver_pages.append(page_entry)

    silver_doc: Dict[str, Any] = {
        "doc_id": doc_id,
        "category": overall_category,
        "tax_year": None,  # fill from metadata or extraction
        "pages": silver_pages,
    }

    silver_path = output_dir / f"{doc_id}.silver.json"
    with silver_path.open("w", encoding="utf-8") as f:
        json.dump(silver_doc, f, indent=2)

    logger.info("Silver document written to %s", silver_path)
    return silver_doc


async def run_silver_and_store(
    storage_conn: str,
    bronze_artifacts: BronzeArtifacts,
    page_texts: List[str],
    page_images_base64: Optional[List[Optional[str]]],
    tables: Optional[Any],
) -> SilverDocument:
    """
    Build the Silver structured document for a PDF and store it to the Silver container.

    - Runs LLM over all pages (batched with limited concurrency).
    - Picks a document-level category + confidence.
    - Adds per-page results.
    - For "Medical expenses", adds a MedicalLines array.
    - Validates via silver_validator and chooses folder: valid / needs_review / rejected.
    - Uploads JSON to the Silver container.
    """

    logger.info("[SILVER] Starting Silver for docId=%s", bronze_artifacts.doc_id)

    if not page_texts:
        raise ValueError("page_texts cannot be empty")

    # 1. LLM for all pages
    page_results = await _call_llm_for_all_pages(
        page_texts=page_texts,
        page_images_base64=page_images_base64,
        tables=tables,
    )

    # 2. Document-level LLM category / confidence (before rule-based override)
    llm_doc_category, llm_doc_conf = _majority_category_and_confidence(page_results)

    # 3. Rule-based classification override (if stronger)
    joined_text = "\n\n".join(page_texts)

    final_category = llm_doc_category
    final_conf = llm_doc_conf

    rule_cat, rule_conf_raw = rule_based_classification(joined_text)

    # Ensure numeric
    try:
        rule_conf = float(rule_conf_raw)
    except Exception:
        logger.warning(
            "[SILVER] rule_based_classification returned non-numeric confidence=%r",
            rule_conf_raw,
        )
        rule_conf = 0.0

    # Only override if rule-based is stronger
    if rule_cat and rule_conf > final_conf:
        logger.info(
            "[SILVER] Rule override: LLM '%s'(%.2f) → rule '%s'(%.2f)",
            llm_doc_category,
            llm_doc_conf,
            rule_cat,
            rule_conf,
        )
        final_category = rule_cat
        final_conf = rule_conf

    # 4. Merge basic fields
    merged_fields = _merge_extracted_fields(page_results)
    page_results_payload = _build_page_results_payload(page_results)

    # 5. Simple "receipt stitching" for Medical expenses
    cat_lower = final_category.lower()
    if "medical" in cat_lower:
        medical_lines = _stitch_medical_lines(page_results, page_texts)
        if medical_lines:
            # Keep them under a dedicated key so Gold can fan-out later
            merged_fields["MedicalLines"] = medical_lines

            # Optional: compute a total if amounts are numeric
            total = 0.0
            count = 0
            for line in medical_lines:
                amt = line.get("Amount")
                if isinstance(amt, (int, float)):
                    total += float(amt)
                    count += 1
            if count > 0:
                merged_fields.setdefault("TotalAmount", round(total, 2))

    # 6. Build core structured payload
    structured: Dict[str, Any] = {
        "doc_id": bronze_artifacts.doc_id,
        "client_id": bronze_artifacts.client_id,
        "tax_year": bronze_artifacts.tax_year,
        "category": final_category,
        "confidence": final_conf,
        "storage_uri": bronze_artifacts.raw_pdf_blob_path,
        "page_count": len(page_texts),
        "page_results": page_results_payload,
        "extracted_fields": merged_fields,
    }

    # 7. Validate (soft) and derive status
    validation = validate_silver_document(structured)
    status = validation.status  # "valid" | "needs_review" | "invalid"

    # Choose folder under the Silver container
    if status == "valid":
        subfolder = "valid"
    elif status == "needs_review":
        subfolder = "needs_review"
    else:
        subfolder = "rejected"

    # 8. Upload JSON to Silver container
    silver_container = "silver"
    blob_path = f"{subfolder}/{bronze_artifacts.client_id}/{bronze_artifacts.tax_year}/{bronze_artifacts.doc_id}.json"

    logger.info(
        "[SILVER] Uploading structured JSON to '%s/%s'", silver_container, blob_path
    )

    payload_bytes = json.dumps(structured, ensure_ascii=False, indent=2).encode("utf-8")

    blob_client: BlobClient = BlobClient.from_connection_string(
        conn_str=storage_conn,
        container_name=silver_container,
        blob_name=blob_path,
    )
    try:
        # Ensure container exists (ignore race)
        container_client = ContainerClient.from_connection_string(
            conn_str=storage_conn, container_name=silver_container
        )
        try:
            await container_client.create_container()
        except Exception:
            pass

        # Simple retry for transient failures
        for attempt in range(3):
            try:
                await blob_client.upload_blob(payload_bytes, overwrite=True)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("[SILVER] Blob upload retry %d due to %s", attempt + 1, exc)
                await asyncio.sleep(2 * (attempt + 1))
    finally:
        try:
            await blob_client.close()
        except Exception:
            pass
    logger.info(
        "[SILVER] ✓ Stored at %s (category=%s confidence=%.2f status=%s)",
        blob_path,
        final_category,
        final_conf,
        status,
    )

    return SilverDocument(
        doc_id=bronze_artifacts.doc_id,
        category=final_category,
        confidence=final_conf,
        blob_path=blob_path,
        structured=structured,
        validation=validation,
    )
