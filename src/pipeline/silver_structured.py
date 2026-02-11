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

# from src.pipeline.llm_processing import process_page_with_llm  # REMOVED
from src.pipeline.bronze_store import BronzeArtifacts  # your existing dataclass
from src.pipeline.silver_validator import validate_silver_document, SilverValidationResult
# from src.pipeline.classification_rules import rule_based_classification  # REMOVED

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



# Legacy LLM page calling logic removed.
# TaxExtractionBridge handles this now.


# Legacy merge logic removed.


# Legacy stitching logic removed.


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


# Legacy run_silver_and_store removed.
# Use TaxExtractionBridge in process_document.py instead.
