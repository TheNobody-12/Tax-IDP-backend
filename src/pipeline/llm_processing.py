# src/pipeline/llm_processing.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, List

from src.config.azure_clients import llm_client, openai_deployment_name

logger = logging.getLogger(__name__)

SAFE_SYSTEM_PROMPT = """
You help classify Canadian personal income tax and financial documents and
extract structured information using three sources per page:

1. OCR text (raw text)
2. Azure Document Intelligence JSON (structured tables, key-values, layout)
3. (Optional) page image as base64

You MUST intelligently combine all three inputs:
- Prefer DI JSON for numeric values, amounts, table rows, key-values.
- Prefer OCR text for contextual meaning, names, addresses.
- Use the page image only when OCR and DI disagree.
- NEVER hallucinate amounts or dates; if unsure, leave the field null.

You MUST respond with a single valid JSON object, no comments, no trailing text:

{
  "category": "<one of the valid categories below>",
  "confidence": 0.0-1.0,
  "extracted_fields": { ... },
  "corrected_text": "<optional cleaned-up text or summary>",
  "tables": [ ... ]
}

Valid categories (exact strings):

- "Slips"
- "Medical expenses"
- "Charitable donations"
- "Political donations"
- "Child care expenses"
- "RRSP contribution"
- "Union and Professional Dues"
- "Property Tax receipt"
- "Rent receipt"
- "Other documents"

You MUST pick exactly one category per page.

FACT TABLE AWARE EXTRACTION
--------------------------------

For each category, favour extracting fields that map cleanly to these fact tables:

1) FactChildcare
   - ExpenseDate
   - ChildName
   - ProviderName
   - City
   - Description
   - Amount

2) FactDonation
   - DonationDate
   - DonorName
   - DoneeName (charity name)
   - Amount

3) FactFHSAContribution
   - SIN
   - Amount

4) FactInvestmentSlip  /  FactSlipBox
   - SIN
   - SlipType (T4, T5, T3, T5008, etc.)
   - BoxNumber (box code on the slip, e.g. "14", "40", "117")
   - Amount
   - Currency
   - Country
   - AdminFee
   - ModeOfHolding
   - SourceFieldName (the label or description from the slip)

5) FactMedicalExpense
   - ExpenseDate
   - PatientName
   - IsPrescription (true/false)
   - PrescriptionId (when found, e.g. N306331, RX codes)
   - VendorName
   - VendorCity
   - Description
   - Amount

6) FactPropertyTax
   - PropertyAddress
   - Municipality
   - RollNumber
   - TaxYearOfBill
   - Amount
   - IsPrincipalResidence (true/false)

7) FactRent
   - PropertyAddress
   - LandlordName
   - MonthsPaid
   - MonthlyRent
   - TotalRent

8) FactRRSPContribution
   - SIN
   - First60DaysAmount
   - RestOfYearAmount
   - IsSpousal (true/false)

9) FactUnionDues
   - OrganizationName (union / professional body)
   - City
   - Description
   - Amount

For T-slips ("Slips" category), focus on:

- SlipType (e.g. "T4", "T5", "T3", "T5008")
- SIN
- box_values: { "<box_number>": amount }  (e.g. "14": 52345.67)

Include both:
- a high-level summary in `extracted_fields`, and
- detailed numeric values (box numbers, amounts) whenever available.

AMOUNT NORMALIZATION RULES
----------------------------

- ALWAYS return clean JSON numbers, never strings, when you are confident.
- Strip "$", "CAD", "C$", and commas from numeric values.
  - Example: "$1,234.56 CAD" → 1234.56
- If the source text clearly shows a range or multiple amounts, pick the best
  single amount that matches the fact-table field.
- If you are not sure about a number, set the field to null instead of guessing.

DATE NORMALIZATION RULES
-------------------------

- Return dates as ISO "YYYY-MM-DD" when possible.
- If a date cannot be reliably determined, set the field to null.

MULTI-PAGE / RECEIPT THINKING
------------------------------

You are called once per PAGE, not per whole PDF.

- Treat each page as one logical record (e.g., one receipt or one slip page).
- Do NOT try to reference other pages.
- Do NOT try to stitch across pages in this call; higher-level logic will do that.
- Just extract the best fields for THIS page only.

RESPONSE FORMAT RULES
-----------------------

- Output MUST be a single JSON object, no leading/trailing text.
- Do NOT wrap JSON in ``` fences.
- No explanations, no commentary, no "reasoning" keys.
- Keys:
    - category: string (exactly one of the allowed categories)
    - confidence: number between 0 and 1
    - extracted_fields: object with the fields above where applicable
    - corrected_text: string (optional; can be empty)
    - tables: array (optional; can be empty)

If you are unsure of the correct category, use "Other documents"
with a lower confidence (e.g. 0.4–0.6).
"""


# Category-specific allowed fields mapping
CATEGORY_FIELD_MAP: Dict[str, List[str]] = {
    "Child care expenses": [
        "ExpenseDate","ChildName","ProviderName","City","Description","Amount"
    ],
    "Charitable donations": [
        "DonationDate","DonorName","DoneeName","Amount"
    ],
    "RRSP contribution": [
        "SIN","Amount","First60DaysAmount","RestOfYearAmount","IsSpousal"
    ],
    "Slips": [
        "SIN","SlipType","BoxNumber","Amount","Currency","Country","AdminFee","ModeOfHolding","SourceFieldName","box_values"
    ],
    "Medical expenses": [
        "ExpenseDate","PatientName","IsPrescription","PrescriptionId","VendorName","VendorCity","Description","Amount"
    ],
    "Property Tax receipt": [
        "PropertyAddress","Municipality","RollNumber","TaxYearOfBill","Amount","IsPrincipalResidence"
    ],
    "Rent receipt": [
        "PropertyAddress","LandlordName","MonthsPaid","MonthlyRent","TotalRent"
    ],
    "Union and Professional Dues": [
        "OrganizationName","City","Description","Amount"
    ],
    # Fallback category accepts anything
    "Other documents": []
}


def _filter_fields_by_category(category: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    allowed = CATEGORY_FIELD_MAP.get(category)
    if not allowed:
        # For Other documents or unknown, return fields as-is
        return fields
    filtered: Dict[str, Any] = {}
    for k in allowed:
        if k in fields:
            filtered[k] = fields[k]
    # Preserve box_values for slips even if not explicitly in fields list
    if category == "Slips" and "box_values" in fields:
        filtered["box_values"] = fields["box_values"]
    return filtered


async def process_page_with_llm(
    page_text: str,
    page_image_base64: Optional[str],
    page_number: int,
    tables: Optional[Any] = None,
) -> Dict[str, Any]:
    """Send a single page to the LLM and parse a robust JSON response."""
    user_payload: Dict[str, Any] = {
        "meta": {"page_number": page_number},
        "ocr_text": page_text or "",
        "tables": tables or [],
    }
    if page_image_base64:
        user_payload["page_image_base64"] = page_image_base64
    try:
        logger.info("[LLM] Sending request to LLM for page %s...", page_number)
        response = await llm_client.chat.completions.create(
            model=openai_deployment_name,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SAFE_SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            temperature=0.1,
            max_tokens=8000,
        )
        raw_json = response.choices[0].message.content or ""
        logger.debug("[LLM] Raw LLM output (first 200 chars): %s", raw_json[:200].replace("\n", "\\n"))
        try:
            parsed = json.loads(raw_json)
        except Exception as e:
            logger.warning("[LLM WARNING] JSON parsing failed for page %s: %s", page_number, e)
            try:
                start = raw_json.find("{"); end = raw_json.rfind("}")
                if start != -1 and end != -1 and end > start:
                    salvaged = raw_json[start:end+1]
                    parsed = json.loads(salvaged)
                    logger.warning("[LLM] JSON successfully salvaged for page %s", page_number)
                else:
                    raise ValueError("Could not find JSON object in response")
            except Exception as salvage_err:
                logger.error("[LLM ERROR] Unable to parse JSON for page %s even after salvage: %s", page_number, salvage_err)
                return {"status": "error","error": f"LLM JSON parse error: {e}","page_number": page_number,"category": None,"confidence": 0.0,"extracted_fields": {},"corrected_text": "","tables": [],"raw_json": raw_json}
        category = parsed.get("category") or "Other documents"
        confidence = float(parsed.get("confidence") or 0.0)
        extracted_fields = parsed.get("extracted_fields") or {}
        if not isinstance(extracted_fields, dict):
            extracted_fields = {}
        # Filter fields by category mapping
        extracted_fields = _filter_fields_by_category(category, extracted_fields)
        corrected_text = parsed.get("corrected_text") or ""
        tables_out = parsed.get("tables") or []
        if not isinstance(tables_out, list):
            tables_out = []
        return {
            "status": "ok",
            "category": category,
            "confidence": confidence,
            "extracted_fields": extracted_fields,
            "corrected_text": corrected_text,
            "tables": tables_out,
            "page_number": page_number,
            "raw_json": raw_json,
            "llm_used": True,
        }
    except Exception as e:
        logger.exception("[LLM ERROR] Unexpected error on page %s: %s", page_number, e)
        return {
            "status": "error",
            "error": str(e),
            "page_number": page_number,
            "category": None,
            "confidence": 0.0,
            "extracted_fields": {},
            "corrected_text": "",
            "tables": [],
            "raw_json": "",
            "llm_used": True,
        }
