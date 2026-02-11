
"""
Silver layer validation.

This module validates a Silver document *before* it is written to Gold (SQL).

It supports two input shapes:

1) A dict-like "structured" payload, as generated in silver_structured.run_silver_and_store()
   (keys like doc_id, client_id, tax_year, category, confidence, extracted_fields, ...)

2) A SilverDocument-like object with attributes:
       doc_id, client_id, tax_year, category, confidence, structured (dict)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)

# Confidence thresholds
HARD_MIN_CONFIDENCE = 0.30   # below this → invalid
REVIEW_CONFIDENCE = 0.80     # below this (but above hard min) → needs_review

_AMOUNT_REGEX = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

@dataclass
class SilverValidationResult:
    doc_id: str
    client_id: str
    tax_year: int
    category: str
    confidence: float

    status: Literal["valid", "needs_review", "invalid"]
    errors: List[str]
    warnings: List[str]

    # Fields after any normalisation / cleaning (single-record view)
    normalized_fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convenience helper when callers want a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "null", "NULL"):
            return None
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _clean_amount_like(value: Any) -> Optional[float]:
    """
    Fuzzy "amount" normaliser:

    - Strips currency text like "$", "CAD", "C$", etc.
    - Finds the first numeric substring using regex.
    - Returns None if nothing numeric is found.
    """
    if value is None:
        return None
    txt = str(value)
    # remove common currency tokens
    for token in ["$", "CAD", "C$", "cad", "Cad"]:
        txt = txt.replace(token, " ")
    txt = txt.replace(",", " ")
    m = _AMOUNT_REGEX.search(txt)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "").replace(" ", ""))
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value in (None, "", "null", "NULL"):
            return None
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return None


def _normalize_date_iso(value: Any) -> Optional[str]:
    """
    Best-effort date normalizer. Tries ISO first, then a couple of common
    text date formats. Returns ISO string (YYYY-MM-DD) or None.
    """
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    # raw ISO
    try:
        parts = text.split("T")[0]
        d = datetime.fromisoformat(parts)
        return d.date().isoformat()
    except Exception:
        pass

    try:
        import dateutil.parser  # type: ignore
        d = dateutil.parser.parse(text, dayfirst=False)
        return d.date().isoformat()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Soft, category-specific normalisers (mostly warnings, not hard errors)
# ---------------------------------------------------------------------------

def _soft_validate_donation(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    charity_name = (fields.get("DoneeName") or fields.get("charity_name") or "").strip()
    total_amount = _clean_amount_like(
        fields.get("Amount")
        or fields.get("TotalAmount")
        or fields.get("total_amount")
    )
    donation_date_raw = (
        fields.get("DonationDate")
        or fields.get("donation_date")
    )
    donation_date = _normalize_date_iso(donation_date_raw)

    if not charity_name:
        warnings.append("Donation: charity/donee name is missing or uncertain.")

    if total_amount is None or total_amount <= 0:
        warnings.append("Donation: amount is missing or non-positive; manual review recommended.")

    if donation_date_raw and not donation_date:
        warnings.append(f"Donation: 'donation_date' is not a valid date: {donation_date_raw!r}")

    return {
        "DoneeName": charity_name or None,
        "DonationDate": donation_date,
        "Amount": total_amount,
        "DonationType": fields.get("DonationType"),
        "DonorName": fields.get("DonorName"),
        "ReceiptNumber": fields.get("ReceiptNumber"),
        "Currency": fields.get("Currency") or "CAD",
    }


def _soft_validate_rrsp(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    sin = (fields.get("SIN") or "").strip()
    first_60 = _clean_amount_like(fields.get("First60DaysAmount"))
    rest_of_year = _clean_amount_like(fields.get("RestOfYearAmount"))

    if not sin:
        warnings.append("RRSP: SIN not found; please verify manually.")

    if (first_60 is None) and (rest_of_year is None):
        warnings.append("RRSP: no clear contribution amounts found.")

    return {
        "SIN": sin or None,
        "First60DaysAmount": first_60,
        "RestOfYearAmount": rest_of_year,
        "IsSpousal": bool(fields.get("IsSpousal")) if fields.get("IsSpousal") is not None else None,
    }


def _soft_validate_investment(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    slip_type = (fields.get("SlipType") or fields.get("slip_type") or "Unknown").strip()
    sin = (fields.get("SIN") or "").strip()

    box_values = fields.get("box_values") or fields.get("BoxValues") or {}
    if not isinstance(box_values, dict):
        box_values = {}

    norm_box_values: Dict[str, Optional[float]] = {}
    for box, val in box_values.items():
        norm_box_values[str(box)] = _clean_amount_like(val)

    if not norm_box_values:
        warnings.append("Investment: no numeric box values found; may be incomplete.")

    return {
        "SlipType": slip_type or "Unknown",
        "SIN": sin or None,
        "BoxValues": norm_box_values,
    }


def _soft_validate_medical(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    expense_date_raw = (
        fields.get("ExpenseDate")
        or fields.get("expense_date")
    )
    expense_date = _normalize_date_iso(expense_date_raw)
    amount = _clean_amount_like(fields.get("Amount"))

    if amount is None or amount <= 0:
        warnings.append("Medical: amount is missing or suspicious; manual review recommended.")

    if expense_date_raw and not expense_date:
        warnings.append(f"Medical: 'expense_date' could not be parsed: {expense_date_raw!r}")

    return {
        "ExpenseDate": expense_date,
        "PatientName": fields.get("PatientName"),
        "IsPrescription": bool(fields.get("IsPrescription")) if fields.get("IsPrescription") is not None else None,
        "VendorName": fields.get("VendorName"),
        "VendorCity": fields.get("VendorCity"),
        "Description": fields.get("Description"),
        "Amount": amount,
    }


def _soft_validate_childcare(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    expense_date_raw = fields.get("ExpenseDate") or fields.get("expense_date")
    expense_date = _normalize_date_iso(expense_date_raw)
    amount = _clean_amount_like(fields.get("Amount") or fields.get("amount"))

    if amount is None or amount <= 0:
        warnings.append("Childcare: amount appears missing or non-positive.")

    if expense_date_raw and not expense_date:
        warnings.append(f"Childcare: 'expense_date' could not be parsed: {expense_date_raw!r}")

    return {
        "ExpenseDate": expense_date,
        "ChildName": fields.get("ChildName"),
        "ProviderName": fields.get("ProviderName"),
        "City": fields.get("City"),
        "Description": fields.get("Description"),
        "Amount": amount,
    }


def _soft_validate_fhsa(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    sin = (fields.get("SIN") or "").strip()
    amount = _clean_amount_like(fields.get("Amount"))

    if not sin:
        warnings.append("FHSA: SIN not found; please verify manually.")

    if amount is None or amount <= 0:
        warnings.append("FHSA: contribution amount appears missing or non-positive.")

    return {
        "SIN": sin or None,
        "Amount": amount,
        "Institution": fields.get("Institution"),
        "PlanNumber": fields.get("PlanNumber"),
    }


def _soft_validate_property_tax(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    amount = _clean_amount_like(fields.get("Amount"))
    if amount is None or amount <= 0:
        warnings.append("Property tax: amount appears missing or non-positive.")

    return {
        "PropertyAddress": fields.get("PropertyAddress"),
        "Municipality": fields.get("Municipality"),
        "RollNumber": fields.get("RollNumber"),
        "TaxYearOfBill": fields.get("TaxYearOfBill"),
        "Amount": amount,
        "IsPrincipalResidence": bool(fields.get("IsPrincipalResidence"))
        if fields.get("IsPrincipalResidence") is not None
        else None,
    }


def _soft_validate_rent(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    total_rent = _clean_amount_like(
        fields.get("TotalRent") or fields.get("total_rent") or fields.get("Amount")
    )
    monthly_rent = _clean_amount_like(fields.get("MonthlyRent"))

    if total_rent is None and monthly_rent is not None:
        months = _safe_int(fields.get("MonthsPaid"))
        if months:
            total_rent = monthly_rent * months

    if total_rent is None or total_rent <= 0:
        warnings.append("Rent: total rent is missing or non-positive; manual review recommended.")

    return {
        "PropertyAddress": fields.get("PropertyAddress"),
        "LandlordName": fields.get("LandlordName"),
        "MonthsPaid": _safe_int(fields.get("MonthsPaid")),
        "MonthlyRent": monthly_rent,
        "TotalRent": total_rent,
    }


def _soft_validate_union_dues(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    amount = _clean_amount_like(fields.get("Amount"))
    if amount is None or amount <= 0:
        warnings.append("Union dues: amount appears missing or non-positive.")

    return {
        "OrganizationName": fields.get("OrganizationName") or fields.get("union_name"),
        "City": fields.get("City"),
        "Description": fields.get("Description"),
        "Amount": amount,
    }


def _soft_validate_generic(fields: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    """
    Fallback for "Slips" or "Other documents": we don't enforce strong rules,
    but we normalise numeric-looking fields where possible.
    """
    slip_type = (fields.get("SlipType") or fields.get("slip_type") or fields.get("category") or "Unknown").strip()

    box_values = fields.get("box_values") or fields.get("BoxValues") or {}
    if not isinstance(box_values, dict):
        # try to derive from keys like "Box14", "box_40", etc.
        box_values = {}
        for key, val in fields.items():
            if isinstance(key, str) and key.lower().startswith("box"):
                box_values[key] = val

    norm_box_values: Dict[str, Any] = {}
    for box, val in box_values.items():
        amt = _clean_amount_like(val)
        norm_box_values[str(box)] = amt if amt is not None else val

    # Only warn if it looks like a slip but has no boxes
    # "Other documents" includes receipts, invoices, etc. which naturally have no boxes.
    looks_like_slip = any(
        term in slip_type.lower() or term in str(fields.get("category_guess", "")).lower()
        for term in ["slip", "t4", "t5", "t3", "t2202", "rc62", "tax", "relevé", "statement"]
    )

    if not norm_box_values and looks_like_slip:
        warnings.append("Generic slip: no 'box_values' found; data may be incomplete.")

    return {
        "SlipType": slip_type or "Unknown",
        "BoxValues": norm_box_values,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _extract_structured_and_meta(silver_obj: Any) -> Dict[str, Any]:
    """
    Normalise different input shapes to a single dict:

    Returns dict with keys:
      - doc_id, client_id, tax_year, category, confidence
      - fields (dict)
    """
    if isinstance(silver_obj, dict):
        structured = silver_obj
    else:
        structured = getattr(silver_obj, "structured", {}) or {}

    doc_id = structured.get("doc_id") or getattr(silver_obj, "doc_id", "") or ""
    client_id = structured.get("client_id") or getattr(silver_obj, "client_id", "") or ""
    tax_year = structured.get("tax_year") or getattr(silver_obj, "tax_year", 0) or 0
    category = structured.get("category") or getattr(silver_obj, "category", "Other documents") or "Other documents"
    confidence = structured.get("confidence")
    if confidence is None:
        confidence = getattr(silver_obj, "confidence", 0.0)

    fields = structured.get("extracted_fields") or structured.get("fields") or {}

    return {
        "doc_id": doc_id,
        "client_id": client_id,
        "tax_year": tax_year,
        "category": category,
        "confidence": float(confidence or 0.0),
        "fields": fields if isinstance(fields, dict) else {},
    }


def validate_silver_document(silver_obj: Any) -> SilverValidationResult:
    """
    Validate a Silver document.

    Accepts either:
      - a 'structured' dict, or
      - a SilverDocument-like object (with .structured)

    Returns SilverValidationResult; never raises for business-rule failures.
    """

    meta = _extract_structured_and_meta(silver_obj)
    doc_id = meta["doc_id"]
    client_id = meta["client_id"]
    tax_year_raw = meta["tax_year"]
    category = meta["category"] or "Other documents"
    confidence = meta["confidence"]
    fields = meta["fields"]

    errors: List[str] = []
    warnings: List[str] = []

    # common checks
    if not doc_id:
        errors.append("Common: 'doc_id' is missing.")
    if not client_id:
        errors.append("Common: 'client_id' is missing.")

    try:
        ty = int(tax_year_raw)
    except Exception:
        warnings.append(f"Common: 'tax_year' is invalid: {tax_year_raw!r}")
        ty = 0

    current_year = datetime.now().year
    if ty and not (1970 <= ty <= current_year + 1):
        warnings.append(f"Common: 'tax_year' {ty} is outside expected range 1970–{current_year+1}.")

    if confidence < 0 or confidence > 1:
        warnings.append(f"Common: 'confidence' {confidence} is outside 0–1; clamping.")
        confidence = max(0.0, min(1.0, confidence))

    if confidence < HARD_MIN_CONFIDENCE:
        errors.append(
            f"Common: model confidence {confidence:.2f} is below hard minimum {HARD_MIN_CONFIDENCE:.2f}."
        )
    elif confidence < REVIEW_CONFIDENCE:
        warnings.append(
            f"Common: model confidence {confidence:.2f} is below review threshold {REVIEW_CONFIDENCE:.2f}."
        )

    # category-specific (all "soft" – mostly warnings)
    cat_lower = category.lower()
    if "donation" in cat_lower and "political" not in cat_lower:
        normalized_fields = _soft_validate_donation(fields, errors, warnings)
    elif "rrsp" in cat_lower:
        normalized_fields = _soft_validate_rrsp(fields, errors, warnings)
    elif any(x in cat_lower for x in ("t5", "t3", "t5008", "investment")):
        normalized_fields = _soft_validate_investment(fields, errors, warnings)
    elif "medical" in cat_lower:
        normalized_fields = _soft_validate_medical(fields, errors, warnings)
    elif "child" in cat_lower:
        normalized_fields = _soft_validate_childcare(fields, errors, warnings)
    elif "fhsa" in cat_lower:
        normalized_fields = _soft_validate_fhsa(fields, errors, warnings)
    elif "property tax" in cat_lower:
        normalized_fields = _soft_validate_property_tax(fields, errors, warnings)
    elif "rent" in cat_lower:
        normalized_fields = _soft_validate_rent(fields, errors, warnings)
    elif "union" in cat_lower:
        normalized_fields = _soft_validate_union_dues(fields, errors, warnings)
    else:
        normalized_fields = _soft_validate_generic(fields, errors, warnings)

    # status: only truly broken things are "invalid"
    if errors:
        status: Literal["valid", "needs_review", "invalid"] = "invalid"
    elif confidence < REVIEW_CONFIDENCE or warnings:
        status = "needs_review"
    else:
        status = "valid"

    result = SilverValidationResult(
        doc_id=doc_id,
        client_id=client_id,
        tax_year=ty,
        category=category,
        confidence=confidence,
        status=status,
        errors=errors,
        warnings=warnings,
        normalized_fields=normalized_fields,
    )

    logger.info(
        "[SILVER-VALIDATION] docId=%s category=%s status=%s errors=%d warnings=%d conf=%.2f",
        result.doc_id,
        result.category,
        result.status,
        len(errors),
        len(warnings),
        result.confidence,
    )

    return result
def validate_silver(silver_obj: Any) -> Dict[str, Any]:
    """Alias for validate_silver_document returning a dict for legacy compatibility."""
    res = validate_silver_document(silver_obj)
    d = res.to_dict()
    d["is_valid"] = (res.status == "valid")
    return d
