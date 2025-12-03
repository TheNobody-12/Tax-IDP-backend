from typing import Dict

def rule_based_classification(text: str) -> Dict[str, object]:
    """
    Simple keyword-based classifier that returns one of the business categories:
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
    """
    t = text.lower()

    # --- Slips (T4, T5, T3, etc.) ---
    if any(k in t for k in ["t4 ", "t4a", "t4e", "t4(rsp)", "t4 rif", "t5 ", "t3 ", "t5008", "t2200", "t5013", "t5018", "t5007"]):
        return {"category": "Slips", "confidence": 0.9}
    if "tax slip" in t or "information slip" in t:
        return {"category": "Slips", "confidence": 0.85}

    # --- Medical expenses ---
    if any(k in t for k in ["medical receipt", "medical expense", "prescription", "rx ", "pharmacy", "drug mart", "drug store", "dentist", "dental", "optometrist"]):
        return {"category": "Medical expenses", "confidence": 0.9}

    # --- Charitable donations ---
    if "official donation receipt" in t and "charity" in t:
        return {"category": "Charitable donations", "confidence": 0.95}
    if "registered charity" in t or "charitable donation" in t:
        return {"category": "Charitable donations", "confidence": 0.9}

    # --- Political donations ---
    if "political contribution" in t or "political donation" in t:
        return {"category": "Political donations", "confidence": 0.9}
    if "elections canada" in t or "campaign contribution" in t:
        return {"category": "Political donations", "confidence": 0.9}

    # --- Child care expenses ---
    if any(k in t for k in ["child care", "childcare", "daycare", "day care", "nursery school", "before and after school program"]):
        return {"category": "Child care expenses", "confidence": 0.9}

    # --- RRSP contribution ---
    if "rrsp" in t and "contribution" in t:
        return {"category": "RRSP contribution", "confidence": 0.95}
    if "registered retirement savings plan" in t and "receipt" in t:
        return {"category": "RRSP contribution", "confidence": 0.9}

    # --- Union and Professional Dues ---
    if "union dues" in t:
        return {"category": "Union and Professional Dues", "confidence": 0.95}
    if any(k in t for k in ["professional dues", "membership dues", "annual dues", "college of ", "association of", "bar association", "society of"]):
        return {"category": "Union and Professional Dues", "confidence": 0.85}

    # --- Property Tax receipt ---
    if "property tax" in t or "tax bill" in t and "property" in t:
        return {"category": "Property Tax receipt", "confidence": 0.9}
    if any(k in t for k in ["interim tax bill", "final tax bill", "assessment roll", "roll number"]):
        return {"category": "Property Tax receipt", "confidence": 0.9}

    # --- Rent receipt ---
    if "rent receipt" in t or "rental receipt" in t:
        return {"category": "Rent receipt", "confidence": 0.95}
    if ("landlord" in t and "tenant" in t) or ("rent" in t and "premises" in t):
        return {"category": "Rent receipt", "confidence": 0.9}

    # Fallback
    return {"category": "Other documents", "confidence": 0.5}
