"""
Prompt factory for Canadian tax document extraction.
Supports multiple document types with specialized extraction rules.
"""



from typing import Tuple, Optional, Dict, Any, Type
import logging
import datetime
import json
from pydantic import BaseModel
from src.extraction.classifier import DocumentCategory, CATEGORY_PROCESSOR_MAP

logger = logging.getLogger(__name__)

def _get_schema_instructions(model: Optional[Type[BaseModel]]) -> str:
    """Generate prompt instructions from Pydantic model fields."""
    if not model:
        return ""
    
    try:
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        
        lines = ["\nTarget Fields:"]
        for field_name, info in props.items():
            req_str = "(Required)" if field_name in required else "(Optional)"
            desc = info.get("description", "")
            lines.append(f"- {field_name} {req_str}: {desc}")
            
        return "\n".join(lines)
    except Exception:
        return ""

# ============================================================================
# Dynamic Prompt Loading
# ============================================================================

_PROMPT_CACHE: Dict[str, Any] = {}
_CACHE_TTL = datetime.timedelta(minutes=5)

def _get_prompt_override(processor_name: str) -> Optional[Tuple[str, str]]:
    """
    Check if there is an active prompt override in the database.
    Returns (system_prompt, user_prompt_template) or None.
    Uses simple caching to avoid DB spam.
    """
    global _PROMPT_CACHE
    
    now = datetime.datetime.now()
    if processor_name in _PROMPT_CACHE:
        entry = _PROMPT_CACHE[processor_name]
        if now - entry["ts"] < _CACHE_TTL:
            return entry["prompts"]

    try:
        from src.pipeline.db import get_sql_conn
    except ImportError:
        # Fallback locally
        try:
            from bookkeeper.src.pipeline.db import get_sql_conn
        except ImportError:
            return None

    try:
        conn = get_sql_conn()
        cur = conn.cursor()
        # Check for Enabled overrides only
        cur.execute("""
            SELECT SystemPrompt, UserPrompt 
            FROM gold.ProcessorConfig 
            WHERE Name = ? AND Enabled = 1 
              AND (SystemPrompt IS NOT NULL OR UserPrompt IS NOT NULL)
        """, (processor_name,))
        row = cur.fetchone()
        conn.close()

        prompts = None
        if row:
            sys_p, user_p =row[0], row[1]
            if sys_p and user_p:
               prompts = (sys_p, user_p)
        
        _PROMPT_CACHE[processor_name] = {
            "ts": now,
            "prompts": prompts
        }
        return prompts
        
    except Exception as e:
        logger.warning(f"Failed to load prompt overrides for {processor_name}: {e}")
        return None

# ============================================================================
# Prompt Generators
# ============================================================================

def build_medical_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    """
    Build prompts for medical expense extraction.
    Includes context from previous chunks to handle duplicates/continuations.
    """
    override = _get_prompt_override("medical_expense")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            # Safe format with strict keys that match what is expected
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content,
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception as e:
            logger.error(f"Failed to format medical override prompt: {e}. Falling back.")

    system_prompt = """You are a specialized tax assistant for Canadian Personal Income Tax (CRA).
Your task is to extract ELIGIBLE MEDICAL EXPENSES for the Medical Expense Tax Credit using a STRICT PAGE-WISE approach.

PRIMARY OBJECTIVE:
Perform a page-by-page extraction where EACH page in the document gets its own separate extraction record. Do NOT consolidate transactions across pages. Extract exactly what is documented on each individual page.

PAGE-WISE EXTRACTION RULES:
1. ONE RECORD PER PAGE: Every page should have Exactly One extraction entry.
2. AMOUNT EXTRACTION:
   - Extract the specific amount shown on the current page.
   - If Page X is a prescription label showing "Patient Pays: $2.83", extract $2.83 for Page X.
   - If Page Y is a terminal/Visa receipt showing "Total: $14.83", extract $14.83 for Page Y.
   - Do NOT sum pages together or try to "consolidate" them into one record. The user wants to see the raw data per page.
3. LINKING & RELATIONSHIPS:
   - Use the 'comments' and 'duplicate_reference' fields to explain relationships.
   - If Page 4 is a payment for items on Pages 1 and 3, extract the amount for Page 4 and add comment: "Total payment for pages 1 and 3".
4. CATEGORIZATION:
   - Assign the most specific category to each page.
   - Payment receipts without item details should be categorized as "Payment Confirmation" or the broad medical category.

Example for Pages 1-4:
- Page 1 (Prescription Label): {"amount": 2.83, "page_numbers": [1], "comments": "Prescription item"}
- Page 2 (Blank): {"amount": null, "page_numbers": [2], "expense_type": "Irrelevant / Non-medical"}
- Page 3 (Receipt for Kit): {"amount": 12.00, "page_numbers": [3], "comments": "Surgical kit item"}
- Page 4 (Visa Terminal Receipt): {"amount": 14.83, "page_numbers": [4], "comments": "Total payment for pages 1 and 3"}

ELIGIBILITY RULES:
Mark eligible=true when:
- Prescriptions, Dental, Vision care, Physiotherapy, Chiropractic, Registered Massage Therapy
- Medical devices, hearing aids, glasses/contacts
- Lab work, diagnostic tests
- Hospital services, clinic fees

Mark eligible=false when:
- Cosmetic procedures (unless medically necessary)
- Gym memberships, fitness classes
- Over-the-counter vitamins/supplements (unless prescribed)
- Appointment reminders, lab results (non-financial pages)
- PARKING for medical visits if distance < 80km one-way from home

MEDICAL TRAVEL & PARKING (Ontario/CRA Rules):
- Parking is eligible ONLY if distance traveled is 80 km or more one-way.
- For parking receipts: Mark eligible=false if distance < 80km, add comment explaining.

DUPLICATE DETECTION:
- Use 'duplicate_reference' to indicate if a page is a direct duplicate of another page (e.g., same receipt scanned twice).
- If one page is an itemized bill and another is a payment for it, they are NOT duplicates in the page-wise view; they are different pieces of evidence for the same transaction. Use comments to link them.

DATE FORMAT:
- Use DD-MM-YYYY whenever possible.

RX_NUMBER:
- Capture prescription number if shown on the page.

COMMENTS REQUIREMENTS:
Explain:
- Why something is ineligible.
- Relationships between pages (e.g., "This payment confirms items on pages X and Y").
- If the page is a continuation or blank.
"""
    
    user_prompt = f"""
Analyze the following document chunk and extract medical expenses using a STRICT PAGE-WISE approach.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}

CRITICAL INSTRUCTIONS:
1. Return a JSON object with a list of expenses.
2. Provide EXACTLY ONE extraction entry for EACH page number in this chunk.
3. DO NOT CONSOLIDATE. If a receipt spans pages 1, 3, and 4, you must create 3 separate entries:
   - Entry for Page 1 with its amount.
   - Entry for Page 3 with its amount.
   - Entry for Page 4 with its total amount.
4. Use the 'comments' field to link these pages (e.g., "Total for pages 1 and 3").
5. If a page is blank or irrelevant, create an entry with amount: null and expense_type: "Irrelevant".
"""
    return system_prompt, user_prompt


def build_childcare_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("child_care")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant for Canadian Personal Income Tax (CRA).
Extract CHILDCARE EXPENSES for the Childcare Expense Deduction using a STRICT PAGE-WISE approach.
"""
    
    user = f"""Extract childcare expenses from the following document using a STRICT PAGE-WISE approach.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user


def build_donation_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("donation")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant. Extract Charitable Donations using a STRICT PAGE-WISE approach.
"""
    user = f"""Extract charitable donation receipts using a STRICT PAGE-WISE approach.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user

def build_fhsa_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("fhsa_contribution")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant. Extract FHSA Contributions using a STRICT PAGE-WISE approach."""
    user = f"""Extract First Home Savings Account (FHSA) contributions page-by-page.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user

def build_slips_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("slips")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant. Extract ALL Tax Slips using a STRICT PAGE-WISE approach."""
    user = f"""
    Analyze the following content and extract tax slips page-by-page.
    
    CONTEXT FROM PREVIOUS CHUNKS:
    {prior_context}
    
    DOCUMENT CHUNK:
    {chunk_content}
    """
    return system, user

def build_property_tax_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("property_tax")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant for Canadian Personal Income Tax (CRA).
Extract PROPERTY TAX BILLS using a STRICT PAGE-WISE approach.
"""
    
    user = f"""Extract property tax bills using a STRICT PAGE-WISE approach.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user


def build_rent_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("rent_receipt")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant for Canadian Personal Income Tax (CRA).
Extract RENT RECEIPTS using a STRICT PAGE-WISE approach.
"""
    
    user = f"""Extract rent receipts page-by-page.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user


def build_rrsp_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("rrsp_contribution")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant. Extract RRSP Contributions using a STRICT PAGE-WISE approach."""
    user = f"""Extract RRSP Contribution Receipts page-by-page.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user

def build_union_dues_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("union_dues")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant. Extract Union/Professional Dues using a STRICT PAGE-WISE approach."""
    user = f"""Extract Union or Professional Dues receipts page-by-page.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user

def build_other_document_prompts(chunk_content: str, prior_context: str = "") -> Tuple[str, str]:
    override = _get_prompt_override("other_docs")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(
                chunk_content=chunk_content, 
                context=chunk_content,
                prior_context=prior_context
            )
        except Exception:
            pass

    system = """You are a tax assistant specializing in miscellaneous financial documents.
Perform extraction using a STRICT PAGE-WISE approach. One entry per page."""
    user = f"""
Analyze the following document page-by-page.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}
"""
    return system, user

# ============================================================================
# Factory
# ============================================================================

def get_prompts_for_category(
    category: DocumentCategory, 
    chunk_content: str, 
    prior_context: str = "",
    model_class: Optional[Type[BaseModel]] = None
) -> Tuple[str, str]:
    """
    Factory function to return system and user prompts for a given category.
    """
    if category == DocumentCategory.MEDICAL_EXPENSES:
        sys, user = build_medical_prompts(chunk_content, prior_context)
    
    elif category == DocumentCategory.CHILD_CARE_EXPENSES:
        sys, user = build_childcare_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.CHARITABLE_DONATIONS:
        sys, user = build_donation_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.FHSA_CONTRIBUTION:
        sys, user = build_fhsa_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.SLIPS:
        sys, user = build_slips_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.PROPERTY_TAX_RECEIPT:
        sys, user = build_property_tax_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.RENT_RECEIPT:
        sys, user = build_rent_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.RRSP_CONTRIBUTION:
        sys, user = build_rrsp_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.UNION_PROFESSIONAL_DUES:
        sys, user = build_union_dues_prompts(chunk_content, prior_context)
        
    elif category == DocumentCategory.OTHER_DOCUMENTS:
        sys, user = build_other_document_prompts(chunk_content, prior_context)
        
    else:
        sys, user = build_other_document_prompts(chunk_content)

    # Inject dynamic schema instructions if available and NOT using a DB override
    # logic: check if DB override usage was detected? 
    # Actually, we can just append it to user prompt for clarity if it's dynamic.
    if model_class:
        override = _get_prompt_override(CATEGORY_PROCESSOR_MAP.get(category, ""))
        if not override:
             # Only auto-append instructions if user hasn't provided a custom prompt
             schema_instr = _get_schema_instructions(model_class)
             if schema_instr:
                 user += f"\n\nIMPORTANT: Extract data matching this structure exactly:\n{schema_instr}"

    return sys, user

def build_repair_prompt(malformed_json: str) -> str:
    """
    Build a prompt to repair malformed JSON.
    """
    return f"The following JSON is invalid. Repair it to valid JSON that matches the schema.\n\nMalformed JSON:\n{malformed_json}"

