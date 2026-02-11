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
Your task is to extract ELIGIBLE MEDICAL EXPENSES for the Medical Expense Tax Credit.

RULES:
1. Extract transaction DATE, AMOUNT, VENDOR (Payee/Provider), and DESCRIPTION.
2. Determine ELIGIBILITY based on CRA rules (Prescriptions, Dental, Vision, Health Plans = YES).
   - Cosmetic procedures, gym memberships, over-the-counter vitamins = NO.
3. IGNORE headers, footers, and non-transaction text.
4. If a transaction is a "Balance Forward" or "Previous Balance", IGNORE IT (flag as duplicate or omit).
5. For INSURANCE STATEMENTS: Only extract the "Patient Pays" or "Amount You Paid" column. Do NOT extract the total billed amount if it was covered.
"""
    
    user_prompt = f"""
Analyze the following document chunk and extract medical expenses.

CONTEXT FROM PREVIOUS CHUNKS:
{prior_context}

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
- Return a JSON object with a list of expenses.
- Use explicit field names (date, amount, payee_provider, expense_type).
- If an expense is clearly ineligible, set 'eligible' to false.
- If unsure, set 'eligible' to null.
"""
    return system_prompt, user_prompt

def build_childcare_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("child_care")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract Childcare Expenses."
    user = f"""Extract childcare receipts.
Fields: Date, Child Name, Provider Name, City, Description, Amount.
Content:
{context}
"""
    return system, user

def build_donation_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("donation")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract Charitable Donations."
    user = f"""Extract charitable donation receipts/transactions.
Relaxed Rules:
- Extract ANY payment that appears to be a donation or registration for a charity event.
- Do not strictly require "Official Tax Receipt" phrasing.
- If it's a registration fee, extract it and note in comments.

Fields: Date, Donor, Donee (Charity), Amount.
Content:
{context}
"""
    return system, user

def build_fhsa_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("fhsa_contribution")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract FHSA Contributions."
    user = f"""Extract First Home Savings Account (FHSA) contributions.
Fields: SIN, Amount.
Content:
{context}
"""
    return system, user

def build_slips_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("slips")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract ALL Tax Slips (T4, T5, T3, T5008, etc.) found in the document."
    user = f"""
    Analyze the following content which may contain MULTIPLE tax slips (e.g. on different pages).
    
    INSTRUCTIONS:
    1. Scan the entire text for any tax slip.
    2. Extract EACH slip as a separate item in the list. Do not stop after the first one.
    3. For each slip, capture:
       - Slip Type (T4, T4A, T5, etc.)
       - Issuer (Employer/Payer)
       - SIN (Social Insurance Number)
       - Box Values: Map every box number (14, 22, etc.) to its value.
    
    CRITICAL: If there are 5 pages of slips, I expect 5 items in the list.
    
    Content:
    {context}
    """
    return system, user

def build_property_tax_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("property_tax")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract Property Tax Bills."
    user = f"""Extract Final Property Tax amount.
Fields: Address, Municipality, Roll Number, Year, Amount.
Content:
{context}
"""
    return system, user

def build_rent_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("rent_receipt")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract Rent Receipts."
    user = f"""Extract Rent Information.
Fields: Address, Landlord, Months Paid, Monthly Rent, Total Amount.
Content:
{context}
"""
    return system, user

def build_rrsp_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("rrsp_contribution")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract RRSP Contributions."
    user = f"""Extract RRSP Contribution Receipts.
Fields: SIN, First 60 Doys Amount, Rest of Year Amount, Total Amount.
Content:
{context}
"""
    return system, user

def build_union_dues_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("union_dues")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. Extract Union/Professional Dues."
    user = f"""Extract Union or Professional Dues receipts.
Fields: Organization, Description, Amount.
Content:
{context}
"""
    return system, user

def build_other_document_prompts(context: str) -> Tuple[str, str]:
    override = _get_prompt_override("other_docs")
    if override:
        sys_tmpl, user_tmpl = override
        try:
            return sys_tmpl, user_tmpl.format(context=context, chunk_content=context)
        except Exception:
            pass

    system = "You are a tax assistant. specific for summarizing miscellaneous financial documents."
    user = f"""
Analyze the following document.
It has been classified as 'Other' or 'Miscellaneous'.
Your goal is to extract key financial information in a structured way.

Fields to Extract:
- Date: The main statement date or transaction date.
- Entity: Who is the document from? (Bank, Organization, etc.)
- Description: What is this document? (e.g. "Annual Mortgage Statement", "Utility Bill")
- Amount: The primary financial figure (e.g. Total Paid, Ending Balance).
- Category Guess: What tax category might this belong to?

Content:
{context}
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
        sys, user = build_childcare_prompts(chunk_content)
        
    elif category == DocumentCategory.CHARITABLE_DONATIONS:
        sys, user = build_donation_prompts(chunk_content)
        
    elif category == DocumentCategory.FHSA_CONTRIBUTION:
        sys, user = build_fhsa_prompts(chunk_content)
        
    elif category == DocumentCategory.SLIPS:
        sys, user = build_slips_prompts(chunk_content)
        
    elif category == DocumentCategory.PROPERTY_TAX_RECEIPT:
        sys, user = build_property_tax_prompts(chunk_content)
        
    elif category == DocumentCategory.RENT_RECEIPT:
        sys, user = build_rent_prompts(chunk_content)
        
    elif category == DocumentCategory.RRSP_CONTRIBUTION:
        sys, user = build_rrsp_prompts(chunk_content)
        
    elif category == DocumentCategory.UNION_PROFESSIONAL_DUES:
        sys, user = build_union_dues_prompts(chunk_content)
        
    elif category == DocumentCategory.OTHER_DOCUMENTS:
        sys, user = build_other_document_prompts(chunk_content)
        
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

