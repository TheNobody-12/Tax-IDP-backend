"""
LLM-based document classifier.
Classifies entire documents into tax categories using first few pages.
"""

import os
import re
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.extraction.clients.azure_ai import AzureAIClient

logger = logging.getLogger(__name__)

class DocumentCategory(str, Enum):
    """Valid document categories for Canadian tax documents"""
    SLIPS = "Slips"
    MEDICAL_EXPENSES = "Medical expenses"
    CHARITABLE_DONATIONS = "Charitable donations"
    POLITICAL_DONATIONS = "Political donations"
    CHILD_CARE_EXPENSES = "Child care expenses"
    RRSP_CONTRIBUTION = "RRSP contribution"
    FHSA_CONTRIBUTION = "FHSA contribution"
    UNION_PROFESSIONAL_DUES = "Union and Professional Dues"
    PROPERTY_TAX_RECEIPT = "Property Tax receipt"
    RENT_RECEIPT = "Rent receipt"
    OTHER_DOCUMENTS = "Other documents"


# Mapping from category to processor name
CATEGORY_PROCESSOR_MAP = {
    DocumentCategory.SLIPS: "slips",
    DocumentCategory.MEDICAL_EXPENSES: "medical_expense",
    DocumentCategory.CHARITABLE_DONATIONS: "donation",
    DocumentCategory.POLITICAL_DONATIONS: "donation",
    DocumentCategory.CHILD_CARE_EXPENSES: "child_care",
    DocumentCategory.RRSP_CONTRIBUTION: "rrsp_contribution",
    DocumentCategory.FHSA_CONTRIBUTION: "fhsa_contribution",
    DocumentCategory.UNION_PROFESSIONAL_DUES: "union_dues",
    DocumentCategory.PROPERTY_TAX_RECEIPT: "property_tax",
    DocumentCategory.RENT_RECEIPT: "rent_receipt",
    DocumentCategory.OTHER_DOCUMENTS: "other_docs",
}


@dataclass
class ClassificationResult:
    """Result of document classification"""
    category: Any  # Was DocumentCategory, now dynamic str
    confidence: float
    reasoning: str
    pages_analyzed: int
    processor_name: Optional[str] = None
    
    def __post_init__(self):
        # We handle processor_name mapping in classify() now
        pass


CLASSIFICATION_SYSTEM_PROMPT = """You are a Canadian tax document classifier.

Analyze the document sample and classify it into exactly ONE category.

Valid categories:
- "Slips" - T4, T4A, T5, T3, T5008, employment/investment income slips
- "Medical expenses" - Pharmacy receipts, dental bills, optometry, hospital, prescriptions, attendant care
- "Charitable donations" - Donation receipts from registered charities
- "Political donations" - Political party contribution receipts
- "Child care expenses" - Daycare, nanny, after-school care receipts
- "RRSP contribution" - RRSP contribution receipts, statements
- "FHSA contribution" - First Home Savings Account statements
- "Union and Professional Dues" - Union membership fees, professional association dues
- "Property Tax receipt" - Municipal property tax bills/receipts
- "Rent receipt" - Rental payment receipts for primary residence
- "Other documents" - Cannot determine or doesn't fit above categories

Rules:
- Choose the SINGLE most appropriate category for the ENTIRE document
- If document contains multiple types, choose the predominant one
- If unsure, use "Other documents" with lower confidence

Respond with ONLY a JSON object:
{"category": "<category_name>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}"""

class ClassificationResponse(Enum):
    # Dummy class for Structured Output if needed, but we'll use a direct Pydantic model
    pass

from pydantic import BaseModel
class ClassificationModel(BaseModel):
    category: str
    confidence: float
    reasoning: str

class DocumentClassifier:
    """
    LLM-based document classifier using AzureAIClient.
    Uses first few pages for quick classification.
    """
    
    def __init__(
        self,
        ai_client: Optional[AzureAIClient] = None,
        model: Optional[str] = None,
        max_pages: int = 3,
        max_chars: int = 8000
    ):
        self._ai_client = ai_client
        self._model = model
        self.max_pages = max_pages
        self.max_chars = max_chars
    
    def _get_ai_client(self) -> AzureAIClient:
        if self._ai_client is None:
            self._ai_client = AzureAIClient()
        return self._ai_client
    
    def _get_model(self) -> str:
        if self._model:
            return self._model
        return self._get_ai_client().fast_model
    

    def _extract_sample(self, markdown: str) -> tuple[str, int]:
        """Extract first N pages from markdown, limited by character count"""
        page_pattern = r'(PAGE\s+(\d+)\s+START)'
        parts = re.split(page_pattern, markdown)
        
        # re.split with groups returns groups too.
        # pattern has 2 groups: (PAGE N START) and (N).
        # We need to handle this.
        # But for simple sampling, let's just grab the first N chars and verify page count roughly.
        
        # Fallback simplified logic
        if not markdown:
            return "", 0
            
        sample = markdown[:self.max_chars]
        pages = len(re.findall(r'PAGE\s+\d+\s+START', sample))
        return sample, max(1, pages)

    def _get_dynamic_prompt(self) -> str:
        """Build a system prompt dynamically from registered processors."""
        from src.extraction.base import PROCESSOR_REGISTRY
        
        processors = PROCESSOR_REGISTRY.get_all()
        # Sort for stability
        processors.sort(key=lambda p: p.display_name)
        
        lines = []
        lines.append("You are a document classifier.")
        lines.append("\nAnalyze the document sample and classify it into exactly ONE category.")
        lines.append("\nValid categories:")
        
        valid_display_names = []
        
        for p in processors:
            name = p.display_name
            desc = p.description or ""
            lines.append(f'- "{name}" : {desc}')
            valid_display_names.append(name)
            
        lines.append("\nRules:")
        lines.append("- Choose the SINGLE most appropriate category for the ENTIRE document")
        lines.append("- If document contains multiple types, choose the predominant one")
        lines.append("- If unsure, use 'Other documents' with lower confidence")
        lines.append("\nRespond with ONLY a JSON object:")
        lines.append('{"category": "<category_name>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}')
        lines.append(f"\nThe <category_name> MUST be one of: {json.dumps(valid_display_names)}")

        return "\n".join(lines)
    
    async def classify(self, markdown_content: str, ai_client: Optional[Any] = None) -> ClassificationResult:
        """
        Classify a document based on its markdown content.
        Only uses first few pages for speed.
        """
        sample, pages_analyzed = self._extract_sample(markdown_content)
        
        if not sample.strip():
            return ClassificationResult(
                category=DocumentCategory.OTHER_DOCUMENTS,
                confidence=0.0,
                reasoning="No content to analyze",
                pages_analyzed=0
            )
        
        ai_client = ai_client or self._get_ai_client()
        # Use configured override model, or ask client for its fast model
        model = self._model or getattr(ai_client, "fast_model", None)
        
        user_prompt = f"Classify this document (first {pages_analyzed} pages):\n\n{sample}"

        try:
            prompt = self._get_dynamic_prompt()
            
            result, _ = await ai_client.extract_data(
                model=model,
                system_prompt=prompt,
                user_prompt=user_prompt,
                response_model=ClassificationModel
            )
            
            if result:
                category_str = result.category
                
                # Dynamic matching
                # Find matching processor by display_name
                from src.extraction.base import PROCESSOR_REGISTRY
                all_procs = PROCESSOR_REGISTRY.get_all()
                
                matched_proc = None
                # Exact match
                for p in all_procs:
                    if p.display_name.lower() == category_str.lower():
                        matched_proc = p
                        break
                
                # If no exact match (LLM hallucinated?), default to Other
                if matched_proc:
                    # We return the display name as the category enum value
                    # But DocumentCategory is an Enum...
                    # We should probably return the string directly.
                    # But ClassificationResult types category as DocumentCategory.
                    # This is the tricky part.
                    # Ideally we remove DocumentCategory enum restriction from ClassificationResult.
                    # But downstream code expects it?
                    pass

                # Hack: Just return the string. Python enums at runtime...
                # Actually, we should change ClassificationResult.category to str.
                
                return ClassificationResult(
                    category=matched_proc.display_name if matched_proc else "Other documents",
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    pages_analyzed=pages_analyzed,
                    processor_name=matched_proc.name if matched_proc else "other_docs"
                )
                
                return ClassificationResult(
                    category=category,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    pages_analyzed=pages_analyzed
                )
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
        
        return ClassificationResult(
            category=DocumentCategory.OTHER_DOCUMENTS,
            confidence=0.0,
            reasoning="Classification failed",
            pages_analyzed=pages_analyzed
        )
