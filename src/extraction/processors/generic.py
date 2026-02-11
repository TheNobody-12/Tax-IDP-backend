"""
Generic Expense Processor for various tax document types.
Handles simple token-based processing for arbitrary schemas.
"""

import time
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any, Type, Union
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict

from src.extraction.base import BaseProcessor, ProcessorResult, DocumentContext, register_processor
from src.extraction.clients.azure_ai import AzureAIClient
from src.extraction.utils.prompts import get_prompts_for_category
from src.extraction.classifier import DocumentCategory
from src.extraction.utils.markdown import chunk_markdown_by_pages, batch_page_chunks, estimate_tokens, get_encoder
from src.extraction.models import (
    ClientInfo, 
    ChildcareExpense, 
    Donation, 
    FHSAContribution, 
    PropertyTax, 
    RentReceipt, 
    RRSPContribution, 
    UnionDue,
    TaxSlip,
    OtherDocument
)

from src.extraction.utils.models_dynamic import get_model_for_processor

logger = logging.getLogger(__name__)

@dataclass
class SimpleChunk:
    chunk_id: str
    page_numbers: List[int]
    content: str
    token_count: int

def get_response_model(item_model: Type[BaseModel]):
    """Dynamically create a response model for the specific item type"""
    class DynamicResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        client_info: Optional[ClientInfo]
        expenses: List[item_model]
    return DynamicResponse

class GenericExpenseProcessor(BaseProcessor):
    """
    Generic processor for various tax document types.
    """
    
    def __init__(
        self,
        category: DocumentCategory,
        item_model: Type[BaseModel],
        output_name: str,
        ai_client: Optional[AzureAIClient] = None,
        model: Optional[str] = None
    ):
        self._category = category
        self._item_model = item_model
        self._output_name = output_name
        self._ai_client = ai_client
        self._model = model
        
        # Performance settings
        self.max_input_tokens_per_minute = 25000
        self.min_request_interval = 0.2
        self.batch_token_threshold = 10000
        
        # For Slips, force 1 page per batch to ensure no items are missed
        if self._category == DocumentCategory.SLIPS:
            self.max_pages_per_batch = 1
        else:
            self.max_pages_per_batch = 10
            
        self.encoder = None
        self.tokens_used_window = 0
        self.window_start = time.monotonic()
        self.last_request_time = 0.0

    @property
    def name(self) -> str:
        return self._output_name

    @property
    def display_name(self) -> str:
        return self._category.value
    
    @property
    def description(self) -> str:
        return f"Extract {self._category.value}"
    
    @property
    def output_format(self) -> str:
        return "csv"

    def _get_ai_client(self) -> AzureAIClient:
        if self._ai_client is None:
            self._ai_client = AzureAIClient()
        return self._ai_client
    
    def _create_chunks(self, markdown_content: str) -> List[SimpleChunk]:
        if self.encoder is None:
            from src.extraction.utils.markdown import get_encoder
            self.encoder = get_encoder()
            
        page_chunks = chunk_markdown_by_pages(markdown_content)
        batched = batch_page_chunks(
            page_chunks,
            token_threshold=self.batch_token_threshold,
            max_pages_per_batch=self.max_pages_per_batch,
            encoder=self.encoder
        )
        
        chunks = []
        for i, batch_content in enumerate(batched):
            import re
            page_nums = [int(m) for m in re.findall(r'PAGE\s+(\d+)\s+START', batch_content)]
            token_count = estimate_tokens(batch_content, self.encoder)
            chunks.append(SimpleChunk(
                chunk_id=f"chunk_{i+1:03d}",
                page_numbers=page_nums if page_nums else [i+1],
                content=batch_content,
                token_count=token_count
            ))
        return chunks

    async def process(self, context: DocumentContext, output_dir: str, ai_client: Optional[Any] = None) -> ProcessorResult:
        ai_client = ai_client or self._get_ai_client()
        encoder = self.encoder or get_encoder()
        
        chunks = self._create_chunks(context.markdown_content)
        logger.info(f"Processing {len(chunks)} chunks for {self._category.value}")
        
        all_items: List[BaseModel] = []
        client_info: Optional[ClientInfo] = None
        errors: List[str] = []
        
        results_lock = asyncio.Lock()
        
        # Dynamic model loading
        DynamicItemModel = get_model_for_processor(self._output_name, self._item_model)
        ResponseModel = get_response_model(DynamicItemModel)

        # Use local lock for this request's loop
        rate_limit_lock = asyncio.Lock()

        async def process_chunk(i: int, chunk: SimpleChunk):
            nonlocal client_info
            
            async with results_lock:
                 prior_context = f"Found {len(all_items)} items so far." if all_items else ""

            system_prompt, user_prompt = get_prompts_for_category(
                self._category,
                chunk.content,
                prior_context=prior_context,
                model_class=DynamicItemModel
            )
            
            chunk_tokens = estimate_tokens(system_prompt + user_prompt, encoder) + 1024
            
            async with rate_limit_lock:
                now = time.monotonic()
                if now - self.window_start >= 60:
                    self.window_start = now
                    self.tokens_used_window = 0
                
                if self.tokens_used_window + chunk_tokens > self.max_input_tokens_per_minute:
                    sleep_time = 60 - (now - self.window_start)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    self.window_start = time.monotonic()
                    self.tokens_used_window = 0
                
                since_last = time.monotonic() - self.last_request_time
                if since_last < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - since_last)
                
                self.tokens_used_window += chunk_tokens
                self.last_request_time = time.monotonic()
            
            result, _ = await ai_client.extract_data(
                model=self._model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ResponseModel
            )
            
            if result:
                async with results_lock:
                    for item in result.expenses:
                        if not getattr(item, "page_numbers", None):
                            # Default to chunk pages if model supports it
                            if hasattr(item, "page_numbers"):
                                item.page_numbers = list(chunk.page_numbers)
                    all_items.extend(result.expenses)
                    if not client_info and result.client_info:
                        client_info = result.client_info
                logger.debug(f"Chunk {i} complete: {len(result.expenses)} items")
            else:
                errors.append(f"Chunk {i} failed")

        sem = asyncio.Semaphore(4)
        async def sem_process(i, chunk):
            async with sem:
                await process_chunk(i, chunk)
        
        tasks = [sem_process(i, chunk) for i, chunk in enumerate(chunks, 1)]
        await asyncio.gather(*tasks)

        return ProcessorResult(
            processor_name=self.name,
            items_extracted=len(all_items),
            output_file=None,
            data=[item.model_dump() for item in all_items],
            client_info=client_info.model_dump() if client_info else None,
            errors=errors
        )

# Register Processors
def register_generic_processors():
    processors = [
        (DocumentCategory.CHILD_CARE_EXPENSES, ChildcareExpense, "child_care"),
        (DocumentCategory.CHARITABLE_DONATIONS, Donation, "donation"),
        (DocumentCategory.POLITICAL_DONATIONS, Donation, "political_donation"),
        (DocumentCategory.FHSA_CONTRIBUTION, FHSAContribution, "fhsa_contribution"),
        (DocumentCategory.PROPERTY_TAX_RECEIPT, PropertyTax, "property_tax"),
        (DocumentCategory.RENT_RECEIPT, RentReceipt, "rent_receipt"),
        (DocumentCategory.RRSP_CONTRIBUTION, RRSPContribution, "rrsp_contribution"),
        (DocumentCategory.UNION_PROFESSIONAL_DUES, UnionDue, "union_dues"),
        (DocumentCategory.SLIPS, TaxSlip, "slips"),
        (DocumentCategory.OTHER_DOCUMENTS, OtherDocument, "other_docs"),
    ]
    
    for cat, model, name in processors:
        p = GenericExpenseProcessor(cat, model, name)
        register_processor(p)
