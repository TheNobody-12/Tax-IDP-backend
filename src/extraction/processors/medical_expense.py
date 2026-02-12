import json
import asyncio
import time
import logging
from typing import List, Optional, Tuple, Dict, Any, Type
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, create_model

from src.extraction.base import BaseProcessor, ProcessorResult, DocumentContext, register_processor
from src.extraction.clients.azure_ai import AzureAIClient
from src.extraction.utils.prompts import get_prompts_for_category
from src.extraction.classifier import DocumentCategory
from src.extraction.utils.export import post_process_expenses
from src.extraction.utils.markdown import chunk_markdown_by_pages, batch_page_chunks, estimate_tokens, get_encoder
from src.extraction.models import MedicalExpense, ClientInfo
from src.extraction.utils.models_dynamic import get_model_for_processor

logger = logging.getLogger(__name__)

@dataclass
class SimpleChunk:
    """A simple token-based chunk of pages"""
    chunk_id: str
    page_numbers: List[int]
    content: str
    token_count: int


class MedicalExtractionResponse(BaseModel):
    """Strict response model for Medical Expenses"""
    model_config = ConfigDict(extra="forbid")
    client_info: Optional[ClientInfo]
    expenses: List[MedicalExpense]


def format_prior_expenses_summary(expenses: List[Dict[str, Any]]) -> str:
    """Format prior expenses as a compact one-line-each summary for context"""
    if not expenses:
        return ""
    
    lines = []
    for exp in expenses:
        amount = exp.get("amount", 0)
        if amount is not None and amount != 0:
            amount_str = f"${amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
        else:
            amount_str = "$0"
        
        payee = exp.get('payee_provider') or 'Unknown'
        lines.append(
            f"Pg{exp.get('page_numbers', '?')}: {payee[:30]} | {amount_str}"
        )
    
    return "\n".join(lines)


class MedicalExpenseProcessor(BaseProcessor):
    """
    Processor for extracting medical expenses from tax documents.
    Uses simple token-based batching (no LLM chunking overhead).
    """
    
    def __init__(
        self,
        ai_client: Optional[AzureAIClient] = None,
        model: Optional[str] = None,
        max_input_tokens_per_minute: int = 25000,
        min_request_interval: float = 2.0,
        batch_token_threshold: int = 10000,
        max_pages_per_batch: int = 10
    ):
        self._ai_client = ai_client
        self._model = model
        self.max_input_tokens_per_minute = max_input_tokens_per_minute
        self.min_request_interval = min_request_interval
        self.batch_token_threshold = batch_token_threshold
        self.max_pages_per_batch = max_pages_per_batch
        self.encoder = None
        
        self.tokens_used_window = 0
        self.window_start = time.monotonic()
        self.last_request_time = 0.0
    
    @property
    def name(self) -> str:
        return "medical_expense"

    @property
    def display_name(self) -> str:
        return "Medical expenses"
    
    @property
    def description(self) -> str:
        return "Extract medical expenses for Canadian tax credits"
    
    @property
    def output_format(self) -> str:
        return "csv"
    
    def _get_ai_client(self) -> AzureAIClient:
        if self._ai_client is None:
            self._ai_client = AzureAIClient()
        return self._ai_client
    
    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = get_encoder()
        return self.encoder
    
    def _create_chunks(self, markdown_content: str) -> List[SimpleChunk]:
        """Create chunks using token-based batching (fast, no LLM calls)"""
        encoder = self._get_encoder()
        
        # Split by pages
        page_chunks = chunk_markdown_by_pages(markdown_content)
        
        # Batch by tokens
        batched = batch_page_chunks(
            page_chunks,
            token_threshold=self.batch_token_threshold,
            max_pages_per_batch=self.max_pages_per_batch,
            encoder=encoder
        )
        
        # Convert to SimpleChunk objects
        chunks = []
        for i, batch_content in enumerate(batched):
            import re
            # Regex to find all page markers in this batch
            page_nums = [int(m) for m in re.findall(r'PAGE\s+(\d+)\s+START', batch_content, re.IGNORECASE)]
            token_count = estimate_tokens(batch_content, encoder)
            
            # Fallback for page numbers if regex fails (should not happen if markdown.py is consistent)
            if not page_nums:
                 # Try finding just numbers near "PAGE"
                 soft_matches = re.findall(r'PAGE\s*(\d+)', batch_content, re.IGNORECASE)
                 if soft_matches:
                     page_nums = [int(m) for m in soft_matches]
            
            chunks.append(SimpleChunk(
                chunk_id=f"chunk_{i+1:03d}",
                page_numbers=page_nums if page_nums else [i+1],
                content=batch_content,
                token_count=token_count
            ))
        
        return chunks
    
        return chunks
    
    async def process(self, context: DocumentContext, output_dir: str, ai_client: Optional[Any] = None) -> ProcessorResult:
        """Process document for medical expenses"""
        if ai_client:
            self._ai_client = ai_client
        else:
            ai_client = self._get_ai_client()
        encoder = self._get_encoder()
        
        # Create chunks using simple token-based batching
        chunks = self._create_chunks(context.markdown_content)
        # Verify page extraction
        for i, c in enumerate(chunks):
            if not c.page_numbers:
                # Fallback: if regex failed, assume sequential?
                # But we don't know page count per chunk easily.
                # Actually, simple chunking is by page FIRST.
                # So if batching works, we just need to preserve original pages.
                # Let's trust regex for now but log warning.
                logger.warning(f"Chunk {i} has no page numbers extracted! Content start: {c.content[:50]}")
        
        logger.info(f"Processing {len(chunks)} chunks for medical expenses")
        
        all_expenses: List[MedicalExpense] = []
        all_expenses_dicts: List[Dict[str, Any]] = []
        client_info: Optional[ClientInfo] = None
        errors: List[str] = []
        
        results_lock = asyncio.Lock()
        
        # Resolve dynamic model once
        ItemModel = get_model_for_processor(self.name, MedicalExpense)

        # Use local lock for this request's loop
        rate_limit_lock = asyncio.Lock()

        async def process_single_chunk(i: int, chunk: SimpleChunk):
            nonlocal client_info
            
            async with results_lock:
                current_context_dicts = list(all_expenses_dicts)
            
            prior_context = ""
            if current_context_dicts:
                prior_summary = format_prior_expenses_summary(current_context_dicts)
                prior_context = f"\n\nPRIOR EXPENSES ({len(current_context_dicts)} found so far):\n{prior_summary}"
            
            # Pass ItemModel to prompt factory
            system_prompt, user_prompt = get_prompts_for_category(
                DocumentCategory.MEDICAL_EXPENSES,
                chunk.content,
                prior_context=prior_context,
                model_class=ItemModel
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
                        logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                        await asyncio.sleep(sleep_time)
                    self.window_start = time.monotonic()
                    self.tokens_used_window = 0
                
                since_last = time.monotonic() - self.last_request_time
                if since_last < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - since_last)
                
                self.tokens_used_window += chunk_tokens
                self.last_request_time = time.monotonic()
            
            logger.info(f"Chunk {i}/{len(chunks)} start processing")
            
            extracted_data = await self._extract_with_retry(
                ai_client, system_prompt, user_prompt, i, errors, ItemModel=ItemModel
            )
            
            if extracted_data:
                async with results_lock:
                    # Backfill page numbers if missing
                    for exp in extracted_data.expenses:
                        if not exp.page_numbers:
                            exp.page_numbers = list(chunk.page_numbers)
                            
                    all_expenses.extend(extracted_data.expenses)
                    for exp in extracted_data.expenses:
                        all_expenses_dicts.append(exp.model_dump())
                    if not client_info and extracted_data.client_info:
                        client_info = extracted_data.client_info
                logger.info(f"Chunk {i} complete: {len(extracted_data.expenses)} expenses")
            
            return len(extracted_data.expenses) if extracted_data else 0

        # Run chunks sequentially for medical to maintain context? 
        # Actually, they can run in parallel, but context helps.
        # Let's run with limited concurrency to balance speed and context.
        # Identify provider for logging
        provider_name = self._ai_client.__class__.__name__.replace("AIClient", "")

        sem = asyncio.Semaphore(1)
        async def sem_process(i, chunk):
            async with sem:
                logger.info(f"[{provider_name}] Starting extraction for chunk {i}/{len(chunks)}...")
                count = await process_single_chunk(i, chunk)
                logger.info(f"[{provider_name}] Completed chunk {i}/{len(chunks)}. Found {count} expenses.")
                return count
        
        tasks = [sem_process(i, chunk) for i, chunk in enumerate(chunks, 1)]
        await asyncio.gather(*tasks)
        
        logger.info(f"[{provider_name}] All chunks processed. Total expenses found: {len(all_expenses)}")
        
        # Post-process
        processed = post_process_expenses(all_expenses)
        
        return ProcessorResult(
            processor_name=self.name,
            items_extracted=len(processed),
            output_file=None,
            data=[exp.model_dump() for exp in processed],
            client_info=client_info.model_dump() if client_info else None,
            errors=errors
        )
    
    async def _extract_with_retry(
        self,
        ai_client: AzureAIClient,
        system_prompt: str,
        user_prompt: str,
        chunk_index: int,
        errors: List[str],
        ItemModel: Type[BaseModel],
        max_retries: int = 3
    ) -> Optional[MedicalExtractionResponse]:
        """Extract with retry logic"""
        
        if ItemModel is MedicalExpense:
            ResponseModel = MedicalExtractionResponse
        else:
            # Create dynamic wrapper
            ResponseModel = create_model(
                "MedicalExtractionResponseDynamic",
                client_info=(Optional[ClientInfo], None),
                expenses=(List[ItemModel], ...),
                __config__=ConfigDict(extra="forbid")
            )

        for attempt in range(max_retries):
            try:
                result, _ = await ai_client.extract_data(
                    model=self._model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=ResponseModel
                )
                return result
                
            except Exception as e:
                error_str = str(e)
                if "content_filter" in error_str.lower():
                    logger.warning(f"Chunk {chunk_index}: Content filter trigger, skipping")
                    return None
                elif attempt < max_retries - 1:
                    logger.warning(f"Chunk {chunk_index}: Error '{error_str[:50]}', retry {attempt + 1}")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"Chunk {chunk_index} failed after {max_retries} attempts")
                    errors.append(f"Chunk {chunk_index}: {error_str[:100]}")
                    return None
        
        return None


# Create and register the processor
_medical_expense_processor = MedicalExpenseProcessor()
register_processor(_medical_expense_processor)
