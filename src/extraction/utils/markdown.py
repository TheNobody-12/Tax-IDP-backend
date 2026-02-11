import re
from typing import Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None

def get_encoder():
    """Get the tiktoken encoder for cl100k_base"""
    if tiktoken:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    return None

def estimate_tokens(text: str, encoder=None) -> int:
    """Estimate the number of tokens in a string"""
    if encoder:
        return len(encoder.encode(text))
    return max(1, len(text) // 4)

def chunk_markdown_by_pages(markdown: str, chunk_size: int = 3000) -> list[str]:
    """
    Split markdown into manageable chunks for LLM processing.
    Attempts to split at logical boundaries (page markers, sections).
    """
    # Split by explicit PAGE markers first; treat each page as its own chunk
    sections = [
        s.strip()
        for s in re.split(r"\n(?=PAGE \d+ START\n)", markdown)
        if s.strip()
    ]
    if len(sections) > 1:
        return sections
        
    # Fall back to page breaks when markers are missing
    sections = [s.strip() for s in markdown.split("---\n")]
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section:
            continue
        section_block = f"{section}\n---\n"
        if len(current_chunk) + len(section_block) < chunk_size:
            current_chunk += section_block
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section_block
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def batch_page_chunks(
    page_chunks: list[str], 
    token_threshold: int = 6000, 
    max_pages_per_batch: int = 8,
    encoder=None
) -> list[str]:
    """Merge small page chunks into larger batches to reduce request overhead"""
    batched = []
    current = []
    current_tokens = 0
    
    for page_chunk in page_chunks:
        page_tokens = estimate_tokens(page_chunk, encoder=encoder)
        
        # If a single page is already large, send it alone
        if page_tokens > token_threshold:
            if current:
                batched.append("\n---\n".join(current))
                current = []
                current_tokens = 0
            batched.append(page_chunk)
            continue
            
        # Check if adding this page exceeds limits
        if current and (
            current_tokens + page_tokens > token_threshold 
            or len(current) >= max_pages_per_batch
        ):
            batched.append("\n---\n".join(current))
            current = [page_chunk]
            current_tokens = page_tokens
        else:
            current.append(page_chunk)
            current_tokens += page_tokens
            
    if current:
        batched.append("\n---\n".join(current))
        
    return batched
