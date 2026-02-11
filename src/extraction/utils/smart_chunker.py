"""
Smart chunking for tax documents.
Groups related pages (same vendor, date, account) into logical chunks.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """A logical chunk of document content"""
    chunk_id: str
    page_numbers: List[int]
    content: str
    vendor_hint: Optional[str] = None
    date_hint: Optional[str] = None
    account_hint: Optional[str] = None
    chunk_type: str = "receipt"  # receipt, table, statement, other


def extract_vendor_from_text(text: str) -> Optional[str]:
    """Extract vendor name from page content"""
    vendors = [
        "Vision Care Centre", "Shoppers Drug Mart", "Hilltop Dentistry",
        "John Shute Optometry", "Davidson Shouldice", "Rexall", "Costco Pharmacy",
        "Walmart Pharmacy", "Loblaw", "Meaford Long Term Care"
    ]
    text_upper = text.upper()
    for vendor in vendors:
        if vendor.upper() in text_upper:
            return vendor
    
    # Try to extract from first few lines
    lines = text.split('\n')[:5]
    for line in lines:
        line = line.strip()
        if len(line) > 3 and line[0].isupper() and not line.startswith("PAGE"):
            return line[:50]
    
    return None


def extract_date_from_text(text: str) -> Optional[str]:
    """Extract date from page content"""
    # Common date patterns
    patterns = [
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # DD-MM-YYYY or MM/DD/YYYY
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',     # YYYY-MM-DD
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2}[\s,]+\d{4})',  # Month DD, YYYY
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # DD Month YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_account_from_text(text: str) -> Optional[str]:
    """Extract account number from page content"""
    patterns = [
        r'Account\s*[:#]?\s*(\d+)',
        r'In Account With:\s*(\d+)',
        r'Acct\s*#?\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def parse_page_chunks(markdown: str) -> List[Dict]:
    """Split markdown into page-level chunks with metadata"""
    # Split by PAGE markers
    pages = []
    pattern = r'PAGE\s+(\d+)\s+START\n(.*?)PAGE\s+\d+\s+END'
    
    for match in re.finditer(pattern, markdown, re.DOTALL):
        page_num = int(match.group(1))
        content = match.group(2).strip()
        
        pages.append({
            "page_number": page_num,
            "content": content,
            "vendor": extract_vendor_from_text(content),
            "date": extract_date_from_text(content),
            "account": extract_account_from_text(content),
            "token_estimate": len(content) // 4
        })
    
    return pages


def group_related_pages(pages: List[Dict], max_tokens_per_chunk: int = 8000) -> List[DocumentChunk]:
    """Group related pages into logical chunks"""
    if not pages:
        return []
    
    chunks = []
    current_group = [pages[0]]
    current_tokens = pages[0]["token_estimate"]
    
    for page in pages[1:]:
        # Check if this page should be grouped with current
        should_group = False
        prev = current_group[-1]
        
        # Same vendor
        if page["vendor"] and prev["vendor"] and page["vendor"] == prev["vendor"]:
            should_group = True
        
        # Same account number
        if page["account"] and prev["account"] and page["account"] == prev["account"]:
            should_group = True
        
        # Same date (likely same visit)
        if page["date"] and prev["date"] and page["date"] == prev["date"]:
            should_group = True
        
        # Consecutive pages with no vendor (might be continuation)
        if page["page_number"] == prev["page_number"] + 1 and not page["vendor"]:
            should_group = True
        
        # Check token limit
        if should_group and current_tokens + page["token_estimate"] <= max_tokens_per_chunk:
            current_group.append(page)
            current_tokens += page["token_estimate"]
        else:
            # Finalize current chunk
            chunks.append(_create_chunk_from_pages(current_group, len(chunks) + 1))
            current_group = [page]
            current_tokens = page["token_estimate"]
    
    # Don't forget the last group
    if current_group:
        chunks.append(_create_chunk_from_pages(current_group, len(chunks) + 1))
    
    return chunks


def _create_chunk_from_pages(pages: List[Dict], chunk_index: int) -> DocumentChunk:
    """Create a DocumentChunk from a list of pages"""
    page_numbers = [p["page_number"] for p in pages]
    content_parts = []
    
    for p in pages:
        content_parts.append(f"PAGE {p['page_number']} START\n{p['content']}\nPAGE {p['page_number']} END")
    
    combined_content = "\n---\n".join(content_parts)
    
    # Use hints from first page with data
    vendor = next((p["vendor"] for p in pages if p["vendor"]), None)
    date = next((p["date"] for p in pages if p["date"]), None)
    account = next((p["account"] for p in pages if p["account"]), None)
    
    return DocumentChunk(
        chunk_id=f"chunk_{chunk_index:03d}",
        page_numbers=page_numbers,
        content=combined_content,
        vendor_hint=vendor,
        date_hint=date,
        account_hint=account
    )


def smart_chunk_document(markdown: str, max_tokens_per_chunk: int = 8000) -> List[DocumentChunk]:
    """Main entry point for smart chunking"""
    pages = parse_page_chunks(markdown)
    if not pages:
        # Fallback: treat entire content as one chunk
        return [DocumentChunk(
            chunk_id="chunk_001",
            page_numbers=[1],
            content=markdown,
            chunk_type="other"
        )]
    
    print(f"📄 Parsed {len(pages)} pages from document")
    
    chunks = group_related_pages(pages, max_tokens_per_chunk)
    
    print(f"📦 Grouped into {len(chunks)} semantic chunks:")
    for chunk in chunks:
        pages_str = ",".join(map(str, chunk.page_numbers))
        print(f"   - {chunk.chunk_id}: pages [{pages_str}] - {chunk.vendor_hint or 'Unknown vendor'}")
    
    return chunks
