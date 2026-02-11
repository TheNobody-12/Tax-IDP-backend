import os
import asyncio
from pathlib import Path
from typing import Optional
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

class AzureDIClient:
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("AZURE_DI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_DI_KEY")
        
        if not self.endpoint:
            raise RuntimeError("Missing AZURE_DI_ENDPOINT")
        if not self.api_key:
            raise RuntimeError("Missing AZURE_DI_KEY")
            
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version=os.getenv("AZURE_DI_API_VERSION", "2024-11-30")
        )

    async def _analyze_document(self, pdf_path: str):
        with open(pdf_path, "rb") as f:
            poller = await self.client.begin_analyze_document(
                "prebuilt-layout",
                body=f,
            )
        return await poller.result()

    def convert_pdf_to_markdown(
        self,
        pdf_path: str,
        markdown_cache_path: Optional[str] = None,
        reuse_markdown: bool = True
    ) -> str:
        """
        Convert PDF to markdown using Azure Document Intelligence.
        """
        md_path = Path(markdown_cache_path) if markdown_cache_path else Path(pdf_path).with_suffix(".md")
        if reuse_markdown and md_path.exists():
            return md_path.read_text(encoding="utf-8")

        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(self._analyze_document(pdf_path))

        pages = []
        for page in result.pages or []:
            lines = [line.content for line in (page.lines or []) if line.content]
            page_text = "\n".join(lines).strip()
            if not page_text:
                continue
            pages.append(
                f"PAGE {page.page_number} START\n{page_text}\nPAGE {page.page_number} END"
            )

        markdown = "\n---\n".join(pages)
        
        try:
            md_path.write_text(markdown, encoding="utf-8")
        except OSError:
            pass

        return markdown
