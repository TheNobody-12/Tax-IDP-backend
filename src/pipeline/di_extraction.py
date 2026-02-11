# di_extraction.py
import os
from typing import List, Tuple, Any
import logging

from azure.core.polling import LROPoller
from src.config.azure_clients import credential, di_endpoint, di_api_version

logger = logging.getLogger(__name__)
logger.info("[DI] di_extraction module loaded (endpoint=%s api_version=%s)", di_endpoint, di_api_version)

import base64
from io import BytesIO
from PIL import Image


async def extract_with_azure_di(local_file_path: str, file_url: str | None = None) -> Tuple[List[str], list, List[str], Any, str]:
    """
    Extract text + tables + DI page images for LLM multimodal classification.
    Includes explicit PAGE X START/END markers for downstream processing.

    RETURNS:
        page_texts          -> List[str] (with markers)
        tables              -> List[dict]
        page_images_base64  -> List[str or None]
        raw_di_result       -> DI response object
        markdown_content    -> Full document markdown
    """

    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    async with DocumentIntelligenceClient(
        endpoint=di_endpoint,
        credential=credential,
        api_version=di_api_version,
    ) as client:
        if file_url:
            logger.info("[DI] Analyzing via URL: %s", file_url)
            # Use AnalyzeDocumentRequest if sdk supports it, or kwargs or urlSource param
            # For recent SDKs, analyze_request or similar is used.
            # We try standard pattern:
            # Error "missing 1 required positional argument: 'body'" suggests body is required.
            # When using URL, the body should be the request model containing urlSource.
            poller: LROPoller = await client.begin_analyze_document(
                model_id="prebuilt-layout",
                body={"urlSource": file_url}
                # content_type might be inferred or needed as application/json, but let's try just body first or with kwargs if needed.
                # Actually, standard behavior for JSON body in Azure SDK is usually enough.
            )
        else:
            if not local_file_path:
                raise ValueError("Either local_file_path or file_url must be provided")

            with open(local_file_path, "rb") as f:
                file_bytes = f.read()
            
            logger.info("[DI] Running OCR + Layout: %s", local_file_path)
            poller: LROPoller = await client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=file_bytes,
                content_type="application/pdf"
            )
            
        result = await poller.result()

    # Extract PAGE TEXTS with markers
    page_texts: List[str] = []
    markdown_pages: List[str] = []
    
    for page in result.pages or []:
        lines = []
        if hasattr(page, "lines"):
            for line in page.lines:
                if getattr(line, "content", None):
                    lines.append(line.content)

        page_text = "\n".join(lines) if lines else ""
        marker_text = f"PAGE {page.page_number} START\n{page_text}\nPAGE {page.page_number} END"
        page_texts.append(marker_text)
        markdown_pages.append(marker_text)

    markdown_content = "\n\n---\n\n".join(markdown_pages)

    # Extract TABLES
    tables = []
    for table in getattr(result, "tables", []) or []:
        t = {
            "rowCount": table.row_count,
            "columnCount": table.column_count,
            "cells": []
        }
        for cell in table.cells or []:
            t["cells"].append({
                "rowIndex": getattr(cell, "row_index", None),
                "columnIndex": getattr(cell, "column_index", None),
                "content": getattr(cell, "content", None),
            })
        tables.append(t)

    # Extract PAGE IMAGES
    page_images_base64: List[str] = []
    for page in result.pages or []:
        if hasattr(page, "image") and page.image and getattr(page.image, "data", None):
            try:
                img = Image.open(BytesIO(page.image.data))
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                page_images_base64.append(b64)
            except Exception as e:
                logger.warning(f"[DI] Could not decode DI image on page: {e}")
                page_images_base64.append(None)
        else:
            page_images_base64.append(None)

    if len(page_images_base64) < len(page_texts):
        page_images_base64.extend([None] * (len(page_texts) - len(page_images_base64)))
    elif len(page_images_base64) > len(page_texts):
        page_images_base64 = page_images_base64[:len(page_texts)]

    logger.info(
        "[DI] Extracted %d pages, %d tables, %d images",
        len(page_texts),
        len(tables),
        len(page_images_base64),
    )

    return page_texts, tables, page_images_base64, result, markdown_content
