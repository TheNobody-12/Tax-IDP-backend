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


async def extract_with_azure_di(local_file_path: str) -> Tuple[List[str], list, List[str], Any]:
    """
    Extract text + tables + DI page images for LLM multimodal classification.

    RETURNS:
        page_texts          -> List[str]
        tables              -> List[dict]
        page_images_base64  -> List[str or None]
        raw_di_result       -> DI response object
    """

    # --------------------------------------------------------
    # 1) Read raw file bytes
    # --------------------------------------------------------
    with open(local_file_path, "rb") as f:
        file_bytes = f.read()

    logger.info("[DI] Running OCR + Layout: %s", local_file_path)

    # --------------------------------------------------------
    # 2) Kick off DI Layout extraction  (Correct usage)
    # --------------------------------------------------------
    # Build a loop-scoped DI client so we don't reuse closed event loops
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient  # local import to bind to current loop
    async with DocumentIntelligenceClient(
        endpoint=di_endpoint,
        credential=credential,
        api_version=di_api_version,
    ) as client:
        poller: LROPoller = await client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=file_bytes,
            content_type="application/pdf"
        )
        result = await poller.result()

    # --------------------------------------------------------
    # 3) Extract PAGE TEXTS
    # --------------------------------------------------------
    page_texts: List[str] = []
    for page in result.pages or []:
        lines = []
        if hasattr(page, "lines"):
            for line in page.lines:
                if getattr(line, "content", None):
                    lines.append(line.content)

        page_texts.append("\n".join(lines) if lines else "")

    # --------------------------------------------------------
    # 4) Extract TABLES
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 5) Extract PAGE IMAGES DIRECTLY FROM DI (SAFE METHOD)
    # --------------------------------------------------------
    page_images_base64: List[str] = []

    for page in result.pages or []:
        # DI page.image.data may exist depending on SDK version
        if hasattr(page, "image") and page.image and getattr(page.image, "data", None):
            try:
                # Decode raw DI image bytes with Pillow
                img = Image.open(BytesIO(page.image.data))
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                page_images_base64.append(b64)

            except Exception as e:
                logger.warning(f"[DI] Could not decode DI image on page: {e}")
                page_images_base64.append(None)
        else:
            # No DI image → fallback None
            page_images_base64.append(None)

    # --------------------------------------------------------------------
    # Sanity alignment: ensure images list matches text list length
    # --------------------------------------------------------------------
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

    return page_texts, tables, page_images_base64, result
