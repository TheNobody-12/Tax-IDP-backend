# src/pipeline/bronze_store.py
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from azure.storage.blob.aio import BlobClient


BRONZE_CONTAINER = "bronze"


@dataclass
class BronzeArtifacts:
    client_id: str
    tax_year: int
    file_name: str
    doc_id: str
    raw_pdf_blob_path: str
    raw_json_path: str
    raw_ocr_path: str
    raw_layout_path: str
    manifest_path: str


def _safe_json(obj: Any) -> Any:
    """Make anything JSON-serializable."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))


async def write_bronze(
    *,
    storage_conn: str,
    client_id: str,
    tax_year: int,
    file_name: str,
    local_pdf_path: str,
    page_texts: List[str],
    tables: List[Any],
    page_images_base64: List[str],
    di_result: Any,
    file_size_bytes: int,
) -> BronzeArtifacts:
    """
    Given DI output, write all Bronze artifacts to the 'bronze' container.
    Assumes the raw PDF is already stored at bronze/{clientId}/{taxYear}/{fileName}.
    """
    # Use filename without extension as doc_id
    name_without_ext = file_name.rsplit(".", 1)[0]
    doc_id = name_without_ext

    raw_pdf_blob_path = f"{client_id}/{tax_year}/{file_name}"
    raw_json_path = f"raw_json/{client_id}/{tax_year}/{doc_id}.raw.json"
    raw_ocr_path = f"raw_ocr/{client_id}/{tax_year}/{doc_id}.ocr.json"
    raw_layout_path = f"raw_layout/{client_id}/{tax_year}/{doc_id}.layout.json"
    manifest_path = f"manifest/{client_id}/{tax_year}/{doc_id}.manifest.json"

    async def _upload_json(path: str, payload: Dict[str, Any]):
        blob = BlobClient.from_connection_string(
            conn_str=storage_conn,
            container_name=BRONZE_CONTAINER,
            blob_name=path,
        )
        await blob.upload_blob(
            json.dumps(_safe_json(payload), indent=2),
            overwrite=True,
        )

    # Raw JSON aggregates everything we might ever need
    if hasattr(di_result, "to_dict"):
        di_dict = di_result.to_dict()
    else:
        di_dict = _safe_json(di_result)

    raw_json = {
        "docId": doc_id,
        "clientId": client_id,
        "taxYear": tax_year,
        "fileName": file_name,
        "page_texts": page_texts,
        "tables": tables,
        "page_images_base64": page_images_base64,
        "di_result": di_dict,
    }

    raw_ocr = {
        "docId": doc_id,
        "clientId": client_id,
        "taxYear": tax_year,
        "page_texts": page_texts,
        "content": di_dict.get("content") if isinstance(di_dict, dict) else None,
        "pages": di_dict.get("pages") if isinstance(di_dict, dict) else None,
    }

    raw_layout = {
        "docId": doc_id,
        "clientId": client_id,
        "taxYear": tax_year,
        "tables": tables,
        "pages": di_dict.get("pages") if isinstance(di_dict, dict) else None,
    }

    await _upload_json(raw_json_path, raw_json)
    await _upload_json(raw_ocr_path, raw_ocr)
    await _upload_json(raw_layout_path, raw_layout)

    manifest = {
        "docId": doc_id,
        "clientId": client_id,
        "taxYear": tax_year,
        "fileName": file_name,
        "rawPdfPath": raw_pdf_blob_path,
        "rawJsonPath": raw_json_path,
        "rawOcrPath": raw_ocr_path,
        "rawLayoutPath": raw_layout_path,
        "status": "Extracted",
        "fileSizeBytes": file_size_bytes,
        "extractedAt": datetime.utcnow().isoformat(),
        "modelVersion": di_dict.get("modelVersion")
        if isinstance(di_dict, dict)
        else di_dict.get("model_version")
        if isinstance(di_dict, dict)
        else None,
    }

    await _upload_json(manifest_path, manifest)

    return BronzeArtifacts(
        client_id=client_id,
        tax_year=tax_year,
        file_name=file_name,
        doc_id=doc_id,
        raw_pdf_blob_path=raw_pdf_blob_path,
        raw_json_path=raw_json_path,
        raw_ocr_path=raw_ocr_path,
        raw_layout_path=raw_layout_path,
        manifest_path=manifest_path,
    )
