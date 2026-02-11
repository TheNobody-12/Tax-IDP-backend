from __future__ import annotations
import os, json, logging, asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import AzureError
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_ACCOUNT_CONN = os.getenv("AzureWebJobsStorage") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
_BRONZE_CONTAINER = os.getenv("AZURE_BRONZE_CONTAINER", "bronze")
_SILVER_CONTAINER = os.getenv("AZURE_SILVER_CONTAINER", "silver")

async def _client() -> BlobServiceClient:
    # Create a fresh client per call to avoid cross-loop reuse issues.
    if not _ACCOUNT_CONN:
        raise RuntimeError("AzureWebJobsStorage / AZURE_STORAGE_CONNECTION_STRING not set")
    return BlobServiceClient.from_connection_string(_ACCOUNT_CONN)


def _bronze_blob_path(doc_id: str) -> str:
    return f"{doc_id}/document.pdf"


def _bronze_di_blob_path(doc_id: str) -> str:
    return f"{doc_id}/di.json"


def _silver_prefix(category: str | None, client_id: str | None, tax_year: int | None) -> str:
    cat = (category or "Uncategorized").replace("/", "_")
    client = (client_id or "UnknownClient").replace("/", "_")
    year = str(tax_year or "UnknownYear")
    return f"{cat}/{client}/{year}"


def silver_blob_paths(doc_id: str, category: str | None, client_id: str | None, tax_year: int | None) -> Dict[str, str]:
    prefix = _silver_prefix(category, client_id, tax_year)
    return {"pdf": f"{prefix}/{doc_id}.pdf", "json": f"{prefix}/{doc_id}.silver.json"}


async def upload_bronze_pdf(doc_id: str, local_pdf: str) -> str:
    async with await _client() as bs:
        blob_path = _bronze_blob_path(doc_id)
        data = Path(local_pdf).read_bytes()
        meta = {"doc_id": doc_id}
        await _retry_upload(bs.get_blob_client(_BRONZE_CONTAINER, blob_path), data, content_type="application/pdf", metadata=meta)
        logger.info("[STORAGE] Uploaded bronze PDF %s/%s", _BRONZE_CONTAINER, blob_path)
        return f"{_BRONZE_CONTAINER}:{blob_path}"


async def upload_bronze_di(doc_id: str, di_doc: Dict[str, Any]) -> str:
    """Upload DI JSON to bronze container for a given doc_id."""
    async with await _client() as bs:
        blob_path = _bronze_di_blob_path(doc_id)
        data = json.dumps(di_doc, indent=2).encode("utf-8")
        meta = {"doc_id": doc_id}
        await _retry_upload(bs.get_blob_client(_BRONZE_CONTAINER, blob_path), data, content_type="application/json", metadata=meta)
        logger.info("[STORAGE] Uploaded bronze DI %s/%s", _BRONZE_CONTAINER, blob_path)
        return f"{_BRONZE_CONTAINER}:{blob_path}"


async def upload_silver_artifacts(doc_id: str, silver_doc: Dict[str, Any], local_pdf: str, client_id: str | None, tax_year: int | None) -> Dict[str, str]:
    async with await _client() as bs:
        category = silver_doc.get("category")
        paths = silver_blob_paths(doc_id, category, client_id, tax_year)
        meta = {"doc_id": doc_id, "client_id": str(client_id or ""), "tax_year": str(tax_year or ""), "category": str(category or "")}
        await _retry_upload(bs.get_blob_client(_SILVER_CONTAINER, paths["pdf"]), Path(local_pdf).read_bytes(), content_type="application/pdf", metadata=meta)
        await _retry_upload(bs.get_blob_client(_SILVER_CONTAINER, paths["json"]), json.dumps(silver_doc, indent=2).encode("utf-8"), content_type="application/json", metadata=meta)
        logger.info("[STORAGE] Uploaded silver artifacts under %s/%s", _SILVER_CONTAINER, paths["pdf"].rsplit("/", 1)[0])
        return paths


async def upload_silver_json(blob_ref: str, silver_doc: Dict[str, Any]) -> str:
    """
    Overwrite an existing silver JSON blob. blob_ref format: 'container:path.json' or just 'path.json' (defaults to silver container).
    """
    if ":" in blob_ref:
        container, blob = blob_ref.split(":", 1)
    else:
        container, blob = _SILVER_CONTAINER, blob_ref
    async with await _client() as bs:
        await _retry_upload(bs.get_blob_client(container, blob), json.dumps(silver_doc, indent=2).encode("utf-8"))
        logger.info("[STORAGE] Updated silver JSON %s/%s", container, blob)
        return f"{container}:{blob}"


async def download_blob_json(container: str, blob: str) -> Optional[Dict[str, Any]]:
    async with await _client() as bs:
        bc = bs.get_blob_client(container, blob)
        try:
            if not await bc.exists():
                return None
            data = await (await bc.download_blob()).readall()
            return json.loads(data.decode("utf-8"))
        except AzureError as e:
            logger.warning("[STORAGE] Failed to download %s/%s: %s", container, blob, e)
            return None


async def download_silver_json(blob_ref: str) -> Optional[Dict[str, Any]]:
    if ":" in blob_ref:
        container, blob = blob_ref.split(":", 1)
    else:
        container, blob = _SILVER_CONTAINER, blob_ref
    return await download_blob_json(container, blob)


async def move_silver_category(doc_id: str, old_category: str | None, new_category: str, client_id: str | None, tax_year: int | None) -> Dict[str, str]:
    async with await _client() as bs:
        old_paths = silver_blob_paths(doc_id, old_category, client_id, tax_year)
        new_paths = silver_blob_paths(doc_id, new_category, client_id, tax_year)
        for key in ("pdf", "json"):
            src = old_paths[key]
            dst = new_paths[key]
            src_bc = bs.get_blob_client(_SILVER_CONTAINER, src)
            if not await src_bc.exists():
                continue
            dst_bc = bs.get_blob_client(_SILVER_CONTAINER, dst)
            await dst_bc.start_copy_from_url(src_bc.url)
            try:
                props = await dst_bc.get_blob_properties()
                if getattr(props, 'copy', None) and props.copy.status != "success":
                    logger.info("[STORAGE] Copy in progress for %s", dst)
            except Exception:
                pass
            try:
                await src_bc.delete_blob()
            except Exception as e:
                logger.warning("[STORAGE] Failed to delete old blob %s: %s", src, e)
        logger.info("[STORAGE] Moved silver blobs to %s", new_category)
        return new_paths


async def _retry_upload(bc, data: bytes, attempts: int = 5, base_delay: float = 0.5, content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None):
    """Upload with exponential backoff and optional content-type/metadata."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            await bc.upload_blob(
                data=data,
                overwrite=True,
                content_settings=(None if content_type is None else ContentSettings(content_type=content_type)),
                metadata=metadata,
            )
            return
        except AzureError as e:
            last_err = e
            delay = base_delay * (2 ** i)
            logger.warning("[STORAGE] Upload attempt %s failed for %s: %s. Retrying in %.2fs", i + 1, bc.blob_name, e, delay)
            try:
                await asyncio.sleep(delay)
            except Exception:
                pass
    if last_err:
        raise last_err


async def delete_document_blobs(doc_id: str, category: str | None, client_id: str | None, tax_year: int | None):
    """Delete all blobs associated with a document (Bronze & Silver)."""
    async with await _client() as bs:
        # Bronze
        for blob_path in [_bronze_blob_path(doc_id), _bronze_di_blob_path(doc_id)]:
            try:
                await bs.get_blob_client(_BRONZE_CONTAINER, blob_path).delete_blob()
                logger.info(f"[STORAGE] Deleted bronze blob {_BRONZE_CONTAINER}/{blob_path}")
            except Exception as e:
                logger.warning(f"[STORAGE] Failed to delete bronze blob {blob_path}: {e}")

        # Silver
        paths = silver_blob_paths(doc_id, category, client_id, tax_year)
        for key, blob_path in paths.items():
            try:
                await bs.get_blob_client(_SILVER_CONTAINER, blob_path).delete_blob()
                logger.info(f"[STORAGE] Deleted silver blob {_SILVER_CONTAINER}/{blob_path}")
            except Exception as e:
                logger.warning(f"[STORAGE] Failed to delete silver blob {blob_path}: {e}")

async def generate_blob_sas_url(container: str, blob_name: str, expiry_minutes: int = 60) -> str:
    """Generate a read-only SAS URL for a blob."""
    if not _ACCOUNT_CONN:
        raise ValueError("Storage connection string not set")
    
    # Parse connection string manually to avoid client overhead for SAS gen
    # Quick hack: assume standard conn string format
    kv = dict(item.split('=', 1) for item in _ACCOUNT_CONN.split(';') if item)
    account_name = kv.get('AccountName')
    account_key = kv.get('AccountKey')

    if not account_name or not account_key:
        # Fallback to loading client if complex conn string (e.g. emulator)
        # But generate_blob_sas needs key. If emulator, it's well known.
        raise ValueError("Could not parse AccountName/AccountKey from connection string")

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes)
    )
    
    # Construct URL
    # Handle emulator specific URL construction if needed, but assuming standard Azure for now:
    # https://<account>.blob.core.windows.net/<container>/<blob>?<sas>
    # If emulator: http://127.0.0.1:10000/<account>/<container>/<blob>?<sas>
    
    if "127.0.0.1" in _ACCOUNT_CONN or "localhost" in _ACCOUNT_CONN:
        # Emulator format guess
        blob_url = f"http://127.0.0.1:10000/{account_name}/{container}/{blob_name}"
    else:
        blob_endpoint = kv.get('BlobEndpoint', f"https://{account_name}.blob.core.windows.net")
        blob_url = f"{blob_endpoint}/{container}/{blob_name}"

    return f"{blob_url}?{sas_token}"
