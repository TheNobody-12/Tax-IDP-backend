# src/pipeline/silver_review_store.py

import json
from azure.storage.blob.aio import BlobClient

SILVER_CONTAINER = "silver"


async def save_silver_review_blob(
    storage_conn: str,
    client_id: str,
    tax_year: int,
    doc_id: str,
    silver_payload: dict,
    validation_status: str,
):
    """
    Store a document in:
        silver/review/...   → needs human review
        silver/rejected/... → invalid slips
    """

    if validation_status == "invalid":
        folder = "rejected"
    else:
        folder = "needs_review"

    blob_path = f"{folder}/{client_id}/{tax_year}/{doc_id}.json"

    blob_client = BlobClient.from_connection_string(
        conn_str=storage_conn,
        container_name=SILVER_CONTAINER,
        blob_name=blob_path,
    )

    await blob_client.upload_blob(
        json.dumps(silver_payload, indent=2),
        overwrite=True,
    )

    return blob_path
