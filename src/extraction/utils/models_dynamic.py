
import json
import logging
import datetime
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, create_model

from src.extraction.processors.dynamic import create_pydantic_model_from_schema

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_TTL = datetime.timedelta(minutes=5)

def get_model_for_processor(processor_name: str, default_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Get the Pydantic model for a processor.
    Checks DB for SchemaDefinition overrides.
    If override exists, creates a dynamic model.
    Otherwise returns default_model.
    """
    global _MODEL_CACHE
    
    now = datetime.datetime.now()
    if processor_name in _MODEL_CACHE:
        entry = _MODEL_CACHE[processor_name]
        if now - entry["ts"] < _CACHE_TTL:
            return entry["model"]

    try:
        from src.pipeline.db import get_sql_conn
    except ImportError:
        try:
            from bookkeeper.src.pipeline.db import get_sql_conn
        except ImportError:
            return default_model

    try:
        conn = get_sql_conn()
        cur = conn.cursor()
        cur.execute("SELECT SchemaDefinition FROM gold.ProcessorConfig WHERE Name = ?", (processor_name,))
        row = cur.fetchone()
        conn.close()

        if row and row[0]:
            schema_json = row[0]
            # Verify if it differs from default?
            # actually, if present, we trust it matches the user intent.
            # But we should only use it if it's "valid".
            
            try:
                # We need to construct a model that mimics the default one but with potentially different fields.
                # However, our dynamic processor builder creates a flattened model.
                # Our system processors usually expect specific structures. 
                # Specifically, they wrap it in StructuredExtraction.
                # So we just need the INNER model.
                
                # Check if schema is empty/default
                s = json.loads(schema_json)
                if not s.get("fields"):
                     # No fields defined, usage default
                     model = default_model
                else:
                    dynamic_model = create_pydantic_model_from_schema(f"{processor_name}Dynamic", schema_json)
                    if dynamic_model:
                        model = dynamic_model
                    else:
                        model = default_model
            except Exception as e:
                logger.warning(f"Failed to parse schema override for {processor_name}: {e}")
                model = default_model
        else:
            model = default_model
            
        _MODEL_CACHE[processor_name] = {
            "ts": now,
            "model": model
        }
        return model
        
    except Exception as e:
        logger.warning(f"Error loading model override for {processor_name}: {e}")
        return default_model

def clear_model_cache(processor_name: Optional[str] = None):
    """Clear the model cache, optionally for a specific processor only."""
    global _MODEL_CACHE
    if processor_name:
        if processor_name in _MODEL_CACHE:
            del _MODEL_CACHE[processor_name]
            logger.info(f"Cleared model cache for {processor_name}")
    else:
        _MODEL_CACHE.clear()
        logger.info("Cleared all model cache")
