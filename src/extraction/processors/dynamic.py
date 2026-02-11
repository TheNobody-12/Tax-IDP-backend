
import json
import logging
import os
from typing import List, Dict, Any, Optional
from pydantic import create_model, BaseModel, Field

from src.extraction.base import BaseProcessor, ProcessorResult, DocumentContext, register_processor, PROCESSOR_REGISTRY
from src.extraction.clients.azure_ai import AzureAIClient
from src.extraction.utils.markdown import chunk_markdown_by_pages
from src.extraction.utils.prompts import (
    build_medical_prompts,
    build_childcare_prompts,
    build_donation_prompts,
    build_fhsa_prompts,
    build_slips_prompts,
    build_property_tax_prompts,
    build_rent_prompts,
    build_rrsp_prompts,
    build_union_dues_prompts,
    build_other_document_prompts
)
from src.extraction.models import (
    MedicalExpense, ChildcareExpense, Donation, FHSAContribution, 
    TaxSlip, PropertyTax, RentReceipt, RRSPContribution, UnionDue, OtherDocument
)

logger = logging.getLogger(__name__)

def create_pydantic_model_from_schema(name: str, schema_json: str):
    """
    Dynamically create a Pydantic model from a JSON schema definition.
    Expected JSON format:
    {
        "fields": [
            {"name": "field_name", "type": "string|float|int|bool", "description": "desc"}
        ]
    }
    """
    try:
        schema = json.loads(schema_json)
        fields = {}
        for f in schema.get("fields", []):
            fname = f.get("name")
            ftype_str = f.get("type", "string").lower()
            fdesc = f.get("description", "")
            
            if ftype_str == "float" or ftype_str == "number":
                ftype = (Optional[float], Field(default=None, description=fdesc))
            elif ftype_str == "int" or ftype_str == "integer":
                ftype = (Optional[int], Field(default=None, description=fdesc))
            elif ftype_str == "bool" or ftype_str == "boolean":
                ftype = (Optional[bool], Field(default=None, description=fdesc))
            elif ftype_str == "array":
                # simplified handling: list of strings or list of any
                if fname == "page_numbers":
                    ftype = (Optional[List[int]], Field(default_factory=list, description=fdesc))
                else:
                    ftype = (Optional[List[str]], Field(default_factory=list, description=fdesc))
            else:
                ftype = (Optional[str], Field(default=None, description=fdesc))
            
            fields[fname] = ftype
            
        return create_model(f"{name}Model", **fields)
    except Exception as e:
        logger.error(f"Failed to create model for {name}: {e}")
        return None

class DynamicProcessor(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._name = config["Name"]
        self._description = config.get("Description", "")
        self._system_prompt = config.get("SystemPrompt", "")
        self._user_prompt = config.get("UserPrompt", "")
        self._schema_def = config.get("SchemaDefinition", "{}")
        self._model = create_pydantic_model_from_schema(self._name, self._schema_def)
        
    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._config.get("DisplayName") or self._name

    @property
    def description(self) -> str:
        return self._description or "Dynamic Processor"

    @property
    def output_format(self) -> str:
        return "json"

    async def process(self, context: DocumentContext, output_dir: str, ai_client: Optional[Any] = None) -> ProcessorResult:
        logger.info(f"Dynamic processing for {self.name}")
        client = ai_client or AzureAIClient()
        
        # Simple processing: treat whole doc as one or split by pages?
        # For simplicity, let's just attempt extraction on the first 10 pages or reasonable chunk
        # In a real dynamic processor, we might want configuration for chunking strategy.
        # Fallback: simple page chunking.
        
        chunks = chunk_markdown_by_pages(context.markdown_content)
        # Limit to first 20 pages for now to avoid explosion
        chunks = chunks[:20]
        
        all_data = []
        errors = []
        
        # We need a response container
        # Since extract_data expects a model that HAS a list of items or single item.
        # Let's wrap our dynamic model in a list if possible?
        # Actually AzureAIClient.extract_data expects 'response_model' to be the outer container.
        
        # We need to construct a container model: class Response(BaseModel): items: List[DynamicModel]
        ContainerModel = create_model(
            f"{self._name}Response", 
            items=(List[self._model], Field(default_factory=list))
        )

        for i, chunk in enumerate(chunks):
            user_msg = self._user_prompt.format(chunk_content=chunk, context=chunk)
            try:
                result, _ = await client.extract_data(
                    model=None, # default
                    system_prompt=self._system_prompt,
                    user_prompt=user_msg,
                    response_model=ContainerModel
                )
                if result and result.items:
                    for item in result.items:
                        all_data.append(item.model_dump())
            except Exception as e:
                errors.append(f"Chunk {i} failed: {e}")

        # Save result
        output_file = os.path.join(output_dir, f"{self.name}_{context.doc_id}.json")
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)

        return ProcessorResult(
            processor_name=self.name,
            items_extracted=len(all_data),
            output_file=output_file,
            data=all_data,
            errors=errors
        )

def load_dynamic_processors():
    """Load enabled dynamic processors from DB and register them."""
    try:
        from src.pipeline.db import get_sql_conn
        conn = get_sql_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM gold.ProcessorConfig WHERE Enabled = 1 AND IsSystem = 0")
        columns = [column[0] for column in cur.description]
        rows = cur.fetchall()
        conn.close()
        
        count = 0
        for row in rows:
            config = dict(zip(columns, row))
            try:
                proc = DynamicProcessor(config)
                PROCESSOR_REGISTRY.register(proc)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load dynamic processor {config.get('Name')}: {e}")
        
        logger.info(f"Loaded {count} dynamic processors")
        
    except Exception as e:
        logger.error(f"Failed to load dynamic processors: {e}")

def seed_system_processors():
    """Seed the DB with built-in processors if they don't exist."""
    try:
        from src.pipeline.db import get_sql_conn
        conn = get_sql_conn()
        cur = conn.cursor()
        
        # Helper to get schema
        def get_schema_json(model_class):
            schema = model_class.model_json_schema()
            props = schema.get("properties", {})
            fields = []
            for name, prop in props.items():
                p_type = prop.get("type", "string")
                if "anyOf" in prop:
                     for t in prop["anyOf"]:
                         if t.get("type") != "null":
                             p_type = t.get("type")
                             break
                fields.append({
                    "name": name,
                    "type": p_type,
                    "description": prop.get("description", "") or prop.get("title", "")
                })
            return json.dumps({"fields": fields})

        dummy_ctx = "{chunk_content}"
        dummy_prior = "{prior_context}"
        
        processors_data = [
            ("medical_expense", "Medical Expenses", build_medical_prompts, True, MedicalExpense),
            ("child_care", "Child Care", build_childcare_prompts, False, ChildcareExpense),
            ("donation", "Charitable Donations", build_donation_prompts, False, Donation),
            ("fhsa_contribution", "FHSA Contribution", build_fhsa_prompts, False, FHSAContribution),
            ("slips", "Tax Slips", build_slips_prompts, False, TaxSlip),
            ("property_tax", "Property Tax", build_property_tax_prompts, False, PropertyTax),
            ("rent_receipt", "Rent Receipt", build_rent_prompts, False, RentReceipt),
            ("rrsp_contribution", "RRSP Contribution", build_rrsp_prompts, False, RRSPContribution),
            ("union_dues", "Union Dues", build_union_dues_prompts, False, UnionDue),
            ("other_docs", "Other Documents", build_other_document_prompts, False, OtherDocument),
        ]
        
        for name, display, func, has_prior, model_cls in processors_data:
            try:
                if has_prior:
                    sys_p, user_p = func(dummy_ctx, dummy_prior)
                else:
                    sys_p, user_p = func(dummy_ctx)
                
                schema_json = get_schema_json(model_cls)
                    
                cur.execute("""
                    MERGE gold.ProcessorConfig AS target
                    USING (SELECT ? AS Name) AS source
                    ON (target.Name = source.Name)
                    WHEN MATCHED THEN
                        UPDATE SET 
                            DisplayName = ?, 
                            IsSystem = 1,
                            SystemPrompt = ?,
                            UserPrompt = ?,
                            SchemaDefinition = ?,
                            UpdatedAt = GETUTCDATE()
                    WHEN NOT MATCHED THEN
                        INSERT (Name, DisplayName, Description, SystemPrompt, UserPrompt, SchemaDefinition, Enabled, IsSystem)
                        VALUES (?, ?, ?, ?, ?, ?, 1, 1);
                """, (name, display, sys_p, user_p, schema_json, name, display, f"Built-in processor for {display}", sys_p, user_p, schema_json))
            except Exception as e:
                logger.error(f"Error seeding {name}: {e}")
                
        conn.commit()
        conn.close()
        logger.info("[DB] System processors seed check complete")
    except Exception as e:
        logger.error(f"Failed to seed system processors: {e}")
