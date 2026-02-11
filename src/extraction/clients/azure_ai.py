import os
import json
import asyncio
import logging
from typing import Optional, Type, TypeVar, List, Dict, Any
from pydantic import BaseModel, ValidationError
from openai import OpenAI, AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)

# Generic type for the response model
T = TypeVar("T", bound=BaseModel)

class AzureAIClient:
    """
    Client aligned with the user's working POC.
    Uses sync OpenAI client for Azure paths (v1 behavior) wraped in async helpers.
    """
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        # Check if regular OpenAI should be used
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Check for placeholder key
        use_regular_openai = (
            openai_api_key and 
            "sk-proj-" in openai_api_key and 
            "your-key-here" not in openai_api_key
        )

        if use_regular_openai:
            logger.info("Initializing with regular OpenAI (Async)")
            self.async_client = AsyncOpenAI(api_key=openai_api_key)
            self.sync_client = None
            self.deployment = os.getenv("OPENAI_MODEL", "gpt-4o")
            self.deployment_fast = os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini")
            self.use_responses_api = False
        else:
            logger.info("Initializing with Azure OpenAI (POC Path - Sync)")
            self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
            self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5.1-chat"
            self.deployment_fast = os.getenv("AZURE_OPENAI_DEPLOYMENT_FAST") or self.deployment
            
            # Using synchronous OpenAI client as it's proven to work without api-version conflicts
            self.sync_client = OpenAI(
                base_url=self.endpoint, 
                api_key=self.api_key,
                max_retries=0
            )
            self.async_client = None
            self.use_responses_api = True

    @property
    def fast_model(self) -> str:
        return self.deployment_fast

    @property
    def strong_model(self) -> str:
        return self.deployment

    def _extract_text(self, response) -> str:
        """Extract text from either Responses API or Chat Completion API response (Mirrors POC)"""
        # 1. Direct output_text (Azure specific)
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        
        # 2. Output list (Responses API)
        output = getattr(response, "output", None)
        if isinstance(output, list):
            parts = []
            for item in output:
                content_items = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                for content in content_items or []:
                    text = content.get("text") if isinstance(content, dict) else getattr(content, "text", None)
                    if text:
                        parts.append(text)
            return "".join(parts)
            
        # 3. Choices (Standard Chat Completions)
        if hasattr(response, "choices") and response.choices:
            return (response.choices[0].message.content or "").strip()
            
        return ""

    async def repair_json(self, model: str, raw_text: str, response_model: Type[T]) -> T:
        """Use the LLM to repair malformed JSON into a valid model instance"""
        from ..utils.prompts import build_repair_prompt
        repair_prompt = build_repair_prompt(raw_text)
        json_schema = response_model.model_json_schema()
        
        try:
            if self.use_responses_api:
                # Wrap sync call in thread
                response = await asyncio.to_thread(
                    self.sync_client.responses.create,
                    model=model,
                    max_output_tokens=2048,
                    input=[
                        {"role": "system", "content": "You output valid JSON only. No commentary."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "json_repair",
                            "strict": False,
                            "schema": json_schema,
                        }
                    },
                )
                response_text = self._extract_text(response)
            else:
                completion = await self.async_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You output valid JSON only. No commentary."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                response_text = completion.choices[0].message.content

            if not response_text:
                raise ValueError("Empty repair response")
            
            data = json.loads(response_text)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            raise e

    async def extract_data(
        self,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        max_output_tokens: int = 4096,
        max_retries: int = 5
    ) -> tuple[Optional[T], Optional[str]]:
        """
        Structured extraction using the best available API (Responses or Chat Completions).
        Mirrors the retry and parsing logic from the working POC.
        """
        model = model or self.deployment
        json_schema = response_model.model_json_schema()
        
        for attempt in range(max_retries):
            try:
                if self.use_responses_api:
                    # POC Path: using sync_client.responses.create wrapped in thread
                    response = await asyncio.to_thread(
                        self.sync_client.responses.create,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "structured_extraction",
                                "strict": False,
                                "schema": json_schema,
                            }
                        },
                    )
                    content = self._extract_text(response)
                    response_id = getattr(response, "id", None)
                else:
                    # Regular OpenAI Path
                    enhanced_system_prompt = f"{system_prompt}\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(json_schema, indent=2)}\n\nOutput only the JSON object."
                    
                    completion = await self.async_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": enhanced_system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_output_tokens,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    content = completion.choices[0].message.content
                    response_id = completion.id

                if not content:
                    logger.warning(f"Empty content from LLM on attempt {attempt + 1}")
                    continue

                # Parse and validate with cleanup logic from POC
                try:
                    parsed_dict = None
                    try:
                        parsed_dict = json.loads(content)
                    except json.JSONDecodeError:
                        # Basic JSON cleanup attempt from POC
                        start = content.find("{")
                        end = content.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            parsed_dict = json.loads(content[start:end + 1])
                        else:
                            # Use repair logic if cleanup fails
                            return await self.repair_json(model, content, response_model), response_id
                    
                    if parsed_dict:
                        instance = response_model.model_validate(parsed_dict)
                        return instance, response_id
                        
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Validation/Parsing failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        # Final attempt: try one last repair
                        try:
                            return await self.repair_json(model, content, response_model), response_id
                        except Exception:
                            return None, response_id
                    continue
                
            except RateLimitError as e:
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.warning(f"LLM error: {e}, retrying...")
                await asyncio.sleep(2)
        
        return None, None
