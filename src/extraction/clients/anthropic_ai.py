import os
import json
import asyncio
import logging
from typing import Optional, Type, TypeVar, List, Dict, Any, Tuple
from pydantic import BaseModel, ValidationError
import anthropic

logger = logging.getLogger(__name__)

# Generic type for the response model
T = TypeVar("T", bound=BaseModel)

class AnthropicAIClient:
    """
    Anthropic Client for structured extraction using Claude 3.5.
    Aligned with AzureAIClient interface.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            # We don't raise here, but features relying on it will fail.
            # This allows the app to start even if only one provider is configured.
            logger.warning("ANTHROPIC_API_KEY not found. Anthropic features will be unavailable.")
            self.client = None
        else:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
        self.model_fast = os.getenv("ANTHROPIC_MODEL_FAST", "claude-3-5-haiku-20241022")

    @property
    def fast_model(self) -> str:
        return self.model_fast

    @property
    def strong_model(self) -> str:
        return self.model

    def _extract_text(self, response) -> str:
        """Extract text from Anthropic response content blocks"""
        if hasattr(response, "content") and isinstance(response.content, list):
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        return str(getattr(response, "content", ""))

    async def repair_json(self, model: str, raw_text: str, response_model: Type[T]) -> T:
        """Use Claude to repair malformed JSON"""
        if not self.client:
            raise RuntimeError("Anthropic client NOT initialized")

        from ..utils.prompts import build_repair_prompt
        repair_prompt = build_repair_prompt(raw_text)
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=2048,
                system="You output valid JSON only. No commentary.",
                messages=[{"role": "user", "content": repair_prompt}],
            )
            response_text = self._extract_text(response)
            
            # Clean markdown if present
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()
            
            data = json.loads(response_text)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(f"Anthropic JSON repair failed: {e}")
            raise e

    async def extract_data(
        self,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        max_output_tokens: int = 4096,
        max_retries: int = 3
    ) -> tuple[Optional[T], Optional[str]]:
        """
        Structured extraction using Claude's system prompt and XML-ish tool-less JSON enforcement.
        Claude 3.5 is excellent at following JSON schemas.
        """
        if not self.client:
            raise RuntimeError("Anthropic client NOT initialized")

        model = model or self.model
        json_schema = response_model.model_json_schema()
        
        # Enhanced instructions for Anthropic
        enhanced_system_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond with a valid JSON object matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}\n\n"
            f"Rules:\n"
            f"- Output ONLY the JSON object.\n"
            f"- Do not include any preamble or postscript.\n"
            f"- Ensure all fields are escaped correctly."
        )

        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=max_output_tokens,
                    system=enhanced_system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0
                )
                
                content = self._extract_text(response)
                response_id = getattr(response, "id", None)

                if not content:
                    continue

                # Parse and validate
                try:
                    # Basic cleanup
                    json_str = content.strip()
                    if "```json" in json_str:
                        json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()
                    
                    if not (json_str.startswith("{") or json_str.startswith("[")):
                        start = json_str.find("{")
                        end = json_str.rfind("}")
                        if start != -1 and end != -1:
                            json_str = json_str[start:end+1]

                    data = json.loads(json_str)
                    instance = response_model.model_validate(data)
                    return instance, response_id
                        
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Anthropic parsing failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        try:
                            return await self.repair_json(model, content, response_model), response_id
                        except Exception:
                            return None, response_id
                    continue
                
            except Exception as e:
                logger.warning(f"Anthropic API error: {e}, retrying...")
                await asyncio.sleep(2)
        
        return None, None
