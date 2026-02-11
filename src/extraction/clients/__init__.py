from typing import Optional, Any
from .azure_ai import AzureAIClient
from .anthropic_ai import AnthropicAIClient

def get_llm_client(provider: str = "azure", **kwargs) -> Any:
    """
    Factory to get the appropriate LLM client based on provider name.
    Supported providers: 'azure', 'openai', 'anthropic'
    """
    provider = provider.lower()
    if provider in ("azure", "openai"):
        return AzureAIClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicAIClient(**kwargs)
    else:
        # Fallback to Azure as it's the POC default
        return AzureAIClient(**kwargs)
