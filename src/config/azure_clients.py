import os
import logging
from typing import Callable

import dotenv
import tiktoken
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Load local .env when running locally (Functions / Flask / Docker dev)
# --------------------------------------------------------------------
dotenv.load_dotenv(dotenv.find_dotenv())

LOG_INIT = os.getenv("LOG_AZURE_CLIENT_INIT", "true").lower() == "true"
if LOG_INIT:
    logger.info("Initializing Azure clients...")

# --------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------


def _build_credential() -> DefaultAzureCredential | ClientSecretCredential:
    """
    Build a single long-lived Azure credential for both:
      - Azure OpenAI (via AAD token) when API key not provided
      - Azure Document Intelligence when DI key not provided
    """
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")

    if tenant_id and client_id and client_secret:
        if LOG_INIT:
            logger.info("Environment is configured for ClientSecretCredential")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    if LOG_INIT:
        logger.info("Using DefaultAzureCredential")
    # Disable interactive browser (not usable in Functions/containers)
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


credential = _build_credential()

# --------------------------------------------------------------------
# Azure OpenAI (LLM) client – SINGLE async client
# --------------------------------------------------------------------

openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # model name
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")

if not openai_endpoint:
    raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT in environment")
if not openai_deployment:
    raise RuntimeError("Missing AZURE_OPENAI_DEPLOYMENT in environment")

# Scope for Cognitive Services / Azure OpenAI
_COGNITIVE_SCOPE = os.getenv(
    "AZURE_COGNITIVE_SCOPE",
    "https://cognitiveservices.azure.com/.default",
)


def _aad_token_provider() -> str:
    """
    Token provider callback for AsyncAzureOpenAI.

    IMPORTANT:
    - This does NOT create any aiohttp sessions.
    - It just uses the shared `credential` to fetch an AAD token.
    """
    token = credential.get_token(_COGNITIVE_SCOPE)
    return token.token


# Single shared async Azure OpenAI client
llm_client: AsyncAzureOpenAI
if openai_api_key:
    if LOG_INIT:
        logger.info("Initializing Azure OpenAI client using API key auth.")
    llm_client = AsyncAzureOpenAI(
        api_version=openai_api_version,
        azure_endpoint=openai_endpoint,
        api_key=openai_api_key,
    )
else:
    if LOG_INIT:
        logger.info("Initializing Azure OpenAI client using AAD token provider.")
    llm_client = AsyncAzureOpenAI(
        api_version=openai_api_version,
        azure_endpoint=openai_endpoint,
        azure_ad_token_provider=_aad_token_provider,
    )

# Expose deployment name so the rest of the pipeline can use it
openai_deployment_name: str = openai_deployment

# --------------------------------------------------------------------
# Azure Document Intelligence client – SINGLE async client
# --------------------------------------------------------------------

di_endpoint = os.getenv("AZURE_DI_ENDPOINT")
di_api_version = os.getenv("AZURE_DI_API_VERSION", "2024-11-30")
di_key = os.getenv("AZURE_DI_KEY")

if not di_endpoint:
    raise RuntimeError("Missing AZURE_DI_ENDPOINT in environment")

if di_key:
    di_credential = AzureKeyCredential(di_key)
    if LOG_INIT:
        logger.info("Initializing Document Intelligence client with API key.")
else:
    di_credential = credential
    if LOG_INIT:
        logger.info("Initializing Document Intelligence client with AAD credential.")

document_intelligence_client: DocumentIntelligenceClient = DocumentIntelligenceClient(
    endpoint=di_endpoint,
    credential=di_credential,
    api_version=di_api_version,
)

# --------------------------------------------------------------------
# Tokenizer (for prompt budgeting etc.)
# --------------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")

if LOG_INIT:
    logger.info("✓ Azure OpenAI client initialized")
    logger.info("✓ Document Intelligence client initialized")
    logger.info("✓ All clients ready!")
