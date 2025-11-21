"""OpenAI/Azure OpenAI client service with enhanced model selection and performance optimizations."""
import logging
from typing import Optional, Any, Dict
from openai import AsyncAzureOpenAI, AsyncOpenAI
from app.core.config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_DEFAULT,
    MODEL_ALIASES, DEFAULT_AZURE_DEPLOYMENT, _clean_env, _normalize_azure_endpoint,
    AVAILABLE_MODELS, DEFAULT_MODEL
)

logger = logging.getLogger(__name__)

# Global async client state with improved connection pooling
_async_azure_client: Optional[AsyncAzureOpenAI] = None
_async_openai_client: Optional[AsyncOpenAI] = None

def get_azure_client() -> Optional[AsyncAzureOpenAI]:
    """Get Async Azure OpenAI client if configured with optimized settings."""
    global _async_azure_client
    if _async_azure_client is None and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        try:
            _async_azure_client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=_normalize_azure_endpoint(AZURE_OPENAI_ENDPOINT),
                # Optimized performance settings
                max_retries=2,  # Reduced from 3 for faster failure
                timeout=40,     # Optimized: reduced from 60 for faster responses
                default_headers={
                    "User-Agent": "Azure-OpenAI-Stock-Tool/1.0",
                    "Connection": "keep-alive"  # Connection pooling
                },
                # Connection pooling settings - httpx async client handles this automatically
                http_client=None  # Use default httpx async client with connection pooling
            )
            logger.info("Async Azure OpenAI client initialized with optimizations at %s", _normalize_azure_endpoint(AZURE_OPENAI_ENDPOINT))
        except Exception as e:
            logger.error("Failed to initialize Async Azure OpenAI client: %s", e)
            _async_azure_client = None
    return _async_azure_client

def get_openai_client() -> Optional[AsyncOpenAI]:
    """Get Async standard OpenAI client if configured with optimized settings."""
    global _async_openai_client
    if _async_openai_client is None and OPENAI_API_KEY:
        try:
            kwargs = {
                "api_key": OPENAI_API_KEY,
                "max_retries": 2,  # Reduced from 3
                "timeout": 40,     # Optimized: reduced from 60 for faster responses
                "default_headers": {
                    "User-Agent": "Azure-OpenAI-Stock-Tool/1.0",
                    "Connection": "keep-alive"
                }
            }
            if (OPENAI_BASE_URL or "").strip():
                kwargs["base_url"] = OPENAI_BASE_URL.strip()
            _async_openai_client = AsyncOpenAI(**kwargs)
            logger.info("Async OpenAI client initialized with optimizations")
        except Exception as e:
            logger.error("Failed to initialize Async OpenAI client: %s", e)
            _async_openai_client = None
    return _async_openai_client

def get_client_for_model(model_key: str, timeout: Optional[int] = None) -> tuple[Any, str, Dict[str, Any]]:
    """Get the appropriate client and resolved model/deployment name with config for a given model key."""
    model_config = AVAILABLE_MODELS.get(model_key)

    if not model_config:
        # Fallback to default model
        model_key = DEFAULT_MODEL
        model_config = AVAILABLE_MODELS.get(model_key, {})

    provider = model_config.get("provider", "azure")

    # Prepare optimized configuration
    config = {
        "timeout": timeout or model_config.get("timeout", 40),  # Optimized: reduced from 60
        "max_completion_tokens": model_config.get("max_completion_tokens", 800),  # Optimized: reduced from 1500
        "temperature": model_config.get("temperature", 0.7)
    }

    if provider == "azure":
        client = get_azure_client()
        if not client:
            # Fallback to OpenAI if Azure not available
            client = get_openai_client()
            if client:
                # Use OpenAI equivalent model
                resolved_model = model_config.get("model", "gpt-4o-mini")
                return client, resolved_model, config
            else:
                raise RuntimeError("No AI client available")

        deployment = model_config.get("deployment")
        if not deployment:
            raise RuntimeError(f"Azure deployment not configured for model {model_key}")

        return client, deployment, config

    elif provider == "openai":
        client = get_openai_client()
        if not client:
            # Fallback to Azure if available
            client = get_azure_client()
            if client:
                # Try to find Azure equivalent
                deployment = model_config.get("deployment") or DEFAULT_AZURE_DEPLOYMENT
                if deployment:
                    return client, deployment, config
            raise RuntimeError("No OpenAI client available")

        model_name = model_config.get("model", "gpt-4o-mini")
        return client, model_name, config

    else:
        raise RuntimeError(f"Unknown provider: {provider}")

def get_client():
    """Get the default client (for backward compatibility)."""
    client, _, _ = get_client_for_model(DEFAULT_MODEL)
    return client

def get_provider() -> Optional[str]:
    """Get the current provider type."""
    if get_azure_client():
        return "azure"
    elif get_openai_client():
        return "openai"
    return None

def resolve_deployment(name: Optional[str]) -> str:
    """Resolve deployment name for Azure (legacy function)."""
    n = _clean_env(name).lower()
    if not n:
        return DEFAULT_AZURE_DEPLOYMENT
    return MODEL_ALIASES.get(n, _clean_env(name))

def resolve_model(name: Optional[str]) -> str:
    """Legacy function - resolve model/deployment name."""
    if not name:
        return DEFAULT_MODEL

    # Try to find in available models
    if name in AVAILABLE_MODELS:
        _, resolved, _ = get_client_for_model(name)
        return resolved

    # Fallback to old behavior
    _, resolved, _ = get_client_for_model(DEFAULT_MODEL)
    return resolved

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available models with their configurations."""
    available = {}

    for model_key, config in AVAILABLE_MODELS.items():
        provider = config.get("provider")
        is_available = False

        if provider == "azure":
            is_available = bool(get_azure_client() and config.get("deployment"))
        elif provider == "openai":
            is_available = bool(get_openai_client())

        if is_available or config.get("deployment"):  # Include if deployment exists
            available[model_key] = {
                "display_name": config.get("display_name", model_key),
                "description": config.get("description", ""),
                "provider": provider,
                "available": is_available,
                "default": config.get("default", False)
            }

    return available
