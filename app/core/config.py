"""Application configuration and environment variables."""
import os
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment from .env if present
load_dotenv()

# ---------- OpenAI/Azure config ----------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Standard OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Enhanced deployment configurations
AZURE_OPENAI_DEPLOYMENT_4_1 = os.getenv("AZURE_OPENAI_DEPLOYMENT_4_1")
AZURE_OPENAI_DEPLOYMENT_4_1_MINI = os.getenv("AZURE_OPENAI_DEPLOYMENT_4_1_MINI")
AZURE_OPENAI_DEPLOYMENT_O3 = os.getenv("AZURE_OPENAI_DEPLOYMENT_O3")
AZURE_OPENAI_DEPLOYMENT_GPT5 = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5")
AZURE_OPENAI_DEPLOYMENT_OSS_120B = os.getenv("AZURE_OPENAI_DEPLOYMENT_OSS_120B")

# Model Selection Configuration
AVAILABLE_MODELS = {
    # Azure deployed models (your deployment)
    "gpt-oss-120b": {
        "provider": "azure",
        "deployment": AZURE_OPENAI_DEPLOYMENT_OSS_120B,
        "display_name": "GPT OSS 120B (Azure)",
        "description": "Large open-source model deployed on Azure",
        "default": True,  # Set as default since you deployed it
        "timeout": 45,  # Optimized: reduced from 120s for faster responses
        "max_completion_tokens": 600,  # Optimized: reduced from 1000 for speed
        "temperature": 0.3  # Lower temperature for more focused responses
    },

    # OpenAI models
    "gpt-5": {
        "provider": "openai",
        "model": "gpt-5",
        "display_name": "GPT-5",
        "description": "Latest GPT-5 model from OpenAI",
        "timeout": 40,  # Optimized: reduced from 60s
        "temperature": 1.0,  # GPT-5 only supports default temperature
        "temperature_fixed": True,  # Flag to indicate temperature cannot be changed
        "completion_token_param": "max_completion_tokens",
        "max_completion_tokens": 2000  # Optimized: reduced from 4000 for speed
    },
    "o3": {
        "provider": "openai",
        "model": "o3",
        "display_name": "O3",
        "description": "OpenAI O3 reasoning model",
        "timeout": 60,  # Optimized: reduced from 90s (reasoning needs more time)
        "temperature": 1.0,  # O3 may also have temperature restrictions
        "temperature_fixed": True,  # Flag to indicate temperature cannot be changed
        "completion_token_param": "max_completion_tokens",
        "max_completion_tokens": 2000  # Optimized: reduced from 4000
    },
    "gpt-4.1": {
        "provider": "openai",
        "model": "gpt-4.1",
        "display_name": "GPT-4.1",
        "description": "Enhanced GPT-4.1 model",
        "timeout": 35,  # Optimized: reduced from 60s
        "max_completion_tokens": 800  # Added explicit limit for speed
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "display_name": "GPT-4o",
        "description": "GPT-4 Omni multimodal model",
        "timeout": 35,  # Optimized: reduced from 60s
        "max_completion_tokens": 800  # Added explicit limit for speed
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "description": "Faster, cost-effective GPT-4o variant",
        "timeout": 25,  # Optimized: reduced from 30s
        "max_completion_tokens": 600  # Added explicit limit for speed
    }
}

# Get default model
DEFAULT_MODEL = next((k for k, v in AVAILABLE_MODELS.items() if v.get("default")), "gpt-oss-120b")

# RAG Configuration
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() in {"1", "true", "yes"}
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Cache Configuration
QUOTE_CACHE_SIZE = int(os.getenv("QUOTE_CACHE_SIZE", "1024"))
QUOTE_TTL_SECONDS = int(os.getenv("QUOTE_TTL_SECONDS", "60"))
CONV_CACHE_SIZE = int(os.getenv("CONV_CACHE_SIZE", "1000"))
CONV_TTL_SECONDS = int(os.getenv("CONV_TTL_SECONDS", str(60 * 60 * 4)))  # 4 hours
NEWS_CACHE_SIZE = int(os.getenv("NEWS_CACHE_SIZE", "1024"))
NEWS_TTL_SECONDS = int(os.getenv("NEWS_TTL_SECONDS", "300"))  # 5 minutes
# New: cache for extracted article content (reduces repeat fetch/parse)
ARTICLE_CACHE_SIZE = int(os.getenv("ARTICLE_CACHE_SIZE", "2048"))
ARTICLE_TTL_SECONDS = int(os.getenv("ARTICLE_TTL_SECONDS", str(60 * 60)))  # 1 hour

# Conversation Management - Optimized for performance
MAX_CONV_MESSAGES = int(os.getenv("MAX_CONV_MESSAGES", "50"))
CONV_SUMMARY_THRESHOLD = int(os.getenv("CONV_SUMMARY_THRESHOLD", "20"))
MAX_TOKENS_PER_TURN = int(os.getenv("MAX_TOKENS_PER_TURN", "6000"))  # Reduced from 8000 for faster responses
RAG_MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "3"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

# Performance optimizations
ENABLE_RESPONSE_CACHE = os.getenv("ENABLE_RESPONSE_CACHE", "true").lower() in {"1", "true", "yes"}
RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "300"))  # 5 minutes
SIMPLE_QUERY_CACHE_TTL = int(os.getenv("SIMPLE_QUERY_CACHE_TTL", "60"))  # 1 minute for simple queries
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
SEMANTIC_CACHE_MAX_ITEMS = int(os.getenv("SEMANTIC_CACHE_MAX_ITEMS", "128"))
SEMANTIC_CACHE_TTL = int(os.getenv("SEMANTIC_CACHE_TTL", str(60 * 15)))  # 15 minutes
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))

# Simple query patterns that don't need RAG/tools
SIMPLE_QUERY_PATTERNS = [
    r'(?i)^(hi|hello|hey|good morning|good afternoon).*',
    r'(?i)^(what|how|who|when|where|why)\s+(is|are|do|does|can|will|would).*',
    r'(?i)^(tell me|explain|describe).*',
    r'(?i).*(stock price|current price|quote).*',
    r'(?i)^(thanks|thank you|bye|goodbye).*'
]

# Streaming optimizations
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "64"))  # Smaller chunks for faster streaming
MAX_STREAM_CHUNKS = int(os.getenv("MAX_STREAM_CHUNKS", "500"))  # Prevent runaway streams

# Tool execution optimizations
ENABLE_PARALLEL_TOOLS = os.getenv("ENABLE_PARALLEL_TOOLS", "true").lower() in {"1", "true", "yes"}
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "10"))  # 10 seconds max per tool

# Model-specific optimizations
FAST_MODEL_FOR_SIMPLE = os.getenv("FAST_MODEL_FOR_SIMPLE", "gpt-4o-mini")

# News / RAG concurrency tuning
NEWS_FETCH_MAX_WORKERS = int(os.getenv("NEWS_FETCH_MAX_WORKERS", "6"))  # concurrent article fetches
RAG_MAX_WORKERS = int(os.getenv("RAG_MAX_WORKERS", "4"))  # concurrent rag queries per news batch
# RAG strategy: 'symbol' (one query for all items), 'item' (one query per article), or 'auto'
RAG_STRATEGY = os.getenv("RAG_STRATEGY", "symbol").strip().lower()
RAG_MAX_PER_ITEM = int(os.getenv("RAG_MAX_PER_ITEM", "3"))  # when doing per-item, cap items enriched

# Financial compliance and guardrails configuration
FINANCIAL_GUARDRAILS = {
    "investment_advice_prohibited": True,
    "require_data_sources": True,
    "require_timezone_currency": True,
    "require_confidence_threshold": True,
    "default_timezone": "Asia/Tokyo",
    "default_currency": "JPY"
}

# Model-specific system prompts with compliance guardrails
MODEL_SYSTEM_PROMPTS = {
    "gpt-oss-120b": """You are an expert financial analyst who writes unbiased, journalistic answers grounded in the provided search results. You see the final query plus structured findings assembled by another system; respond only to the latest user query, drawing on those findings without restating the agent’s internal reasoning.

Authoring rules (follow exactly):
- Produce an accurate, detailed, and comprehensive answer that is fully self-contained. Do not reference prior assistant responses.
- Cite supporting statements with `[index]` immediately after the sentence, using the source indices supplied by the tool and never fabricating or reordering citations. When citing multiple sources for a single sentence, merge them inside one bracket separated by commas (e.g., `[1,2]`), never as `[1][2]`. Use at most three citations per sentence and do not add spaces before the bracket.
- Write in Markdown without starting the response with a heading. Use `##` level headings for main sections and bold text (e.g., `**Key Metrics**`) for subsection labels when helpful.
- Prefer unordered lists for highlights. Use ordered lists only when presenting ranked data. Never mix list styles or nest them together.
- When comparing items, present the comparison as a Markdown table with labeled columns.
- Wrap code snippets in fenced blocks with a language tag. Wrap math in LaTeX using `\(` and `\)` for inline expressions and `\[` and `\]` for display math. Never use single dollar signs.
- Keep paragraphs concise (no more than three sentences) to mirror Perplexity/GPT readability. Bold sparingly; use italics for gentle emphasis.
- Omit URLs and bibliographies in the final output.

Financial compliance guardrails:
1. Treat all outputs as educational information, not investment advice.
2. Always note data currency (timestamp, timezone, currency) when summarizing market figures. Default timezone: Asia/Tokyo; default currency: JPY.
3. If real-time data is unavailable, state that explicitly and offer the latest accessible alternative (e.g., prior close).
4. If confidence is low, acknowledge the limitation and recommend additional tool runs or data collection.

If the query cannot be answered or is based on an incorrect premise, explain why using the same formatting rules.""",

    "default": """You are an expert financial analyst who writes unbiased, journalistic answers grounded in the provided search results. You receive the final user query plus curated evidence gathered by another system; rely on that evidence and respond only to the most recent query without repeating earlier assistant messages.

Authoring rules (follow exactly):
- Deliver an accurate, thorough, and self-contained answer. Do not mention the planning agent or prior conversational replies.
- Cite search results using [index] at the end of sentences when needed, for example "Ice is less dense than water[1,2]." NO SPACE between the last word and the citation, and merge multiple sources into a single bracket (never output `[1][2]`).
- Cite at most three sources per sentence and do not invent or reorder citations.
- Use Markdown formatting: begin with narrative text (no opening heading), then organize content with `##` headings and bolded subsection titles where appropriate.
- Favor unordered lists for bullet points; use ordered lists exclusively for rankings, and never mix or nest list styles.
- Render comparisons as Markdown tables with meaningful headers.
- Encapsulate code in fenced blocks with language annotations and express math using `\(`/`\)` or `\[`/`\]`. Avoid single-dollar math delimiters.
- Keep paragraphs tight (≤3 sentences) to maintain clarity similar. Apply bold sparingly; reserve italics for light emphasis.
- Do not output raw URLs or bibliographies in the conclusion.

Tool Usage Rules:

For LIVE/CURRENT financial data:
- Use get_stock_quote → current stock price or snapshot
- Use get_historical_prices → historical stock/index data
- Use get_company_profile → company details, financials, and fundamentals
- Use get_augmented_news → latest articles and headlines with enhanced context
- Use get_risk_assessment → real-time stock risk metrics

For CURRENT EVENTS, NEWS & UNKNOWN INFORMATION:
- ALWAYS use web_search or perplexity_search → for any recent events, breaking news, or information you're uncertain about
- Use web_search → for real-time information, current events, recent developments
- Use perplexity_search → for comprehensive analysis with AI synthesis and citations
- Use financial_context_search → for financial news and market analysis

For EDUCATIONAL/CONCEPTUAL topics:
- Use rag_search → financial concepts, ratios, indicators (P/E, RSI, etc.) from knowledge base
- If rag_search doesn't provide sufficient information, follow up with web_search for additional context
- Use augmented_rag_search → complex topics requiring both knowledge base and current information

For ANY UNCERTAIN OR INCOMPLETE KNOWLEDGE:
- NEVER provide partial or potentially outdated information
- ALWAYS use web_search or perplexity_search to get current, accurate information
- If you don't know something with confidence, search for it immediately
- Prefer recent, verified sources over assumptions

Financial compliance guardrails:
1. Provide educational context only—no personalized investment advice.
2. Report data with explicit timestamps, timezones (default Asia/Tokyo), and currencies (default JPY) whenever relevant.
3. If live data cannot be retrieved, acknowledge the gap and offer the most recent available value.
4. Flag low-confidence findings and suggest follow-up analysis or tool usage when appropriate.

If the query is unanswerable or rests on a faulty premise, say so clearly while adhering to the formatting requirements.
Current date: Saturday, September 27, 2025
You are trained on data up to October 2025.
"""
}

def get_system_prompt_for_model(model_key: str) -> str:
    """Get optimized system prompt for specific model."""
    return MODEL_SYSTEM_PROMPTS.get(model_key, MODEL_SYSTEM_PROMPTS["default"])

# Risk Assessment
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.015"))

# News Configuration
NEWS_USER_AGENT = os.getenv("NEWS_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# DDGS (DuckDuckGo Search) Configuration
DDGS_REGION = os.getenv("DDGS_REGION", "jp-jp")  # Japan region for Japanese financial markets
DDGS_SAFESEARCH = os.getenv("DDGS_SAFESEARCH", "moderate")
DDGS_TIMELIMIT = os.getenv("DDGS_TIMELIMIT")  # Optional time limit: d (day), w (week), m (month), y (year)

# Brave Search Configuration (High-Quality Source)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")  # Brave Search API key for enhanced search quality

# Finnhub Configuration (Real-time Stock Data)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")  # Finnhub API key for real-time stock data

# Alpha Vantage Configuration (News & Sentiment)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Alpha Vantage for news/sentiment (25 requests/day free tier)

# AWS Configuration (LocalStack or production)
USE_LOCALSTACK = os.getenv("USE_LOCALSTACK", "false").lower() in {"1", "true", "yes"}
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # Set to http://localhost:4566 for LocalStack
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "test")  # Dummy for LocalStack
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "test")  # Dummy for LocalStack

# S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "stocktool-knowledge")
S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() in {"1", "true", "yes"}

# DynamoDB Configuration
DYNAMODB_TABLE_CONVERSATIONS = os.getenv("DYNAMODB_TABLE_CONVERSATIONS", "stocktool-conversations")
DYNAMODB_TABLE_CACHE = os.getenv("DYNAMODB_TABLE_CACHE", "stocktool-stock-cache")
DYNAMODB_ENABLED = os.getenv("DYNAMODB_ENABLED", "false").lower() in {"1", "true", "yes"}

# SQS Configuration
SQS_QUEUE_ANALYSIS = os.getenv("SQS_QUEUE_ANALYSIS", "stocktool-analysis-queue")
SQS_ENABLED = os.getenv("SQS_ENABLED", "false").lower() in {"1", "true", "yes"}

# SNS Configuration
SNS_TOPIC_NOTIFICATIONS = os.getenv("SNS_TOPIC_NOTIFICATIONS", "stocktool-notifications")
SNS_ENABLED = os.getenv("SNS_ENABLED", "false").lower() in {"1", "true", "yes"}

# CloudWatch Configuration
CLOUDWATCH_NAMESPACE = os.getenv("CLOUDWATCH_NAMESPACE", "StockTool")
CLOUDWATCH_ENABLED = os.getenv("CLOUDWATCH_ENABLED", "false").lower() in {"1", "true", "yes"}

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Auth Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "refresh_token")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN")

# Admin Configuration
ADMIN_USERNAMES = {u.strip().lower() for u in os.getenv("ADMIN_USERNAMES", "admin").split(",") if u.strip()}

# CORS Configuration  
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "*")

# Validation patterns
TICKER_RE = re.compile(r"^[\^]?[A-Z0-9][A-Z0-9.-]{0,9}$")

def _clean_env(v: Optional[str]) -> str:
    """Clean environment variable values."""
    s = (v or "").strip().strip('"').strip("'")
    # treat template placeholders like <your> as empty
    if s.startswith("<") and s.endswith(">"):
        return ""
    if "</" in s or s.startswith("<"):
        return ""
    return s

def _normalize_azure_endpoint(ep: Optional[str]) -> str:
    """Helper to normalize Azure endpoint if a full path was provided."""
    s = _clean_env(ep)
    if not s:
        return s
    # Strip trailing slashes
    while s.endswith('/'):
        s = s[:-1]
    # If the env contains '/openai/deployments/', cut back to base '/'
    m = re.search(r"/openai/.*$", s)
    if m:
        s = s[: m.start()]  # keep just the resource base URL
    return s

# Build model aliases
def build_model_aliases() -> Dict[str, str]:
    """Build model alias mapping for deployment resolution."""
    MODEL_ALIASES: Dict[str, str] = {}
    
    # Clean deployment names
    deployment_4_1 = _clean_env(AZURE_OPENAI_DEPLOYMENT_4_1)
    deployment_4_1_mini = _clean_env(AZURE_OPENAI_DEPLOYMENT_4_1_MINI)
    deployment_o3 = _clean_env(AZURE_OPENAI_DEPLOYMENT_O3)
    deployment_gpt5 = _clean_env(AZURE_OPENAI_DEPLOYMENT_GPT5)
    deployment_oss_120b = _clean_env(AZURE_OPENAI_DEPLOYMENT_OSS_120B)
    
    if deployment_4_1:
        for a in ("gpt-4.1", "gpt-4-1"):
            MODEL_ALIASES[a.lower()] = deployment_4_1
    if deployment_4_1_mini:
        for a in ("gpt-4.1-mini", "gpt-4-1-mini"):
            MODEL_ALIASES[a.lower()] = deployment_4_1_mini
    if deployment_o3:
        for a in ("o3", "gpt-o3", "gpt o3"):
            MODEL_ALIASES[a.lower()] = deployment_o3
    if deployment_gpt5:
        for a in ("gpt-5", "gpt 5", "gpt5"):
            MODEL_ALIASES[a.lower()] = deployment_gpt5
    if deployment_oss_120b:
        for a in ("gpt-oss-120b", "oss-120b", "gpt oss 120b"):
            MODEL_ALIASES[a.lower()] = deployment_oss_120b
    
    # Fallback generic aliases
    MODEL_ALIASES.setdefault("gpt-4.1", "gpt-4-1")
    MODEL_ALIASES.setdefault("gpt-4-1", "gpt-4-1")
    MODEL_ALIASES.setdefault("gpt-4.1-mini", "gpt-4-1-mini")
    MODEL_ALIASES.setdefault("gpt-4-1-mini", "gpt-4-1-mini")
    MODEL_ALIASES.setdefault("o3", "o3")
    MODEL_ALIASES.setdefault("gpt-o3", "o3")
    MODEL_ALIASES.setdefault("gpt o3", "o3")
    MODEL_ALIASES.setdefault("gpt-5", "gpt-5")
    MODEL_ALIASES.setdefault("gpt 5", "gpt-5")
    MODEL_ALIASES.setdefault("gpt5", "gpt-5")
    MODEL_ALIASES.setdefault("gpt-oss-120b", "gpt-oss-120b")
    MODEL_ALIASES.setdefault("oss-120b", "gpt-oss-120b")
    MODEL_ALIASES.setdefault("gpt oss 120b", "gpt-oss-120b")
    
    return MODEL_ALIASES

# Get model aliases
MODEL_ALIASES = build_model_aliases()

# Compute default Azure deployment
DEFAULT_AZURE_DEPLOYMENT = (
    _clean_env(AZURE_OPENAI_DEPLOYMENT)
    or _clean_env(AZURE_OPENAI_DEPLOYMENT_OSS_120B)
    or MODEL_ALIASES.get("gpt-oss-120b")
    or _clean_env(AZURE_OPENAI_DEPLOYMENT_4_1)
    or _clean_env(AZURE_OPENAI_DEPLOYMENT_4_1_MINI)
    or _clean_env(AZURE_OPENAI_DEPLOYMENT_O3)
    or _clean_env(AZURE_OPENAI_DEPLOYMENT_GPT5)
    or MODEL_ALIASES.get("gpt-4.1")
)

def check_model_availability() -> Dict[str, bool]:
    """Check which models are actually available."""
    availability = {}

    # Check Azure models
    azure_available = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

    for model_key, config in AVAILABLE_MODELS.items():
        if config["provider"] == "azure":
            deployment = config.get("deployment")
            availability[model_key] = azure_available and bool(deployment)
        elif config["provider"] == "openai":
            availability[model_key] = bool(OPENAI_API_KEY)
        else:
            availability[model_key] = False

    return availability

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get models with availability status."""
    availability = check_model_availability()

    models = {}
    for model_key, config in AVAILABLE_MODELS.items():
        models[model_key] = {
            **config,
            "available": availability.get(model_key, False)
        }

    return models

# Update model configurations with availability
for model_key in AVAILABLE_MODELS:
    if model_key not in AVAILABLE_MODELS[model_key]:
        AVAILABLE_MODELS[model_key]["available"] = False


# ---------- ML Tool Selection Configuration ----------
# Enable ML-based tool selection (vs rule-based)
ML_TOOL_SELECTION_ENABLED = os.getenv("ML_TOOL_SELECTION_ENABLED", "true").lower() == "true"  # ✅ ENABLED

# Path to trained ML model
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/tool_classifier.pkl")

# Minimum confidence threshold for tool selection
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.3"))

# Maximum number of tools to offer per request
ML_MAX_TOOLS = int(os.getenv("ML_MAX_TOOLS", "5"))

# Embedding model for query understanding
ML_EMBEDDING_MODEL = os.getenv("ML_EMBEDDING_MODEL", "text-embedding-3-small")

# Embedding cache settings
ML_EMBEDDING_CACHE_SIZE = int(os.getenv("ML_EMBEDDING_CACHE_SIZE", "1000"))
ML_EMBEDDING_CACHE_TTL = int(os.getenv("ML_EMBEDDING_CACHE_TTL", "3600"))  # 1 hour

# Tool usage logging for training data collection
ML_LOGGING_ENABLED = os.getenv("ML_LOGGING_ENABLED", "true").lower() == "true"

# Minimum number of logs before training
ML_MIN_TRAINING_SAMPLES = int(os.getenv("ML_MIN_TRAINING_SAMPLES", "100"))

# Recommended number of logs for good accuracy
ML_RECOMMENDED_TRAINING_SAMPLES = int(os.getenv("ML_RECOMMENDED_TRAINING_SAMPLES", "500"))
