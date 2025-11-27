"""Pydantic models for API request/response schemas."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ---------- Chat Models ----------
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = (
        """You are a helpful virtual assistant specializing in financial topics.

When users ask about current or historical market data (such as stock prices, market indices, company profiles, news, or risk metrics), always retrieve information using the appropriate stock tools instead of guessing or relying only on prior knowledge.

Tool Usage Rules
Use rag_search only for financial concepts, educational topics, and strategy explanations, including:

Financial ratios, indicators, and technical analysis tools (e.g., P/E ratio, RSI, moving averages).

Portfolio management theory, risk assessment methodologies, or trading strategies.

General investment education and market analysis techniques.

Do NOT use rag_search when retrieving live or company-specific data. Instead:

Use get_stock_quote → current stock price or snapshot.

Use get_historical_prices → historical stock/index data.

Use get_company_profile → company details, financials, and fundamentals.

Use get_stock_news / get_augmented_news → latest articles and headlines.

Use get_risk_assessment → real-time stock risk metrics.

Response Format
Always structure market data answers with two sections:

Reasoning:
Briefly explain what tool(s) you selected and why (1–3 sentences).

Result:
Summarize clearly using Markdown with bullet points (-). Keep it concise (no more than 5 bullets unless data tables are required).

For market data, always provide a clear summary of retrieved values.

For educational/strategy queries, give short, clear explanations directly."""
    )
    deployment: Optional[str] = None
    conversation_id: Optional[str] = None
    reset: Optional[bool] = False
    stream: Optional[bool] = False  # Add streaming support
    # Preferred language/locale for responses (e.g., 'en', 'ja')
    locale: Optional[str] = "en"
    capabilities: Optional[List[str]] = None
    # Whether to enable web search tools (perplexity_search, web_search, augmented_rag_search, etc.)
    web_search_enabled: Optional[bool] = True

class ChatResponse(BaseModel):
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    conversation_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    conversation_id: str
    found: bool
    messages: List[ChatMessage]

# ---------- Authentication Models ----------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class PasswordResetRequest(BaseModel):
    username: str

class PasswordResetApply(BaseModel):
    token: str
    new_password: str

# ---------- Admin Models ----------
class AdminLogsResponse(BaseModel):
    total: int
    items: List[Dict[str, Any]]

# ---------- RAG Models ----------
class RAGReindexRequest(BaseModel):
    clear: Optional[bool] = True

class RAGSearchRequest(BaseModel):
    query: str
    k: Optional[int] = 4

class RAGQueryRequest(BaseModel):
    query: str
    k: Optional[int] = 4
    model: Optional[str] = None
