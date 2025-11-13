# Project Task Backlog

## Naming (Proposed Project Names)
- MarketScope AI
- EquitySense Assistant
- FinPilot Copilot
- AlphaLens Chat
- StockGrid Insight

Pick one (suggest: MarketScope AI for clarity + professional tone).

## Core Completed
- Chat endpoint with tool calling
- Model alias system (gpt-4.1, gpt-4.1-mini, o3, gpt-5, gpt-oss-120b)
- System prompt enforcing Reasoning + Result pattern
- News (yfinance + RSS fallback + augmented article extraction)
- Risk metrics (volatility, Sharpe, max drawdown, VaR, beta)
- Auth (register/login/JWT + refresh tokens)
- Admin log viewer
- Conversation caching
- RAG plumbing (index/search) optional

## Near-Term Enhancements (Sprint 1)
1. Streaming responses (Server-Sent Events or WebSocket)
2. Multi-ticker comparative query parsing (detect list: AAPL, MSFT, NVDA)
3. News sentiment scoring (simple polarity + aggregate)
4. Rate limiting (per user/IP) to prevent abuse
5. Add unit tests for: ticker validation, risk calculations edge cases, tool call loop fallback

## Sprint 2
1. Portfolio aggregation endpoint (list of tickers → combined stats)
2. Caching layer migration option (Redis) behind interface
3. Add OpenTelemetry traces for tool latency diagnostics
4. Export logs (CSV/JSON) via admin endpoint
5. Add user roles (basic vs pro; gating features)

## Sprint 3 / Stretch
1. Fine-tuned prompt compression for long histories
2. Custom embeddings model selection in UI
3. Pluggable pricing provider fallback (e.g., AlphaVantage)
4. Self-hosted OSS model gateway (serve gpt-oss-120b alias)
5. Batch RAG ingestion endpoint (zip upload)

## Technical Debt
- Secrets committed in current `.env` (rotate immediately; rely on `.env.example`)
- Limited test coverage
- No retry/backoff on external news/article fetch
- Tool call loop max=3 is static (could be adaptive)

## QA / Test Ideas
- Invalid ticker (numeric start, punctuation) → error path
- Illiquid ticker with sparse history → risk metrics degrade gracefully
- Conversation reset flag clears memory
- RAG disabled still allows chat (rag_search returns enabled=false)
- Missing deployment alias falls back to default

## Security Checklist
- Rotate exposed API keys
- Enforce HTTPS in production (Cookie secure flag)
- Add password complexity (length + entropy) validation
- Brute force protection (lockout after N failed logins)

## Metrics to Implement
- Requests per model alias
- Tool latency histogram (quote vs history vs news)
- Tool usage ratio (market queries with tool calls / total market queries)

## Definition of Done (Incremental Features)
- Code + minimal tests + README update
- No critical lint/type errors
- Endpoint documented via OpenAPI
- Manual smoke test script updated

---
Last updated: {{DATE}}
