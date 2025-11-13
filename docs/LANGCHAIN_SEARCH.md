# LangChain Web Search Integration

**Created:** October 1, 2025  
**Purpose:** Provide alternative web search capability using LangChain framework with structured result handling and automatic fallback mechanisms.

---

## Overview

The LangChain Web Search Service offers a robust alternative to the existing Perplexity-style web search, leveraging LangChain's ecosystem for:

- **Multiple search backends** (DuckDuckGo, Brave Search)
- **Structured result parsing** with retry logic
- **AI-powered answer synthesis** using OpenAI models
- **Automatic caching** for improved performance
- **Graceful fallback** between search providers

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  LangChain Web Search                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │  Query Input │───▶│ Search Tools │──▶│   Results    │  │
│  └──────────────┘    └──────────────┘   └──────────────┘  │
│         │                    │                   │         │
│         │            ┌───────▼────────┐          │         │
│         │            │  Brave Search  │          │         │
│         │            │  (if available)│          │         │
│         │            └───────┬────────┘          │         │
│         │                    │ Fallback          │         │
│         │            ┌───────▼────────┐          │         │
│         │            │  DuckDuckGo    │          │         │
│         │            │   (default)    │          │         │
│         │            └────────────────┘          │         │
│         │                                        │         │
│         └─────────────┐               ┌──────────┘         │
│                       │               │                    │
│                ┌──────▼───────────────▼──────┐             │
│                │   Result Parser              │             │
│                │   (JSON/Text formats)        │             │
│                └──────────────┬───────────────┘             │
│                               │                             │
│                    ┌──────────▼──────────┐                  │
│                    │  Cache Layer        │                  │
│                    │  (TTL: 30min)       │                  │
│                    └──────────┬──────────┘                  │
│                               │                             │
│                    ┌──────────▼──────────┐                  │
│                    │  LLM Synthesis      │                  │
│                    │  (OpenAI GPT-4o)    │                  │
│                    └──────────┬──────────┘                  │
│                               │                             │
│                    ┌──────────▼──────────┐                  │
│                    │  Citation Manager   │                  │
│                    └──────────┬──────────┘                  │
│                               │                             │
│                        ┌──────▼──────┐                      │
│                        │   Response  │                      │
│                        └─────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Required Packages

```bash
# Core LangChain packages
pip install langchain==0.1.0
pip install langchain-community==0.0.13
pip install langchain-openai==0.0.2

# Search backend (DuckDuckGo - included by default)
pip install duckduckgo-search

# Optional: Brave Search (premium)
# Requires BRAVE_API_KEY environment variable
```

### Environment Variables

```bash
# Required for AI synthesis
OPENAI_API_KEY=sk-...

# Optional: Premium search backend
BRAVE_API_KEY=BSA...

# Optional: Enable verbose logging
LANGCHAIN_VERBOSE=true
```

---

## Usage

### 1. Basic Search

```python
from app.services.langchain_web_search import langchain_web_search
import asyncio

async def basic_search():
    result = await langchain_web_search(
        query="What is the current price of Tesla stock?",
        max_results=5,
        synthesize_answer=True
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['result_count']}")
    print(f"Time: {result['total_time']:.2f}s")

asyncio.run(basic_search())
```

### 2. Service Class Usage

```python
from app.services.langchain_web_search import LangChainWebSearchService

# Initialize service
service = LangChainWebSearchService(
    model="gpt-4o-mini",
    temperature=0.3,
    max_results=10,
    verbose=False
)

# Perform search
result = await service.search(
    query="Latest AI developments",
    synthesize_answer=True,
    prefer_brave=True
)
```

### 3. Synchronous Wrapper

```python
from app.services.langchain_web_search import langchain_web_search_sync

# Synchronous call
result = langchain_web_search_sync(
    query="Market trends 2025",
    max_results=8
)
```

### 4. Tool Integration

The LangChain search is automatically available as a tool when LangChain is installed:

```python
from app.utils.tools import TOOL_REGISTRY

# Check availability
if "langchain_search" in TOOL_REGISTRY:
    result = TOOL_REGISTRY["langchain_search"](
        query="Stock market analysis",
        max_results=5,
        synthesize_answer=True
    )
```

---

## Response Format

```json
{
  "query": "What is the current price of Apple stock?",
  "answer": "As of October 1, 2025, Apple (AAPL) stock is trading at...",
  "results": [
    {
      "title": "Apple Stock Price Today",
      "url": "https://example.com/aapl",
      "snippet": "Apple stock price information...",
      "source": "brave_search",
      "relevance_score": 0.9,
      "timestamp": null,
      "citation_id": 1
    }
  ],
  "citations": [
    {
      "id": 1,
      "url": "https://example.com/aapl",
      "title": "Apple Stock Price Today",
      "snippet": "Apple stock price..."
    }
  ],
  "search_time": 1.23,
  "synthesis_time": 2.45,
  "total_time": 3.68,
  "result_count": 5,
  "source_engine": "brave"
}
```

---

## Features

### 1. Multi-Backend Support

- **Brave Search**: Premium API with high-quality results
- **DuckDuckGo**: Free fallback with good coverage
- **Automatic Fallback**: If Brave fails, automatically tries DuckDuckGo

### 2. Intelligent Caching

```python
# Search cache: 30 minutes TTL
SEARCH_CACHE = TTLCache(maxsize=500, ttl=1800)

# Synthesis cache: 1 hour TTL
SYNTHESIS_CACHE = TTLCache(maxsize=200, ttl=3600)
```

### 3. Structured Result Parsing

Handles multiple result formats:
- JSON arrays
- Formatted text
- Mixed content

### 4. AI-Powered Synthesis

Uses OpenAI models to generate comprehensive answers:
- Cites sources with `[index]` format
- Markdown formatting
- Concise and accurate

### 5. Error Handling

- Retry logic for failed searches
- Graceful degradation
- Detailed error logging

---

## Configuration Options

### Service Initialization

```python
LangChainWebSearchService(
    openai_api_key: str = None,      # OpenAI API key
    brave_api_key: str = None,       # Brave API key (optional)
    model: str = "gpt-4o-mini",      # LLM model for synthesis
    temperature: float = 0.3,         # Model temperature
    max_results: int = 10,            # Max search results
    verbose: bool = False             # Verbose logging
)
```

### Search Parameters

```python
await service.search(
    query: str,                       # Search query
    max_results: int = None,          # Override default max results
    synthesize_answer: bool = True,   # Generate AI summary
    include_recent: bool = True,      # Prioritize recent content
    prefer_brave: bool = True         # Prefer Brave over DDGS
)
```

---

## Performance Comparison

| Metric | LangChain Search | Perplexity Search |
|--------|-----------------|-------------------|
| **Search Time** | 1-2s | 1-3s |
| **Synthesis Time** | 2-3s | 0s (disabled) |
| **Total Time** | 3-5s | 1-3s |
| **Result Quality** | High (Brave) | High (Brave) |
| **Fallback Support** | ✅ Yes | ✅ Yes |
| **Caching** | ✅ Yes (30min) | ✅ Yes (30min) |
| **AI Synthesis** | ✅ Optional | ❌ Disabled |
| **Citation Format** | `[number]` | `[number]` |

---

## Advantages

### vs. Perplexity Search:

1. **Structured Framework**: LangChain provides robust abstractions
2. **Better Error Handling**: Automatic retry and fallback logic
3. **Extensible**: Easy to add new search backends
4. **Optional Synthesis**: Can disable AI synthesis for speed
5. **Better Parsing**: Handles multiple result formats

### vs. Direct API Calls:

1. **Abstraction Layer**: Consistent interface across backends
2. **Built-in Caching**: Reduces API calls
3. **Retry Logic**: Automatic recovery from failures
4. **Tool Integration**: Works seamlessly with LangChain agents

---

## Limitations

1. **Additional Dependency**: Requires LangChain packages (~50MB)
2. **Slower with Synthesis**: AI synthesis adds 2-3s overhead
3. **OpenAI Dependency**: Synthesis requires OpenAI API key
4. **Learning Curve**: LangChain framework has some complexity

---

## Testing

### Run Test Suite

```bash
# All tests
python test_langchain_search.py

# Basic test only
python test_langchain_search.py --mode basic

# Custom query
python test_langchain_search.py --query "Your custom query here"

# Comparison test
python test_langchain_search.py --mode comparison
```

### Test Cases

1. ✅ Basic search functionality
2. ✅ Recent news search
3. ✅ Japanese language queries
4. ✅ Search without synthesis (fast mode)
5. ✅ Brave Search preference
6. ✅ Service class direct usage

---

## Integration with Existing System

### As a Tool

LangChain search is automatically added to `TOOL_REGISTRY` when available:

```python
# Check if available
if LANGCHAIN_SEARCH_AVAILABLE:
    print("✅ LangChain search is available")
```

### Tool Specification

```json
{
  "name": "langchain_search",
  "description": "ALTERNATIVE SEARCH: LangChain-powered web search with structured result parsing and retry logic. Use this as an alternative to perplexity_search for research queries.",
  "parameters": {
    "query": "Search query in any language",
    "max_results": 8,
    "synthesize_answer": true,
    "prefer_brave": true
  }
}
```

### Usage in Chat

The AI can automatically use LangChain search:

```
User: "What are the latest developments in quantum computing?"

AI: [Uses langchain_search tool]
    Query: "latest quantum computing developments 2025"
    
Response: "Recent breakthroughs in quantum computing include...[1,2,3]"
```

---

## Best Practices

### 1. Choose the Right Mode

```python
# Fast mode - no synthesis
result = await langchain_web_search(
    query="Quick fact check",
    synthesize_answer=False  # Faster, just raw results
)

# Comprehensive mode - with synthesis
result = await langchain_web_search(
    query="Complex research question",
    synthesize_answer=True   # Slower, AI-generated answer
)
```

### 2. Use Caching Effectively

```python
# Same query within 30 minutes will use cache
result1 = await langchain_web_search(query="Market trends")
result2 = await langchain_web_search(query="Market trends")  # Cache hit!
```

### 3. Prefer Brave for Quality

```python
# High-quality results with Brave Search
result = await langchain_web_search(
    query="Technical analysis",
    prefer_brave=True  # Uses Brave if API key available
)
```

### 4. Handle Errors Gracefully

```python
try:
    result = await langchain_web_search(query="...")
    if result['result_count'] == 0:
        print("No results found")
    elif "error" in result['answer'].lower():
        print("Search error occurred")
except Exception as e:
    print(f"Search failed: {e}")
```

---

## Troubleshooting

### Issue: "LangChain not available"

**Solution:**
```bash
pip install langchain langchain-community langchain-openai duckduckgo-search
```

### Issue: "No OpenAI API key"

**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
```

Or disable synthesis:
```python
result = await langchain_web_search(
    query="...",
    synthesize_answer=False
)
```

### Issue: "Brave Search not working"

**Solution:**
```bash
# Set Brave API key
export BRAVE_API_KEY="BSA..."

# Or use DuckDuckGo (default fallback)
result = await langchain_web_search(
    query="...",
    prefer_brave=False
)
```

### Issue: "Slow responses"

**Solution:**
```python
# Disable synthesis for faster results
result = await langchain_web_search(
    query="...",
    synthesize_answer=False,  # Skip AI synthesis
    max_results=5              # Reduce result count
)
```

---

## Future Enhancements

1. **Additional Backends**
   - Google Search integration
   - Bing Search support
   - SerpAPI integration

2. **Advanced Features**
   - Query rewriting with LLM
   - Multi-step research chains
   - Automatic source validation

3. **Performance Optimizations**
   - Parallel backend querying
   - Streaming synthesis results
   - Smart cache invalidation

4. **Quality Improvements**
   - Result deduplication
   - Relevance scoring
   - Source authority ranking

---

## Summary

The LangChain Web Search integration provides a **robust, extensible alternative** to the existing search system with:

✅ **Multi-backend support** (Brave, DuckDuckGo)  
✅ **Automatic retry and fallback**  
✅ **AI-powered synthesis** (optional)  
✅ **Comprehensive caching**  
✅ **Structured result parsing**  
✅ **Easy integration** with existing tools  

**When to use:**
- Research queries requiring comprehensive answers
- When you need structured result handling
- For extensibility with other LangChain tools
- When Brave Search premium quality is needed

**When to use Perplexity Search instead:**
- Speed is critical (no synthesis overhead)
- Simple fact-checking queries
- When LangChain dependencies are unavailable

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [DuckDuckGo Search API](https://github.com/deedy5/duckduckgo_search)
- [Brave Search API](https://brave.com/search/api/)
- [OpenAI API](https://platform.openai.com/docs)
