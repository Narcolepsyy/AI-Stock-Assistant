# LangChain Search Quick Start Guide

## Installation (30 seconds)

```bash
# Install LangChain dependencies
pip install -r requirements-langchain.txt

# Or install individually
pip install langchain langchain-community langchain-openai duckduckgo-search
```

## Setup (10 seconds)

```bash
# Required: Set OpenAI API key for synthesis
export OPENAI_API_KEY="sk-..."

# Optional: Set Brave API key for premium search
export BRAVE_API_KEY="BSA..."
```

## Usage Examples

### 1. Simple Query (Async)

```python
from app.services.langchain_web_search import langchain_web_search
import asyncio

async def main():
    result = await langchain_web_search(
        query="What is the PE ratio of Apple?",
        max_results=5
    )
    print(result['answer'])

asyncio.run(main())
```

### 2. Simple Query (Sync)

```python
from app.services.langchain_web_search import langchain_web_search_sync

result = langchain_web_search_sync(
    query="Latest Tesla news",
    max_results=5
)
print(f"Found {result['result_count']} results")
```

### 3. Fast Mode (No AI Synthesis)

```python
result = await langchain_web_search(
    query="Quick fact check",
    synthesize_answer=False  # Skip AI synthesis for speed
)
# Returns raw search results only
```

### 4. With Service Class

```python
from app.services.langchain_web_search import LangChainWebSearchService

service = LangChainWebSearchService(
    model="gpt-4o-mini",
    max_results=10
)

result = await service.search(
    query="Market analysis",
    prefer_brave=True
)
```

## Test It

```bash
# Run basic test
python test_langchain_search.py --mode basic

# Run all tests
python test_langchain_search.py

# Custom query
python test_langchain_search.py --query "Your question here"
```

## Check Integration

```python
from app.utils.tools import TOOL_REGISTRY

# Verify LangChain search is available
if "langchain_search" in TOOL_REGISTRY:
    print("✅ LangChain search is integrated!")
    
    # Test it
    result = TOOL_REGISTRY["langchain_search"](
        query="Test query",
        max_results=3
    )
    print(f"Got {result['result_count']} results")
else:
    print("❌ LangChain search not available")
```

## Troubleshooting

**Issue:** Import error
```bash
# Solution: Install dependencies
pip install langchain langchain-community langchain-openai duckduckgo-search
```

**Issue:** "No OpenAI API key"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="sk-..."

# Or use without synthesis
result = await langchain_web_search(query="...", synthesize_answer=False)
```

**Issue:** Slow responses
```python
# Solution: Disable synthesis
result = await langchain_web_search(
    query="...",
    synthesize_answer=False,  # Faster!
    max_results=5              # Fewer results
)
```

## Quick Comparison

| Feature | LangChain Search | Perplexity Search |
|---------|-----------------|-------------------|
| Speed (no synthesis) | ~1-2s | ~1-3s |
| Speed (with synthesis) | ~3-5s | N/A |
| AI Synthesis | ✅ Optional | ❌ Disabled |
| Multi-backend | ✅ Yes | ✅ Yes |
| Easy to extend | ✅ Yes | ⚠️ Custom |

## When to Use

✅ **Use LangChain Search when:**
- You need comprehensive AI-generated answers
- Research queries requiring synthesis
- Want to leverage LangChain ecosystem
- Need structured result handling

⚡ **Use Perplexity Search when:**
- Speed is critical
- Simple fact-checking
- Don't need AI synthesis

## Next Steps

1. **Read full documentation**: `LANGCHAIN_SEARCH.md`
2. **Run tests**: `python test_langchain_search.py`
3. **Integrate with your app**: See examples above
4. **Customize**: Modify service parameters as needed

---

**That's it!** 🎉 You're ready to use LangChain web search in your application.
