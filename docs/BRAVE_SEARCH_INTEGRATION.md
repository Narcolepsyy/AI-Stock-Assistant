# Brave Search API Integration

## Overview
The perplexity web search service now supports **dual search engine integration** with both Brave Search API and DuckDuckGo for comprehensive, high-quality search results.

## Features

### 🔍 **Dual Search Engine Support**
- **Brave Search API**: High-quality, ad-free search results with excellent relevance
- **DuckDuckGo**: Privacy-focused search as backup/complement
- **Concurrent Execution**: Both searches run simultaneously for faster results
- **Smart Deduplication**: Removes duplicate URLs across providers
- **Quality Prioritization**: Brave results prioritized due to higher quality

### 🎯 **Enhanced Search Quality**
- **Better Coverage**: Multiple search engines provide more comprehensive results
- **Higher Relevance**: Brave Search typically provides more accurate results
- **Recency Support**: Brave Search supports time-based filtering (past week/month/year)
- **Fallback Resilience**: If one service fails, the other continues working

### ⚡ **Performance Optimizations**
- **Async Concurrent Searches**: Multiple providers searched simultaneously
- **Smart Result Distribution**: Results split between providers (e.g., 4 from each for 8 total)
- **Relevance Scoring**: Results sorted by quality and relevance scores
- **Intelligent Fallback**: Graceful degradation if API keys are missing

## Configuration

### 1. **Get Brave Search API Key**
- Visit: https://api.search.brave.com/app/keys
- Sign up and get your free API key
- Free tier provides: 2,000 queries/month

### 2. **Configure Environment**
Add to your `.env` file:
```bash
# Brave Search API Configuration
BRAVE_API_KEY="your_brave_api_key_here"
```

### 3. **Restart Application**
```bash
# Restart your FastAPI server
uvicorn main:app --reload
```

## Usage

The integration is **completely transparent** - no code changes needed:

```python
# All existing web search calls now use both engines automatically
result = perplexity_web_search(
    query="Latest AI developments 2025",
    max_results=8,
    synthesize_answer=True,
    include_recent=True
)
```

## Search Provider Comparison

| Feature | Brave Search | DuckDuckGo |
|---------|--------------|------------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Recency Filters** | ✅ Yes | ❌ No |
| **Rate Limits** | 2K/month free | Unlimited |
| **Commercial Use** | ✅ Allowed | ✅ Allowed |
| **Privacy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **API Stability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Technical Implementation

### **Architecture**
```
┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │  Enhanced Query │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│        Concurrent Search Tasks          │
├─────────────────┬───────────────────────┤
│  Brave Search   │    DuckDuckGo         │
│  (if API key)   │    (always active)    │
└─────────────────┴───────────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Result Combination              │
│  • Deduplicate URLs                     │
│  • Merge and sort by relevance          │
│  • Assign citation IDs                  │
└─────────────────────────────────────────┘
```

### **Smart Fallback Logic**
1. **Both APIs Available**: Use both for comprehensive results
2. **Only DuckDuckGo**: Falls back gracefully to single provider
3. **API Failures**: Individual failures don't break the entire search
4. **Rate Limiting**: Automatically handles API limits

## Benefits for Users

### 🎯 **Better Search Results**
- **Higher Accuracy**: Brave Search provides more relevant results
- **Comprehensive Coverage**: Two engines find more sources
- **Recent Content**: Time-based filtering for current events
- **Reduced Bias**: Multiple sources provide balanced perspectives

### 🚀 **Improved Performance**
- **Faster Response**: Concurrent searches reduce wait time
- **Better Reliability**: Redundancy prevents search failures
- **Quality Scoring**: Best results bubble to the top

### 💡 **Enhanced AI Responses**
- **More Sources**: Better synthesis with diverse information
- **Higher Confidence**: More data points increase answer quality  
- **Current Information**: Recent results improve factual accuracy

## Testing

Run the integration test:
```bash
source .venv/bin/activate
python test_brave_search_integration.py
```

Expected output:
- ✅ Brave API detection (with/without key)
- 📊 Source distribution across providers
- ⏱️ Performance metrics
- 💡 Quality indicators

## Monitoring

The service logs important events:
- Brave Search API availability
- Search performance metrics
- Error handling and fallbacks
- Provider result distribution

Check logs for:
```
INFO: Brave Search API enabled
INFO: Search completed - Brave: 4 results, DuckDuckGo: 4 results
WARNING: Brave Search API rate limit reached, using DuckDuckGo only
```

## Migration Notes

- **Backward Compatible**: All existing code continues to work
- **No Breaking Changes**: Same API interface maintained
- **Gradual Rollout**: Can be enabled/disabled via environment variables
- **Zero Downtime**: Hot-swappable configuration

## Cost Considerations

- **Brave Search**: 2,000 queries/month free, then paid tiers
- **DuckDuckGo**: Completely free but rate-limited
- **Recommended**: Start with free tier, upgrade if needed
- **Fallback**: Always works even without Brave API key