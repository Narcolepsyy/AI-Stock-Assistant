# Web Search Integration for AI Stock Tool

## Overview

The AI Stock Tool now includes comprehensive web search capabilities that augment the RAG (Retrieval-Augmented Generation) system with current information from the internet. This enables the AI to provide responses grounded in both internal knowledge and real-time web data.

## Features Added

### 1. Web Search Service (`app/services/web_search_service.py`)

A robust web search service that provides:

- **DuckDuckGo Integration**: Primary search engine (no API key required)
- **Fallback Bing Search**: Secondary search option for better coverage
- **Content Extraction**: Full page content extraction with BeautifulSoup
- **Caching**: TTL-based caching for performance optimization
- **Async/Concurrent**: Full async support with concurrent content fetching

#### Key Functions:
```python
# Basic web search
web_search(query: str, max_results: int = 8, include_content: bool = True)

# News-specific search
web_search_news(query: str, max_results: int = 8, days_back: int = 7)
```

### 2. Enhanced RAG Service (`app/services/enhanced_rag_service.py`)

Combines knowledge base and web search for augmented context:

- **Augmented Search**: Merges KB and web results with relevance scoring
- **Financial Context**: Specialized search for financial topics
- **Source Diversity**: Ensures mix of internal and external sources
- **Context Formatting**: Formats results for LLM consumption

#### Key Functions:
```python
# General augmented search
augmented_rag_search(query: str, kb_k: int = 3, web_results: int = 5)

# Financial-specific search with news
financial_context_search(query: str, symbol: Optional[str] = None)
```

### 3. New AI Tools

Four new tools added to the AI's capabilities:

#### `web_search`
- **Purpose**: Search the web for current information
- **Use Cases**: Recent events, breaking news, current data
- **Parameters**: query, max_results, include_content

#### `web_search_news`
- **Purpose**: Find recent news articles
- **Use Cases**: Latest news on specific topics
- **Parameters**: query, max_results, days_back

#### `augmented_rag_search`
- **Purpose**: Comprehensive search combining KB + web
- **Use Cases**: Complex queries needing multiple sources
- **Parameters**: query, kb_k, web_results, include_web

#### `financial_context_search`
- **Purpose**: Specialized financial context with news
- **Use Cases**: Stock analysis, financial research
- **Parameters**: query, symbol, include_news

### 4. Enhanced UI Indicators

The frontend now shows:

- **Tool Activity Panel**: Visual indicators for active searches
- **Web Search Icons**: Globe icon for web searches, newspaper for news
- **Source Types**: Different colors/icons for KB vs web vs news
- **Real-time Status**: Running/completed/error states with animations

## Usage Examples

### Basic Web Search
```python
# Search for current events
result = web_search("Apple iPhone 16 release date 2024")

# Returns:
{
    "query": "Apple iPhone 16 release date 2024",
    "results": [
        {
            "title": "Apple iPhone 16 Release Date...",
            "url": "https://...",
            "content": "Full page content...",
            "relevance_score": 0.95
        }
    ],
    "count": 5,
    "duration_seconds": 2.3
}
```

### Augmented RAG Search
```python
# Combine knowledge base with web search
result = augmented_rag_search("technical analysis indicators")

# Returns both KB knowledge and current web information
{
    "combined_chunks": [
        {
            "source": "knowledge_base",
            "title": "Technical Analysis Guide",
            "content": "RSI is a momentum indicator...",
            "type": "knowledge_base"
        },
        {
            "source": "web_search",
            "title": "Latest Technical Analysis Trends 2024",
            "content": "New AI-powered indicators...",
            "type": "web_search",
            "url": "https://..."
        }
    ],
    "sources": {
        "knowledge_base": {"count": 2},
        "web_search": {"count": 3}
    }
}
```

### Financial Context Search
```python
# Financial analysis with recent news
result = financial_context_search(
    "Apple earnings analysis", 
    symbol="AAPL",
    include_news=True
)

# Includes KB knowledge, web search, and recent news
{
    "combined_chunks": [...],
    "sources": {
        "knowledge_base": {"count": 2},
        "web_search": {"count": 2},
        "recent_news": {"count": 2}
    },
    "symbol": "AAPL"
}
```

## AI Behavior Changes

### Enhanced System Prompt

The AI now follows enhanced tool usage rules:

1. **For LIVE/CURRENT data**: Use existing stock tools (get_stock_quote, etc.)
2. **For EDUCATIONAL topics**: Use rag_search or financial_context_search
3. **For CURRENT EVENTS**: Use web_search or web_search_news
4. **For COMPLEX QUERIES**: Use augmented_rag_search for comprehensive context

### Intelligent Tool Selection

The AI automatically chooses the best tool based on query type:

- **"Latest news on Tesla"** → `web_search_news`
- **"What is P/E ratio"** → `rag_search` (knowledge base)
- **"Tesla earnings analysis"** → `financial_context_search` (comprehensive)
- **"Current AI developments"** → `web_search` (real-time info)

## Technical Architecture

### Search Flow
```
User Query → AI Model → Tool Selection → Search Service
                                     ↓
Knowledge Base ← Enhanced RAG ← Web Search
                     ↓
             Relevance Scoring
                     ↓
           Formatted Response
```

### Caching Strategy
- **Web Search Cache**: 5-minute TTL for search results
- **Article Content Cache**: 1-hour TTL for extracted content
- **Knowledge Base**: Persistent with change detection

### Performance Optimizations
- **Concurrent Fetching**: Multiple articles fetched simultaneously
- **Content Limits**: Configurable character limits per source
- **Result Ranking**: Weighted scoring between KB and web sources
- **Timeouts**: Reasonable timeouts to prevent hanging

## Configuration

### Environment Variables
```bash
# Web search settings
NEWS_USER_AGENT="Mozilla/5.0 ..."
ARTICLE_CACHE_SIZE=2048
ARTICLE_TTL_SECONDS=3600
NEWS_FETCH_MAX_WORKERS=6

# RAG integration
RAG_ENABLED=true
KNOWLEDGE_DIR=knowledge
CHROMA_DIR=.chroma
```

### Customization Options
- **Search Providers**: Can add more search engines
- **Content Extractors**: Configurable extraction rules
- **Cache Settings**: Adjustable TTL and size limits
- **Concurrency**: Configurable worker limits

## Benefits

### For Users
- **Current Information**: Access to real-time data and events
- **Comprehensive Answers**: Responses combine multiple information sources
- **Source Transparency**: Clear indication of information sources
- **Better Context**: Enhanced understanding through diverse perspectives

### For Developers
- **Modular Design**: Services can be used independently
- **Extensible**: Easy to add new search providers or extractors
- **Performance**: Optimized caching and async operations
- **Monitoring**: Built-in logging and error handling

## Future Enhancements

### Planned Features
1. **More Search Providers**: Google, specialized financial APIs
2. **Advanced Filtering**: Date ranges, source types, domains
3. **Semantic Ranking**: Better relevance scoring with embeddings
4. **User Preferences**: Customizable source weights and preferences
5. **Search Analytics**: Usage statistics and optimization insights

### Integration Opportunities
1. **Real-time Alerts**: Web monitoring for specific queries
2. **Trend Analysis**: Track information changes over time
3. **Source Verification**: Cross-reference information across sources
4. **Personalized Results**: User-specific search customization

## Testing and Validation

The system has been tested with:
- ✅ Basic web search functionality
- ✅ News search with time filtering
- ✅ Augmented RAG combining multiple sources
- ✅ Financial context search with symbol-specific news
- ✅ UI indicators and tool activity display
- ✅ Error handling and fallback mechanisms

## Summary

The web search integration transforms the AI Stock Tool from a knowledge base assistant into a comprehensive research tool that can provide current, accurate information by combining internal knowledge with real-time web data. This enables more informed financial analysis and broader general assistance capabilities.