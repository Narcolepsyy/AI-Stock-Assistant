# Perplexity Web Search Enhancements with BM25 + Azure GPT OSS 120B + Text Embeddings

## Overview

This document summarizes the enhanced Perplexity-style web search service that now incorporates advanced ranking algorithms and your Azure-deployed models for superior search results and answer synthesis.

## Key Enhancements

### 1. BM25 Lexical Ranking System 🔍
- **Added**: BM25 (Best Match 25) scoring algorithm for improved lexical relevance
- **Implementation**: `rank-bm25` library with Okapi BM25 variant
- **Features**:
  - Text preprocessing and tokenization
  - Document corpus ranking against query terms
  - Normalized scoring (0-1 range)
  - Fallback handling for edge cases

### 2. Semantic Similarity with Azure Text Embeddings 🧠
- **Integration**: Azure OpenAI text embeddings for semantic understanding
- **Model Used**: Your deployed `text-embedding-3-large` model
- **Features**:
  - Query and document embedding generation
  - Cosine similarity calculation
  - Batch processing for efficiency
  - Fallback implementation when scikit-learn unavailable

### 3. Azure GPT OSS 120B Answer Synthesis 🤖
- **Primary Model**: Your Azure-deployed GPT OSS 120B model
- **Features**:
  - Prioritizes your custom deployment over OpenAI
  - Optimized prompts for large model
  - Enhanced citation requirements
  - Graceful fallback to OpenAI if Azure fails

### 4. Hybrid Ranking System 📊
- **Multi-Signal Scoring**: Combines multiple relevance signals
- **Weights**:
  - BM25 (Lexical): 40%
  - Semantic Similarity: 40%
  - Original Relevance: 10%
  - Content Quality: 10%
- **Normalization**: All scores normalized to 0-1 range for fair combination

## Technical Implementation

### Enhanced SearchResult Data Structure
```python
@dataclass
class SearchResult:
    # Original fields
    title: str
    url: str
    snippet: str
    content: str = ""
    relevance_score: float = 0.0
    
    # Enhanced scoring fields
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0
    embedding_vector: Optional[List[float]] = None
```

### New Dependencies Added
```
rank-bm25>=0.2.2        # BM25 ranking algorithm
html2text>=2020.1.16    # Enhanced content extraction
nltk>=3.8.1             # Text processing (optional)
scikit-learn>=1.3.0     # Cosine similarity (with fallback)
```

### Configuration Integration
- Uses existing Azure OpenAI configuration
- Leverages `AZURE_OPENAI_DEPLOYMENT_OSS_120B` for synthesis
- Utilizes `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` for semantic scoring
- Maintains backward compatibility with OpenAI fallbacks

## Search Process Flow

1. **Enhanced Web Search** (DDGS with improved relevance filtering)
2. **Content Extraction** (BeautifulSoup + html2text optimization)
3. **BM25 Lexical Scoring** (Query-document matching)
4. **Semantic Similarity Scoring** (Azure embeddings + cosine similarity)
5. **Hybrid Score Calculation** (Weighted combination of all signals)
6. **Result Re-ranking** (Sort by combined score)
7. **Answer Synthesis** (Azure GPT OSS 120B with enhanced prompts)
8. **Citation Building** (Source tracking and reference generation)

## Performance Optimizations

### BM25 Optimizations
- Efficient text preprocessing without heavy NLTK dependencies
- Batch processing for multiple documents
- Normalized scoring for consistent ranking

### Embedding Optimizations
- Batch embedding generation for efficiency
- Connection reuse and proper cleanup
- Timeout handling for large models
- Fallback cosine similarity implementation

### GPT OSS 120B Optimizations
- Longer timeout (120s) for large model
- Optimized token limits (1000 max_tokens)
- Lower temperature (0.3) for focused responses
- Model-specific system prompts

## Fallback Mechanisms

### Dependency Fallbacks
- **No scikit-learn**: Custom cosine similarity implementation
- **No numpy**: Pure Python vector operations
- **No NLTK**: Basic regex-based tokenization

### Service Fallbacks
- **Azure GPT OSS 120B fails**: Falls back to OpenAI GPT-4o-mini
- **Azure embeddings fail**: Continues with BM25 + original scoring
- **BM25 fails**: Uses original relevance scoring

## Testing and Validation

### Test Scripts Created
1. **`test_enhanced_perplexity.py`**: Comprehensive test suite
2. **`validate_enhancements.py`**: Simple validation script

### Test Coverage
- Dependency availability testing
- BM25 scoring functionality
- Azure model integration
- Embedding generation and similarity
- End-to-end search workflow
- Performance benchmarking

## Usage Examples

### Basic Usage (Synchronous)
```python
from app.services.perplexity_web_search import perplexity_web_search

result = perplexity_web_search(
    query="Tesla stock analysis 2024",
    max_results=5,
    synthesize_answer=True,
    include_recent=True
)

# Access enhanced scores
for source in result['sources']:
    print(f"BM25: {source['bm25_score']:.3f}")
    print(f"Semantic: {source['semantic_score']:.3f}")
    print(f"Combined: {source['combined_score']:.3f}")
```

### Advanced Usage (Async)
```python
from app.services.perplexity_web_search import get_perplexity_service

service = get_perplexity_service()
response = await service.perplexity_search(
    query="AI market trends",
    max_results=8,
    synthesize_answer=True
)
```

## Configuration Requirements

### Environment Variables Needed
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Model Deployments
AZURE_OPENAI_DEPLOYMENT_OSS_120B=gpt-oss-120b
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
```

## Benefits of Enhanced System

### Improved Relevance
- **BM25**: Better lexical matching for keyword-based queries
- **Semantic**: Understanding of query intent and context
- **Hybrid**: Best of both lexical and semantic ranking

### Better Answer Quality
- **GPT OSS 120B**: More capable model for complex synthesis
- **Enhanced Citations**: More thorough source attribution
- **Fallback Safety**: Reliable operation even if components fail

### Performance Gains
- **Batch Processing**: Efficient embedding generation
- **Connection Reuse**: Reduced API call overhead
- **Smart Caching**: Leverages existing cache infrastructure

### Scalability
- **Modular Design**: Easy to adjust weights and add new signals
- **Async Operations**: Non-blocking operations for better throughput
- **Resource Management**: Proper cleanup and connection handling

## Monitoring and Debugging

### Enhanced Logging
- BM25 scoring details
- Embedding generation status
- Model selection and fallback events
- Performance timing for each stage

### Response Metadata
```python
{
    "search_time": 2.5,      # Time for web search
    "synthesis_time": 3.2,   # Time for answer generation
    "total_time": 6.1,       # Total processing time
    "confidence_score": 0.85, # Overall confidence
    "method": "perplexity_enhanced"
}
```

## Future Enhancement Opportunities

1. **Query Expansion**: Use embeddings for query enhancement
2. **Result Clustering**: Group similar results for diversity
3. **Personalization**: User-specific ranking adjustments
4. **Real-time Learning**: Adaptive weight tuning based on feedback
5. **Multi-modal Integration**: Image and video search capabilities

## Conclusion

The enhanced Perplexity web search service now provides:
- **Superior Ranking**: BM25 + semantic similarity hybrid approach
- **Better Answers**: Azure GPT OSS 120B with optimized prompts  
- **Robust Fallbacks**: Multiple layers of reliability
- **Performance**: Optimized for your Azure infrastructure
- **Scalability**: Ready for production deployment

Your search service is now equipped with state-of-the-art ranking algorithms and leverages your premium Azure deployments for the best possible search and synthesis experience.