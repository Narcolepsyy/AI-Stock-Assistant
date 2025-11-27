"""Perplexity-style web search service with answer synthesis and source citations."""
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import OrderedDict
import aiohttp
import json
import re
import os
import hashlib
from contextlib import suppress
from urllib.parse import quote_plus, urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup
import html2text
from dataclasses import dataclass, field
from openai import AsyncOpenAI, AsyncAzureOpenAI
from app.services.openai_client import get_client, get_client_for_model
from ddgs import DDGS
# Import get_openai_client from the module or using the local function
# Enhanced ranking imports
from rank_bm25 import BM25Okapi
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback cosine similarity implementation
    SKLEARN_AVAILABLE = False
    import math
from app.core.config import (
    AZURE_OPENAI_DEPLOYMENT_OSS_120B, 
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    get_system_prompt_for_model,
    DEFAULT_MODEL,
    DDGS_REGION,
    DDGS_SAFESEARCH,
    DDGS_TIMELIMIT,
    BRAVE_API_KEY
)

logger = logging.getLogger(__name__)

# Enhanced LRU Cache with TTL implementation
class LRUCacheWithTTL:
    """LRU Cache with TTL support for better memory management."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and is not expired."""
        if key not in self._cache:
            return None
        
        # Check TTL
        if self._is_expired(key):
            self._remove(key)
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting LRU items if necessary."""
        current_time = datetime.now().timestamp()
        
        if key in self._cache:
            # Update existing item
            self._cache[key] = value
            self._timestamps[key] = current_time
            self._cache.move_to_end(key)
        else:
            # Add new item
            if len(self._cache) >= self.max_size:
                # Evict LRU item
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
            
            self._cache[key] = value
            self._timestamps[key] = current_time
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        
        timestamp = self._timestamps[key]
        return (datetime.now().timestamp() - timestamp) > self.ttl_seconds
    
    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear_expired(self) -> int:
        """Clear all expired entries and return count of removed items."""
        expired_keys = [key for key in self._cache if self._is_expired(key)]
        for key in expired_keys:
            self._remove(key)
        return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()

# Performance optimizations: Enhanced in-memory caches with TTL
_embeddings_cache = LRUCacheWithTTL(max_size=200, ttl_seconds=3600)  # 1 hour TTL
_search_cache = LRUCacheWithTTL(max_size=100, ttl_seconds=1800)      # 30 min TTL
_content_cache = LRUCacheWithTTL(max_size=150, ttl_seconds=7200)     # 2 hour TTL
_query_enhancement_cache = LRUCacheWithTTL(max_size=300, ttl_seconds=1800)  # 30 min TTL for synthesized queries


def _build_search_cache_key(
    query: str,
    max_results: int,
    include_recent: bool,
    time_limit: Optional[str]
) -> str:
    """Build a deterministic cache key for search results."""
    payload = json.dumps(
        {
            "query": query,
            "max_results": max_results,
            "include_recent": include_recent,
            "time_limit": time_limit or "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _get_cache_key(f"search::{payload}")


def _serialize_search_results(results: List["SearchResult"]) -> List[Dict[str, Any]]:
    """Serialize search results for cache storage (exclude heavy fields)."""
    serialized: List[Dict[str, Any]] = []
    for item in results:
        serialized.append(
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "relevance_score": item.relevance_score,
                "timestamp": item.timestamp,
                "source": item.source,
                "citation_id": item.citation_id,
            }
        )
    return serialized


def _deserialize_search_results(data: Optional[List[Dict[str, Any]]]) -> List["SearchResult"]:
    """Deserialize cached search results back into SearchResult objects."""
    if not data:
        return []

    results: List[SearchResult] = []
    for entry in data:
        results.append(
            SearchResult(
                title=entry.get("title", ""),
                url=entry.get("url", ""),
                snippet=entry.get("snippet", ""),
                relevance_score=entry.get("relevance_score", 0.0),
                timestamp=entry.get("timestamp", ""),
                source=entry.get("source", ""),
                citation_id=entry.get("citation_id", 0),
            )
        )

    return results

# Domain priors for enhanced ranking precision
DOMAIN_PRIORS = {
    # High-quality sources (1.2-1.3x boost)
    'high_quality': {
        'wikipedia.org': 1.2, 'britannica.com': 1.2, 'nature.com': 1.3,
        'science.org': 1.3, 'pubmed.ncbi.nlm.nih.gov': 1.3, 'arxiv.org': 1.2,
        'ieee.org': 1.2, 'acm.org': 1.2, 'springer.com': 1.2,
        'mit.edu': 1.3, 'stanford.edu': 1.3, 'harvard.edu': 1.3,
        'oxford.ac.uk': 1.3, 'cambridge.org': 1.3
    },
    # Financial sources (1.1-1.3x boost for financial queries)
    'financial': {
        'bloomberg.com': 1.2, 'reuters.com': 1.2, 'wsj.com': 1.2,
        'ft.com': 1.2, 'marketwatch.com': 1.1, 'yahoo.com': 1.0,
        'sec.gov': 1.3, 'federalreserve.gov': 1.3, 'imf.org': 1.2,
        'worldbank.org': 1.2, 'investopedia.com': 1.1
    },
    # Technical sources (1.1-1.2x boost for tech queries)
    'technical': {
        'stackoverflow.com': 1.1, 'github.com': 1.1, 'docs.python.org': 1.2,
        'developer.mozilla.org': 1.2, 'w3.org': 1.2, 'ietf.org': 1.2,
        'techcrunch.com': 1.0, 'arstechnica.com': 1.1, 'wired.com': 1.0
    },
    # News sources (1.0-1.1x boost for current events)
    'news': {
        'bbc.com': 1.1, 'cnn.com': 1.0, 'npr.org': 1.1, 'pbs.org': 1.1,
        'ap.org': 1.1, 'reuters.com': 1.1, 'theguardian.com': 1.1,
        'nytimes.com': 1.1, 'washingtonpost.com': 1.0
    },
    # Low-quality sources (0.6-0.7x penalty)
    'low_quality': {
        'quora.com': 0.7, 'answers.com': 0.6, 'ehow.com': 0.6,
        'wikihow.com': 0.7, 'ask.com': 0.6, 'chacha.com': 0.5
    }
}

# Malicious domains denylist for security
MALICIOUS_DOMAINS = {
    'malware.com', 'phishing.net', 'spam.org', 'virus.co',
    'badsite.ru', 'malicious.tk', 'trojan.ml', 'scam.site'
}

# Configuration constants for Brave Search (high-quality source)
MIN_SEARCH_RESULTS_THRESHOLD = 3  # Minimum results before DDGS fallback
BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_QUALITY_BONUS = 0.2  # Quality bonus for Brave results

# Enhanced domain quality assessment with regional and topic-specific scoring
TRUSTED_FINANCIAL_DOMAINS = {
    # Tier 1: Premium financial sources (highest quality)
    'tier1': {
        'bloomberg.com', 'reuters.com', 'ft.com', 'wsj.com', 'economist.com',
        'sec.gov', 'federalreserve.gov', 'treasury.gov', 'bis.org', 'imf.org'
    },
    # Tier 2: Major financial news and analysis
    'tier2': {
        'cnbc.com', 'marketwatch.com', 'barrons.com', 'forbes.com',
        'businessinsider.com', 'seekingalpha.com', 'morningstar.com',
        'nasdaq.com', 'nyse.com', 'finance.yahoo.com', 'investing.com'
    },
    # Tier 3: Specialized and regional financial sources
    'tier3': {
        'nikkei.com', 'japantimes.co.jp', 'asahi.com', 'mainichi.jp',
        'ecb.europa.eu', 'bankofengland.co.uk', 'fool.com', 'zacks.com',
        'marketbeat.com', 'statista.com', 'investopedia.com'
    },
    # Academic and research institutions
    'academic': {
        'ssrn.com', 'nber.org', 'bls.gov', 'census.gov', 'oecd.org',
        'mckinsey.com', 'pwc.com', 'deloitte.com', 'ey.com', 'kpmg.com',
        'bcg.com', 'bain.com'
    }
}

# Additional trusted Japanese financial/corporate domains
TRUSTED_JP_FINANCIAL_DOMAINS = {
    'netbk.co.jp',            # 住信SBIネット銀行 (NEOBANK)
    'sbigroup.co.jp',         # SBIグループ
    'kabutore.biz',           # 投資/企業分析
    'kabutan.jp',             # 株探
    'irbank.net',             # IRバンク
    'tdnet.info',             # TDnet (適時開示)
    'xbrl.tdnet.info',        # TDnet XBRL
    'toyokeizai.net',         # 東洋経済
    'diamond.jp',             # ダイヤモンド
    'itmedia.co.jp',          # ITmedia
}

# Enhanced untrusted domains with categories
UNTRUSTED_DOMAINS = {
    # Social media and forums (can be noisy for financial data)
    'social': {
        'reddit.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'youtube.com', 'tiktok.com', 'pinterest.com'
    },
    # Ad-heavy/clickbait sites
    'clickbait': {
        'buzzfeed.com', 'clickhole.com', 'upworthy.com', 'listverse.com'
    },
    # Personal blogs and unverified content
    'blogs': {
        'medium.com', 'wordpress.com', 'blogspot.com', 'wix.com',
        'squarespace.com', 'tumblr.com'
    },
    # Wiki-style (can be unreliable for real-time financial data)
    'wiki': {
        'wikipedia.org', 'fandom.com', 'wikia.com'
    },
    # Content farms and low-quality aggregators
    'farms': {
        'ehow.com', 'wikihow.com', 'answers.com', 'ask.com'
    }
}

# Financial search verticals for enhanced relevance
FINANCIAL_SEARCH_VERTICALS = {
    'news': ['earnings', 'financial results', 'market news', 'stock news'],
    'web': ['analysis', 'forecast', 'prediction', 'outlook', 'research'],
    'videos': ['earnings call', 'investor presentation', 'conference']
}

class BraveSearchClient:
    """High-quality Brave Search API client for enhanced search results with proper lifecycle management."""
    
    def __init__(self):
        """Initialize Brave Search client with API key and rate limiting."""
        # Normalize API key to avoid whitespace/newline-induced auth failures
        self.api_key = (BRAVE_API_KEY or "").strip()
        self.base_url = BRAVE_API_BASE_URL
        self.timeout = aiohttp.ClientTimeout(total=10, connect=3)
        # Fix: Correct Brave API headers format
        self.headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'Accept-Language': 'ja-JP,ja;q=0.9,en;q=0.8',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
        }
        if self.api_key:
            self.headers['X-Subscription-Token'] = self.api_key
        self._session = None
        
        # Rate limiting for Brave free tier (1 query/second)
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # 1 second between requests
        self._rate_lock = asyncio.Lock()  # Add lock for thread-safe rate limiting
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @property
    def is_closed(self) -> bool:
        """Check if the client is closed."""
        return self._closed
        
    @property
    def is_available(self) -> bool:
        """Check if Brave Search API is available."""
        return bool(self.api_key) and not self._closed
    
    async def _get_session(self) -> Optional[aiohttp.ClientSession]:
        """Get or create HTTP session for Brave API."""
        if not self.is_available:
            return None
            
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=5, 
                limit_per_host=3,
                ttl_dns_cache=300,
                use_dns_cache=True,
                force_close=True,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
                connector=connector
            )
        return self._session
    
    async def search(
        self, 
        query: str, 
        count: int = 10, 
        freshness: Optional[str] = None,
        country: str = "ALL"
    ) -> List[Dict[str, Any]]:
        """
        Perform high-quality search using Brave Search API with enhanced parameters.
        
        Args:
            query: Search query
            count: Number of results (max 20)
            freshness: Time filter ('pd'=day, 'pw'=week, 'pm'=month, 'py'=year)
            country: Country filter (default: "ALL")
            
        Returns:
            List of search results with high relevance scores and quality filtering
        """
        if not self.is_available:
            logger.debug("Brave Search API not available")
            return []
            
        session = await self._get_session()
        if not session:
            return []
        
        # Validate and sanitize parameters to avoid 422 errors
        valid_countries = {'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'IT', 'ES', 'JP', 'KR', 'CN', 'ALL'}
        valid_freshness = {'pd', 'pw', 'pm', 'py'}
        valid_languages = {'en', 'ja', 'zh', 'ko', 'de', 'fr', 'es', 'it'}
        try:
            # Thread-safe rate limiting
            async with self._rate_lock:
                current_time = time.time()
                time_since_last_request = current_time - self._last_request_time
                
                if time_since_last_request < self._min_request_interval:
                    sleep_time = self._min_request_interval - time_since_last_request
                    logger.debug(f"Brave rate limiting: waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                
                self._last_request_time = time.time()
            
            # Determine optimal locale (country + language) for the query
            try:
                locale_country, locale_lang = self._get_optimal_locale(query, country)
            except Exception:
                # Fallbacks if detection fails
                locale_country, locale_lang = (country if country != "ALL" else "US", "en")

            # Build simple search parameters with locale (use only supported params)
            params = {
                'q': query,
                'count': min(max(count, 1), 20),  # Ensure valid range 1-20
                'safesearch': 'moderate'
            }

            # Add country parameter if specified and valid
            if locale_country and locale_country != "ALL" and locale_country in valid_countries:
                params['country'] = locale_country

            # Add language hints for Brave to improve non-English results
            if locale_lang and locale_lang in valid_languages:
                # Normalize and map UI language to region-specific where useful
                ui_lang_map = {
                    'ja': 'ja-JP',
                    'zh': 'zh-CN',
                    'ko': 'ko-KR',
                    'en': 'en-US',
                    'de': 'de-DE',
                    'fr': 'fr-FR',
                    'es': 'es-ES',
                    'it': 'it-IT'
                }
                # search_lang expects ISO 639-1; ensure 'ja' for Japanese
                search_lang = 'ja' if locale_lang in ('ja', 'jp') else locale_lang
                params['search_lang'] = search_lang
                # Avoid sending ui_lang to reduce validation errors; rely on Accept-Language header instead

            # If the query is clearly Japanese, force JP/JA parameters (ensure 'ja' not 'jp')
            if any(ord(c) > 127 for c in query):
                params['country'] = 'JP'
                # Brave expects ISO 639-1 language code; use 'ja' for Japanese
                params['search_lang'] = 'ja'
                # Do not send ui_lang
            
            # Add freshness filter if specified
            if freshness:
                params['freshness'] = freshness
            
            logger.debug(f"Brave Search query: '{query}' with params: {params}")
            
            async with session.get(self.base_url, params=params) as response:
                logger.debug(f"Brave Search response: {response.status}")
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        # Quick diagnostic: log available top-level keys and results count
                        if isinstance(data, dict):
                            web_info = data.get('web')
                            results_count = len(web_info.get('results', [])) if isinstance(web_info, dict) and isinstance(web_info.get('results'), list) else 0
                            logger.debug(f"Brave payload keys: {list(data.keys())}; web.results: {results_count}")
                        
                        # Schema validation: Check for required fields
                        if not isinstance(data, dict):
                            logger.warning("Brave API returned non-dict response")
                            return []
                        
                        web_data = data.get('web')
                        if not isinstance(web_data, dict):
                            logger.warning("Brave API response missing 'web' field")
                            return []
                        
                        results_data = web_data.get('results')
                        if not isinstance(results_data, list):
                            logger.warning("Brave API response missing 'web.results' list")
                            return []
                        
                        raw_results = self._parse_brave_results(data, query)
                        
                        # If Brave returned 200 but zero results, try a fallback query without locale hints
                        if not raw_results:
                            logger.info("Brave 200 OK but 0 results; retrying without locale/language hints")
                            minimal_params = {
                                'q': query,
                                'count': min(count, 10),
                                'safesearch': 'moderate'
                            }
                            # Gentle backoff before retrying
                            await asyncio.sleep(self._min_request_interval + 0.2)
                            async with session.get(self.base_url, params=minimal_params) as retry_response:
                                if retry_response.status == 200:
                                    try:
                                        data2 = await retry_response.json()
                                        raw_results = self._parse_brave_results(data2, query)
                                    except Exception as e:
                                        logger.debug(f"Fallback parse after 0-results failed: {e}")
                                        raw_results = []
                        
                        # Apply post-retrieval quality filtering and reranking
                        quality_results = self._apply_quality_filtering(raw_results, query)
                        reranked_results = self._rerank_by_quality(quality_results, query)
                        
                        logger.info(f"Brave Search: {len(raw_results)} raw → {len(quality_results)} filtered → {len(reranked_results)} final")
                        return reranked_results
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Brave Search API JSON decode error: {e}")
                        return []
                    except Exception as e:
                        logger.warning(f"Brave Search API response parsing error: {e}")
                        return []
                elif response.status == 422:
                    # Handle parameter validation errors by retrying with relaxed params
                    try:
                        error_body = await response.text()
                    except Exception:
                        error_body = "<unavailable>"
                    logger.warning(
                        "Brave Search API parameter validation error (422), retrying with relaxed params; body=%s",
                        error_body[:500]
                    )
                    if "SUBSCRIPTION_TOKEN_INVALID" in error_body:
                        logger.warning("Brave API token appears invalid. Check BRAVE_API_KEY value/config.")
                    
                    # Retry with minimal parameters
                    minimal_params = {
                        'q': query,
                        'count': min(count, 10)  # Reduce count
                    }
                    # Respect rate limiting before retry
                    await asyncio.sleep(self._min_request_interval + 0.2)
                    async with session.get(self.base_url, params=minimal_params) as retry_response:
                        if retry_response.status == 200:
                            try:
                                data = await retry_response.json()
                                raw_results = self._parse_brave_results(data, query)
                                quality_results = self._apply_quality_filtering(raw_results, query)
                                reranked_results = self._rerank_by_quality(quality_results, query)
                                logger.info(f"Brave Search retry: {len(reranked_results)} results")
                                return reranked_results
                            except Exception as e:
                                logger.warning(f"Brave Search retry also failed: {e}")
                                return []
                        elif retry_response.status == 422:
                            try:
                                retry_error_body = await retry_response.text()
                            except Exception:
                                retry_error_body = "<unavailable>"
                            logger.warning(
                                "Brave Search retry failed with 422; body=%s. Will try bare-minimum params.",
                                retry_error_body[:500]
                            )
                            # Final ultra-minimal attempt with only 'q'
                            await asyncio.sleep(self._min_request_interval + 0.2)
                            async with session.get(self.base_url, params={'q': query}) as retry2:
                                if retry2.status == 200:
                                    try:
                                        data = await retry2.json()
                                        raw_results = self._parse_brave_results(data, query)
                                        quality_results = self._apply_quality_filtering(raw_results, query)
                                        reranked_results = self._rerank_by_quality(quality_results, query)
                                        logger.info(f"Brave Search ultra-minimal retry: {len(reranked_results)} results")
                                        return reranked_results
                                    except Exception as e:
                                        logger.warning(f"Brave ultra-minimal retry parse failed: {e}")
                                else:
                                    logger.warning(f"Brave ultra-minimal retry failed with status: {retry2.status}")
                        elif retry_response.status == 429:
                            # Wait and make one final retry attempt
                            logger.warning("Brave Search retry hit rate limit (429); backing off and retrying once")
                            await asyncio.sleep(self._min_request_interval + 0.8)
                            async with session.get(self.base_url, params=minimal_params) as retry2:
                                if retry2.status == 200:
                                    try:
                                        data = await retry2.json()
                                        raw_results = self._parse_brave_results(data, query)
                                        quality_results = self._apply_quality_filtering(raw_results, query)
                                        reranked_results = self._rerank_by_quality(quality_results, query)
                                        logger.info(f"Brave Search second retry: {len(reranked_results)} results")
                                        return reranked_results
                                    except Exception as e:
                                        logger.warning(f"Brave Search second retry parse failed: {e}")
                                else:
                                    logger.warning(f"Brave Search second retry failed with status: {retry2.status}")
                        else:
                            logger.warning(f"Brave Search retry failed with status: {retry_response.status}")
                            return []
                
                elif response.status == 401:
                    logger.warning("Brave Search API unauthorized (401) - check API key")
                    return []
                elif response.status == 403:
                    logger.warning("Brave Search API forbidden (403) - subscription token or plan issue")
                    return []
                elif response.status == 429:
                    logger.warning("Brave Search API rate limit exceeded; waiting before one retry")
                    await asyncio.sleep(self._min_request_interval + 0.8)
                    try:
                        async with session.get(self.base_url, params=params) as retry_response:
                            if retry_response.status == 200:
                                try:
                                    data = await retry_response.json()
                                    raw_results = self._parse_brave_results(data, query)
                                    quality_results = self._apply_quality_filtering(raw_results, query)
                                    reranked_results = self._rerank_by_quality(quality_results, query)
                                    logger.info(f"Brave Search post-429 retry: {len(reranked_results)} results")
                                    return reranked_results
                                except Exception as e:
                                    logger.warning(f"Brave Search post-429 parse failed: {e}")
                            else:
                                logger.warning(f"Brave Search post-429 retry failed with status: {retry_response.status}")
                    except Exception as e:
                        logger.debug(f"Brave Search post-429 retry error: {e}")
                    return []
                elif response.status >= 500:
                    # Handle server errors with retry logic
                    logger.warning(f"Brave Search API server error: {response.status}, implementing retry...")
                    
                    # Implement exponential backoff with jitter
                    for retry_attempt in range(2):  # Max 2 retries for 5xx errors
                        retry_delay = (2 ** retry_attempt) + random.uniform(0, 1)  # Exponential backoff + jitter
                        logger.debug(f"Retrying Brave Search in {retry_delay:.2f}s (attempt {retry_attempt + 1}/2)")
                        await asyncio.sleep(retry_delay)
                        
                        # Retry with same parameters
                        try:
                            async with session.get(self.base_url, params=params) as retry_response:
                                if retry_response.status == 200:
                                    try:
                                        data = await retry_response.json()
                                        if self._validate_response_schema(data):
                                            raw_results = self._parse_brave_results(data, query)
                                            quality_results = self._apply_quality_filtering(raw_results, query)
                                            reranked_results = self._rerank_by_quality(quality_results, query)
                                            logger.info(f"Brave Search retry succeeded: {len(reranked_results)} results")
                                            return reranked_results
                                    except Exception as e:
                                        logger.warning(f"Brave Search retry parsing failed: {e}")
                                        continue
                                elif retry_response.status >= 500:
                                    logger.debug(f"Retry attempt {retry_attempt + 1} still failed with {retry_response.status}")
                                    continue
                                else:
                                    logger.debug(f"Retry failed with different error: {retry_response.status}")
                                    break
                        except asyncio.TimeoutError:
                            logger.debug(f"Retry attempt {retry_attempt + 1} timed out")
                            continue
                        except Exception as e:
                            logger.debug(f"Retry attempt {retry_attempt + 1} failed: {e}")
                            continue
                    
                    logger.warning("All Brave Search retry attempts failed")
                    return []
                else:
                    # Try to get error details from response body
                    try:
                        error_text = await response.text()
                        logger.warning(f"Brave Search API error {response.status}: {error_text[:200]}")
                    except:
                        logger.warning(f"Brave Search API error: {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.warning("Brave Search API timeout")
            return []
        except Exception as e:
            logger.debug(f"Brave Search API error: {e}")
            return []
    
    def _validate_response_schema(self, data: Dict[str, Any]) -> bool:
        """Validate Brave Search API response schema."""
        if not isinstance(data, dict):
            return False
        
        # Accept either web or news sections as valid result carriers
        web_data = data.get('web')
        news_data = data.get('news')
        
        has_web = isinstance(web_data, dict) and isinstance(web_data.get('results'), list)
        has_news = isinstance(news_data, dict) and isinstance(news_data.get('results'), list)
        
        return has_web or has_news
    
    def _parse_brave_results(self, data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Parse Brave Search API response into standardized format with enhanced quality scoring."""
        results: List[Dict[str, Any]] = []

        def _normalize_url(url: str) -> str:
            """Normalize URLs for consistent citation & deduplication.
            - Ensure scheme (default https)
            - Collapse multiple slashes in path
            - Remove default ports (:80, :443)
            - Strip tracking query params (utm_*, fbclid, gclid, msclkid, ref, ref_src, referrer, source, code)
            - Remove trailing slash (except root)
            - Lowercase hostname
            """
            if not url:
                return ""
            url = url.strip()
            if url.startswith('//'):
                url = 'https:' + url
            if not url.lower().startswith(('http://', 'https://')):
                url = 'https://' + url  # assume https for bare domains
            try:
                parsed = urlparse(url)
                scheme = parsed.scheme.lower() if parsed.scheme else 'https'
                # Prefer https when original was http (heuristic): upgrade unless localhost or internal
                if scheme == 'http' and not parsed.netloc.startswith(('localhost', '127.0.0.1')):
                    scheme = 'https'
                netloc = parsed.netloc.lower()
                # Remove default ports
                if netloc.endswith(':80'):
                    netloc = netloc[:-3]
                elif netloc.endswith(':443'):
                    netloc = netloc[:-4]
                # Collapse duplicate slashes in path
                path = re.sub(r'/+', '/', parsed.path or '/')
                if path != '/' and path.endswith('/'):
                    path = path[:-1]
                # Clean query params
                tracking_keys = {'fbclid','gclid','msclkid','ref','ref_src','referrer','source','code'}
                # Any key starting with utm_ also removed
                query_items = []
                for k, v in parse_qsl(parsed.query, keep_blank_values=False):
                    lk = k.lower()
                    if lk.startswith('utm_') or lk in tracking_keys:
                        continue
                    query_items.append((k, v))
                query = urlencode(query_items, doseq=True)
                normalized = urlunparse((scheme, netloc, path, '', query, ''))
                return normalized
            except Exception:
                return url  # fallback to original
        
        # Helper to parse a list of result entries
        def parse_entries(entries: List[Dict[str, Any]]):
            for result in entries:
                title = (result.get('title') or '').strip()
                raw_url = (result.get('url') or result.get('link') or result.get('resolved_url') or '').strip()
                url = _normalize_url(raw_url)
                description = (result.get('description') or result.get('snippet') or '').strip()
                if not title or not url:
                    continue
                quality_score = self._calculate_brave_quality_score(result, query, title, description)
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': description,
                    'relevance_score': quality_score,
                    'source': 'brave_search',
                    'timestamp': datetime.now().isoformat(),
                    'search_engine': 'Brave',
                    'raw_data': result
                })
        
        # Extract web results
        web_results = data.get('web', {}).get('results', [])
        if isinstance(web_results, list) and web_results:
            parse_entries(web_results)

        # Fallback: also parse news results if present
        if not results:
            news_results = data.get('news', {}).get('results', [])
            if isinstance(news_results, list) and news_results:
                parse_entries(news_results)

        logger.debug(f"Brave Search parsed {len(results)} results with quality scoring")
        return results
    
    def _optimize_financial_query(self, query: str) -> str:
        """Optimize query for financial search with domain-specific terms."""
        enhanced_query = query.strip()
        
        # Add financial context terms if not present
        financial_indicators = ['stock', 'share', 'financial', 'earnings', 'revenue', 'market']
        has_financial_context = any(term in enhanced_query.lower() for term in financial_indicators)
        
        if not has_financial_context:
            # Check if query contains company names or symbols
            company_patterns = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'META']
            company_names = ['Apple', 'Microsoft', 'Tesla', 'Google', 'Amazon', 'Nvidia', 'Meta']
            
            has_company = any(pattern in enhanced_query for pattern in company_patterns + company_names)
            if has_company:
                enhanced_query += " stock analysis financial"
        
        # Add site filters for high-quality sources
        if self._is_financial_query(enhanced_query):
            # Boost trusted financial domains in search ranking
            trusted_sites = ['site:bloomberg.com OR site:reuters.com OR site:wsj.com']
            # Note: This is commented as it may be too restrictive, but can be enabled for ultra-high quality
            # enhanced_query += f" ({' OR '.join(trusted_sites)})"
        
        return enhanced_query
    
    def _get_optimal_locale(self, query: str, default_country: str) -> tuple[str, str]:
        """Determine optimal country and language based on query content with improved detection."""
        # Use improved language detection
        detected_lang = self._detect_language(query)
        
        query_lower = query.lower()
        
        # Language-specific optimizations with better validation
        if detected_lang == 'ja' or 'japan' in query_lower or '日本' in query:
            return 'JP', 'ja'
        elif detected_lang == 'zh' or 'china' in query_lower or '中国' in query:
            return 'CN', 'zh'
        elif detected_lang == 'ko' or 'korea' in query_lower or '韓国' in query:
            return 'KR', 'ko'
        elif detected_lang == 'de' or any(term in query_lower for term in ['germany', 'deutsch', 'deutschland']):
            return 'DE', 'de'
        elif detected_lang == 'fr' or any(term in query_lower for term in ['france', 'français', 'french']):
            return 'FR', 'fr'
        elif detected_lang == 'es' or any(term in query_lower for term in ['spain', 'español', 'spanish']):
            return 'ES', 'es'
        elif detected_lang == 'it' or any(term in query_lower for term in ['italy', 'italian', 'italiano']):
            return 'IT', 'it'
        elif any(term in query_lower for term in ['europe', 'eu', 'european']):
            return 'GB', 'en'  # Use UK for European financial markets
        elif any(term in query_lower for term in ['nasdaq', 'nyse', 'usa', 'america']):
            return 'US', 'en'
        
        # Default based on input or fallback with validation
        if default_country and default_country != "ALL":
            # Map detected language to appropriate country if default_country is generic
            if detected_lang == 'ja':
                return 'JP', 'ja'
            elif detected_lang == 'zh':
                return 'CN', 'zh'
            elif detected_lang == 'ko':
                return 'KR', 'ko'
            elif detected_lang == 'de':
                return 'DE', 'de'
            elif detected_lang == 'fr':
                return 'FR', 'fr'
            elif detected_lang == 'es':
                return 'ES', 'es'
            elif detected_lang == 'it':
                return 'IT', 'it'
            else:
                return default_country, 'en'
        
        # Final fallback to US for financial queries
        return 'US', 'en'
    
    def _is_time_sensitive_query(self, query: str) -> bool:
        """Check if query requires recent/time-sensitive results."""
        time_sensitive_terms = [
            'latest', 'recent', 'current', 'today', 'now', 'breaking',
            'earnings', 'results', 'report', 'news', 'update', 'forecast',
            '2025', '2024', 'this year', 'this quarter', 'q1', 'q2', 'q3', 'q4'
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in time_sensitive_terms)
    
    def _is_financial_query(self, query: str) -> bool:
        """Check if query is finance/business related."""
        financial_terms = [
            'stock', 'share', 'market', 'financial', 'investment', 'analysis',
            'earnings', 'revenue', 'profit', 'nasdaq', 'nyse', 'trading', 'price',
            'forecast', 'prediction', 'valuation', 'dividend', 'portfolio'
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in financial_terms)
    
    def _calculate_brave_quality_score(self, raw_result: Dict, query: str, title: str, description: str) -> float:
        """Calculate enhanced quality score for Brave results with improved tiered domain scoring."""
        base_score = 0.4  # Reduced base score to allow more spread
        
        # Factor 1: Enhanced domain quality with tiers
        domain = urlparse(raw_result.get('url', '')).netloc.lower()
        domain_bonus = 0.0
        
        # Check tiered trusted domains
        all_trusted_domains = set()
        for tier, domains in TRUSTED_FINANCIAL_DOMAINS.items():
            all_trusted_domains.update(domains)
        
        all_untrusted_domains = set()
        for category, domains in UNTRUSTED_DOMAINS.items():
            all_untrusted_domains.update(domains)
        
        if domain in TRUSTED_FINANCIAL_DOMAINS.get('tier1', set()):
            domain_bonus = 0.25  # Highest boost for tier 1 sources
        elif domain in TRUSTED_FINANCIAL_DOMAINS.get('tier2', set()):
            domain_bonus = 0.20  # High boost for tier 2 sources
        elif domain in TRUSTED_FINANCIAL_DOMAINS.get('tier3', set()):
            domain_bonus = 0.15  # Good boost for tier 3 sources
        elif domain in TRUSTED_FINANCIAL_DOMAINS.get('academic', set()):
            domain_bonus = 0.18  # Academic sources boost
        elif domain in TRUSTED_JP_FINANCIAL_DOMAINS or domain.endswith('.co.jp'):
            # Boost reputable Japanese corporate/financial sites
            domain_bonus = max(domain_bonus, 0.15)
        elif domain in all_untrusted_domains:
            # Differentiate penalties by category
            if domain in UNTRUSTED_DOMAINS.get('social', set()):
                domain_bonus = -0.3  # Moderate penalty for social media
            elif domain in UNTRUSTED_DOMAINS.get('clickbait', set()):
                domain_bonus = -0.5  # Heavy penalty for clickbait
            elif domain in UNTRUSTED_DOMAINS.get('farms', set()):
                domain_bonus = -0.4  # Heavy penalty for content farms
            else:
                domain_bonus = -0.3  # Default penalty
        elif domain.endswith('.gov'):
            domain_bonus = 0.22  # High boost for government sources
        elif domain.endswith('.edu'):
            domain_bonus = 0.18  # Good boost for educational sources
        elif domain.endswith(('.org', '.int')):
            domain_bonus = 0.10  # Moderate boost for organizations
        
        # Factor 2: Query relevance (unchanged)
        query_lower = query.lower()
        title_lower = title.lower()
        desc_lower = description.lower()
        
        relevance_bonus = 0.0
        if query_lower in title_lower:
            relevance_bonus += 0.1
        if query_lower in desc_lower:
            relevance_bonus += 0.05
        
        # Check for exact keyword matches
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        desc_words = set(desc_lower.split())
        
        title_overlap = len(query_words & title_words) / len(query_words) if query_words else 0
        desc_overlap = len(query_words & desc_words) / len(query_words) if query_words else 0
        
        relevance_bonus += title_overlap * 0.1 + desc_overlap * 0.05
        
        # Factor 3: Content quality indicators (EN + JA)
        quality_indicators = [
            # English
            'analysis', 'report', 'research', 'study', 'data', 'statistics', 'investor relations', 'ir',
            # Japanese (common finance/corporate terms)
            '戦略', '経営', '決算', '短信', '発表', '統合報告書', '有価証券報告書', '顧客', '基盤', '成長', 'IR', '投資家', '方針'
        ]
        spam_indicators = ['click here', 'buy now', 'free trial', 'limited time']
        
        content_text = (title + ' ' + description).lower()
        quality_bonus = 0.0
        # Case-insensitive match for English indicators; simple inclusion for Japanese
        for indicator in quality_indicators:
            try:
                if indicator.lower() in content_text:
                    quality_bonus += 0.02
            except Exception:
                # For non-ascii, fallback to simple containment check without lowercasing
                if indicator in (title + ' ' + description):
                    quality_bonus += 0.02
        spam_penalty = sum(-0.05 for indicator in spam_indicators if indicator in content_text)
        
        # Factor 4: Length and completeness (longer descriptions often indicate quality)
        length_bonus = min(len(description) / 200, 0.05)  # Up to 0.05 bonus for good descriptions
        
        # Factor 5: Language match
        language_bonus = 0.0
        if any(ord(c) > 127 for c in query):  # Non-ASCII query (e.g., Japanese)
            if any(ord(c) > 127 for c in title + description):  # Non-ASCII content
                language_bonus = 0.12  # Slightly higher for better separation
        
        # Calculate final score with improved scaling
        final_score = base_score + domain_bonus + relevance_bonus + quality_bonus + spam_penalty + length_bonus + language_bonus
        
        # Apply Brave quality bonus as a multiplicative factor instead of additive
        if final_score > 0:
            final_score *= (1.0 + BRAVE_QUALITY_BONUS)
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, final_score))
    
    def _apply_quality_filtering(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply post-retrieval quality filtering with improved domain checking."""
        filtered_results = []
        
        # Flatten untrusted domains for easier checking
        all_untrusted_domains = set()
        for category, domains in UNTRUSTED_DOMAINS.items():
            all_untrusted_domains.update(domains)
        
        is_japanese_query = any(ord(c) > 127 for c in query)
        
        for result in results:
            url = result.get('url', '')
            domain = urlparse(url).netloc.lower()
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            # Filter 1: Remove untrusted domains (with category-based exceptions)
            if domain in all_untrusted_domains:
                # Allow some social media if it's official corporate accounts for financial queries
                if (domain in UNTRUSTED_DOMAINS.get('social', set()) and 
                    self._is_likely_official_account(title, snippet)):
                    pass  # Allow official accounts
                else:
                    logger.debug(f"Filtered untrusted domain: {domain}")
                    continue
            
            # Filter 2: Remove obviously irrelevant results
            # For Japanese queries, be permissive to avoid dropping useful JP corporate/IR pages
            if not is_japanese_query:
                if not self._is_content_relevant(query, title, snippet):
                    logger.debug(f"Filtered irrelevant content: {title[:50]}...")
                    continue
            
            # Filter 3: Remove spam/low-quality indicators
            if self._has_spam_indicators(title, snippet):
                logger.debug(f"Filtered spam content: {title[:50]}...")
                continue
            
            # Filter 4: Minimum quality threshold (more lenient for Japanese queries)
            min_threshold = 0.2 if is_japanese_query else 0.4
            if result.get('relevance_score', 0) < min_threshold:
                logger.debug(f"Filtered low-quality result: {result.get('relevance_score', 0):.2f}")
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _is_likely_official_account(self, title: str, snippet: str) -> bool:
        """Check if social media content is likely from an official corporate account."""
        content = (title + ' ' + snippet).lower()
        
        # Official indicators
        official_indicators = [
            'official', 'verified', 'corporate', 'investor relations',
            'press release', 'earnings', 'sec filing', 'quarterly report',
            'annual report', 'ceo', 'cfo', 'company announcement'
        ]
        
        return any(indicator in content for indicator in official_indicators)
    
    def _rerank_by_quality(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank results by comprehensive quality score with improved domain assessment."""
        def quality_score(result: Dict[str, Any]) -> float:
            base_score = result.get('relevance_score', 0.5)
            url = result.get('url', '')
            domain = urlparse(url).netloc.lower()
            
            # Enhanced quality multipliers based on domain tiers
            multiplier = 1.0
            
            # Tiered trusted domain bonuses
            if domain in TRUSTED_FINANCIAL_DOMAINS.get('tier1', set()):
                multiplier *= 1.4  # Highest boost for premium sources
            elif domain in TRUSTED_FINANCIAL_DOMAINS.get('tier2', set()):
                multiplier *= 1.3  # High boost for major financial news
            elif domain in TRUSTED_FINANCIAL_DOMAINS.get('tier3', set()):
                multiplier *= 1.2  # Good boost for specialized sources
            elif domain in TRUSTED_FINANCIAL_DOMAINS.get('academic', set()):
                multiplier *= 1.25 # Academic sources get good boost
            
            # Official sources boost
            if domain.endswith('.gov'):
                multiplier *= 1.3  # Government sources are highly trusted
            elif domain.endswith('.edu'):
                multiplier *= 1.2  # Educational sources are reliable
            elif domain.endswith(('.org', '.int')):
                multiplier *= 1.1  # Organizations get modest boost
            
            # Penalty for potentially unreliable sources
            all_untrusted_domains = set()
            for category, domains in UNTRUSTED_DOMAINS.items():
                all_untrusted_domains.update(domains)
                
            if domain in all_untrusted_domains:
                multiplier *= 0.7  # Penalty for untrusted domains
            
            return base_score * multiplier
        
        # Sort by quality score (descending)
        reranked = sorted(results, key=quality_score, reverse=True)
        
        # Log quality distribution for monitoring
        if reranked:
            scores = [quality_score(r) for r in reranked]
            avg_score = sum(scores) / len(scores)
            logger.info(f"Brave reranking: avg quality {avg_score:.3f}, range {min(scores):.3f}-{max(scores):.3f}")
        
        return reranked
    
    def _is_content_relevant(self, query: str, title: str, snippet: str) -> bool:
        """Enhanced relevance checking for content quality."""
        query_lower = query.lower()
        content_lower = (title + ' ' + snippet).lower()
        
        # If query contains non-ASCII (e.g., Japanese), use more permissive matching
        if any(ord(c) > 127 for c in query):
            # Split on whitespace to get keyword-like segments (common in JP queries)
            segments = [seg.strip() for seg in query.split() if len(seg.strip()) >= 2]
            # Additionally, extract Kanji/Kana sequences of length >= 2
            try:
                jp_tokens = __import__('re').findall(r'[\u3040-\u30FF\u4E00-\u9FFF]{2,}', query)
            except Exception:
                jp_tokens = []
            tokens = list({*segments, *jp_tokens})
            if not tokens:
                return False
            matches = sum(1 for t in tokens if t in content_lower)
            # For Japanese, 1 match among a few tokens is often sufficient
            if len(tokens) <= 3:
                return matches >= 1
            match_ratio = matches / max(1, len(tokens))
            return match_ratio >= 0.2  # Lower threshold for JP
        
        # English/Latin scripts: Extract meaningful words (exclude stop words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_words = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
        
        if not query_words:
            return False
        
        # Check for word overlap
        matches = sum(1 for word in query_words if word in content_lower)
        match_ratio = matches / len(query_words)
        
        return match_ratio >= 0.25  # Slightly more lenient (was 0.3)
    
    def _has_spam_indicators(self, title: str, snippet: str) -> bool:
        """Check for spam/low-quality content indicators."""
        content = (title + ' ' + snippet).lower()
        
        spam_patterns = [
            'click here', 'buy now', 'free trial', 'limited time offer',
            'act now', 'call now', 'order today', '100% free',
            'no credit card', 'risk free', 'money back guarantee',
            'amazing results', 'miracle cure', 'secret revealed'
        ]
        
        return any(pattern in content for pattern in spam_patterns)
    
    def _detect_language(self, text: str) -> str:
        """Improved language detection with better thresholds and more language support."""
        if not text:
            return 'en'
        
        text_len = len(text)
        if text_len == 0:
            return 'en'
        
        # Count different character sets
        japanese_chars = sum(1 for char in text if 
                           '\u3040' <= char <= '\u309F' or  # Hiragana
                           '\u30A0' <= char <= '\u30FF' or  # Katakana
                           '\u4E00' <= char <= '\u9FAF')    # Kanji
        
        # Chinese specific characters (not in Japanese)
        chinese_chars = sum(1 for char in text if 
                          '\u4E00' <= char <= '\u9FAF' and  # CJK Unified Ideographs
                          char not in '\u4E00\u4E01\u4E03\u4E07\u4E08\u4E09\u4E0A\u4E0B\u4E0D\u4E0E\u4E10')  # Common JP kanji
        
        # Korean characters
        korean_chars = sum(1 for char in text if 
                         '\uAC00' <= char <= '\uD7AF' or  # Hangul syllables
                         '\u1100' <= char <= '\u11FF')    # Hangul Jamo
        
        # Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        
        # Cyrillic characters
        cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        
        # Thai characters
        thai_chars = sum(1 for char in text if '\u0E00' <= char <= '\u0E7F')
        
        # Calculate ratios with better thresholds
        japanese_ratio = japanese_chars / text_len
        chinese_ratio = chinese_chars / text_len
        korean_ratio = korean_chars / text_len
        arabic_ratio = arabic_chars / text_len
        cyrillic_ratio = cyrillic_chars / text_len
        thai_ratio = thai_chars / text_len
        
        # Use higher thresholds to avoid false positives
        # A text needs at least 5% of characters from a script to be considered that language
        min_threshold = 0.05
        
        # Prioritize detection based on character density
        if japanese_ratio > min_threshold and japanese_ratio >= chinese_ratio:
            return 'ja'
        elif korean_ratio > min_threshold:
            return 'ko'
        elif chinese_ratio > min_threshold and chinese_ratio > japanese_ratio:
            return 'zh'
        elif arabic_ratio > min_threshold:
            return 'ar'
        elif cyrillic_ratio > min_threshold:
            return 'ru'
        elif thai_ratio > min_threshold:
            return 'th'
        
        # Fallback: check for common language indicators in Latin script
        # Use word boundaries to avoid false matches
        import re
        text_lower = text.lower()
        
        # Helper function to check for whole words only
        def has_whole_words(text_to_check, words):
            """Check if text contains any of the words as whole words (with word boundaries)."""
            for word in words:
                # Use word boundaries to match whole words only
                if re.search(r'\b' + re.escape(word) + r'\b', text_to_check):
                    return True
            return False
        
        # German indicators (whole words only)
        if has_whole_words(text_lower, ['der', 'die', 'das', 'und', 'ich', 'ist', 'mit', 'nicht', 'sie', 'auch', 'auf']):
            return 'de'
        # French indicators (whole words only)
        elif has_whole_words(text_lower, ['le', 'la', 'les', 'et', 'un', 'une', 'est', 'sont', 'dans', 'pour', 'avec']):
            # Avoid false positive from English "analysis" containing "al"
            if not has_whole_words(text_lower, ['stock', 'price', 'company', 'market', 'analysis', 'financial']):
                return 'fr'
        # Spanish indicators (whole words only)
        elif has_whole_words(text_lower, ['el', 'la', 'los', 'las', 'y', 'un', 'una', 'es', 'son', 'está', 'están']):
            # Avoid false positive from English words containing Spanish words
            if not has_whole_words(text_lower, ['stock', 'price', 'company', 'market', 'analysis', 'financial']):
                return 'es'
        # Italian indicators (whole words only)
        elif has_whole_words(text_lower, ['il', 'la', 'i', 'le', 'di', 'che', 'è', 'sono', 'per', 'con', 'da']):
            if not has_whole_words(text_lower, ['stock', 'price', 'company', 'market', 'analysis', 'financial']):
                return 'it'
        
        # Default to English
        return 'en'
    
    async def close(self):
        """Explicitly close HTTP session and mark client as closed."""
        self._closed = True
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Error closing Brave client session: {e}")
            finally:
                self._session = None
    
    def log_quality_metrics(self, results: List[Dict[str, Any]], query: str):
        """Log quality metrics for monitoring and improvement."""
        if not results:
            return
        
        # Domain distribution analysis
        domain_counts = {}
        quality_by_domain = {}
        
        for result in results:
            url = result.get('url', '')
            domain = urlparse(url).netloc.lower()
            score = result.get('relevance_score', 0)
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if domain not in quality_by_domain:
                quality_by_domain[domain] = []
            quality_by_domain[domain].append(score)
        
        # Calculate average quality by domain
        domain_quality = {}
        for domain, scores in quality_by_domain.items():
            domain_quality[domain] = sum(scores) / len(scores)
        
        # Log metrics for monitoring
        total_results = len(results)
        avg_quality = sum(r.get('relevance_score', 0) for r in results) / total_results
        trusted_count = sum(1 for r in results if urlparse(r.get('url', '')).netloc.lower() in TRUSTED_FINANCIAL_DOMAINS)
        
        logger.info(f"Brave Quality Metrics - Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        logger.info(f"  Total results: {total_results}, Avg quality: {avg_quality:.3f}")
        logger.info(f"  Trusted domains: {trusted_count}/{total_results} ({trusted_count/total_results*100:.1f}%)")
        logger.info(f"  Domain distribution: {dict(list(domain_counts.items())[:5])}")  # Top 5 domains
        
        # Log low-quality domains for potential addition to denylist
        low_quality_domains = [domain for domain, quality in domain_quality.items() 
                              if quality < 0.6 and domain not in UNTRUSTED_DOMAINS]
        if low_quality_domains:
            logger.warning(f"Low-quality domains detected: {low_quality_domains}")
        
        return {
            'total_results': total_results,
            'avg_quality': avg_quality,
            'trusted_ratio': trusted_count / total_results,
            'domain_distribution': domain_counts,
            'low_quality_domains': low_quality_domains
        }

def _get_cache_key(text: str) -> str:
    """Generate cache key from text."""
    return hashlib.md5(text.encode()).hexdigest()

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid - legacy function for compatibility."""
    return (datetime.now().timestamp() - timestamp) < 3600  # 1 hour default TTL

# Global OpenAI client instance for reuse
_openai_client = None

def get_openai_client() -> AsyncOpenAI:
    """Get a reusable OpenAI client for answer synthesis."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for answer synthesis")
        
        _openai_client = AsyncOpenAI(
            api_key=api_key,
            timeout=30.0
        )
    
    return _openai_client

@dataclass
class SearchResult:
    """Enhanced search result with content extraction and advanced ranking."""
    title: str
    url: str
    snippet: str
    content: str = ""
    relevance_score: float = 0.0  # Legacy field, no longer used in ranking
    timestamp: str = ""
    source: str = ""
    word_count: int = 0
    citation_id: int = 0
    # Enhanced scoring fields
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0
    embedding_vector: Optional[List[float]] = None
    original_position: Optional[int] = None  # Track original search ranking
    # New ranking signal fields
    domain_boost: float = 1.0
    recency_boost: float = 1.0
    snippet_title_boost: float = 1.0
    # NLI verification metadata
    nli_status: str = "unknown"
    nli_confidence: float = 0.0
    nli_reason: str = ""
    nli_last_claim: str = ""

@dataclass  
class PerplexityResponse:
    """Perplexity-style response with synthesized answer and citations."""
    query: str
    synthesized_query: str
    answer: str
    sources: List[SearchResult]
    citations: Dict[int, Any]  # citation_id -> source info (string or structured dict)
    confidence_score: float = 0.0
    search_time: float = 0.0
    synthesis_time: float = 0.0
    total_time: float = 0.0
    verification_notes: List[str] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)

class PerplexityWebSearchService:
    """Enhanced web search service with Perplexity-like answer synthesis and proper lifecycle management."""
    
    def __init__(self):
        self.max_content_length = 8000  # Max chars per source
        self.max_synthesis_sources = 8  # Max sources for synthesis
        self.timeout = aiohttp.ClientTimeout(total=15, connect=5)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        self._session = None  # Reusable session
        self._closed = False
        # NLI verification resources
        self._nli_client = None
        self._nli_model = None
        self._nli_provider = None
        self._nli_lock = asyncio.Lock()
    
    def _log_stage_timing(self, stage: str, duration: float, query: str) -> None:
        """Utility to log stage timing with consistent formatting."""
        try:
            logger.info(
                "Timing | %s | %.3fs | query='%s'",
                stage,
                duration,
                query[:120].replace("\n", " ") if query else ""
            )
        except Exception as timing_error:
            logger.debug(f"Failed to log timing for {stage}: {timing_error}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @property
    def is_closed(self) -> bool:
        """Check if the service is closed."""
        return self._closed
    
    async def close(self):
        """Explicitly close the service and all resources."""
        self._closed = True
        await self._close_session()
    
    async def _close_session(self):
        """Properly close the aiohttp session."""
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Session cleanup error: {e}")
            finally:
                self._session = None
        # Close dedicated NLI client if we created one (Azure async client)
        if self._nli_client and self._nli_provider == "azure":
            try:
                await self._nli_client.close()
            except Exception as e:
                logger.debug(f"NLI client cleanup error: {e}")
        # Reset NLI state regardless of provider
        self._nli_client = None
        self._nli_model = None
        self._nli_provider = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session."""
        if self._closed:
            raise RuntimeError("Service is closed")
            
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,  # Total connection pool size
                limit_per_host=5,  # Connections per host
                ttl_dns_cache=300,
                use_dns_cache=True,
                force_close=True,  # Force close connections after use
                enable_cleanup_closed=True  # Enable cleanup of closed connections
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout, 
                headers=self.headers,
                connector=connector
            )
        return self._session

    async def _get_nli_client(self) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """Lazily resolve an async client and model for NLI verification."""
        if self._closed:
            return None, None, None

        async with self._nli_lock:
            if self._nli_client and self._nli_model:
                return self._nli_client, self._nli_model, self._nli_provider

            try:
                openai_client = get_openai_client()
                if openai_client:
                    self._nli_client = openai_client
                    self._nli_model = "gpt-4o-mini"
                    self._nli_provider = "openai"
                    return self._nli_client, self._nli_model, self._nli_provider
            except Exception as openai_err:
                logger.debug(f"OpenAI NLI client init failed: {openai_err}")

            self._nli_client = None
            self._nli_model = None
            self._nli_provider = None
            return None, None, None

    @staticmethod
    def _normalize_result_url(url: Optional[str]) -> str:
        """Normalize URLs for deduplication and consistent citation mapping."""
        if not url:
            return ""
        normalized = url.strip()
        if normalized.startswith("//"):
            normalized = "https:" + normalized
        if not normalized.lower().startswith(("http://", "https://")):
            normalized = "https://" + normalized
        try:
            parsed = urlparse(normalized)
            scheme = parsed.scheme.lower() if parsed.scheme else "https"
            netloc = parsed.netloc.lower()
            if netloc.endswith(":80"):
                netloc = netloc[:-3]
            elif netloc.endswith(":443"):
                netloc = netloc[:-4]
            path = re.sub(r"/+", "/", parsed.path or "/")
            if path != "/" and path.endswith("/"):
                path = path[:-1]
            tracking_keys = {"fbclid", "gclid", "msclkid", "ref", "ref_src", "referrer", "source", "code"}
            query_items: List[Tuple[str, str]] = []
            for key, value in parse_qsl(parsed.query, keep_blank_values=False):
                lower_key = key.lower()
                if lower_key.startswith("utm_") or lower_key in tracking_keys:
                    continue
                query_items.append((key, value))
            query_str = urlencode(query_items, doseq=True)
            return urlunparse((scheme, netloc, path, "", query_str, ""))
        except Exception:
            return normalized

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on normalized URL while preserving order."""
        seen_keys: set[str] = set()
        deduplicated: List[SearchResult] = []
        for result in results:
            if not isinstance(result, SearchResult):
                continue
            normalized_url = self._normalize_result_url(result.url)
            result.url = normalized_url
            dedup_key = normalized_url or (result.title.strip().lower() if result.title else "")
            if dedup_key and dedup_key in seen_keys:
                continue
            if dedup_key:
                seen_keys.add(dedup_key)
            deduplicated.append(result)
        return deduplicated
        
    async def perplexity_search(
        self,
        query: str,
        max_results: int = 8,
        synthesize_answer: bool = True,
    include_recent: bool = False,
        time_limit: Optional[str] = None
    ) -> PerplexityResponse:
        """
        Perform Perplexity-style search with answer synthesis.
        
        Args:
            query: Search query
            max_results: Maximum search results to process
            synthesize_answer: Whether to generate synthesized answer
            include_recent: Whether to prioritize recent content
            time_limit: DDGS time limit ('d'=day, 'w'=week, 'm'=month, 'y'=year)
            
        Returns:
            PerplexityResponse with synthesized answer and citations
        """
        start_time = datetime.now()
        synthesized_query = query
        
        try:
            # Step 1: Enhanced web search with content extraction
            search_start = datetime.now()
            stage_start = time.perf_counter()
            try:
                search_results, synthesized_query = await self._enhanced_web_search(query, max_results, include_recent, time_limit)
            finally:
                self._log_stage_timing("enhanced_web_search", time.perf_counter() - stage_start, query)
            search_time = (datetime.now() - search_start).total_seconds()
            
            # Step 2: Content extraction and enhancement
            content_stage_start = time.perf_counter()
            try:
                enhanced_results = await self._extract_and_enhance_content(search_results)
            finally:
                self._log_stage_timing("extract_and_enhance_content", time.perf_counter() - content_stage_start, query)
            
            # Step 3: Enhanced ranking with BM25 and semantic similarity
            if enhanced_results:
                # Calculate BM25 scores
                enhanced_results = self._calculate_bm25_scores(query, enhanced_results)
                
                # Calculate semantic similarity scores using Azure embeddings
                semantic_stage_start = time.perf_counter()
                try:
                    enhanced_results = await self._calculate_semantic_scores(query, enhanced_results)
                finally:
                    self._log_stage_timing("calculate_semantic_scores", time.perf_counter() - semantic_stage_start, query)
                
                # Combine all ranking signals
                enhanced_results = self._calculate_combined_scores(enhanced_results)

                # Re-sort by combined score (descending)
                enhanced_results.sort(key=lambda x: (x.combined_score or 0.0), reverse=True)

                # Remove duplicate URLs while preserving ranking order
                deduped_results = self._deduplicate_results(enhanced_results)
                if len(deduped_results) != len(enhanced_results):
                    logger.debug(
                        "Deduplicated %d duplicate sources based on URL",
                        len(enhanced_results) - len(deduped_results)
                    )
                enhanced_results = deduped_results

                # Guard against empty list after deduplication
                if not enhanced_results:
                    logger.warning("No unique search results remained after deduplication")
                elif not any((r.combined_score or 0.0) > 0 for r in enhanced_results):
                    logger.warning("Combined ranking produced all zero scores; retaining original ordering for citation IDs")
                else:
                    # Reassign citation IDs sequentially according to new ranking so synthesis & citations align with quality order
                    for idx, r in enumerate(enhanced_results):
                        r.citation_id = idx + 1

                # Debug log top ranked sources with component scores
                try:
                    top_debug = []
                    for r in enhanced_results[:5]:
                        top_debug.append({
                            'citation_id': r.citation_id,
                            'title': (r.title or '')[:60],
                            'domain_boost': getattr(r, 'domain_boost', None),
                            'bm25': round(getattr(r, 'bm25_score', 0.0), 4),
                            'semantic': round(getattr(r, 'semantic_score', 0.0), 4),
                            'combined': round(getattr(r, 'combined_score', 0.0), 4)
                        })
                    logger.debug(f"Top ranked sources after combined scoring: {top_debug}")
                except Exception as dbg_err:
                    logger.debug(f"Failed logging top ranked sources: {dbg_err}")
            
            # Step 4: Generate ranked citation summary for downstream models
            summary_stage_start = time.perf_counter()
            try:
                answer = self._summarize_ranked_citations(enhanced_results)
            finally:
                self._log_stage_timing("summarize_citations", time.perf_counter() - summary_stage_start, query)

            answer = self._merge_adjacent_citations(answer)

            synthesis_time = 0.0  # Legacy field retained for compatibility
            verification_notes: List[str] = []
            verification_details: Dict[str, Any] = {"provider": None, "evaluations": []}

            # Step 5: Build citations with verification metadata
            citations = self._build_citations(enhanced_results)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence based on content quality and relevance
            confidence_score = self._calculate_confidence(enhanced_results, answer)
            
            return PerplexityResponse(
                query=query,
                synthesized_query=synthesized_query,
                answer=answer,
                sources=enhanced_results,
                citations=citations,
                confidence_score=confidence_score,
                search_time=search_time,
                synthesis_time=synthesis_time,
                total_time=total_time,
                verification_notes=verification_notes,
                verification_details=verification_details
            )
        
        finally:
            # Always cleanup session resources
            try:
                await self._close_session()
            except Exception as cleanup_error:
                logger.debug(f"Session cleanup error: {cleanup_error}")
    
    async def _enhanced_web_search(
        self,
        query: str,
        max_results: int,
        include_recent: bool,
        time_limit: Optional[str] = None
    ) -> Tuple[List[SearchResult], str]:
        """Perform enhanced web search prioritizing Brave as a high-quality source.

        Returns both the enhanced list of `SearchResult` objects and the synthesized query
        that was ultimately executed so downstream consumers can surface it to users.
        """
        cache_key = _build_search_cache_key(query, max_results, include_recent, time_limit)
        cached_payload = _search_cache.get(cache_key)
        cached_enhanced_query: Optional[str] = None
        if cached_payload:
            payload_results: Optional[List[Dict[str, Any]]] = None
            if isinstance(cached_payload, dict):
                payload_results = cached_payload.get("results")
                cached_enhanced_query = cached_payload.get("enhanced_query")
            elif isinstance(cached_payload, list):
                payload_results = cached_payload

            cached_results = _deserialize_search_results(payload_results)
            if cached_results:
                for idx, item in enumerate(cached_results):
                    item.citation_id = idx + 1
                logger.debug(
                    "Enhanced web search cache hit for query '%s' (results=%d)",
                    query[:80],
                    len(cached_results),
                )
                return cached_results, cached_enhanced_query or query

        search_results: List[SearchResult] = []
        enhanced_query = query
        
        try:
            # Initialize Brave client for high-quality results with proper lifecycle
            async with BraveSearchClient() as brave_client:
                # OPTIMIZATION: Skip LLM query enhancement for speed (saves 8-20s per search)
                # Use rule-based enhancement instead which is instant
                enhance_stage_start = time.perf_counter()
                try:
                    # Skip expensive LLM enhancement, use fast rule-based enhancement
                    enhanced_query = self._fallback_enhance_query(query, include_recent)
                    logger.debug(f"Using fast rule-based query enhancement: '{query}' -> '{enhanced_query}'")
                finally:
                    self._log_stage_timing("enhance_search_query", time.perf_counter() - enhance_stage_start, query)
                
                # Strategy: Try Brave first for high-quality results, then supplement with DDGS
                brave_results: List[SearchResult] = []
                ddgs_results: List[SearchResult] = []
                
                brave_freshness = None
                if time_limit:
                    freshness_map = {'d': 'pd', 'w': 'pw', 'm': 'pm', 'y': 'py'}
                    brave_freshness = freshness_map.get(time_limit)
                elif include_recent:
                    brave_freshness = 'pw'  # Past week for recent content
                
                brave_raw: List[Dict[str, Any]] = []
                brave_task: Optional[asyncio.Task] = None
                ddgs_task: Optional[asyncio.Task] = None
                
                if brave_client.is_available:
                    brave_task = asyncio.create_task(
                        brave_client.search(
                            query=enhanced_query,
                            count=max_results,
                            freshness=brave_freshness
                        )
                    )
                    try:
                        done, _ = await asyncio.wait({brave_task}, timeout=0.65)
                        if done:
                            brave_raw = done.pop().result()
                        else:
                            ddgs_task = asyncio.create_task(
                                self._direct_ddgs_search(
                                    enhanced_query,
                                    max(max_results, MIN_SEARCH_RESULTS_THRESHOLD),
                                    time_limit
                                )
                            )
                            brave_raw = await brave_task
                    except Exception as brave_error:
                        logger.warning(f"Brave Search failed: {brave_error}")
                        brave_raw = []
                else:
                    logger.debug("Brave Search unavailable; using DDGS fallback only")
                    ddgs_task = asyncio.create_task(
                        self._direct_ddgs_search(
                            enhanced_query,
                            max(max_results, MIN_SEARCH_RESULTS_THRESHOLD),
                            time_limit
                        )
                    )

                if brave_raw:
                    try:
                        brave_client.log_quality_metrics(brave_raw, enhanced_query)
                    except Exception as metric_error:
                        logger.debug(f"Brave quality metrics logging failed: {metric_error}")
                    for idx, result in enumerate(brave_raw):
                        normalized_url = self._normalize_result_url(result.get('url', ''))
                        search_result = SearchResult(
                            title=result.get('title', ''),
                            url=normalized_url,
                            snippet=result.get('snippet', ''),
                            content='',  # Will be filled by content extraction
                            relevance_score=result.get('relevance_score', 0.8),
                            timestamp=result.get('timestamp', datetime.now().isoformat()),
                            source='brave_search',
                            citation_id=idx + 1
                        )
                        brave_results.append(search_result)
                    logger.info(f"Brave Search (enhanced quality): {len(brave_results)} results")

                # Step 2: Supplement with DDGS if needed (fallback strategy)
                remaining_needed = max_results - len(brave_results)
                need_ddgs = remaining_needed > 0 or len(brave_results) < MIN_SEARCH_RESULTS_THRESHOLD
                ddgs_raw: List[Dict[str, Any]] = []
                if need_ddgs:
                    if ddgs_task is None:
                        ddgs_task = asyncio.create_task(
                            self._direct_ddgs_search(
                                enhanced_query,
                                max(remaining_needed, MIN_SEARCH_RESULTS_THRESHOLD),
                                time_limit
                            )
                        )
                    try:
                        ddgs_raw = await ddgs_task
                    except asyncio.CancelledError:
                        ddgs_raw = []
                    except Exception as e:
                        logger.warning(f"DDGS search failed: {e}")
                        ddgs_raw = []
                else:
                    if ddgs_task:
                        ddgs_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await ddgs_task

                if ddgs_raw:
                    start_citation_id = len(brave_results) + 1
                    for idx, result in enumerate(ddgs_raw):
                        normalized_url = self._normalize_result_url(result.get('url', ''))
                        search_result = SearchResult(
                            title=result.get('title', ''),
                            url=normalized_url,
                            snippet=result.get('snippet', ''),
                            content='',
                            relevance_score=result.get('relevance_score', 0.6),  # Lower base score than Brave
                            timestamp=datetime.now().isoformat(),
                            source='ddgs_search',
                            citation_id=start_citation_id + idx
                        )
                        ddgs_results.append(search_result)
                    logger.info(f"DDGS Search (supplemental): {len(ddgs_results)} results")
            
            # Step 3: Combine and prioritize results (Brave first, then DDGS)
            search_results = brave_results + ddgs_results
            
            # Step 4: Deduplicate by URL while preserving Brave priority
            deduplicated_results = self._deduplicate_results(search_results)
            
            # Step 5: Final sorting with Brave quality prioritization
            deduplicated_results.sort(key=lambda x: (
                # Primary: Source priority (Brave > DDGS)
                1.0 if x.source == 'brave_search' else 0.5,
                # Secondary: Relevance score
                x.relevance_score
            ), reverse=True)
            
            # Limit to requested number of results
            final_results = deduplicated_results[:max_results]
            
            # Update citation IDs for final ordering
            for idx, result in enumerate(final_results):
                result.citation_id = idx + 1
            
            logger.info(f"Enhanced search completed: {len(final_results)} total results "
                       f"({len(brave_results)} from Brave, {len(ddgs_results)} from DDGS)")

            _search_cache.put(
                cache_key,
                {
                    "results": _serialize_search_results(final_results),
                    "enhanced_query": enhanced_query
                }
            )
            return final_results, enhanced_query
                
        except Exception as e:
            logger.error(f"Enhanced web search failed: {e}")
            # Fallback to DDGS-only search
            try:
                raw_results = await self._direct_ddgs_search(enhanced_query, max_results, time_limit)
                
                for idx, result in enumerate(raw_results):
                    normalized_url = self._normalize_result_url(result.get('url', ''))
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        url=normalized_url,
                        snippet=result.get('snippet', ''),
                        content='',
                        relevance_score=result.get('relevance_score', 0.5),
                        timestamp=datetime.now().isoformat(),
                        source='ddgs_fallback',
                        citation_id=idx + 1
                    )
                    search_results.append(search_result)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                search_results = []

        if search_results:
            search_results = self._deduplicate_results(search_results)
            for idx, result in enumerate(search_results):
                result.citation_id = idx + 1
            _search_cache.put(
                cache_key,
                {
                    "results": _serialize_search_results(search_results),
                    "enhanced_query": enhanced_query
                }
            )
        
        return search_results, enhanced_query
    
    async def _direct_ddgs_search(self, query: str, max_results: int, time_limit: Optional[str] = None) -> List[Dict[str, Any]]:
        """Direct DDGS search implementation with improved async handling."""
        import random

        def _normalize_url(url: str) -> str:
            if not url:
                return ''
            url = url.strip()
            if url.startswith('//'):
                url = 'https:' + url
            if not url.lower().startswith(('http://', 'https://')):
                url = 'https://' + url
            try:
                parsed = urlparse(url)
                scheme = parsed.scheme.lower() if parsed.scheme else 'https'
                if scheme == 'http' and not parsed.netloc.startswith(('localhost', '127.0.0.1')):
                    scheme = 'https'
                netloc = parsed.netloc.lower()
                if netloc.endswith(':80'):
                    netloc = netloc[:-3]
                elif netloc.endswith(':443'):
                    netloc = netloc[:-4]
                path = re.sub(r'/+', '/', parsed.path or '/')
                if path != '/' and path.endswith('/'):
                    path = path[:-1]
                tracking_keys = {'fbclid','gclid','msclkid','ref','ref_src','referrer','source','code'}
                query_items = []
                for k,v in parse_qsl(parsed.query, keep_blank_values=False):
                    lk = k.lower()
                    if lk.startswith('utm_') or lk in tracking_keys:
                        continue
                    query_items.append((k,v))
                query_str = urlencode(query_items, doseq=True)
                return urlunparse((scheme, netloc, path, '', query_str, ''))
            except Exception:
                return url

        def _search_ddgs():
            """Run DDGS search in thread with better error handling."""
            try:
                # Small delay to avoid rate limiting
                time.sleep(random.uniform(0.05, 0.1))
                
                # Determine optimal region for this query
                optimal_region = self._get_optimal_ddgs_region(query)
                
                # Configure DDGS - headers not supported in constructor
                with DDGS() as ddgs:
                    # Build search parameters
                    search_params = {
                        'query': query,
                        'region': optimal_region,
                        'safesearch': DDGS_SAFESEARCH,
                        'max_results': max_results
                    }
                    
                    # Add time limit - prioritize parameter, then config, then none
                    effective_timelimit = time_limit or DDGS_TIMELIMIT
                    if effective_timelimit:
                        search_params['timelimit'] = effective_timelimit
                    
                    logger.debug(f"DDGS search params: region={optimal_region}, safesearch={DDGS_SAFESEARCH}, timelimit={effective_timelimit}")
                    
                    search_results = list(ddgs.text(**search_params))
                    
                    # Convert to standardized format
                    results = []
                    for result in search_results:
                        title = result.get('title', '')
                        url = _normalize_url(result.get('href', ''))
                        snippet = result.get('body', '')
                        
                        # Filter irrelevant results
                        if not self._is_result_relevant(query, title, snippet):
                            continue  # Skip irrelevant results
                        
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'relevance_score': 0.8  # Base score for DDGS results
                        })
                    
                    return results
                    
            except Exception as e:
                logger.debug(f"DDGS search error: {e}")
                return []
        
        # Use asyncio.to_thread for better event loop integration (Python 3.9+)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_ddgs),
                timeout=10.0
            )
            return results
        except AttributeError:
            # Fallback for Python < 3.9 using run_in_executor
            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, _search_ddgs),
                timeout=10.0
            )
            return results
        except asyncio.TimeoutError:
            logger.warning(f"DDGS search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"DDGS search failed: {e}")
            return []
    
    def _is_result_relevant(self, query: str, title: str, snippet: str) -> bool:
        """Check if a search result is relevant to the query - made less strict."""
        query_lower = query.lower()
        text_to_check = (title + ' ' + snippet).lower()
        
        # Extract key terms from query for relevance checking
        query_words = set(query_lower.split())
        
        # Check for basic relevance - at least one query word should appear
        basic_relevance = any(word in text_to_check for word in query_words if len(word) > 2)
        
        # If no basic relevance, reject
        if not basic_relevance:
            return False
        
        # For financial/stock queries, apply more lenient filtering
        financial_terms = ['stock', 'share', 'market', 'financial', 'investment', 'analysis', 
                          'earnings', 'revenue', 'profit', 'nasdaq', 'nyse', 'trading', 'price']
        stock_symbols = ['msft', 'tsla', 'aapl', 'googl', 'amzn', 'nvda', 'meta']
        company_names = ['microsoft', 'tesla', 'apple', 'google', 'amazon', 'nvidia', 'meta']
        
        is_financial_query = (
            any(term in query_lower for term in financial_terms) or
            any(symbol in query_lower for symbol in stock_symbols) or
            any(company in query_lower for company in company_names)
        )
        
        if is_financial_query:
            # For financial queries, be more lenient - just check for obviously irrelevant content
            has_financial_content = (
                any(term in text_to_check for term in financial_terms) or
                any(symbol in text_to_check for symbol in stock_symbols) or
                any(company in text_to_check for company in company_names) or
                'price' in text_to_check or 'chart' in text_to_check or 'quote' in text_to_check or
                'news' in text_to_check or 'company' in text_to_check or 'business' in text_to_check
            )
            
            # Only filter out obviously irrelevant content
            clearly_irrelevant_patterns = [
                'recipe', 'cooking', 'food', 'restaurant menu',
                'movie plot', 'film review', 'celebrity gossip',
                'game score', 'sports match', 'weather forecast'
            ]
            
            is_clearly_irrelevant = any(pattern in text_to_check for pattern in clearly_irrelevant_patterns)
            
            # Be more permissive - allow if has basic relevance and financial content, or not clearly irrelevant
            return (has_financial_content or not is_clearly_irrelevant) and basic_relevance
        
        # For Japanese financial queries, additional filtering
        if any(ord(c) > 127 for c in query):  # Japanese query
            japanese_financial_terms = ['銀行', '金融', '株式', '投資', '経済', '市場', '企業', '会社']
            
            is_japanese_financial = any(term in query for term in japanese_financial_terms)
            if is_japanese_financial:
                # Allow results with Japanese financial terms or English equivalents
                has_relevant_content = (
                    any(term in text_to_check for term in japanese_financial_terms) or
                    any(term in text_to_check for term in financial_terms) or
                    basic_relevance
                )
                return has_relevant_content
        
        return basic_relevance
    
    async def _enhance_search_query(self, query: str, include_recent: bool) -> str:
        """Enhance search query using LLM for intelligent optimization."""
        cache_key = _get_cache_key(f"enhanced_query::{query}::{include_recent}")
        cached_query = _query_enhancement_cache.get(cache_key)
        if cached_query:
            logger.debug("Using cached enhanced query for '%s'", query[:80])
            return cached_query

        enhanced_query = await self._llm_synthesize_query(query, include_recent)

        if not enhanced_query:
            enhanced_query = self._fallback_enhance_query(query, include_recent)

        _query_enhancement_cache.put(cache_key, enhanced_query)
        return enhanced_query
    
    def _fallback_enhance_query(self, query: str, include_recent: bool) -> str:
        """Fallback rule-based query enhancement with locale awareness."""
        enhanced_query = query
        
        # Detect the language of the query using the robust detection method
        detected_lang = self._detect_language(query)
        current_year = datetime.now().year
        
        # Language-specific enhancement logic
        if detected_lang == 'ja':
            # Japanese query enhancement
            # Simplify Japanese queries by extracting keywords from complex sentences
            enhanced_query = self._simplify_japanese_query(query)
            
            # Add Japanese recency terms if needed
            if include_recent:
                japanese_recency_terms = ['最新', '最近', '2025', str(current_year)]
                if not any(term in enhanced_query for term in japanese_recency_terms):
                    enhanced_query += f" {current_year}年"
            
            # Add Japanese analysis terms for financial queries
            financial_terms_ja = ['株価', '株式', '企業', '金融', '投資']
            analysis_terms_ja = ['分析', 'レポート', 'ニュース']
            if any(term in query for term in financial_terms_ja):
                if not any(term in enhanced_query for term in analysis_terms_ja):
                    enhanced_query += " 分析"
            
            return enhanced_query
        
        elif detected_lang == 'zh':
            # Chinese query enhancement
            if include_recent:
                chinese_recency_terms = ['最新', '最近', '2025', str(current_year)]
                if not any(term in enhanced_query for term in chinese_recency_terms):
                    enhanced_query += f" {current_year}年"
            
            # Add Chinese analysis terms for financial queries
            financial_terms_zh = ['股票', '股价', '公司', '金融', '投资']
            analysis_terms_zh = ['分析', '报告', '新闻']
            if any(term in query for term in financial_terms_zh):
                if not any(term in enhanced_query for term in analysis_terms_zh):
                    enhanced_query += " 分析"
            
            return enhanced_query
        
        elif detected_lang == 'ko':
            # Korean query enhancement
            if include_recent:
                korean_recency_terms = ['최신', '최근', '2025', str(current_year)]
                if not any(term in enhanced_query for term in korean_recency_terms):
                    enhanced_query += f" {current_year}년"
            
            # Add Korean analysis terms for financial queries
            financial_terms_ko = ['주가', '주식', '기업', '금융', '투자']
            analysis_terms_ko = ['분석', '보고서', '뉴스']
            if any(term in query for term in financial_terms_ko):
                if not any(term in enhanced_query for term in analysis_terms_ko):
                    enhanced_query += " 분석"
            
            return enhanced_query
        
        else:
            # English and other languages - use original enhancement logic
            # Add recency terms if requested
            if include_recent:
                recency_terms = ['latest', 'recent', '2025', str(current_year)]
                if not any(term in query.lower() for term in recency_terms):
                    enhanced_query += f" latest {current_year}"
            
            # Add analysis terms for better context
            analysis_terms = ['analysis', 'review', 'overview', 'report']
            if not any(term in query.lower() for term in analysis_terms):
                if any(word in query.lower() for word in ['stock', 'company', 'financial']):
                    enhanced_query += " analysis"
            
            return enhanced_query
    
    async def _llm_synthesize_query(self, query: str, include_recent: bool) -> str:
        """Use the configured default model to enhance the search query before fallback."""
        normalized_query = (query or "").strip()
        if not normalized_query:
            return ""

        try:
            client, model_name, model_config = get_client_for_model(DEFAULT_MODEL)
        except Exception as client_error:
            logger.debug(f"Query synthesis client unavailable: {client_error}")
            return ""

        recency_guidance = (
            "If relevant, add recency cues (e.g., 'latest 2025', 'recent developments') while keeping the language consistent."
            if include_recent
            else "Avoid adding unnecessary date qualifiers unless the original query already implies them."
        )

        system_prompt = (
            "You specialize in rewriting user queries to perform better on web search engines. "
            "Return exactly one improved query string with no explanations, bullet points, or quotations. "
            "Preserve key entities, tickers, and language (English vs. Japanese) from the original request."
        )

        user_prompt = (
            "Rewrite the following query so that a general web search retrieves the most relevant and high-quality results.\n"
            "- Keep the intent identical.\n"
            "- Add clarifying keywords only when they improve precision.\n"
            "- Prefer concise phrasing (<= 18 words).\n"
            f"- {recency_guidance}\n\n"
            f"Original query:\n{normalized_query}"
        )

        temperature = min(0.35, float(model_config.get("temperature", 0.3) or 0.3))
        max_tokens = max(32, min(96, int(model_config.get("max_completion_tokens", 128) or 128)))
        timeout_seconds = max(8, min(20, int(model_config.get("timeout", 30) or 30)))

        def _call_model():
            return client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_call_model),
                timeout=timeout_seconds
            )
        except Exception as call_error:
            logger.debug(f"Query synthesis call failed: {call_error}")
            return ""

        if not response or not getattr(response, "choices", None):
            logger.debug("Query synthesis returned no choices")
            return ""

        raw_content = (response.choices[0].message.content or "").strip()
        if not raw_content:
            return ""

        first_line = next((line.strip() for line in raw_content.splitlines() if line.strip()), "")
        if not first_line:
            return ""

        cleaned = re.sub(r"^\d+[\).\-]\s*", "", first_line)
        cleaned = re.sub(r"^(?:optimized|enhanced|refined|rewritten|revised)\s+query\s*[:：\-]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip("`'\"")

        if not cleaned:
            return ""

        # Guard against verbose responses by selecting the shortest candidate if commas separate options
        if " | " in cleaned:
            cleaned = min((part.strip() for part in cleaned.split("|") if part.strip()), key=len, default=cleaned)

        if cleaned.lower() == normalized_query.lower():
            return ""

        logger.debug("LLM-enhanced query (%s): '%s' -> '%s'", model_name, normalized_query[:80], cleaned[:80])
        return cleaned
    
    def _simplify_japanese_query(self, query: str) -> str:
        """Simplify Japanese queries by extracting keywords from complex sentences."""
        # Remove common Japanese sentence patterns and particles
        simplified = query
        
        # Remove polite request endings that don't help with search
        japanese_endings = [
            'してください', 'をお願いします', 'を教えてください', 'について教えて',
            'を検索してください', 'を調べてください', 'の情報を', 'について',
            'ください', 'をお願い', 'を教えて', 'を検索', 'を調べ'
        ]
        
        for ending in japanese_endings:
            if ending in simplified:
                simplified = simplified.replace(ending, '')
        
        # Remove common particles and connecting words that don't help with search
        # Use word boundaries to avoid removing parts of important words
        import re
        
        # Remove standalone particles
        particles_to_remove = ['と', 'の', 'を', 'に', 'で', 'から', 'まで', 'が', 'は', 'も', 'へ']
        
        for particle in particles_to_remove:
            # Remove particle when it's connecting words
            simplified = re.sub(rf'{particle}(?=\S)', ' ', simplified)
        
        # Remove specific connective phrases
        simplified = re.sub(r'関連', '', simplified)  # Remove "related to"
        simplified = re.sub(r'。$', '', simplified)    # Remove final period
        
        # Clean up extra spaces
        simplified = ' '.join(simplified.split())
        
        # If the simplified query is too short, return the original
        if len(simplified.strip()) < 5:
            return query
        
        return simplified.strip()
    
    def _get_optimal_ddgs_region(self, query: str) -> str:
        """Determine optimal DDGS region based on query characteristics."""
        # Check if query contains Japanese characters
        has_japanese = any('\u3040' <= char <= '\u309F' or  # Hiragana
                          '\u30A0' <= char <= '\u30FF' or  # Katakana
                          '\u4E00' <= char <= '\u9FAF'     # Kanji
                          for char in query)
        
        # Check for explicit region indicators
        query_lower = query.lower()
        
        if has_japanese or 'japan' in query_lower or 'japanese' in query_lower or '日本' in query:
            return 'jp-jp'  # Japan
        elif 'china' in query_lower or 'chinese' in query_lower or '中国' in query:
            return 'cn-zh'  # China
        elif 'korea' in query_lower or 'korean' in query_lower or '韓国' in query:
            return 'kr-kr'  # Korea
        elif 'europe' in query_lower or 'european' in query_lower:
            return 'eu-en'  # Europe
        elif 'usa' in query_lower or 'america' in query_lower or 'us ' in query_lower:
            return 'us-en'  # United States
        
        # Default to configured region or Japan (your primary market)
        return DDGS_REGION or 'jp-jp'
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if SKLEARN_AVAILABLE:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            v1 = np.array(vec1).reshape(1, -1)
            v2 = np.array(vec2).reshape(1, -1)
            return cosine_similarity(v1, v2)[0][0]
        else:
            # Fallback implementation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
    
    def _detect_language(self, text: str) -> str:
        """Improved language detection with better thresholds and more language support."""
        if not text:
            return 'en'
        
        text_len = len(text)
        if text_len == 0:
            return 'en'
        
        # Count different character sets
        japanese_chars = sum(1 for char in text if 
                           '\u3040' <= char <= '\u309F' or  # Hiragana
                           '\u30A0' <= char <= '\u30FF' or  # Katakana
                           '\u4E00' <= char <= '\u9FAF')    # Kanji
        
        # Chinese specific characters (not in Japanese)
        chinese_chars = sum(1 for char in text if 
                          '\u4E00' <= char <= '\u9FAF' and  # CJK Unified Ideographs
                          char not in '\u4E00\u4E01\u4E03\u4E07\u4E08\u4E09\u4E0A\u4E0B\u4E0D\u4E0E\u4E10')  # Common JP kanji
        
        # Korean characters
        korean_chars = sum(1 for char in text if 
                         '\uAC00' <= char <= '\uD7AF' or  # Hangul syllables
                         '\u1100' <= char <= '\u11FF')    # Hangul Jamo
        
        # Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        
        # Cyrillic characters
        cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        
        # Thai characters
        thai_chars = sum(1 for char in text if '\u0E00' <= char <= '\u0E7F')
        
        # Calculate ratios with better thresholds
        japanese_ratio = japanese_chars / text_len
        chinese_ratio = chinese_chars / text_len
        korean_ratio = korean_chars / text_len
        arabic_ratio = arabic_chars / text_len
        cyrillic_ratio = cyrillic_chars / text_len
        thai_ratio = thai_chars / text_len
        
        # Use higher thresholds to avoid false positives
        # A text needs at least 5% of characters from a script to be considered that language
        min_threshold = 0.05
        
        # Prioritize detection based on character density
        if japanese_ratio > min_threshold and japanese_ratio >= chinese_ratio:
            return 'ja'
        elif korean_ratio > min_threshold:
            return 'ko'
        elif chinese_ratio > min_threshold and chinese_ratio > japanese_ratio:
            return 'zh'
        elif arabic_ratio > min_threshold:
            return 'ar'
        elif cyrillic_ratio > min_threshold:
            return 'ru'
        elif thai_ratio > min_threshold:
            return 'th'
        
        # Fallback: check for common language indicators in Latin script
        # Use word boundaries to avoid false matches
        import re
        text_lower = text.lower()
        
        # Helper function to check for whole words only
        def has_whole_words(text_to_check, words):
            """Check if text contains any of the words as whole words (with word boundaries)."""
            for word in words:
                # Use word boundaries to match whole words only
                if re.search(r'\b' + re.escape(word) + r'\b', text_to_check):
                    return True
            return False
        
        # German indicators (whole words only)
        if has_whole_words(text_lower, ['der', 'die', 'das', 'und', 'ich', 'ist', 'mit', 'nicht', 'sie', 'auch', 'auf']):
            return 'de'
        # French indicators (whole words only)
        elif has_whole_words(text_lower, ['le', 'la', 'les', 'et', 'un', 'une', 'est', 'sont', 'dans', 'pour', 'avec']):
            # Avoid false positive from English "analysis" containing "al"
            if not has_whole_words(text_lower, ['stock', 'price', 'company', 'market', 'analysis', 'financial']):
                return 'fr'
        # Spanish indicators (whole words only)
        elif has_whole_words(text_lower, ['el', 'la', 'los', 'las', 'y', 'un', 'una', 'es', 'son', 'está', 'están']):
            # Avoid false positive from English words containing Spanish words
            if not has_whole_words(text_lower, ['stock', 'price', 'company', 'market', 'analysis', 'financial']):
                return 'es'
        # Italian indicators (whole words only)
        elif any(word in text_lower for word in ['il', 'la', 'i', 'le', 'di', 'che', 'è', 'sono', 'per', 'con', 'da']):
            if not has_whole_words(text_lower, ['stock', 'price', 'company', 'market', 'analysis', 'financial']):
                return 'it'
        
        # Default to English
        return 'en'
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Enhanced text preprocessing for BM25 scoring with Japanese support."""
        try:
            if not text:
                return []
            
            # Detect language for appropriate processing
            language = self._detect_language(text)
            
            if language == 'ja':
                # Japanese text processing
                return self._preprocess_japanese_text(text)
            else:
                # English/Latin text processing
                return self._preprocess_english_text(text)
                
        except Exception:
            # Minimal fallback
            return [word.lower() for word in text.split()[:50] if len(word) > 1]
    
    def _preprocess_japanese_text(self, text: str) -> List[str]:
        """Japanese-specific text preprocessing for BM25."""
        try:
            # Try to use MeCab for proper Japanese tokenization
            try:
                import MeCab
                tagger = MeCab.Tagger('-Owakati')  # Wakati-gaki mode (word segmentation)
                tokenized = tagger.parse(text).strip()
                tokens = [token for token in tokenized.split() if len(token) > 1]
                return tokens[:100]
            except ImportError:
                # MeCab not available, fall through to alternative method
                pass
                # Fallback: Character-based n-grams for Japanese
                # Remove punctuation but keep Japanese characters
                clean_text = re.sub(r'[^\w\s]', '', text)
                
                # Generate character bi-grams and tri-grams for better matching
                tokens = []
                
                # Add individual words (space-separated)
                words = [w for w in clean_text.split() if len(w) > 1]
                tokens.extend(words)
                
                # Add character bi-grams for continuous text
                for i in range(len(clean_text) - 1):
                    if clean_text[i] != ' ' and clean_text[i+1] != ' ':
                        bigram = clean_text[i:i+2]
                        if len(bigram.strip()) == 2:
                            tokens.append(bigram.strip())
                
                # Add character tri-grams for better context
                for i in range(len(clean_text) - 2):
                    if all(c != ' ' for c in clean_text[i:i+3]):
                        trigram = clean_text[i:i+3]
                        if len(trigram.strip()) == 3:
                            tokens.append(trigram.strip())
                
                return list(set(tokens))[:100]  # Remove duplicates and limit
                
        except Exception:
            # Final fallback for Japanese
            return [char for char in text if '\u3040' <= char <= '\u9FAF'][:50]
    
    def _preprocess_english_text(self, text: str) -> List[str]:
        """English-specific text preprocessing for BM25."""
        try:
            # Single regex pass for better performance
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            
            # Optimize: List comprehension with length filter
            tokens = [word for word in text.split() 
                     if 2 < len(word) < 20 and word.isalpha()]
            
            return tokens[:100]  # Limit tokens to prevent excessive processing
        except Exception:
            # Minimal fallback
            return [word.lower() for word in text.split()[:50] if len(word) > 2]
    
    def _calculate_bm25_scores(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Calculate BM25 relevance scores with performance optimizations."""
        if not results:
            return results
        
        try:
            # Optimize: Prepare documents more efficiently
            documents = []
            for result in results:
                # Optimize: Smarter text combination with priorities
                text_parts = []
                
                # Prioritize title (most important)
                if result.title:
                    text_parts.append(result.title * 2)  # Weight title higher
                
                # Add snippet
                if result.snippet:
                    text_parts.append(result.snippet)
                
                # Add limited content (performance optimization)
                if result.content:
                    text_parts.append(result.content[:500])  # Reduced from 1000
                
                full_text = " ".join(text_parts)
                tokenized = self._preprocess_text(full_text)
                documents.append(tokenized)
            
            if not documents or not any(documents):
                # Fallback to relevance scores
                for result in results:
                    result.bm25_score = result.relevance_score
                return results
            
            # Create BM25 index with optimizations
            bm25 = BM25Okapi(documents)
            
            # Optimize: Process query tokens once
            query_tokens = self._preprocess_text(query)
            if query_tokens:
                scores = bm25.get_scores(query_tokens)
                
                # Fix: Proper BM25 score normalization
                # BM25 scores can be negative, so we need min-max normalization
                if hasattr(scores, 'size') and scores.size > 0:
                    min_score = scores.min()
                    max_score = scores.max()
                    if max_score > min_score:
                        # Min-max normalization to 0-1 range
                        scores = (scores - min_score) / (max_score - min_score)
                    else:
                        # All scores are the same, set to 0.5
                        scores = scores * 0 + 0.5
                elif scores:  # Handle list case
                    min_score = min(scores)
                    max_score = max(scores)
                    if max_score > min_score:
                        # Min-max normalization to 0-1 range
                        scores = [(score - min_score) / (max_score - min_score) for score in scores]
                    else:
                        # All scores are the same, set to 0.5
                        scores = [0.5] * len(scores)
                
                # Update results efficiently
                for result, score in zip(results, scores):
                    result.bm25_score = float(score)
            
        except Exception as e:
            logger.debug(f"BM25 scoring failed: {e}")
            # Continue without BM25 scores if it fails
            for result in results:
                result.bm25_score = result.relevance_score
        
        return results
    
    async def _get_azure_embeddings_client(self):
        """Get Azure OpenAI embeddings client for semantic similarity."""
        if not AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT or not AZURE_OPENAI_API_KEY:
            return None
        
        try:
            # Try Azure client first (using AzureOpenAI for proper Azure support)
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION or "2024-02-01",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            # Convert to async client
            from openai import AsyncAzureOpenAI
            async_client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION or "2024-02-01",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            return async_client
        except Exception as e:
            logger.debug(f"Failed to create Azure embeddings client: {e}")
            # Fallback to standard AsyncOpenAI without api_version
            try:
                fallback_client = AsyncOpenAI(
                    api_key=AZURE_OPENAI_API_KEY,
                    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT}"
                )
                return fallback_client
            except Exception as fallback_e:
                logger.debug(f"Fallback embeddings client also failed: {fallback_e}")
                return None
    
    async def _calculate_semantic_scores(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Calculate semantic similarity scores using Azure text embeddings with improved rerank window and batching."""
        if not results:
            return results
        
        # Implement rerank window for cost efficiency - only embed top N results
        RERANK_WINDOW_SIZE = 15  # Reduce embeddings cost by limiting to top candidates
        working_results = results[:RERANK_WINDOW_SIZE] if len(results) > RERANK_WINDOW_SIZE else results
        
        # Check cache for query embedding
        query_cache_key = _get_cache_key(f"query_embedding:{query}")
        query_embedding = _embeddings_cache.get(query_cache_key)
        
        if query_embedding is not None:
            logger.debug("Using cached query embedding")
        
        try:
            embeddings_client = await self._get_azure_embeddings_client()
            if not embeddings_client:
                logger.debug("Azure embeddings client not available, skipping semantic scoring")
                # Set semantic scores to relevance scores for non-processed results
                for result in results[RERANK_WINDOW_SIZE:]:
                    result.semantic_score = result.relevance_score
                return results
            
            # Get query embedding (from cache or fresh)
            if query_embedding is None:
                query_task = embeddings_client.embeddings.create(
                    input=query,
                    model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                    timeout=20.0
                )
            
            # Prepare batch text with Azure batch size limits (max 16 for text-embedding-ada-002)
            BATCH_SIZE = 12  # Conservative batch size for Azure limits
            texts_to_embed = []
            cached_embeddings = {}
            
            for i, result in enumerate(working_results):
                # Optimize: Use only most relevant text parts (title + beginning of content)
                text_parts = [result.title]
                if result.snippet:
                    text_parts.append(result.snippet)
                if result.content:
                    # Limit content to first 300 chars for embedding efficiency
                    text_parts.append(result.content[:300])
                
                full_text = " ".join(text_parts)
                
                # Check cache for this text using new LRU cache
                text_cache_key = _get_cache_key(f"doc_embedding:{full_text}")
                cached_embedding = _embeddings_cache.get(text_cache_key)
                
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                    continue
                
                texts_to_embed.append((i, full_text))
            
            # Process query and new documents with improved batching
            tasks = []
            if query_embedding is None:
                tasks.append(query_task)
            
            # Process documents in batches for better Azure compliance
            embedding_batches = []
            if texts_to_embed:
                # Split into batches respecting Azure limits
                for i in range(0, len(texts_to_embed), BATCH_SIZE):
                    batch = texts_to_embed[i:i + BATCH_SIZE]
                    batch_texts = [text for _, text in batch]
                    
                    docs_task = embeddings_client.embeddings.create(
                        input=batch_texts,
                        model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                        timeout=30.0
                    )
                    embedding_batches.append((docs_task, batch))
            
            # Await query embedding first
            if query_embedding is None and tasks:
                query_response = await tasks[0]
                
                # Handle failures gracefully
                if isinstance(query_response, Exception):
                    logger.debug(f"Query embedding failed: {query_response}")
                    for result in results:
                        result.semantic_score = result.relevance_score
                    return results
                
                query_embedding = query_response.data[0].embedding
                
                # Cache query embedding using LRU cache
                _embeddings_cache.put(query_cache_key, query_embedding)
            
            # Process document embeddings in batches
            if embedding_batches:
                batch_tasks = [batch_task for batch_task, _ in embedding_batches]
                batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for (batch_task, batch_docs), batch_response in zip(embedding_batches, batch_responses):
                    if not isinstance(batch_response, Exception):
                        for (doc_idx, doc_text), embedding_data in zip(batch_docs, batch_response.data):
                            # Cache document embedding using LRU cache
                            doc_cache_key = _get_cache_key(f"doc_embedding:{doc_text}")
                            _embeddings_cache.put(doc_cache_key, embedding_data.embedding)
                            cached_embeddings[doc_idx] = embedding_data.embedding
                    else:
                        logger.debug(f"Embedding batch failed: {batch_response}")
            
            # Calculate similarities using cached and fresh embeddings
            if SKLEARN_AVAILABLE and len(cached_embeddings) == len(working_results):
                # Use sklearn's optimized cosine similarity for batch processing
                import numpy as np
                
                doc_embeddings = []
                for i in range(len(working_results)):
                    doc_embeddings.append(cached_embeddings[i])
                
                doc_embeddings_array = np.array(doc_embeddings)
                query_embedding_array = np.array([query_embedding])
                
                similarities = cosine_similarity(query_embedding_array, doc_embeddings_array)[0]
                
                for i, (result, similarity) in enumerate(zip(working_results, similarities)):
                    result.semantic_score = float(similarity)
                    result.embedding_vector = cached_embeddings[i]
                    
                # Set semantic scores to relevance scores for non-processed results
                for result in results[RERANK_WINDOW_SIZE:]:
                    result.semantic_score = result.relevance_score
            else:
                # Fallback to manual calculation
                for i, result in enumerate(working_results):
                    if i in cached_embeddings:
                        similarity = self._cosine_similarity(query_embedding, cached_embeddings[i])
                        result.semantic_score = float(similarity)
                        result.embedding_vector = cached_embeddings[i]
                    else:
                        result.semantic_score = result.relevance_score
                        
                # Set semantic scores to relevance scores for non-processed results
                for result in results[RERANK_WINDOW_SIZE:]:
                    result.semantic_score = result.relevance_score
            
            await embeddings_client.close()
            
        except Exception as e:
            logger.debug(f"Semantic scoring failed: {e}")
            # Continue without semantic scores if it fails
            for result in results:
                result.semantic_score = result.relevance_score
        
        return results
    
    async def _verify_citations_nli(
        self,
        answer: str,
        results: List[SearchResult]
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Use OpenAI NLI to verify each cited claim against source excerpts."""
        verification_notes: List[str] = []
        verification_details: Dict[str, Any] = {"provider": None, "evaluations": []}

        try:
            nli_client, nli_model, provider = await self._get_nli_client()
            if not nli_client or not nli_model:
                return answer, verification_notes, verification_details

            verification_details["provider"] = provider or "openai"

            citation_map: Dict[str, SearchResult] = {
                str(result.citation_id): result
                for result in results
                if getattr(result, "citation_id", None)
            }

            if not citation_map:
                return answer, verification_notes, verification_details

            # Extract sentences that contain explicit citations
            seen_pairs = set()
            claim_pairs: List[Tuple[str, str]] = []
            raw_sentences: List[str] = []

            for paragraph in re.split(r"\n+", answer):
                raw_sentences.extend(re.split(r"(?<=[.!?。！？])\s+", paragraph))

            for sentence in raw_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                refs = re.findall(r'\[(\d+)\]', sentence)
                if not refs:
                    continue
                claim_text = re.sub(r'\s*\[(\d+)\]\s*', ' ', sentence).strip()
                claim_text = re.sub(r'\s+', ' ', claim_text)
                if not claim_text:
                    continue
                for ref in refs:
                    if ref not in citation_map:
                        continue
                    key = (ref, claim_text)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    claim_pairs.append((ref, claim_text))

            if not claim_pairs:
                return answer, verification_notes, verification_details

            MAX_EVALS = 8
            status_priority = {
                "contradicted": 3,
                "unsupported": 2,
                "unknown": 1,
                "supported": 0,
            }

            for ref, claim_text in claim_pairs[:MAX_EVALS]:
                result = citation_map.get(ref)
                if not result:
                    continue

                evidence_excerpt = self._extract_evidence_excerpt(result, claim_text)
                if not evidence_excerpt:
                    evidence_excerpt = (result.snippet or "")[:500]
                if not evidence_excerpt:
                    continue

                try:
                    evaluation = await self._run_nli_evaluation(
                        nli_client,
                        nli_model,
                        claim_text,
                        evidence_excerpt,
                        ref
                    )
                except Exception as eval_error:
                    logger.debug(f"NLI evaluation error for citation [{ref}]: {eval_error}")
                    continue

                eval_status = evaluation.get("status", "unknown")
                eval_confidence = evaluation.get("confidence", 0.0)
                eval_reason = evaluation.get("reason", "")

                # Update result metadata using severity + confidence heuristics
                current_status = (result.nli_status or "unknown").lower()
                current_priority = status_priority.get(current_status, -1)
                new_priority = status_priority.get(eval_status, -1)
                should_update = False

                if eval_status == "supported" and current_status in {"unknown", "supported"}:
                    should_update = eval_confidence >= getattr(result, "nli_confidence", 0.0)
                elif new_priority > current_priority:
                    should_update = True
                elif new_priority == current_priority and eval_confidence > getattr(result, "nli_confidence", 0.0):
                    should_update = True

                if should_update:
                    result.nli_status = eval_status
                    result.nli_confidence = eval_confidence
                    result.nli_reason = eval_reason
                    result.nli_last_claim = claim_text

                verification_details["evaluations"].append({
                    "citation_id": ref,
                    "claim": claim_text,
                    "status": eval_status,
                    "confidence": eval_confidence,
                    "reason": eval_reason,
                    "evidence_excerpt": evidence_excerpt[:300],
                })

                claim_excerpt = claim_text[:160] + ("…" if len(claim_text) > 160 else "")

                if eval_status == "contradicted":
                    verification_notes.append(
                        f"Claim '{claim_excerpt}' with citation [{ref}] appears contradicted or unsupported ({eval_reason})."
                    )
                elif eval_status in {"unsupported", "unknown"} and eval_confidence < 0.6:
                    verification_notes.append(
                        f"Evidence for citation [{ref}] on '{claim_excerpt}' is inconclusive ({eval_reason})."
                    )

            # Append verification notes to answer if any
            if verification_notes:
                unique_notes = list(dict.fromkeys(verification_notes))
                note_section = "\n".join(f"- {note}" for note in unique_notes)
                answer = answer.rstrip() + "\n\n**Verification notes:**\n" + note_section
                verification_notes = unique_notes

            return answer, verification_notes, verification_details

        except Exception as e:
            logger.debug(f"NLI verification failed: {e}")
            return answer, verification_notes, verification_details

    @staticmethod
    def _merge_adjacent_citations(text: str) -> str:
        """Collapse sequences like [1][2][3] into a single bracket [1,2,3]."""
        if not text:
            return text

        try:
            pattern = re.compile(r'(?:\[\d+\]){2,}')

            def _repl(match) -> str:
                numbers = re.findall(r'\d+', match.group())
                if not numbers:
                    return match.group()
                return f"[{','.join(numbers)}]"

            return pattern.sub(_repl, text)
        except Exception:
            return text

    def _ensure_citations_in_answer(self, answer: str, results: List[SearchResult]) -> str:
        """Ensure the synthesized answer contains inline citations referencing search results."""
        try:
            if not answer or not results:
                return self._merge_adjacent_citations(answer)

            citation_ids = [str(r.citation_id) for r in results if getattr(r, "citation_id", None)]
            if not citation_ids:
                return self._merge_adjacent_citations(answer)

            existing_refs = set(re.findall(r'\[(\d+)\]', answer))
            valid_id_set = set(citation_ids)
            valid_refs = {ref for ref in existing_refs if ref in citation_ids}
            if valid_refs:
                return answer

            paragraphs = answer.split('\n\n')
            if not paragraphs:
                return answer

            id_index = 0
            total_ids = len(citation_ids)
            updated_paragraphs: List[str] = []

            for paragraph in paragraphs:
                stripped = paragraph.strip()
                if not stripped:
                    updated_paragraphs.append(paragraph)
                    continue

                paragraph_refs = re.findall(r'\[(\d+)\]', paragraph)
                if paragraph_refs and any(ref in valid_id_set for ref in paragraph_refs):
                    updated_paragraphs.append(paragraph)
                    continue

                citation_id = citation_ids[id_index]
                id_index = (id_index + 1) % total_ids

                # Preserve original paragraph spacing while appending citation marker
                has_trailing_space = paragraph.endswith(' ')
                appended = paragraph.rstrip() + f" [{citation_id}]"
                if has_trailing_space:
                    appended += ' '
                updated_paragraphs.append(appended)

            updated_answer = '\n\n'.join(updated_paragraphs)

            # If paragraphs approach failed to add any valid refs, fallback to line-level injection
            updated_refs = re.findall(r'\[(\d+)\]', updated_answer)
            if not any(ref in valid_id_set for ref in updated_refs):
                lines = answer.split('\n')
                updated_lines: List[str] = []
                id_index = 0

                for line in lines:
                    stripped = line.strip()
                    line_refs = re.findall(r'\[(\d+)\]', line)
                    if not stripped or any(ref in valid_id_set for ref in line_refs):
                        updated_lines.append(line)
                        continue

                    citation_id = citation_ids[id_index]
                    id_index = (id_index + 1) % total_ids
                    updated_lines.append(line.rstrip() + f" [{citation_id}]")

                updated_answer = '\n'.join(updated_lines)

            return self._merge_adjacent_citations(updated_answer)
        except Exception as e:
            logger.debug(f"Failed to enforce citations: {e}")
            return self._merge_adjacent_citations(answer)

    def _extract_evidence_excerpt(self, result: SearchResult, claim: str, max_length: int = 900) -> str:
        """Extract the most relevant excerpt from a result's content for NLI evaluation."""
        text = (result.content or "").strip()
        if not text:
            return (result.snippet or "").strip()[:max_length]

        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        if not sentences:
            sentences = [text]

        keywords = {
            token.lower()
            for token in re.findall(r"[\w\u3040-\u30FF\u4E00-\u9FFF]+", claim)
            if len(token) > 2
        }

        def sentence_score(sentence: str) -> int:
            lower = sentence.lower()
            return sum(1 for kw in keywords if kw in lower)

        ranked = sorted(sentences, key=sentence_score, reverse=True)
        excerpt_parts: List[str] = []

        for sentence in ranked:
            if not sentence.strip():
                continue
            excerpt_parts.append(sentence.strip())
            if len(" ".join(excerpt_parts)) >= max_length * 0.6:
                break

        excerpt = " ".join(excerpt_parts).strip()
        if not excerpt:
            excerpt = sentences[0].strip()

        return excerpt[:max_length]

    async def _run_nli_evaluation(
        self,
        client: AsyncOpenAI,
        model: str,
        claim: str,
        evidence: str,
        citation_id: str
    ) -> Dict[str, Any]:
        """Call OpenAI to determine if evidence supports the claim."""
        system_prompt = (
            "You are a rigorous fact-checking assistant."
            " Review the claim and the evidence excerpt."
            " Respond with strict JSON using keys verdict, confidence, reason, quote."
            " verdict must be one of ['SUPPORTED','CONTRADICTED','INSUFFICIENT']."
            " confidence is a float between 0 and 1."
            " reason must be concise (<= 40 words)."
            " quote should be a short phrase from the evidence that best justifies the verdict."
        )

        user_prompt = (
            f"Citation ID: [{citation_id}]\n"
            f"Claim: {claim}\n\n"
            f"Evidence excerpt:\n{evidence.strip()}"
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=320
        )

        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Empty NLI response")

        content = response.choices[0].message.content.strip()
        parsed = self._parse_nli_json(content)
        if not parsed:
            raise ValueError(f"Unable to parse NLI JSON: {content}")

        verdict_raw = str(parsed.get("verdict", "")).strip().lower()
        if verdict_raw in {"supported", "support", "entails", "entailed"}:
            status = "supported"
        elif verdict_raw in {"contradicted", "contradicts", "refuted", "refutes", "opposed"}:
            status = "contradicted"
        else:
            status = "unsupported" if verdict_raw in {"insufficient", "inconclusive", "unknown"} else "unknown"

        try:
            confidence = float(parsed.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        reason = str(parsed.get("reason", "")).strip()
        quote = str(parsed.get("quote", "")).strip()

        return {
            "status": status,
            "confidence": confidence,
            "reason": reason,
            "quote": quote,
        }

    def _parse_nli_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object from model output."""
        text = text.strip()
        if not text:
            return None

        # If already JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt to locate JSON-like substring
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None

        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    
    def _calculate_combined_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Calculate enhanced combined ranking with domain priors, recency, and snippet-title matching."""
        if not results:
            return results
        
        # Core scoring weights (remove relevance_score completely)
        bm25_weight = 0.4       # Lexical relevance
        semantic_weight = 0.4   # Semantic similarity  
        content_weight = 0.1    # Content quality
        signals_weight = 0.1    # Domain priors + recency + snippet match
        
        # Calculate score statistics for normalization
        bm25_scores = [r.bm25_score for r in results if hasattr(r, 'bm25_score')]
        semantic_scores = [r.semantic_score for r in results if hasattr(r, 'semantic_score')]
        
        # Robust normalization using min-max scaling
        bm25_min, bm25_max = (min(bm25_scores), max(bm25_scores)) if bm25_scores else (0, 1)
        semantic_min, semantic_max = (min(semantic_scores), max(semantic_scores)) if semantic_scores else (0, 1)
        
        for result in results:
            # 1. Normalize BM25 and semantic scores
            if bm25_max > bm25_min:
                bm25_norm = (result.bm25_score - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_norm = 0.5
                
            if semantic_max > semantic_min:
                semantic_norm = (result.semantic_score - semantic_min) / (semantic_max - semantic_min)
            else:
                semantic_norm = 0.5
            
            # 2. Content quality scoring (independent of relevance_score)
            content_quality = 0.5  # Base quality
            if result.word_count > 0:
                # Optimal range 100-400 words for citations
                if result.word_count < 50:
                    word_bonus = result.word_count / 100  # Penalty for very short
                elif result.word_count <= 400:
                    word_bonus = min(result.word_count / 250, 1.0)  # Optimal range
                else:
                    word_bonus = max(0.7, 1.0 - (result.word_count - 400) / 1000)  # Slight penalty for very long
                content_quality = word_bonus
            
            # 3. Domain priors - check all categories for this domain
            domain_boost = 1.0  # Default no boost
            if result.url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result.url).netloc.lower()
                    
                    # Check all domain categories for matches
                    for category, domains in DOMAIN_PRIORS.items():
                        if domain in domains:
                            domain_boost = max(domain_boost, domains[domain])  # Take highest boost
                            break
                        # Also check for subdomain matches (e.g., subdomain.wikipedia.org)
                        for trusted_domain, boost in domains.items():
                            if domain.endswith('.' + trusted_domain) or domain == trusted_domain:
                                domain_boost = max(domain_boost, boost)
                                break
                except Exception:
                    domain_boost = 1.0
            
            # 4. Recency scoring (for time-sensitive queries)
            recency_boost = 1.0
            if hasattr(result, 'timestamp') and result.timestamp:
                try:
                    from datetime import datetime, timedelta
                    # Simple recency boost - more sophisticated parsing could be added
                    if '2024' in result.timestamp or '2025' in result.timestamp:
                        recency_boost = 1.1  # Recent content gets small boost
                    elif '2023' in result.timestamp:
                        recency_boost = 1.05  # Slightly older content
                    elif any(year in result.timestamp for year in ['2022', '2021', '2020']):
                        recency_boost = 1.0   # Neutral for 2020-2022
                    else:
                        recency_boost = 0.95  # Slight penalty for older content
                except Exception:
                    recency_boost = 1.0
            
            # 5. Snippet-title matching for citation precision  
            snippet_title_boost = 1.0
            if result.title and result.snippet:
                title_words = set(result.title.lower().split())
                snippet_words = set(result.snippet.lower().split())
                
                # Remove common stop words for better matching
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                title_words = title_words - stop_words
                snippet_words = snippet_words - stop_words
                
                if title_words and snippet_words:
                    overlap = len(title_words & snippet_words)
                    total_unique = len(title_words | snippet_words)
                    
                    if total_unique > 0:
                        match_ratio = overlap / total_unique
                        # Boost results where title and snippet have good word overlap
                        snippet_title_boost = 1.0 + (match_ratio * 0.2)  # Up to 20% boost
            
            # 6. Combine all signals
            signal_score = (domain_boost * recency_boost * snippet_title_boost - 1.0) / 2.0  # Normalize to 0-1 range
            signal_score = max(0.0, min(1.0, signal_score))  # Clamp to [0,1]
            
            # 7. Calculate final weighted combination (NO relevance_score)
            combined = (
                bm25_weight * bm25_norm +
                semantic_weight * semantic_norm +
                content_weight * content_quality +
                signals_weight * signal_score
            )
            
            # Apply multiplicative domain boost separately for stronger effect
            combined *= domain_boost
            
            # Ensure final score is bounded
            result.combined_score = max(0.0, min(1.0, combined))
            
            # Store individual signal scores for debugging
            result.domain_boost = domain_boost
            result.recency_boost = recency_boost  
            result.snippet_title_boost = snippet_title_boost
        
        return results
    
    async def _extract_and_enhance_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract and enhance content from search results with optimized performance."""
        enhanced_results = []
        
        # Optimize: Increase concurrent processing but with smart limits
        max_concurrent = min(8, len(results))  # Scale with result count but cap at 8
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_result(result: SearchResult) -> SearchResult:
            async with semaphore:
                return await self._enhance_single_result(result)
        
        # Process all results concurrently with optimized timeout
        tasks = [process_result(result) for result in results]
        enhanced_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed results and optimize sorting
        valid_results = []
        for result in enhanced_results:
            if isinstance(result, SearchResult) and (result.content or result.snippet):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Content extraction failed: {result}")
        
        # Optimized sorting with better scoring weights
        valid_results.sort(key=lambda x: (
            x.relevance_score * 0.5 +  # Increased weight for relevance
            min(x.word_count / 150, 1.0) * 0.2 +  # Reduced content length weight
            (0.3 if x.content else 0.1)  # Content bonus but not too high
        ), reverse=True)
        
        return valid_results[:self.max_synthesis_sources]
    
    async def _enhance_single_result(self, result: SearchResult) -> SearchResult:
        """Enhance a single search result with optimized content extraction."""
        if not result.url or result.content:
            return result  # Skip if no URL or already has content

        cached_content = _content_cache.get(result.url)
        if cached_content:
            cached_text = cached_content.get("content", "")
            if cached_text:
                result.content = cached_text
                result.word_count = cached_content.get(
                    "word_count",
                    len(cached_text.split()),
                )
                logger.debug("Using cached content for %s", result.url)
                return result
        
        try:
            session = await self._get_session()
            
            # Optimize: Shorter timeout and better error handling
            timeout = aiohttp.ClientTimeout(total=8, connect=3)  # Faster timeouts
            
            async with session.get(result.url, timeout=timeout) as response:
                if response.status == 200:
                    # Improved encoding detection and handling
                    content_size_limit = 1024 * 1024  # 1MB limit
                    
                    # Get raw bytes first
                    raw_content = await response.read()
                    if len(raw_content) > content_size_limit:
                        raw_content = raw_content[:content_size_limit]
                    
                    # Try to detect encoding more intelligently
                    html_content = None
                    
                    # First try the response's declared encoding
                    try:
                        if response.charset:
                            html_content = raw_content.decode(response.charset, errors='ignore')
                    except (UnicodeDecodeError, LookupError):
                        pass
                    
                    # Fallback to UTF-8 with error handling
                    if html_content is None:
                        try:
                            html_content = raw_content.decode('utf-8', errors='ignore')
                        except UnicodeDecodeError:
                            # Final fallback - try common encodings
                            for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                                try:
                                    html_content = raw_content.decode(encoding, errors='ignore')
                                    break
                                except UnicodeDecodeError:
                                    continue
                    
                    # If all encoding attempts failed, use utf-8 with replace
                    if html_content is None:
                        html_content = raw_content.decode('utf-8', errors='replace')
                    
                    extracted_content = await self._extract_clean_content(html_content)
                    
                    if extracted_content:
                        # Optimize: Smarter content truncation
                        result.content = extracted_content[:self.max_content_length]
                        result.word_count = len(result.content.split())
                        _content_cache.put(
                            result.url,
                            {
                                "content": result.content,
                                "word_count": result.word_count,
                            },
                        )
                        
                        # Remove artificial relevance boost - let BM25 and semantic scores determine ranking
                        # result.relevance_score = min(result.relevance_score + 0.1, 1.0)  # REMOVED: This inflates all scores to 0.9
                        
        except asyncio.TimeoutError:
            logger.debug(f"Content extraction timed out for {result.url}")
        except Exception as e:
            logger.debug(f"Content extraction failed for {result.url}: {e}")
        
        return result
    
    async def _extract_clean_content(self, html: str) -> str:
        """Extract clean, readable content from HTML with improved encoding and content detection."""
        try:
            # Improved HTML parsing with better encoding handling
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements more comprehensively
            unwanted_tags = [
                'script', 'style', 'nav', 'footer', 'header', 'aside', 'ads',
                'noscript', 'iframe', 'embed', 'object', 'form', 'input',
                'button', 'select', 'textarea', 'label', 'fieldset',
                '.advertisement', '.ad', '.ads', '.sidebar', '.menu',
                '.navigation', '.navbar', '.breadcrumb', '.pagination',
                '.social', '.share', '.comment', '.related', '.popular'
            ]
            
            for selector in unwanted_tags:
                if selector.startswith('.'):
                    # CSS class selector
                    for tag in soup.select(selector):
                        tag.decompose()
                else:
                    # Tag selector
                    for tag in soup(selector):
                        tag.decompose()
            
            # Enhanced main content detection with scoring
            content_candidates = []
            
            # Try various content selectors with scores
            content_selectors = [
                ('main', 10),
                ('article', 9),
                ('[role="main"]', 8),
                ('.main-content', 7),
                ('.content', 6),
                ('#content', 6),
                ('.entry-content', 5),
                ('.post-content', 5),
                ('.article-content', 5),
                ('.text-content', 4),
                ('.body-content', 4)
            ]
            
            for selector, score in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text_length = len(element.get_text(strip=True))
                    if text_length > 100:  # Minimum content threshold
                        content_candidates.append((element, score + (text_length / 100)))
            
            # If no specific content area found, try to find the largest text block
            if not content_candidates:
                all_divs = soup.find_all(['div', 'section', 'p'])
                for div in all_divs:
                    text_length = len(div.get_text(strip=True))
                    if text_length > 200:
                        content_candidates.append((div, text_length / 100))
            
            # Select the best content candidate
            if content_candidates:
                main_content = max(content_candidates, key=lambda x: x[1])[0]
            else:
                # Final fallback to body
                main_content = soup.find('body') or soup
            
            # Enhanced text extraction with better formatting preservation
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_emphasis = False
            h.body_width = 0  # No line wrapping
            h.ignore_tables = False  # Keep tables for financial data
            h.decode_errors = 'ignore'  # Handle encoding issues gracefully
            
            text_content = h.handle(str(main_content))
            
            # Improved text cleaning
            # Fix common encoding issues
            text_content = text_content.replace('\xa0', ' ')  # Non-breaking space
            text_content = text_content.replace('\u2019', "'")  # Smart apostrophe
            text_content = text_content.replace('\u201c', '"').replace('\u201d', '"')  # Smart quotes
            text_content = text_content.replace('\u2013', '-').replace('\u2014', '--')  # Em/en dashes
            
            # Clean up whitespace and newlines more intelligently
            # Preserve paragraph breaks but remove excessive spacing
            lines = text_content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line:  # Non-empty line
                    cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1]:  # Add single empty line between paragraphs
                    cleaned_lines.append('')
            
            # Join lines and normalize whitespace within lines
            text_content = '\n'.join(cleaned_lines)
            text_content = re.sub(r' +', ' ', text_content)  # Multiple spaces to single space
            text_content = re.sub(r'\n\n\n+', '\n\n', text_content)  # Max 2 consecutive newlines
            
            # Quality check: ensure we have substantial and readable content
            word_count = len(text_content.split())
            if word_count > 20 and len(text_content) > 100:
                return text_content.strip()
                
        except Exception as e:
            logger.debug(f"Enhanced HTML parsing failed: {e}")
        
        return ""
    
    async def _synthesize_answer(self, query: str, results: List[SearchResult]) -> str:
        """Synthesize an answer from search results using AI."""
        if not results:
            return "I couldn't find enough reliable information to answer your question."
        
        try:
            # Prepare context from search results
            # OPTIMIZATION: Use top 4 results instead of 6 for faster synthesis (reduces token count)
            context_parts = []
            for result in results[:4]:  # Reduced from 6 to 4 for speed
                content = result.content or result.snippet
                if content:
                    # OPTIMIZATION: Reduced from 1000 to 800 chars per source
                    context_parts.append(f"[{result.citation_id}] {result.title}\n{content[:800]}")
            
            if not context_parts:
                return "I found search results but couldn't extract enough content to provide a comprehensive answer."
            
            context = "\n\n".join(context_parts)
            
            # Create synthesis prompt with emphasis on citations
            synthesis_prompt = f"""You are an expert research assistant. Based on the following search results, provide a comprehensive, accurate answer to the question: "{query}"

SEARCH RESULTS:
{context}

CRITICAL CITATION REQUIREMENTS:
- You MUST use citations [1], [2], [3], etc. for EVERY factual claim, statistic, or piece of information
- Citations should correspond to the source numbers in the search results above
- Include multiple citations per paragraph when drawing from different sources
- Even basic facts should be cited to show source credibility

Instructions:
1. Synthesize information from multiple sources - combine insights rather than just summarizing each source
2. Include specific facts, numbers, dates, and details with proper citations [1], [2], etc.
3. Use citations for ALL factual claims - this is mandatory, not optional
4. If information conflicts between sources, acknowledge this and cite both sources
5. Be thorough but well-organized - use clear structure with headings for complex topics
6. Prioritize the most recent and authoritative information
7. If the question can't be fully answered, clearly explain what information is available and cite sources
8. For financial/market queries, include relevant context about timing and market conditions with citations
9. Use natural, conversational language while maintaining citation accuracy
10. If sources contain Japanese content, include both Japanese terms and English translations with citations

Remember: Every statement of fact must have a citation [X]. This is essential for source credibility.

Provide a complete, well-researched answer with proper citations:"""

            base_system_prompt = (
                "You are an expert research assistant that synthesizes information from multiple sources to "
                "provide accurate, well-cited answers. Always prioritize factual accuracy over speculation. "
                "When information is uncertain or incomplete, clearly state this and explain what additional "
                "information would be needed for a complete answer."
            )

            default_messages = [
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": synthesis_prompt}
            ]

            # OPTIMIZATION: Use fast model (gpt-4o-mini) for synthesis instead of slow OSS 120B
            # This reduces synthesis time from 20s to ~5s (4x faster)
            try:
                # Use gpt-4o-mini for faster synthesis
                try:
                    # Try OpenAI first (usually faster than Azure)
                    openai_client = get_openai_client()
                    response = await asyncio.wait_for(
                        openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=default_messages,
                            max_tokens=600,      # Reduced from 800 for faster generation
                            temperature=0.2
                        ),
                        timeout=10.0  # Aggressive timeout: 10s max for synthesis
                    )
                    logger.info("Used gpt-4o-mini for fast answer synthesis")
                except Exception as openai_error:
                    logger.debug(f"OpenAI synthesis failed: {openai_error}, trying Azure fallback")
                    # Fallback to Azure OSS 120B only if OpenAI fails
                    if AZURE_OPENAI_DEPLOYMENT_OSS_120B and AZURE_OPENAI_API_KEY:
                        azure_client = AsyncAzureOpenAI(
                            api_key=AZURE_OPENAI_API_KEY,
                            api_version=AZURE_OPENAI_API_VERSION or "2024-02-01",
                            azure_endpoint=AZURE_OPENAI_ENDPOINT,
                            timeout=15.0  # Reduced from 60s to 15s for faster failure
                        )
                        
                        system_prompt = get_system_prompt_for_model("gpt-oss-120b")
                        azure_messages = [
                            {
                                "role": "system",
                                "content": f"{system_prompt}\n\n{base_system_prompt}"
                            },
                            {"role": "user", "content": synthesis_prompt}
                        ]

                        response = await asyncio.wait_for(
                            azure_client.chat.completions.create(
                                model=AZURE_OPENAI_DEPLOYMENT_OSS_120B,
                                messages=azure_messages,
                                max_tokens=600,   # Reduced from 800
                                temperature=0.2
                            ),
                            timeout=15.0  # Hard timeout for Azure synthesis
                        )
                        
                        await azure_client.close()
                        logger.info("Used Azure GPT OSS 120B for answer synthesis (fallback)")
                    else:
                        raise ValueError("No synthesis model available")
                    
            except Exception as azure_error:
                logger.warning(f"Azure GPT OSS 120B failed: {azure_error}, falling back to OpenAI")
                try:
                    # Fallback to OpenAI direct if Azure fails
                    openai_client = get_openai_client()
                    response = await openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=default_messages,
                        max_tokens=1200,
                        temperature=0.2
                    )
                except Exception as openai_error:
                    logger.warning(f"OpenAI direct API also failed: {openai_error}, using Azure fallback client")
                    # Final fallback to Azure client service
                    client = get_client()
                    if not client:
                        raise RuntimeError("No fallback AI client available")

                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=default_messages,
                        max_tokens=1200,
                        temperature=0.2
                    )
            
            if not response or not getattr(response, "choices", None):
                logger.error("Synthesis model returned no choices")
                return "I was unable to synthesize a comprehensive answer from the available sources."

            answer = response.choices[0].message.content

            if answer:
                return self._merge_adjacent_citations(answer.strip())
            else:
                return "I was unable to synthesize a comprehensive answer from the available sources."
                
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"I found relevant information but encountered an error while synthesizing the answer. Here are the key sources I found: {', '.join([r.title for r in results[:3]])}"
    
    def _build_citations(self, results: List[SearchResult]) -> Dict[int, Any]:
        """Build citation dictionary from search results with source quality indicators.

        Backward compatibility: historically returned citation_id -> string ("Title - domain [indicator]").
        For richer tool card rendering, we now prefer a structured object:
            {
              citation_id: {
                'title': str,
                'domain': str,
                'url': str,
                'source': str,  # brave_search | ddgs_search | ddgs_fallback
                                'display': str, # legacy formatted string
                                'quality': 'high' | 'normal' | 'fallback'
                                'nli_status': 'supported' | 'contradicted' | 'unsupported' | 'unknown'
                                'nli_confidence': float,
                                'nli_reason': str,
                                'nli_last_claim': str
              }
            }
        The frontend can check type and fallback to string if needed.
        """
        citations: Dict[int, Any] = {}
        seen_keys: set[str] = set()

        for result in results:
            if not result.citation_id:
                continue
            # Normalize URL defensively (in case result created outside parsing helper)
            normalized_url = self._normalize_result_url(result.url)
            result.url = normalized_url
            dedup_key = normalized_url or (result.title.strip().lower() if result.title else "")
            if dedup_key and dedup_key in seen_keys:
                continue
            if dedup_key:
                seen_keys.add(dedup_key)

            domain = urlparse(result.url).netloc if result.url else "Unknown"

            # Quality indicator text & level
            quality = 'normal'
            indicator = ''
            if result.source == 'brave_search':
                indicator = ' [High-Quality Source]'
                quality = 'high'
            elif result.source == 'ddgs_fallback':
                indicator = ' [Fallback]'
                quality = 'fallback'

            nli_status = getattr(result, 'nli_status', 'unknown') or 'unknown'
            nli_confidence = getattr(result, 'nli_confidence', 0.0) or 0.0
            nli_reason = getattr(result, 'nli_reason', '') or ''
            nli_last_claim = getattr(result, 'nli_last_claim', '') or ''

            if nli_status == 'supported' and nli_confidence >= 0.6:
                verification_indicator = ' [Verified]'
            elif nli_status == 'contradicted':
                verification_indicator = ' [Verification Warning]'
            elif nli_status in {'unsupported', 'unknown'}:
                verification_indicator = ' [Not Verified]'
            else:
                verification_indicator = ''

            display = f"{result.title} - {domain}{indicator}{verification_indicator}"
            # Structured entry
            citations[result.citation_id] = {
                'title': result.title,
                'domain': domain,
                'url': result.url,
                'source': result.source,
                'display': display,
                'quality': quality,
                'nli_status': nli_status,
                'nli_confidence': nli_confidence,
                'nli_reason': nli_reason,
                'nli_last_claim': nli_last_claim
            }

        return citations

    def _summarize_ranked_citations(self, results: List[SearchResult]) -> str:
        """Produce a concise textual summary of ranked sources for downstream models."""
        if not results:
            return "No ranked sources were available for this query."

        summary_lines: List[str] = []
        for result in results[: self.max_synthesis_sources]:
            title = (result.title or result.url or "Untitled source").strip()
            snippet = (result.snippet or result.content or "").strip()
            if snippet:
                snippet = re.sub(r"\s+", " ", snippet)
                snippet = snippet[:240].rstrip()
            # Keep user-facing header simple: citation id + title only
            header = f"[{result.citation_id}] {title}"
            if snippet:
                summary_lines.append(f"{header}\n  {snippet}")
            else:
                summary_lines.append(header)

        return "\n\n".join(summary_lines)
    
    def _calculate_confidence(self, results: List[SearchResult], answer: str) -> float:
        """Calculate confidence score based on result quality and answer."""
        if not results:
            return 0.0
        
        # Base confidence on number and quality of sources
        source_score = min(len(results) / 5, 1.0)  # 5+ sources = max score
        
        # Content quality score
        content_results = [r for r in results if r.content]
        content_score = len(content_results) / len(results) if results else 0
        
        # Answer quality score (based on length and citations)
        answer_score = 0.0
        if answer:
            # Check for citations in answer
            citations_found = len(re.findall(r'\[\d+\]', answer))
            citation_score = min(citations_found / 3, 1.0)
            
            # Answer length score
            length_score = min(len(answer) / 500, 1.0)
            
            answer_score = (citation_score * 0.6 + length_score * 0.4)
        
        # Combined confidence
        confidence = (source_score * 0.3 + content_score * 0.4 + answer_score * 0.3)
        
        return min(confidence, 1.0)

# Global service instance
_perplexity_service = None

def get_perplexity_service() -> PerplexityWebSearchService:
    """Get or create the global Perplexity web search service instance."""
    global _perplexity_service
    if _perplexity_service is None:
        _perplexity_service = PerplexityWebSearchService()
    return _perplexity_service

async def cleanup_perplexity_service():
    """Cleanup the global Perplexity service resources with proper error handling."""
    global _perplexity_service, _openai_client
    
    if _perplexity_service:
        try:
            if not _perplexity_service.is_closed:
                await _perplexity_service.close()
        except Exception as e:
            logger.debug(f"Error cleaning up perplexity service: {e}")
        finally:
            _perplexity_service = None
    
    if _openai_client:
        try:
            await _openai_client.close()
        except Exception as e:
            logger.debug(f"Error cleaning up OpenAI client: {e}")
        finally:
            _openai_client = None

# Synchronous wrapper for tools
def perplexity_web_search(
    query: str,
    max_results: int = 5,  # OPTIMIZATION: Reduced from 8 to 5 for faster searches
    synthesize_answer: bool = True,
    include_recent: bool = False,
    time_limit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for Perplexity-style web search with improved event loop handling.
    
    Args:
        query: Search query
        max_results: Maximum search results (default 5 for speed, was 8)
        synthesize_answer: Whether to generate synthesized answer
        include_recent: Whether to prioritize recent content (default False)
        time_limit: DDGS time limit ('d'=day, 'w'=week, 'm'=month, 'y'=year)
        
    Returns:
        Dictionary with search results, synthesized answer, and citations
    """
    async def _async_search():
        service = get_perplexity_service()
        return await service.perplexity_search(
            query=query,
            max_results=max_results,
            synthesize_answer=synthesize_answer,
            include_recent=include_recent,
            time_limit=time_limit
        )
    
    try:
        # Check if we're in an async context more safely
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - need to avoid nested asyncio.run()
            
            # Use asyncio.to_thread if available (Python 3.9+)
            try:
                import sys
                if sys.version_info >= (3, 9):
                    # Use a separate thread with its own event loop
                    def _run_in_new_loop():
                        return asyncio.run(_async_search())
                    
                    # Use a thread pool to run the search
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(_run_in_new_loop)
                        response = future.result(timeout=60)
                else:
                    # For older Python versions, use a more complex approach
                    raise RuntimeError("Use async version when already in event loop")
            except Exception:
                # If we can't handle the async context properly, return an error
                return {
                    'query': query,
                    'synthesized_query': query,
                    'answer': "Cannot perform search from within async context. Please use the async version of this function.",
                    'sources': [],
                    'citations': {},
                    'confidence_score': 0.0,
                    'verification_notes': [],
                    'verification_details': {},
                    'error': "Event loop conflict",
                    'method': 'perplexity_enhanced',
                    'timestamp': datetime.now().isoformat()
                }
                
        except RuntimeError:
            # No active loop, safe to use asyncio.run
            response = asyncio.run(_async_search())
        
        # Convert response to dictionary format
        return {
            'query': response.query,
            'synthesized_query': response.synthesized_query,
            'answer': response.answer,
            'sources': [
                {
                    'title': source.title,
                    'url': source.url,
                    'snippet': source.snippet,
                    'content': source.content,
                    'relevance_score': source.relevance_score,
                    'bm25_score': source.bm25_score,
                    'semantic_score': source.semantic_score,
                    'combined_score': source.combined_score,
                    'timestamp': source.timestamp,
                    'citation_id': source.citation_id,
                    'word_count': source.word_count
                }
                for source in response.sources
            ],
            'citations': response.citations,
            'confidence_score': response.confidence_score,
            'search_time': response.search_time,
            'synthesis_time': response.synthesis_time,
            'total_time': response.total_time,
            'verification_notes': response.verification_notes,
            'verification_details': response.verification_details,
            'method': 'perplexity_enhanced',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Perplexity web search wrapper failed: {e}")
        return {
            'query': query,
            'synthesized_query': query,
            'answer': f"Search failed: {str(e)}",
            'sources': [],
            'citations': {},
            'confidence_score': 0.0,
            'verification_notes': [],
            'verification_details': {},
            'error': str(e),
            'method': 'perplexity_enhanced',
            'timestamp': datetime.now().isoformat()
        }
