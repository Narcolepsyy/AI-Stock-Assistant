import re
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import (
    NEWS_USER_AGENT, ARTICLE_CACHE_SIZE, ARTICLE_TTL_SECONDS,
    NEWS_FETCH_MAX_WORKERS, RAG_STRATEGY, RAG_MAX_PER_ITEM, RAG_MAX_WORKERS,
    TICKER_RE
)
from app.utils.connection_pool import connection_pool
from app.utils.cache_manager import get_cache_manager, CacheType
from app.services.stock.utils import ThreadSafeCache
from app.services.stock.normalization import normalize_symbol
from app.services.stock.providers.yfinance import get_stock_news
from app.services.rag_service import rag_search
from app.services.openai_client import get_client_for_model

logger = logging.getLogger(__name__)
cache_manager = get_cache_manager()

# Pre-compiled regex patterns
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NEWLINE_PATTERN = re.compile(r"\n")

# Thread-safe cache
ARTICLE_CACHE = ThreadSafeCache(maxsize=ARTICLE_CACHE_SIZE, ttl=ARTICLE_TTL_SECONDS)

def extract_article(url: str, timeout: int = 8, max_chars: int = 6000) -> Dict[str, Any]:
    """Fetch and extract the main article text from a URL with caching.

    Caches successful extractions keyed by (url, max_chars) to avoid repeat network + parsing.
    """
    if not (url or '').strip():
        return {"content": None}

    key = ((url or '').strip(), int(max_chars) if isinstance(max_chars, int) else None)
    
    # Check cache first using thread-safe method
    if key in ARTICLE_CACHE:
        cached = ARTICLE_CACHE.get(key)
        if cached:
            # Return a shallow copy to avoid accidental mutation by callers
            return dict(cached) if isinstance(cached, dict) else cached

    try:
        session = connection_pool.get_sync_session()
        resp = session.get(url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=max(2, int(timeout)))
        resp.raise_for_status()
        html = resp.text or ""
    except Exception as e:
        return {"content": None, "error": f"fetch_failed: {e}"}

    # Try readability-lxml first
    try:
        from readability import Document
        doc = Document(html)
        title = doc.short_title() or None
        content_html = doc.summary(html_partial=True)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content_html or html, "lxml")
            # Normalize whitespace without using get_text args flagged by linter
            raw = soup.get_text()
            text = _WHITESPACE_PATTERN.sub(" ", (raw or "")).strip()
        except Exception:
            text = _WHITESPACE_PATTERN.sub(" ", _NEWLINE_PATTERN.sub(" ", content_html or ""))
        text = (text or "").strip()
        if max_chars and isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "..."
        if text:
            result = {"title": title, "content": text}
            # Cache the result using thread-safe method
            ARTICLE_CACHE.set(key, dict(result))
            return result
    except Exception:
        pass

    # Fallback: BeautifulSoup plain text
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        raw = soup.get_text()
        text = _WHITESPACE_PATTERN.sub(" ", raw or "").strip()
        if max_chars and isinstance(max_chars, int) and 0 < max_chars < len(text):
            text = text[:max_chars] + "..."
        result = {"content": text}
        # Cache the result using thread-safe method
        ARTICLE_CACHE.set(key, dict(result))
        return result
    except Exception as e:
        return {"content": None, "error": f"parse_failed: {e}"}

def get_augmented_news(
    symbol: str,
    limit: int = 10,
    include_full_text: bool = True,
    include_rag: bool = True,
    rag_k: int = 3,
    max_chars: int = 6000,
    timeout: int = 8,
) -> Dict[str, Any]:
    """Get news with full article text extraction, parallelized and cached for speed."""
    
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    key = f"aug:{sym}:{int(limit) if limit else 10}:{int(include_full_text)}:{int(include_rag)}:{int(rag_k)}:{int(max_chars)}"
    cached = cache_manager.get(CacheType.STOCK_NEWS, key)
    if cached is not None:
        return cached

    base = get_stock_news(sym, limit=limit)
    items = base.get("items", [])

    # Prepare baseline entries
    enriched: List[Dict[str, Any]] = [dict(it) for it in items]

    # Stage 1: Parallel article extraction
    if include_full_text:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max(1, int(NEWS_FETCH_MAX_WORKERS))) as executor:
            for idx, entry in enumerate(enriched):
                link = (entry.get("link") or "").strip()
                if not link:
                    continue
                futures_map[executor.submit(extract_article, link, timeout=timeout, max_chars=max_chars)] = idx
            for fut in as_completed(futures_map):
                idx = futures_map[fut]
                try:
                    extra = fut.result()
                    if isinstance(extra, dict):
                        if extra.get("title") and not enriched[idx].get("title"):
                            enriched[idx]["title"] = extra.get("title")
                        enriched[idx]["content"] = extra.get("content")
                        if extra.get("error"):
                            enriched[idx]["content_error"] = extra.get("error")
                except Exception as e:
                    enriched[idx]["content_error"] = f"extract_exception: {e}"

    # Stage 2: RAG retrievals (strategy-based)
    if include_rag:
        strategy = (RAG_STRATEGY or "symbol").strip().lower()
        # Auto: use per-item when few items, otherwise one symbol query
        if strategy == "auto":
            strategy = "item" if len(enriched) <= max(1, int(RAG_MAX_PER_ITEM)) else "symbol"

        if strategy == "symbol":
            # One query for all items
            q = f"{sym} latest company news and updates"
            try:
                rs = rag_search(q, int(rag_k))
            except Exception as e:
                rs = {"enabled": False, "error": str(e), "results": []}
            cleaned = []
            for r in (rs.get("results") or [])[: int(rag_k)]:
                try:
                    cleaned.append({
                        "text": (r.get("text") or "")[:1000],
                        "metadata": r.get("metadata"),
                        "score": r.get("score"),
                    })
                except Exception:
                    continue
            for idx in range(len(enriched)):
                enriched[idx]["rag"] = {
                    "enabled": rs.get("enabled", False),
                    "count": len(cleaned),
                    "results": cleaned,
                }
        else:
            # Per-item, optionally cap to RAG_MAX_PER_ITEM
            rag_futures = {}
            with ThreadPoolExecutor(max_workers=max(1, int(RAG_MAX_WORKERS))) as executor:
                # Determine which indices to enrich (cap if configured)
                indices = list(range(len(enriched)))
                max_items = int(RAG_MAX_PER_ITEM) if RAG_MAX_PER_ITEM and RAG_MAX_PER_ITEM > 0 else len(indices)
                for idx in indices[:max_items]:
                    entry = enriched[idx]
                    title = (entry.get("title") or "").strip()
                    if not title:
                        enriched[idx]["rag"] = {"enabled": False, "count": 0, "results": []}
                        continue
                    q = f"{sym} news: {title}"
                    rag_futures[executor.submit(rag_search, q, int(rag_k))] = idx
                for fut in as_completed(rag_futures):
                    idx = rag_futures[fut]
                    try:
                        rs = fut.result()
                    except Exception as e:
                        rs = {"enabled": False, "error": str(e), "results": []}
                    cleaned = []
                    for r in (rs.get("results") or [])[: int(rag_k)]:
                        try:
                            cleaned.append({
                                "text": (r.get("text") or "")[:1000],
                                "metadata": r.get("metadata"),
                                "score": r.get("score"),
                            })
                        except Exception:
                            continue
                    enriched[idx]["rag"] = {
                        "enabled": rs.get("enabled", False),
                        "count": len(cleaned),
                        "results": cleaned,
                    }
                # For any remaining items not enriched (if capped), attach empty rag
                for idx in indices[max_items:]:
                    enriched[idx]["rag"] = {"enabled": False, "count": 0, "results": []}

    result = {
        "symbol": sym,
        "count": len(enriched),
        "items": enriched,
        "source": base.get("source", "yfinance/rss"),
        "augmented": True,
    }
    try:
        cache_manager.set(CacheType.STOCK_NEWS, key, result)
    except Exception:
        pass
    return result

def _analyze_sentiment_simple(text: str) -> str:
    """Simple sentiment analysis for Japanese/English financial news using keyword matching."""
    if not text:
        return "neutral"
    
    text_lower = text.lower()
    
    # Japanese positive keywords
    jp_positive = ["上昇", "急騰", "好調", "増加", "プラス", "回復", "改善", "成長", "拡大", "好材料", "買い"]
    # Japanese negative keywords  
    jp_negative = ["下落", "急落", "悪化", "減少", "マイナス", "低下", "縮小", "悪材料", "売り", "損失", "赤字"]
    
    # English positive keywords
    en_positive = ["rise", "surge", "gain", "up", "increase", "growth", "positive", "strong", "bullish", "buy", "improve"]
    # English negative keywords
    en_negative = ["fall", "drop", "decline", "down", "decrease", "loss", "negative", "weak", "bearish", "sell", "worsen"]
    
    positive_count = 0
    negative_count = 0
    
    # Count Japanese keywords
    for word in jp_positive:
        if word in text:
            positive_count += 1
    for word in jp_negative:
        if word in text:
            negative_count += 1
            
    # Count English keywords
    for word in en_positive:
        if word in text_lower:
            positive_count += 1
    for word in en_negative:
        if word in text_lower:
            negative_count += 1
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def get_nikkei_news_with_sentiment(limit: int = 5) -> Dict[str, Any]:
    """Get recent Nikkei 225 news headlines with 1-sentence summaries and sentiment analysis."""
    
    try:
        # Fetch Nikkei news using existing infrastructure
        news_data = get_augmented_news("^N225", limit=limit, include_full_text=True, include_rag=False)
        items = news_data.get("items", [])
        
        if not items:
            return {
                "symbol": "^N225",
                "count": 0,
                "summaries": [],
                "error": "日経平均（N225）の最新ニュースヘッドラインが取得できませんでした"
            }
        
        # Process each news item
        processed_items = []
        
        for item in items[:limit]:
            title = item.get("title", "")
            content = item.get("content", "")
            publisher = item.get("publisher", "")
            published_at = item.get("published_at", "")
            
            # Create one-sentence summary
            summary_text = title
            if content and len(content) > len(title):
                # Use first sentence of content or truncate
                sentences = content.split("。")  # Japanese sentence delimiter
                if not sentences[0]:
                    sentences = content.split(".")  # English fallback
                if sentences and len(sentences[0].strip()) > 10:
                    summary_text = sentences[0].strip()
                    if not summary_text.endswith("。") and not summary_text.endswith("."):
                        summary_text += "。"
                else:
                    summary_text = content[:100] + "..." if len(content) > 100 else content
            
            # Analyze sentiment using both title and content
            full_text = f"{title} {content}"
            sentiment = _analyze_sentiment_simple(full_text)
            
            # Format sentiment in Japanese
            sentiment_jp = {
                "positive": "ポジティブ", 
                "negative": "ネガティブ", 
                "neutral": "ニュートラル"
            }.get(sentiment, "ニュートラル")
            
            processed_items.append({
                "title": title,
                "summary": summary_text,
                "sentiment": sentiment,
                "sentiment_jp": sentiment_jp,
                "publisher": publisher,
                "published_at": published_at[:10] if published_at else "",
                "link": item.get("link", "")
            })
        
        return {
            "symbol": "^N225",
            "count": len(processed_items),
            "summaries": processed_items,
            "source": "yfinance+sentiment"
        }
        
    except Exception as e:
        logger.error(f"Error fetching Nikkei news with sentiment: {e}")
        return {
            "symbol": "^N225", 
            "count": 0,
            "summaries": [],
            "error": f"ニュース取得エラー: {str(e)}"
        }
