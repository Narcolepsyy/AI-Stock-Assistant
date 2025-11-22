import logging
import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from typing import Dict, Any, List, Optional

from app.core.config import (
    TICKER_RE, QUOTE_CACHE_SIZE, QUOTE_TTL_SECONDS,
    NEWS_CACHE_SIZE, NEWS_TTL_SECONDS, NEWS_USER_AGENT
)
from app.utils.connection_pool import connection_pool
from app.utils.circuit_breaker import get_circuit_breaker, CircuitBreakerError
from app.utils.cache_manager import get_cache_manager, CacheType

from app.services.stock.utils import safe_float, safe_int, to_timestamp_str
from app.services.stock.normalization import normalize_symbol, normalize_period, normalize_interval
from app.services.stock.charts import build_price_chart

logger = logging.getLogger(__name__)
cache_manager = get_cache_manager()

def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """Return latest close price and meta for a ticker using Yahoo Finance with TTL cache."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    # Check cache first
    cached = cache_manager.get(CacheType.STOCK_QUOTES, sym)
    if cached is not None:
        logger.debug("cache hit for %s", sym)
        return cached

    ticker = yf.Ticker(sym)
    
    # Circuit breaker for Yahoo Finance API calls
    yf_breaker = get_circuit_breaker(
        "yahoo_finance_api",
        failure_threshold=5,
        recovery_timeout=120.0,
        expected_exception=(Exception,)
    )
    
    try:
        # Check if circuit breaker allows the call
        if yf_breaker.state.value == "open":
            raise CircuitBreakerError(f"Yahoo Finance API circuit breaker is open")
        
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        # Record success manually for sync function
        yf_breaker._record_success()
    except Exception as e:
        # Record failure for circuit breaker
        yf_breaker._record_failure(e)
        raise ValueError(f"failed to retrieve data for '{sym}': {e}")

    if hist is None or hist.empty:
        raise ValueError(f"No price data found for symbol '{sym}'")

    last_row = hist.tail(1)
    close_val = safe_float(last_row["Close"].iloc[0])
    if close_val is None:
        raise ValueError(f"invalid close price for symbol '{sym}'")

    as_of = to_timestamp_str(last_row.index[-1])

    prev_close = None
    try:
        if len(hist) > 1:
            prev_close = safe_float(hist["Close"].iloc[-2])
    except Exception:
        prev_close = None

    day_open = safe_float(last_row["Open"].iloc[0])
    day_high = safe_float(last_row["High"].iloc[0])
    day_low = safe_float(last_row["Low"].iloc[0])
    volume = safe_int(last_row["Volume"].iloc[0])

    currency = None
    try:
        fi = getattr(ticker, "fast_info", None) or {}
        currency = fi.get("currency") if isinstance(fi, dict) else None
    except Exception:
        currency = None

    info: Dict[str, Any] = {}
    try:
        info = ticker.get_info() or {}
    except Exception as info_err:
        logger.debug("get_info failed for %s: %s", sym, info_err)
        info = {}

    market_cap = safe_int(info.get("marketCap"))
    shares_outstanding = safe_int(info.get("sharesOutstanding"))
    year_high = safe_float(info.get("fiftyTwoWeekHigh"))
    year_low = safe_float(info.get("fiftyTwoWeekLow"))
    eps = safe_float(info.get("trailingEps"))
    pe_ratio = safe_float(info.get("trailingPE"))

    info_day_open = safe_float(info.get("regularMarketOpen"))
    if info_day_open is not None:
        day_open = info_day_open
    info_day_low = safe_float(info.get("dayLow"))
    if info_day_low is not None:
        day_low = info_day_low
    info_day_high = safe_float(info.get("dayHigh"))
    if info_day_high is not None:
        day_high = info_day_high
    info_prev_close = safe_float(info.get("regularMarketPreviousClose"))
    if info_prev_close is not None:
        prev_close = info_prev_close
    info_volume = safe_int(info.get("regularMarketVolume"))
    if info_volume is not None:
        volume = info_volume

    if currency is None:
        currency = info.get("currency")

    change_abs = None
    change_pct = None
    if prev_close is not None and prev_close != 0:
        change_abs = close_val - prev_close
        change_pct = (change_abs / prev_close) * 100.0

    chart_data: Dict[str, Any] = {}
    try:
        chart_data = build_price_chart(ticker)
    except Exception as chart_error:
        logger.debug("chart build failed for %s: %s", sym, chart_error)
        chart_data = {}

    result = {
        "symbol": sym,
        "price": round(close_val, 4),
        "currency": (currency or "USD"),
        "change": round(change_abs, 4) if change_abs is not None else None,
        "change_percent": round(change_pct, 4) if change_pct is not None else None,
        "previous_close": round(prev_close, 4) if prev_close is not None else None,
        "day_open": round(day_open, 4) if day_open is not None else None,
        "day_high": round(day_high, 4) if day_high is not None else None,
        "day_low": round(day_low, 4) if day_low is not None else None,
        "volume": volume,
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
        "year_high": round(year_high, 4) if year_high is not None else None,
        "year_low": round(year_low, 4) if year_low is not None else None,
        "eps": round(eps, 4) if eps is not None else None,
        "pe_ratio": round(pe_ratio, 4) if pe_ratio is not None else None,
        "as_of": as_of,
        "source": "yfinance",
    }

    if chart_data:
        result["chart"] = chart_data

    cache_manager.set(CacheType.STOCK_QUOTES, sym, result)
    return result

def get_company_profile(symbol: str) -> Dict[str, Any]:
    """Return company profile details for a ticker using yfinance."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    t = yf.Ticker(sym)
    info: Dict[str, Any] = {}
    try:
        info = t.get_info() or {}
    except Exception as e:
        logger.debug("get_info failed for %s: %s", sym, e)
        info = {}

    fast = {}
    try:
        fast = getattr(t, "fast_info", None) or {}
    except Exception:
        fast = {}

    return {
        "symbol": sym,
        "longName": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "country": info.get("country"),
        "currency": (fast.get("currency") if isinstance(fast, dict) else None) or info.get("currency") or "USD",
        "summary": info.get("longBusinessSummary") or info.get("summary") or None,
        "market_cap": info.get("marketCap"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "float_shares": info.get("floatShares"),
        "enterprise_value": info.get("enterpriseValue"),
        "book_value": info.get("bookValue"),
        "market_to_book": info.get("priceToBook"),
        "source": "yfinance",
    }

def get_historical_prices(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    limit: Optional[int] = None,
    auto_adjust: bool = False,
) -> Dict[str, Any]:
    """Return historical OHLCV for a ticker."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    allowed_periods = {"5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"}
    allowed_intervals = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}

    # Normalize any natural language inputs before validation
    p = normalize_period(period)
    itv = normalize_interval(interval)

    if p not in allowed_periods:
        raise ValueError(f"invalid period: {period}")
    if itv not in allowed_intervals:
        raise ValueError(f"invalid interval: {interval}")

    t = yf.Ticker(sym)
    try:
        hist = t.history(period=p, interval=itv, auto_adjust=auto_adjust)
    except Exception as e:
        raise ValueError(f"failed to retrieve history for '{sym}': {e}")

    if hist is None or hist.empty:
        raise ValueError(f"No historical data found for symbol '{sym}'")

    if limit and isinstance(limit, int) and limit > 0:
        hist = hist.tail(limit)

    rows: List[Dict[str, Any]] = []
    for idx, row in hist.iterrows():
        try:
            rows.append({
                "date": str(idx),
                "open": float(row.get("Open")),
                "high": float(row.get("High")),
                "low": float(row.get("Low")),
                "close": float(row.get("Close")),
                "volume": int(row.get("Volume", 0)) if not (row.get("Volume") != row.get("Volume")) else None,  # handle NaN
            })
        except Exception:
            continue

    currency = None
    try:
        fi = getattr(t, "fast_info", None) or {}
        currency = fi.get("currency") if isinstance(fi, dict) else None
    except Exception:
        currency = None

    return {
        "symbol": sym, 
        "currency": currency or "USD", 
        "count": len(rows), 
        "interval": itv, 
        "period": p, 
        "rows": rows, 
        "source": "yfinance"
    }

def get_stock_news(symbol: str, limit: int = 10) -> Dict[str, Any]:
    """Return recent news articles for a ticker using yfinance with RSS fallback."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    # Serve from cache if available
    try:
        key = f"{sym}:{int(limit) if limit else 10}"
    except Exception:
        key = f"{sym}:10"
    cached = cache_manager.get(CacheType.STOCK_NEWS, key)
    if cached is not None:
        try:
            if isinstance(cached, dict) and int(cached.get("count") or 0) == 0:
                cache_manager.delete(CacheType.STOCK_NEWS, key)
            else:
                return cached
        except Exception:
            return cached

    items: List[Dict[str, Any]] = []
    # Primary: yfinance
    news_raw = None
    try:
        t = yf.Ticker(sym)
        news_raw = getattr(t, "news", None)
        if callable(news_raw):
            news_raw = news_raw()
    except Exception as e:
        logger.debug("yfinance news retrieval failed for %s: %s", sym, e)
        news_raw = None

    if news_raw:
        for n in news_raw[: max(1, int(limit))]:
            try:
                title = n.get("title") or n.get("headline")
                link = n.get("link") or n.get("url")
                publisher = n.get("publisher") or n.get("provider")
                ts = n.get("providerPublishTime") or n.get("published_at") or n.get("publishTime")
                if isinstance(ts, (int, float)):
                    try:
                        published_at = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
                    except Exception:
                        published_at = str(ts)
                else:
                    published_at = str(ts) if ts else None

                thumb = None
                try:
                    if isinstance(n.get("thumbnail"), dict):
                        res = (n.get("thumbnail") or {}).get("resolutions") or []
                        if res:
                            thumb = res[0].get("url")
                except Exception:
                    thumb = None

                item = {
                    "uuid": n.get("uuid") or n.get("id"),
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published_at": published_at,
                    "type": n.get("type") or "yfinance",
                    "related_tickers": n.get("relatedTickers") or n.get("related_tickers") or [sym],
                    "thumbnail": thumb,
                }
                if item["title"] and item["link"]:
                    items.append(item)
            except Exception:
                continue

    # Fallback: Yahoo Finance RSS if yfinance returned nothing
    if not items:
        try:
            # Prefer JP region/lang for Nikkei ^N225, otherwise default to US/en
            region = "JP" if sym == "^N225" else "US"
            lang = "ja-JP" if sym == "^N225" else "en-US"
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(sym)}&region={region}&lang={lang}"
            # Fetch RSS with timeout and UA to avoid hanging
            try:
                session = connection_pool.get_sync_session()
                resp = session.get(rss_url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=5)
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
            except Exception:
                # As a last resort, try direct parse (may be slower)
                feed = feedparser.parse(rss_url)
            for e in (feed.entries or [])[: max(1, int(limit))]:
                try:
                    title = getattr(e, "title", None)
                    link = getattr(e, "link", None)
                    publisher = None
                    try:
                        publisher = (getattr(getattr(e, "source", None), "title", None)) or getattr(e, "author", None)
                    except Exception:
                        publisher = None
                    if not publisher and link:
                        try:
                            netloc = urlparse(link).netloc
                            publisher = netloc.replace("www.", "") if netloc else None
                        except Exception:
                            publisher = None

                    published_at = None
                    try:
                        pp = getattr(e, "published_parsed", None)
                        if pp:
                            published_at = datetime(*pp[:6], tzinfo=timezone.utc).isoformat()
                        else:
                            published_at = getattr(e, "published", None)
                    except Exception:
                        published_at = getattr(e, "published", None)

                    thumb = None
                    try:
                        media = getattr(e, "media_thumbnail", None) or getattr(e, "media_content", None)
                        if isinstance(media, list) and media:
                            thumb = media[0].get("url")
                        elif isinstance(media, dict):
                            thumb = media.get("url")
                    except Exception:
                        thumb = None

                    item = {
                        "uuid": getattr(e, "id", None) or getattr(e, "guid", None),
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                        "published_at": published_at,
                        "type": "rss",
                        "related_tickers": [sym],
                        "thumbnail": thumb,
                    }
                    if item["title"] and item["link"]:
                        items.append(item)
                except Exception:
                    continue
        except Exception as e:
            logger.debug("RSS fallback failed for %s: %s", sym, e)

    # Last-resort: Google News RSS for Nikkei if still empty
    if not items and sym == "^N225":
        try:
            # Japanese Google News for better coverage
            gnews_url = "https://news.google.com/rss/search?q=%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%20OR%20Nikkei%20225&hl=ja&gl=JP&ceid=JP:ja"
            session = connection_pool.get_sync_session()
            resp = session.get(gnews_url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=5)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for e in (feed.entries or [])[: max(1, int(limit))]:
                try:
                    title = getattr(e, "title", None)
                    link = getattr(e, "link", None)
                    published_at = None
                    try:
                        pp = getattr(e, "published_parsed", None)
                        if pp:
                            published_at = datetime(*pp[:6], tzinfo=timezone.utc).isoformat()
                        else:
                            published_at = getattr(e, "published", None)
                    except Exception:
                        published_at = getattr(e, "published", None)

                    item = {
                        "uuid": getattr(e, "id", None) or getattr(e, "guid", None),
                        "title": title,
                        "publisher": None,
                        "link": link,
                        "published_at": published_at,
                        "type": "google-news-rss",
                        "related_tickers": [sym],
                        "thumbnail": None,
                    }
                    if item["title"] and item["link"]:
                        items.append(item)
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Google News fallback failed for %s: %s", sym, e)

    result = {"symbol": sym, "count": len(items), "items": items, "source": "yfinance+rss" if news_raw else "rss"}
    try:
        if items:
            cache_manager.set(CacheType.STOCK_NEWS, key, result)
    except Exception:
        pass
    return result

def _fill_missing_quarterly_earnings(symbol: str, quarterly_earnings_list: list) -> list:
    """Fill missing quarterly earnings by calculating from EPS data when available."""
    import math
    
    try:
        # Get financials to access EPS data
        financials = get_financials(symbol, freq="quarterly")
        income_statement = financials.get('income_statement', {})
        
        if not income_statement:
            return quarterly_earnings_list
        
        # Create a map of existing earnings data
        earnings_map = {}
        for item in quarterly_earnings_list:
            quarter = item.get('Quarter')
            earnings = item.get('Earnings')
            if quarter and earnings is not None and not (isinstance(earnings, float) and math.isnan(earnings)):
                earnings_map[quarter] = earnings
        
        # Check each period in financials and calculate missing earnings
        for date_str, period_data in income_statement.items():
            if not isinstance(period_data, dict):
                continue
                
            # Convert date to quarter format (YYYYQX)
            try:
                from datetime import datetime
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                quarter = f"{date_obj.year}Q{(date_obj.month-1)//3+1}"
            except:
                continue
            
            # Skip if we already have earnings data for this quarter
            if quarter in earnings_map:
                continue
            
            # Try to calculate earnings from EPS × shares
            eps = period_data.get('Diluted EPS') or period_data.get('Basic EPS')
            shares = period_data.get('Diluted Average Shares') or period_data.get('Basic Average Shares')
            
            if eps is not None and shares is not None and not math.isnan(eps) and not math.isnan(shares):
                calculated_earnings = eps * shares
                
                # Add calculated earnings to the list
                new_item = {'Quarter': quarter, 'Earnings': calculated_earnings}
                
                # Insert in correct chronological order (newest first)
                inserted = False
                for i, existing_item in enumerate(quarterly_earnings_list):
                    existing_quarter = existing_item.get('Quarter', '')
                    if existing_quarter < quarter:
                        quarterly_earnings_list.insert(i, new_item)
                        inserted = True
                        break
                
                if not inserted:
                    quarterly_earnings_list.append(new_item)
                    
                logger.info(f"Calculated missing earnings for {symbol} {quarter}: {calculated_earnings:,.0f} from EPS ({eps}) × shares ({shares:,.0f})")
        
        # Remove duplicates, keeping entries with valid earnings data
        seen_quarters = set()
        deduplicated_list = []
        
        for item in quarterly_earnings_list:
            quarter = item.get('Quarter')
            earnings = item.get('Earnings')
            
            if not quarter:
                continue
                
            if quarter not in seen_quarters:
                seen_quarters.add(quarter)
                deduplicated_list.append(item)
            elif earnings is not None and not (isinstance(earnings, float) and math.isnan(earnings)):
                # Replace existing entry if this one has valid earnings
                for i, existing in enumerate(deduplicated_list):
                    if existing.get('Quarter') == quarter:
                        existing_earnings = existing.get('Earnings')
                        if existing_earnings is None or (isinstance(existing_earnings, float) and math.isnan(existing_earnings)):
                            deduplicated_list[i] = item
                        break
        
        return deduplicated_list
        
    except Exception as e:
        logger.warning(f"Failed to fill missing quarterly earnings for {symbol}: {e}")
        return quarterly_earnings_list

def get_financials(symbol: str, freq: str = "quarterly") -> Dict[str, Any]:
    """Get comprehensive financial statements (income statement, balance sheet, cash flow)."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    ticker = yf.Ticker(sym)
    
    try:
        # Get financial data based on frequency
        if freq.lower() in ["quarterly", "q"]:
            income_stmt = ticker.quarterly_financials
            balance_sheet = ticker.quarterly_balance_sheet
            cash_flow = ticker.quarterly_cashflow
        else:  # annual
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
        
        # Convert to dictionary format for JSON serialization
        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            # Convert to dict with date strings as keys
            result = {}
            for col in df.columns:
                date_key = str(col.date()) if hasattr(col, 'date') else str(col)
                result[date_key] = {}
                for idx in df.index:
                    value = df.loc[idx, col]
                    if pd.notna(value):
                        result[date_key][str(idx)] = float(value) if isinstance(value, (int, float)) else str(value)
            return result
        
        currency = None
        try:
            fi = getattr(ticker, "fast_info", None) or {}
            currency = fi.get("currency") if isinstance(fi, dict) else None
        except Exception:
            currency = None

        return {
            "symbol": sym,
            "frequency": freq,
            "currency": currency or "USD",
            "income_statement": df_to_dict(income_stmt),
            "balance_sheet": df_to_dict(balance_sheet), 
            "cash_flow": df_to_dict(cash_flow),
            "source": "yfinance"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "frequency": freq,
            "error": f"Failed to retrieve financials: {str(e)}",
            "source": "yfinance"
        }

def get_earnings_data(symbol: str) -> Dict[str, Any]:
    """Get earnings history, estimates, and calendar data."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    ticker = yf.Ticker(sym)
    
    try:
        # Get earnings data - use income_stmt instead of deprecated earnings
        # Extract Net Income from financial statements
        annual_income_stmt = ticker.financials
        quarterly_income_stmt = ticker.quarterly_financials
        
        # Convert income statements to earnings-like format
        earnings = None
        quarterly_earnings = None
        
        # Extract Net Income from annual financials
        if annual_income_stmt is not None and not annual_income_stmt.empty:
            try:
                # Look for Net Income row
                net_income_rows = [idx for idx in annual_income_stmt.index if 'net income' in str(idx).lower() or 'net earnings' in str(idx).lower()]
                if net_income_rows:
                    net_income_row = net_income_rows[0]
                    earnings_data = annual_income_stmt.loc[net_income_row]
                    # Convert to DataFrame with Year and Earnings columns
                    earnings = pd.DataFrame({
                        'Year': [col.year for col in earnings_data.index],
                        'Earnings': earnings_data.values
                    }).set_index('Year')
            except Exception:
                earnings = None
        
        # Extract Net Income from quarterly financials
        if quarterly_income_stmt is not None and not quarterly_income_stmt.empty:
            try:
                net_income_rows = [idx for idx in quarterly_income_stmt.index if 'net income' in str(idx).lower() or 'net earnings' in str(idx).lower()]
                if net_income_rows:
                    net_income_row = net_income_rows[0]
                    quarterly_data = quarterly_income_stmt.loc[net_income_row]
                    # Convert to DataFrame with Quarter and Earnings columns
                    quarterly_earnings = pd.DataFrame({
                        'Quarter': [f"{col.year}Q{(col.month-1)//3+1}" for col in quarterly_data.index],
                        'Earnings': quarterly_data.values
                    }).set_index('Quarter')
            except Exception:
                quarterly_earnings = None
        
        earnings_dates = ticker.earnings_dates
        
        def df_to_records(df):
            if df is None or df.empty:
                return []
            # Reset index and convert to records, handling date serialization
            df_copy = df.reset_index()
            # Convert datetime columns to strings for JSON serialization
            for col in df_copy.columns:
                if hasattr(df_copy[col].dtype, 'name') and 'datetime' in str(df_copy[col].dtype):
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
                elif df_copy[col].dtype == 'object':
                    # Check if column contains datetime-like objects
                    try:
                        sample = df_copy[col].dropna().iloc[0] if len(df_copy[col].dropna()) > 0 else None
                        if hasattr(sample, 'strftime'):
                            df_copy[col] = df_copy[col].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else x)
                    except:
                        pass
            return df_copy.to_dict('records')
        
        currency = None
        try:
            fi = getattr(ticker, "fast_info", None) or {}
            currency = fi.get("currency") if isinstance(fi, dict) else None
        except Exception:
            currency = None

        # Fill missing quarterly earnings by calculating from EPS data when available
        quarterly_earnings_list = df_to_records(quarterly_earnings)
        quarterly_earnings_list = _fill_missing_quarterly_earnings(sym, quarterly_earnings_list)

        return {
            "symbol": sym,
            "currency": currency or "USD",
            "annual_earnings": df_to_records(earnings),
            "quarterly_earnings": quarterly_earnings_list,
            "earnings_calendar": df_to_records(earnings_dates),
            "source": "yfinance"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "error": f"Failed to retrieve earnings data: {str(e)}",
            "source": "yfinance"
        }

def get_analyst_recommendations(symbol: str) -> Dict[str, Any]:
    """Get analyst recommendations, price targets, and upgrades/downgrades."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    ticker = yf.Ticker(sym)
    
    try:
        recommendations = ticker.recommendations
        upgrades_downgrades = ticker.upgrades_downgrades
        
        def df_to_records(df):
            if df is None or df.empty:
                return []
            return df.reset_index().to_dict('records')
        
        # Get current price for context
        try:
            info = ticker.get_info()
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_mean = info.get('targetMeanPrice')
            target_median = info.get('targetMedianPrice')
            recommendation_mean = info.get('recommendationMean')
        except Exception:
            current_price = target_high = target_low = target_mean = target_median = recommendation_mean = None
        
        currency = None
        try:
            fi = getattr(ticker, "fast_info", None) or {}
            currency = fi.get("currency") if isinstance(fi, dict) else None
        except Exception:
            currency = None

        return {
            "symbol": sym,
            "currency": currency or "USD",
            "current_price": current_price,
            "price_targets": {
                "high": target_high,
                "low": target_low,
                "mean": target_mean,
                "median": target_median
            },
            "recommendation_mean": recommendation_mean,
            "recommendations_history": df_to_records(recommendations),
            "upgrades_downgrades": df_to_records(upgrades_downgrades),
            "source": "yfinance"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "error": f"Failed to retrieve analyst data: {str(e)}",
            "source": "yfinance"
        }

def get_institutional_holders(symbol: str) -> Dict[str, Any]:
    """Get institutional and mutual fund holders data."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    ticker = yf.Ticker(sym)
    
    try:
        institutional_holders = ticker.institutional_holders
        mutualfund_holders = ticker.mutualfund_holders
        major_holders = ticker.major_holders
        
        def df_to_records(df):
            if df is None or df.empty:
                return []
            return df.to_dict('records')
        
        return {
            "symbol": sym,
            "institutional_holders": df_to_records(institutional_holders),
            "mutualfund_holders": df_to_records(mutualfund_holders),
            "major_holders": df_to_records(major_holders),
            "source": "yfinance"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "error": f"Failed to retrieve holders data: {str(e)}",
            "source": "yfinance"
        }

def get_dividends_splits(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Get dividend and stock split history."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    ticker = yf.Ticker(sym)
    
    try:
        # Get dividend and split data
        dividends = ticker.dividends
        splits = ticker.splits
        
        # Filter by period if specified
        if period != "max":
            p = normalize_period(period)
            try:
                end_date = pd.Timestamp.now()
                if p == "1y":
                    start_date = end_date - pd.DateOffset(years=1)
                elif p == "2y":
                    start_date = end_date - pd.DateOffset(years=2)
                elif p == "5y":
                    start_date = end_date - pd.DateOffset(years=5)
                elif p == "6mo":
                    start_date = end_date - pd.DateOffset(months=6)
                elif p == "3mo":
                    start_date = end_date - pd.DateOffset(months=3)
                elif p == "1mo":
                    start_date = end_date - pd.DateOffset(months=1)
                else:
                    start_date = None
                
                if start_date:
                    dividends = dividends[dividends.index >= start_date]
                    splits = splits[splits.index >= start_date]
            except Exception:
                pass  # Use all data if filtering fails
        
        def series_to_records(series):
            if series is None or series.empty:
                return []
            return [{"date": str(idx.date()), "value": float(val)} for idx, val in series.items()]
        
        currency = None
        try:
            fi = getattr(ticker, "fast_info", None) or {}
            currency = fi.get("currency") if isinstance(fi, dict) else None
        except Exception:
            currency = None

        return {
            "symbol": sym,
            "period": period,
            "currency": currency or "USD", 
            "dividends": series_to_records(dividends),
            "splits": series_to_records(splits),
            "dividend_count": len(dividends),
            "split_count": len(splits),
            "source": "yfinance"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "period": period,
            "error": f"Failed to retrieve dividend/split data: {str(e)}",
            "source": "yfinance"
        }

def get_market_indices(region: str = "global") -> Dict[str, Any]:
    """Get major market indices data (S&P500, Nasdaq, Nikkei, etc.)."""
    
    # Define major indices by region
    indices_map = {
        "us": {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ Composite", 
            "^DJI": "Dow Jones Industrial Average",
            "^RUT": "Russell 2000"
        },
        "japan": {
            "^N225": "Nikkei 225",
            "^TPX": "TOPIX"
        },
        "europe": {
            "^FTSE": "FTSE 100",
            "^GDAXI": "DAX",
            "^FCHI": "CAC 40"
        },
        "asia": {
            "^HSI": "Hang Seng Index",
            "000001.SS": "Shanghai Composite",
            "^STI": "Straits Times Index",
            "^KOSPI": "KOSPI"
        },
        "global": {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^DJI": "Dow Jones",
            "^N225": "Nikkei 225",
            "^FTSE": "FTSE 100",
            "^GDAXI": "DAX"
        }
    }
    
    region_lower = region.lower()
    if region_lower not in indices_map:
        region_lower = "global"
    
    indices = indices_map[region_lower]
    results = []
    
    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            hist = ticker.history(period="2d", interval="1d")
            if hist is None or hist.empty:
                continue
                
            latest = hist.tail(1)
            current_price = float(latest["Close"].iloc[0])
            
            # Calculate change from previous day
            change = None
            change_pct = None
            if len(hist) >= 2:
                previous = hist.iloc[-2]
                prev_close = float(previous["Close"])
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
            
            # Get basic info
            currency = "USD"  # Most indices are in USD or local currency
            try:
                fi = getattr(ticker, "fast_info", None) or {}
                currency = fi.get("currency") if isinstance(fi, dict) else None
            except Exception:
                pass
            
            results.append({
                "symbol": symbol,
                "name": name,
                "price": round(current_price, 2),
                "change": round(change, 2) if change else None,
                "change_pct": round(change_pct, 2) if change_pct else None,
                "currency": currency or "USD",
                "as_of": str(latest.index[-1])
            })
            
        except Exception as e:
            logger.debug(f"Failed to get data for {symbol}: {e}")
            continue
    
    return {
        "region": region,
        "count": len(results),
        "indices": results,
        "source": "yfinance"
    }

def get_market_summary() -> Dict[str, Any]:
    """Get comprehensive market summary including global indices performance and market sentiment.
    
    This is a convenience wrapper around get_market_indices that provides a global market overview.
    """
    return get_market_indices(region="global")

def get_market_cap_details(symbol: str) -> Dict[str, Any]:
    """Get comprehensive market capitalization and valuation metrics for a stock."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    ticker = yf.Ticker(sym)
    
    try:
        info = ticker.get_info()
        
        # Get current price for calculations
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Market cap and share data
        market_cap = info.get('marketCap')
        shares_outstanding = info.get('sharesOutstanding')
        float_shares = info.get('floatShares')
        shares_short = info.get('sharesShort')
        short_ratio = info.get('shortRatio')
        
        # Enterprise value and debt
        enterprise_value = info.get('enterpriseValue')
        total_debt = info.get('totalDebt')
        total_cash = info.get('totalCash')
        
        # Valuation ratios
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        price_to_book = info.get('priceToBook')
        price_to_sales = info.get('priceToSalesTrailing12Months')
        enterprise_to_revenue = info.get('enterpriseToRevenue')
        enterprise_to_ebitda = info.get('enterpriseToEbitda')
        
        # Calculate additional metrics
        calculated_market_cap = None
        if current_price and shares_outstanding:
            calculated_market_cap = current_price * shares_outstanding
            
        # Share statistics
        percent_held_by_insiders = info.get('heldByInsiders')
        percent_held_by_institutions = info.get('heldByInstitutions')
        
        # Format large numbers for readability
        def format_large_number(value):
            if value is None:
                return None
            if value >= 1e12:
                return f"{value/1e12:.2f}T"
            elif value >= 1e9:
                return f"{value/1e9:.2f}B"
            elif value >= 1e6:
                return f"{value/1e6:.2f}M"
            else:
                return value
        
        currency = None
        try:
            fi = getattr(ticker, "fast_info", None) or {}
            currency = fi.get("currency") if isinstance(fi, dict) else None
        except Exception:
            currency = None

        return {
            "symbol": sym,
            "currency": currency or "USD",
            "current_price": current_price,
            "market_cap": market_cap,
            "market_cap_formatted": format_large_number(market_cap),
            "calculated_market_cap": calculated_market_cap,
            "enterprise_value": enterprise_value,
            "enterprise_value_formatted": format_large_number(enterprise_value),
            "shares_data": {
                "shares_outstanding": shares_outstanding,
                "float_shares": float_shares,
                "shares_short": shares_short,
                "short_ratio": short_ratio,
                "percent_held_by_insiders": percent_held_by_insiders,
                "percent_held_by_institutions": percent_held_by_institutions
            },
            "debt_and_cash": {
                "total_debt": total_debt,
                "total_cash": total_cash,
                "net_debt": (total_debt - total_cash) if (total_debt and total_cash) else None
            },
            "valuation_ratios": {
                "pe_ratio": pe_ratio,
                "forward_pe": forward_pe,
                "peg_ratio": peg_ratio,
                "price_to_book": price_to_book,
                "price_to_sales": price_to_sales,
                "enterprise_to_revenue": enterprise_to_revenue,
                "enterprise_to_ebitda": enterprise_to_ebitda
            },
            "source": "yfinance"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "error": f"Failed to retrieve market cap details: {str(e)}",
            "source": "yfinance"
        }
