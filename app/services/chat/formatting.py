import json
from typing import Any, Dict, List

# Human-readable previews for common tool results (module-level for reuse)
def human_preview_company_profile(result: Dict[str, Any]) -> str:
    try:
        sym = (result.get("symbol") or "").upper()
        name = result.get("longName") or ""
        sector = result.get("sector") or ""
        industry = result.get("industry") or ""
        country = result.get("country") or ""
        currency = result.get("currency") or ""
        website = result.get("website") or ""
        parts = []
        title = f"Company profile for {sym}: {name}".strip()
        parts.append(title)
        si = ", ".join([p for p in [sector, industry] if p])
        if si:
            parts.append(si)
        if country:
            parts.append(country)
        if currency:
            parts.append(f"Currency: {currency}")
        if website:
            parts.append(website)
        return "\n".join(parts)
    except Exception:
        try:
            s = json.dumps(result)
            return (s[:280] + ("..." if len(s) > 280 else ""))
        except Exception:
            return ""

def human_preview_quote(result: Dict[str, Any]) -> str:
    try:
        sym = (result.get("symbol") or "").upper()
        price = result.get("price")
        currency = result.get("currency") or ""
        as_of = result.get("as_of") or ""
        if price is not None:
            return f"{sym} latest close: {price} {currency} (as of {as_of})"
        return ""
    except Exception:
        try:
            s = json.dumps(result)
            return (s[:280] + ("..." if len(s) > 280 else ""))
        except Exception:
            return ""

# Japanese versions of preview functions
def human_preview_company_profile_jp(result: Dict[str, Any]) -> str:
    try:
        sym = (result.get("symbol") or "").upper()
        name = result.get("longName") or ""
        sector = result.get("sector") or ""
        industry = result.get("industry") or ""
        country = result.get("country") or ""
        currency = result.get("currency") or ""
        
        parts = []
        if sym and name:
            parts.append(f"**{sym} - {name}**")
        if sector or industry:
            sector_info = f"{sector}" + (f" / {industry}" if industry else "")
            parts.append(f"業種: {sector_info}")
        if country:
            parts.append(f"国: {country}")
        if currency:
            parts.append(f"通貨: {currency}")
        
        formatted = "\n".join(parts)
        
        # Add data source
        formatted += "\n\n**データ出典**: Yahoo Finance 企業情報"
        
        return formatted
    except Exception:
        return human_preview_company_profile(result)

def human_preview_quote_jp(result: Dict[str, Any]) -> str:
    try:
        sym = (result.get("symbol") or "").upper()
        price = result.get("price")
        currency = result.get("currency") or ""
        as_of = result.get("as_of") or ""
        
        if price is not None:
            if currency == "JPY":
                formatted = f"**{sym}**: {price:,.0f}円"
            else:
                formatted = f"**{sym}**: {price:,.2f} {currency}"
        else:
            formatted = f"**{sym}**: データなし"
        
        if as_of:
            formatted += f"\n取得時刻: {as_of}"
        
        return formatted
    except Exception:
        return human_preview_quote(result)

def human_preview_historical_jp(result: Dict[str, Any]) -> str:
    try:
        sym = (result.get("symbol") or "").upper()
        currency = result.get("currency") or ""
        count = result.get("count", 0)
        period = result.get("period", "")
        rows = result.get("rows", [])
        
        if not rows:
            return f"{sym} 過去データ：データなし"
        
        # Get latest row for trend info
        latest = rows[-1] if rows else {}
        if len(rows) > 1:
            prev = rows[-2]
            current_price = latest.get("close")
            prev_price = prev.get("close")
            
            if current_price and prev_price:
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                trend = "上昇" if change > 0 else "下落" if change < 0 else "横ばい"
                currency_jp = {"JPY": "円", "USD": "ドル", "EUR": "ユーロ"}.get(currency, currency)
                
                return f"**{sym} 過去{period}データ：**\n最新終値 {current_price:,} {currency_jp} (前日比 {change:+.1f} {currency_jp}, {change_pct:+.2f}%, {trend})"
        
        return f"**{sym} 過去{period}データ** ({count}件のデータポイント)"
    except Exception:
        try:
            return f"{result.get('symbol', '')} historical data: {result.get('count', 0)} points"
        except Exception:
            return ""

def build_news_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a concise summary for augmented news results suitable for model consumption."""
    sym = (result.get("symbol") or "").upper()
    items = result.get("items") or []
    headlines: List[Dict[str, Any]] = []
    for it in items[:5]:
        try:
            t = (it.get("title") or "").strip()
            pub = (it.get("publisher") or "").strip() or None
            dt = (it.get("published_at") or "").strip() or None
            link = (it.get("link") or "").strip() or None
            content = (it.get("content") or "").strip()
            if len(t) > 160:
                t = t[:160] + "..."
            if content and len(content) > 280:
                content = content[:280] + "..."
            headlines.append({
                "title": t,
                "publisher": pub,
                "published_at": dt,
                "summary": content or None,
                "link": link,
            })
        except Exception:
            continue
    return {
        "symbol": sym,
        "count": len(items),
        "headlines": headlines,
        "note": "summarized for brevity from get_augmented_news"
    }

def human_preview_from_summary(summary: Dict[str, Any]) -> str:
    """Create human preview from news summary."""
    sym = summary.get("symbol") or ""
    lines = [f"Top headlines for {sym}:"]
    for h in summary.get("headlines", [])[:3]:
        bits = [h.get("title") or "Untitled"]
        if h.get("publisher"):
            bits.append(f"({h['publisher']}")
            if h.get("published_at"):
                bits[-1] = bits[-1][:-1] + f", {h['published_at'][:10]})"
        elif h.get("published_at"):
            bits.append(f"({h['published_at'][:10]})")
        lines.append(" - " + " ".join(bits))
    return "\n".join(lines)

def human_preview_historical(result: Dict[str, Any]) -> str:
    try:
        sym = (result.get("symbol") or "").upper()
        currency = result.get("currency") or ""
        count = result.get("count", 0)
        period = result.get("period", "")
        rows = result.get("rows", [])
        
        if not rows:
            return f"{sym} historical data: No data"
        
        # Get latest row for trend info
        latest = rows[-1] if rows else {}
        if len(rows) > 1:
            prev = rows[-2]
            current_price = latest.get("close")
            prev_price = prev.get("close")
            
            if current_price and prev_price:
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                trend = "up" if change > 0 else "down" if change < 0 else "flat"
                
                return f"{sym} {period} data: Latest {current_price} {currency} ({change:+.1f} {currency}, {change_pct:+.2f}%, {trend})"
        
        return f"{sym} {period} data ({count} points)"
    except Exception:
        try:
            return f"{result.get('symbol', '')} historical data: {result.get('count', 0)} points"
        except Exception:
            return ""

def human_preview_nikkei_news_jp(result: Dict[str, Any]) -> str:
    try:
        count = result.get("count", 0)
        summaries = result.get("summaries", [])
        error = result.get("error")
        
        if error:
            return f"日経平均ニュース：{error}"
        
        if not summaries:
            return "日経平均ニュース：該当するニュースがありませんでした"
        
        lines = ["**日経平均の直近ニュース：**"]
        for i, item in enumerate(summaries[:5], 1):
            sentiment = item.get("sentiment_jp", "ニュートラル")
            summary = item.get("summary", "")
            publisher = item.get("publisher", "")
            
            # Format: number. summary (sentiment) - publisher
            line = f"{i}. {summary}"
            if sentiment:
                line += f" ({sentiment})"
            if publisher:
                line += f" - {publisher}"
            lines.append(line)
        
        return "\n".join(lines)
    except Exception:
        try:
            return f"日経平均ニュース：{result.get('count', 0)}件"
        except Exception:
            return "日経平均ニュース取得完了"
