from typing import Optional

# Natural language mappings for timeframes
_NL_PERIOD_MAP = {
    "past week": "5d", "last week": "5d", "previous week": "5d",
    "1 week": "5d", "one week": "5d", "7d": "5d", "five days": "5d",
    "5 days": "5d", "last 5 days": "5d", "past month": "1mo",
    "last month": "1mo", "1 month": "1mo", "one month": "1mo",
    "3 months": "3mo", "three months": "3mo", "6 months": "6mo",
    "six months": "6mo", "1 year": "1y", "one year": "1y",
    "2 years": "2y", "two years": "2y", "5 years": "5y",
    "five years": "5y", "10 years": "10y", "ten years": "10y",
    "ytd": "ytd", "year to date": "ytd", "all time": "max", "max": "max",
    # Fix common confusion: map "1d" period to "5d" (minimum valid period)
    "1d": "5d", "1 day": "5d", "one day": "5d", "today": "5d", "daily": "5d"
}

_NL_INTERVAL_MAP = {
    "daily": "1d", "day": "1d", "1 day": "1d", "one day": "1d",
    "weekly": "1wk", "week": "1wk", "monthly": "1mo", "month": "1mo",
    "hourly": "1h", "hour": "1h",
}

# Known symbol aliases (index and localized names)
_ALIAS_MAP = {
    # Nikkei 225
    "^N225": "^N225",
    "N225": "^N225",
    "NI225": "^N225",
    "NIKKEI": "^N225",
    "NIKKEI225": "^N225",
    "NIKKEI 225": "^N225",
    "日経": "^N225",
    "日経平均": "^N225",
    "日経平均株価": "^N225",
    
    # TOPIX
    "^TPX": "^TPX",
    "TPX": "^TPX",
    "TOPIX": "^TPX",
    "東証": "^TPX",
    
    # Japanese Banking Sector
    "地域銀行セクター指数": "1615.T",
    "銀行セクター指数": "1615.T", 
    "地域銀行指数": "8359.T",
    "銀行セクター": "8355.T",
    "日本銀行セクター": "8306.T",
    "JAPANESE REGIONAL BANKS": "8359.T",
    "JAPANESE BANKS SECTOR": "8306.T",
    
    # Alternative regional bank representatives
    "地方銀行": "8359.T",
    "REGIONAL BANKS JP": "8359.T",
}

def normalize_symbol(raw: Optional[str]) -> str:
    """Normalize various user inputs to a valid Yahoo Finance symbol."""
    s = (raw or "").strip()
    if not s:
        return s
    # Keep original for unicode matching; also build uppercase ascii variant
    s_upper = s.upper()
    # Direct alias hit first (handles Japanese keys)
    if s in _ALIAS_MAP:
        return _ALIAS_MAP[s]
    if s_upper in _ALIAS_MAP:
        return _ALIAS_MAP[s_upper]
    # If starts with caret already, just uppercase rest
    if s.startswith("^"):
        return "^" + s[1:].upper()
    # No alias: return uppercased string
    # Special handling for Japanese stock codes (4 digits)
    if s.isdigit() and len(s) == 4:
        return f"{s}.T"
    return s_upper

def normalize_period(val: Optional[str]) -> str:
    """Normalize natural language period to yfinance format."""
    if not val:
        return "1mo"
    s = str(val).strip().lower()
    return _NL_PERIOD_MAP.get(s, s)

def normalize_interval(val: Optional[str]) -> str:
    """Normalize natural language interval to yfinance format."""
    if not val:
        return "1d"
    s = str(val).strip().lower()
    return _NL_INTERVAL_MAP.get(s, s)
