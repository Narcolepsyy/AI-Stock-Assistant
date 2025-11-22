"""
Stock Service Facade
This module now serves as a facade, importing functionality from the refactored
`app.services.stock` package. This ensures backward compatibility while
delegating actual logic to specialized modules.
"""

import logging
from typing import Dict, Any, List, Optional

# Import from new modules
from app.services.stock.normalization import (
    normalize_symbol as _normalize_symbol,
    normalize_period as _normalize_period,
    normalize_interval as _normalize_interval
)
from app.services.stock.utils import (
    ThreadSafeCache,
    safe_float,
    safe_int,
    to_timestamp_str
)
from app.services.stock.charts import build_price_chart
from app.services.stock.providers.yfinance import (
    get_stock_quote,
    get_company_profile,
    get_historical_prices,
    get_stock_news,
    get_financials,
    get_earnings_data,
    get_analyst_recommendations,
    get_institutional_holders,
    get_dividends_splits,
    get_market_indices,
    get_market_summary,
    get_market_cap_details
)
from app.services.stock.news import (
    get_augmented_news,
    get_nikkei_news_with_sentiment,
    extract_article
)
from app.services.stock.analysis import (
    get_risk_assessment,
    get_technical_indicators,
    check_golden_cross,
    calculate_correlation
)

logger = logging.getLogger(__name__)

# Re-export helper functions if they were public (though they started with _)
# We keep them available just in case, or for internal consistency if needed.
# But generally, external consumers should use the public API functions.

__all__ = [
    "get_stock_quote",
    "get_company_profile",
    "get_historical_prices",
    "get_stock_news",
    "get_augmented_news",
    "get_nikkei_news_with_sentiment",
    "extract_article",
    "get_financials",
    "get_earnings_data",
    "get_analyst_recommendations",
    "get_institutional_holders",
    "get_dividends_splits",
    "get_market_indices",
    "get_market_summary",
    "get_market_cap_details",
    "get_risk_assessment",
    "get_technical_indicators",
    "check_golden_cross",
    "calculate_correlation",
    "build_price_chart",
    "ThreadSafeCache"
]
