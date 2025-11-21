import pandas as pd
from typing import Dict, Any, List, Optional
from app.services.stock.utils import safe_float, safe_int, to_timestamp_str

def history_to_points(df: Optional[pd.DataFrame], limit: int = 600) -> List[Dict[str, Any]]:
    """Convert a historical DataFrame into a compact list of data points."""
    if df is None or df.empty:
        return []

    try:
        data = df.sort_index()
    except Exception:
        data = df

    length = len(data)
    if limit and length > limit:
        step = max(1, length // limit)
        data = data.iloc[::step]

    points: List[Dict[str, Any]] = []
    for idx, row in data.iterrows():
        close_val = safe_float(row.get("Close"))
        if close_val is None:
            continue
        point = {
            "time": to_timestamp_str(idx),
            "close": close_val,
        }
        open_val = safe_float(row.get("Open"))
        high_val = safe_float(row.get("High"))
        low_val = safe_float(row.get("Low"))
        volume_val = safe_int(row.get("Volume"))
        if open_val is not None:
            point["open"] = open_val
        if high_val is not None:
            point["high"] = high_val
        if low_val is not None:
            point["low"] = low_val
        if volume_val is not None:
            point["volume"] = volume_val
        points.append(point)

    return points

def filter_by_start(df: Optional[pd.DataFrame], start: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    """Return subset of dataframe from start timestamp inclusive."""
    if df is None or df.empty or start is None:
        return df
    try:
        return df.loc[df.index >= start]
    except Exception:
        return df

def build_price_chart(ticker: Any) -> Dict[str, Any]:
    """Build multi-range chart data for price visualization."""
    ranges: List[Dict[str, Any]] = []
    timezone_name: Optional[str] = None

    def _append_range(
        key: str,
        label: str,
        df: Optional[pd.DataFrame],
        period: str,
        interval: str,
        limit: int,
    ) -> None:
        nonlocal timezone_name
        if df is None or df.empty:
            return
        points = history_to_points(df, limit=limit)
        if not points:
            return
        idx = getattr(df, "index", None)
        start = None
        end = None
        if idx is not None and len(idx) > 0:
            start = to_timestamp_str(idx[0])
            end = to_timestamp_str(idx[-1])
            if timezone_name is None:
                tz = getattr(idx, "tz", None)
                if tz is not None:
                    try:
                        timezone_name = getattr(tz, "zone", None) or tz.tzname(idx[-1])
                    except Exception:
                        timezone_name = None
        ranges.append({
            "key": key,
            "label": label,
            "period": period,
            "interval": interval,
            "start": start,
            "end": end,
            "points": points,
        })

    intraday = None
    try:
        intraday = ticker.history(period="5d", interval="5m", auto_adjust=False)
    except Exception:
        intraday = None

    if intraday is not None and not intraday.empty:
        try:
            intraday = intraday.sort_index()
            last_date = intraday.index[-1].date()
            same_day = intraday.loc[intraday.index.date == last_date]
        except Exception:
            same_day = intraday
        _append_range("1d", "1D", same_day, "1d", "5m", limit=400)
        _append_range("5d", "5D", intraday, "5d", "5m", limit=600)

    daily = None
    try:
        daily = ticker.history(period="1y", interval="1d", auto_adjust=False)
    except Exception:
        daily = None

    if daily is not None and not daily.empty:
        daily = daily.sort_index()
        last_idx = daily.index[-1]
        one_month_start = last_idx - pd.DateOffset(months=1)
        six_month_start = last_idx - pd.DateOffset(months=6)
        ytd_start = pd.Timestamp(year=last_idx.year, month=1, day=1, tz=last_idx.tz)

        _append_range("1m", "1M", filter_by_start(daily, one_month_start), "1mo", "1d", limit=120)
        _append_range("6m", "6M", filter_by_start(daily, six_month_start), "6mo", "1d", limit=180)
        _append_range("ytd", "YTD", filter_by_start(daily, ytd_start), "ytd", "1d", limit=220)
        _append_range("1y", "1Y", daily, "1y", "1d", limit=260)

    five_year = None
    try:
        five_year = ticker.history(period="5y", interval="1wk", auto_adjust=False)
    except Exception:
        five_year = None

    if five_year is not None and not five_year.empty:
        five_year = five_year.sort_index()
        _append_range("5y", "5Y", five_year, "5y", "1wk", limit=260)

    max_hist = None
    try:
        max_hist = ticker.history(period="max", interval="1mo", auto_adjust=False)
    except Exception:
        max_hist = None

    if max_hist is not None and not max_hist.empty:
        max_hist = max_hist.sort_index()
        _append_range("max", "MAX", max_hist, "max", "1mo", limit=360)

    if not ranges:
        return {}

    default_key = "1d" if any(r.get("key") == "1d" for r in ranges) else ranges[0].get("key")
    chart: Dict[str, Any] = {
        "ranges": ranges,
        "default_range": default_key,
    }
    if timezone_name:
        chart["timezone"] = timezone_name
    return chart
