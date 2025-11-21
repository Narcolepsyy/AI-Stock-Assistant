import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from app.core.config import TICKER_RE, RISK_FREE_RATE
from app.services.stock.normalization import normalize_symbol, normalize_period
from app.services.stock.providers.yfinance import get_historical_prices

logger = logging.getLogger(__name__)

def get_risk_assessment(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    rf_rate: Optional[float] = None,
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute risk metrics (volatility, Sharpe, max drawdown, VaR, beta) for a stock."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")

    hist = get_historical_prices(sym, period=period, interval=interval, auto_adjust=False)
    rows = hist.get("rows", [])
    if not rows or len(rows) < 3:
        return {
            "symbol": sym,
            "period": period,
            "interval": interval,
            "count": len(rows or []),
            "error": "insufficient data",
        }

    df = pd.DataFrame(rows)
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.sort_values("date")
    except Exception:
        pass
    closes = pd.to_numeric(df["close"], errors="coerce").dropna()
    rets = closes.pct_change().dropna()

    if rets.empty:
        return {
            "symbol": sym,
            "period": period,
            "interval": interval,
            "count": len(rows or []),
            "error": "no returns computed",
        }

    # Daily metrics
    mean_daily = float(rets.mean())
    std_daily = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    TRADING_DAYS = 252.0
    vol_ann = float(std_daily * (TRADING_DAYS ** 0.5)) if std_daily == std_daily else None

    # Sharpe
    rf_annual = float(rf_rate) if rf_rate is not None else RISK_FREE_RATE
    rf_daily = rf_annual / TRADING_DAYS
    try:
        sharpe = ((mean_daily - rf_daily) / std_daily) * (TRADING_DAYS ** 0.5)
        sharpe_ratio = float(sharpe)
    except Exception:
        sharpe_ratio = None

    # Max drawdown
    cum = (1.0 + rets).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum / roll_max) - 1.0
    try:
        max_dd = float(drawdown.min())
    except Exception:
        max_dd = None

    # Historical VaR at 95%
    try:
        var_95_daily = float(np.percentile(rets.values, 5))
    except Exception:
        var_95_daily = None

    # Beta vs benchmark
    beta = None
    bench_sym = None
    if (benchmark or "").strip():
        bench_sym = (benchmark or "").strip().upper()
        try:
            bhist = get_historical_prices(bench_sym, period=period, interval=interval, auto_adjust=False)
            brows = bhist.get("rows", [])
            if brows and len(brows) >= 3:
                bdf = pd.DataFrame(brows)
                try:
                    bdf["date"] = pd.to_datetime(bdf["date"], errors="coerce", utc=True)
                    bdf = bdf.sort_values("date")
                except Exception:
                    pass
                bclose = pd.to_numeric(bdf["close"], errors="coerce").dropna()
                brets = bclose.pct_change().dropna()
                joined = pd.concat([rets.reset_index(drop=True), brets.reset_index(drop=True)], axis=1).dropna()
                joined.columns = ["asset", "bench"]
                if len(joined) > 1 and float(joined["bench"].var(ddof=1)) != 0.0:
                    cov = float(joined.cov().iloc[0, 1])
                    var_b = float(joined["bench"].var(ddof=1))
                    beta = cov / var_b
        except Exception:
            beta = None

    return {
        "symbol": sym,
        "period": period,
        "interval": interval,
        "count": int(len(rets)),
        "volatility_annualized": vol_ann,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
        "var_95_daily": var_95_daily,
        "beta": beta,
        "benchmark": bench_sym,
        "risk_free_rate": rf_annual,
        "source": "computed",
    }

def get_technical_indicators(symbol: str, period: str = "3mo", indicators: List[str] = None) -> Dict[str, Any]:
    """Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")
    
    # Default indicators if none specified
    if indicators is None:
        indicators = ["sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "macd", "bb_20"]
    
    # Auto-adjust period if too short for requested indicators
    max_period_needed = 0
    for indicator in indicators:
        if indicator.startswith(("sma_", "ema_", "bb_")):
            period_val = int(indicator.split("_")[1])
            max_period_needed = max(max_period_needed, period_val)
    
    # If period looks too short, auto-extend
    if max_period_needed >= 25 and period in ["1mo", "30d"]:
        period = "3mo"  # Auto-extend for better analysis
    
    try:
        # Get historical data
        hist_data = get_historical_prices(sym, period=period, interval="1d")
        rows = hist_data.get("rows", [])
        
        # Calculate minimum required days based on indicators
        min_required_days = 30  # Default minimum
        for indicator in indicators:
            if indicator.startswith("sma_") or indicator.startswith("ema_"):
                period_val = int(indicator.split("_")[1])
                min_required_days = max(min_required_days, period_val + 5)  # Add buffer
            elif indicator.startswith("bb_"):
                period_val = int(indicator.split("_")[1])
                min_required_days = max(min_required_days, period_val + 5)
        
        if len(rows) < min_required_days:
            # For Japanese users requesting short periods, suggest longer period
            suggested_period = "3mo" if period in ["1mo", "30d"] else "6mo"
            error_msg = f"データが不足しています（{len(rows)}日）。{min_required_days}日以上必要です。"
            if period in ["1mo", "30d"]:
                error_msg += f" より長い期間（{suggested_period}など）をお試しください。"
            
            return {
                "symbol": sym,
                "period": period,
                "actual_days": len(rows),
                "required_days": min_required_days,
                "suggested_period": suggested_period,
                "error": error_msg,
                "source": "yfinance"
            }
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        close_prices = df["close"].astype(float)
        
        results = {"symbol": sym, "period": period, "indicators": {}}
        
        for indicator in indicators:
            try:
                if indicator.startswith("sma_"):
                    period_val = int(indicator.split("_")[1])
                    sma = close_prices.rolling(window=period_val).mean()
                    results["indicators"][indicator] = {
                        "name": f"Simple Moving Average ({period_val})",
                        "current": float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                        "values": [{"date": str(idx.date()), "value": float(val)} for idx, val in sma.dropna().tail(20).items()]
                    }
                
                elif indicator.startswith("ema_"):
                    period_val = int(indicator.split("_")[1])
                    ema = close_prices.ewm(span=period_val).mean()
                    results["indicators"][indicator] = {
                        "name": f"Exponential Moving Average ({period_val})",
                        "current": float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None,
                        "values": [{"date": str(idx.date()), "value": float(val)} for idx, val in ema.dropna().tail(20).items()]
                    }
                
                elif indicator.startswith("rsi_"):
                    period_val = int(indicator.split("_")[1])
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period_val).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period_val).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
                    signal = "neutral"
                    if current_rsi:
                        if current_rsi > 70:
                            signal = "overbought"
                        elif current_rsi < 30:
                            signal = "oversold"
                    
                    results["indicators"][indicator] = {
                        "name": f"Relative Strength Index ({period_val})",
                        "current": current_rsi,
                        "signal": signal,
                        "values": [{"date": str(idx.date()), "value": float(val)} for idx, val in rsi.dropna().tail(20).items()]
                    }
                
                elif indicator == "macd":
                    ema_12 = close_prices.ewm(span=12).mean()
                    ema_26 = close_prices.ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9).mean()
                    histogram = macd_line - signal_line
                    
                    results["indicators"][indicator] = {
                        "name": "MACD (12,26,9)",
                        "current": {
                            "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                            "signal": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
                            "histogram": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
                        },
                        "values": [
                            {
                                "date": str(idx.date()),
                                "macd": float(macd_line.loc[idx]) if not pd.isna(macd_line.loc[idx]) else None,
                                "signal": float(signal_line.loc[idx]) if not pd.isna(signal_line.loc[idx]) else None,
                                "histogram": float(histogram.loc[idx]) if not pd.isna(histogram.loc[idx]) else None
                            }
                            for idx in macd_line.dropna().tail(20).index
                        ]
                    }
                
                elif indicator.startswith("bb_"):
                    period_val = int(indicator.split("_")[1])
                    sma = close_prices.rolling(window=period_val).mean()
                    std = close_prices.rolling(window=period_val).std()
                    upper_band = sma + (std * 2)
                    lower_band = sma - (std * 2)
                    
                    current_price = float(close_prices.iloc[-1])
                    current_upper = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else None
                    current_lower = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else None
                    
                    position = "middle"
                    if current_upper and current_lower:
                        if current_price > current_upper:
                            position = "above_upper"
                        elif current_price < current_lower:
                            position = "below_lower"
                    
                    results["indicators"][indicator] = {
                        "name": f"Bollinger Bands ({period_val},2)",
                        "current": {
                            "upper": current_upper,
                            "middle": float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                            "lower": current_lower,
                            "position": position
                        },
                        "values": [
                            {
                                "date": str(idx.date()),
                                "upper": float(upper_band.loc[idx]) if not pd.isna(upper_band.loc[idx]) else None,
                                "middle": float(sma.loc[idx]) if not pd.isna(sma.loc[idx]) else None,
                                "lower": float(lower_band.loc[idx]) if not pd.isna(lower_band.loc[idx]) else None
                            }
                            for idx in sma.dropna().tail(20).index
                        ]
                    }
                    
            except Exception as e:
                logger.debug(f"Failed to calculate {indicator} for {sym}: {e}")
                continue

        # Check for Golden Cross (5-day SMA crossing above 25-day SMA)
        if "sma_5" in results["indicators"] and "sma_25" in results["indicators"]:
            try:
                sma5_values = results["indicators"]["sma_5"]["values"]
                sma25_values = results["indicators"]["sma_25"]["values"]
                
                if len(sma5_values) >= 2 and len(sma25_values) >= 2:
                    # Get last 2 days data for both SMAs
                    sma5_current = sma5_values[-1]["value"]
                    sma5_prev = sma5_values[-2]["value"]
                    sma25_current = sma25_values[-1]["value"]
                    sma25_prev = sma25_values[-2]["value"]
                    
                    # Check for golden cross (short MA crosses above long MA)
                    golden_cross = (sma5_prev <= sma25_prev and sma5_current > sma25_current)
                    # Check for death cross (short MA crosses below long MA)
                    death_cross = (sma5_prev >= sma25_prev and sma5_current < sma25_current)
                    
                    results["cross_analysis"] = {
                        "golden_cross": golden_cross,
                        "death_cross": death_cross,
                        "sma5_current": sma5_current,
                        "sma25_current": sma25_current,
                        "trend": "上昇トレンド" if sma5_current > sma25_current else "下降トレンド",
                        "analysis_jp": "ゴールデンクロス発生" if golden_cross else "デッドクロス発生" if death_cross else "クロス無し"
                    }
            except Exception as e:
                logger.debug(f"Failed to analyze crosses for {sym}: {e}")
        
        results["source"] = "yfinance+computed"
        return results
        
    except Exception as e:
        return {
            "symbol": sym,
            "period": period,
            "error": f"Failed to calculate technical indicators: {str(e)}",
            "source": "yfinance"
        }

def check_golden_cross(symbol: str, short_period: int = 5, long_period: int = 25, period: str = "3mo") -> Dict[str, Any]:
    """Check for golden cross/death cross between two moving averages."""
    sym = normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("invalid symbol; use letters/numbers with optional '.' or '-' (e.g., AAPL, VOD.L)")
    
    # Ensure we have enough period for analysis
    min_days_needed = max(long_period * 2, 60)  # At least 60 trading days
    if period == "1mo":
        period = "3mo"  # Auto-extend
    
    try:
        indicators = [f"sma_{short_period}", f"sma_{long_period}"]
        result = get_technical_indicators(sym, period=period, indicators=indicators)
        
        if "error" in result:
            return result
        
        sma_short = result["indicators"].get(f"sma_{short_period}")
        sma_long = result["indicators"].get(f"sma_{long_period}")
        
        if not sma_short or not sma_long:
            return {
                "symbol": sym,
                "error": f"移動平均線の計算に失敗しました（{short_period}日・{long_period}日）",
                "source": "yfinance"
            }
        
        # Get recent values for cross detection
        short_values = sma_short["values"][-10:]  # Last 10 days
        long_values = sma_long["values"][-10:]
        
        crosses = []
        for i in range(1, min(len(short_values), len(long_values))):
            prev_short = short_values[i-1]["value"]
            prev_long = long_values[i-1]["value"]
            curr_short = short_values[i]["value"]
            curr_long = long_values[i]["value"]
            
            if prev_short <= prev_long and curr_short > curr_long:
                crosses.append({
                    "date": short_values[i]["date"],
                    "type": "golden_cross",
                    "type_jp": "ゴールデンクロス",
                    "description": f"{short_period}日線が{long_period}日線を上抜け"
                })
            elif prev_short >= prev_long and curr_short < curr_long:
                crosses.append({
                    "date": short_values[i]["date"],
                    "type": "death_cross", 
                    "type_jp": "デッドクロス",
                    "description": f"{short_period}日線が{long_period}日線を下抜け"
                })
        
        # Current status
        current_short = short_values[-1]["value"]
        current_long = long_values[-1]["value"]
        trend = "上昇傾向" if current_short > current_long else "下降傾向"
        
        return {
            "symbol": sym,
            "period": period,
            "short_ma": {
                "period": short_period,
                "current": current_short,
                "name": f"{short_period}日移動平均"
            },
            "long_ma": {
                "period": long_period,
                "current": current_long,
                "name": f"{long_period}日移動平均"
            },
            "current_trend": trend,
            "recent_crosses": crosses,
            "has_golden_cross": any(c["type"] == "golden_cross" for c in crosses[-3:]),  # Last 3 crosses
            "has_death_cross": any(c["type"] == "death_cross" for c in crosses[-3:]),
            "analysis_summary": f"直近のクロス: {len(crosses)}回, 現在のトレンド: {trend}",
            "source": "yfinance+computed"
        }
        
    except Exception as e:
        return {
            "symbol": sym,
            "error": f"ゴールデンクロス分析に失敗: {str(e)}",
            "source": "yfinance"
        }

def _get_japanese_banking_alternatives(symbol: str) -> List[str]:
    """Get alternative Japanese banking sector symbols when the requested one has insufficient data."""
    symbol_lower = symbol.lower()
    
    # Regional bank alternatives  
    if any(term in symbol_lower for term in ["地域", "地方", "regional"]):
        return [
            "8359.T",  # Hachijuni Bank (reliable data)
            "8365.T",  # Toyama Bank  
            "8334.T",  # Gunma Bank
            "8360.T",  # Yamanashi Chuo Bank
        ]
    
    # General banking sector alternatives
    if any(term in symbol_lower for term in ["銀行", "bank"]):
        return [
            "8306.T",  # Mitsubishi UFJ Financial Group
            "8316.T",  # Sumitomo Mitsui Financial Group  
            "8411.T",  # Mizuho Financial Group
            "8359.T",  # Hachijuni Bank (regional representative)
        ]
    
    return []

def calculate_correlation(
    symbol1: str,
    symbol2: str,
    period: str = "6mo",
    interval: str = "1d"
) -> Dict[str, Any]:
    """Calculate correlation coefficient between two stocks/indices based on daily returns."""
    sym1 = normalize_symbol(symbol1)
    sym2 = normalize_symbol(symbol2)
    
    if not sym1 or not TICKER_RE.match(sym1):
        raise ValueError(f"invalid symbol1: {symbol1}")
    if not sym2 or not TICKER_RE.match(sym2):
        raise ValueError(f"invalid symbol2: {symbol2}")
    
    try:
        # Get historical data for both symbols
        hist1 = get_historical_prices(sym1, period=period, interval=interval, auto_adjust=False)
        hist2 = get_historical_prices(sym2, period=period, interval=interval, auto_adjust=False)
        
        rows1 = hist1.get("rows", [])
        rows2 = hist2.get("rows", [])
        
        if len(rows1) < 10 or len(rows2) < 10:
            # Get alternative suggestions for Japanese banking sector
            alternatives2 = _get_japanese_banking_alternatives(symbol2)
            alternatives1 = _get_japanese_banking_alternatives(symbol1)
            
            all_suggestions = []
            if alternatives2:
                all_suggestions.extend([f"{alt} - {symbol2}の代替として" for alt in alternatives2[:3]])
            if alternatives1:  
                all_suggestions.extend([f"{alt} - {symbol1}の代替として" for alt in alternatives1[:2]])
            
            # Add general market alternatives
            if not all_suggestions:
                all_suggestions.extend([
                    "^N225 - 日経平均株価との相関",
                    "^TPX - TOPIX指数との相関", 
                    "8306.T - 主要銀行（三菱UFJ）との相関"
                ])
            
            error_msg = f"データが不足しているため、相関係数を算出できませんでした。\n取得データポイント: {sym1}={len(rows1)}, {sym2}={len(rows2)} (最低10必要)"
            
            if all_suggestions:
                error_msg += f"\n\n💡 代替案をお試しください:\n" + "\n".join(f"• {s}" for s in all_suggestions[:4])
            
            return {
                "symbol1": sym1,
                "symbol2": sym2, 
                "original_symbol1": symbol1,
                "original_symbol2": symbol2,
                "period": period,
                "interval": interval,
                "error": error_msg,
                "suggestions": all_suggestions[:4],
                "source": "yfinance"
            }
        
        # Convert to DataFrames and align dates
        df1 = pd.DataFrame(rows1)
        df2 = pd.DataFrame(rows2)
        
        df1["date"] = pd.to_datetime(df1["date"])
        df2["date"] = pd.to_datetime(df2["date"])
        
        df1 = df1.set_index("date").sort_index()
        df2 = df2.set_index("date").sort_index()
        
        # Get overlapping dates
        common_dates = df1.index.intersection(df2.index)
        if len(common_dates) < 10:
            return {
                "symbol1": sym1,
                "symbol2": sym2,
                "period": period,
                "interval": interval,
                "error": f"Insufficient overlapping data points (got {len(common_dates)})",
                "source": "yfinance"
            }
        
        # Filter to common dates and calculate returns
        df1_aligned = df1.loc[common_dates]
        df2_aligned = df2.loc[common_dates]
        
        # Calculate daily returns (percentage change)
        returns1 = pd.to_numeric(df1_aligned["close"]).pct_change().dropna()
        returns2 = pd.to_numeric(df2_aligned["close"]).pct_change().dropna()
        
        # Ensure same length
        min_len = min(len(returns1), len(returns2))
        if min_len < 5:
            return {
                "symbol1": sym1,
                "symbol2": sym2,
                "period": period,
                "interval": interval,
                "error": f"Insufficient return data for correlation (got {min_len} returns)",
                "source": "yfinance"
            }
        
        returns1 = returns1.tail(min_len)
        returns2 = returns2.tail(min_len)
        
        # Calculate correlation coefficient
        correlation = float(np.corrcoef(returns1, returns2)[0, 1])
        
        # Calculate additional statistics
        volatility1 = float(returns1.std() * (252 ** 0.5))  # Annualized volatility
        volatility2 = float(returns2.std() * (252 ** 0.5))
        
        # Get currency info
        currency1 = hist1.get("currency", "USD")
        currency2 = hist2.get("currency", "USD")
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "非常に強い" if correlation > 0 else "非常に強い負の"
        elif abs_corr >= 0.6:
            strength = "強い" if correlation > 0 else "強い負の"
        elif abs_corr >= 0.4:
            strength = "中程度の" if correlation > 0 else "中程度の負の"
        elif abs_corr >= 0.2:
            strength = "弱い" if correlation > 0 else "弱い負の"
        else:
            strength = "ほとんどない"
            
        interpretation = f"{sym1}と{sym2}の相関は{strength}関係にあります（係数: {correlation:.2f}）。"
        
        return {
            "symbol1": sym1,
            "symbol2": sym2,
            "correlation": round(correlation, 4),
            "interpretation": interpretation,
            "period": period,
            "data_points": min_len,
            "stats": {
                "volatility1_annualized": round(volatility1, 4),
                "volatility2_annualized": round(volatility2, 4)
            },
            "source": "yfinance+computed"
        }
        
    except Exception as e:
        return {
            "symbol1": sym1,
            "symbol2": sym2,
            "error": f"Failed to calculate correlation: {str(e)}",
            "source": "yfinance"
        }
