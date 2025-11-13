# Japanese Stock Market Dashboard

## Overview
Real-time dashboard focused on **Tokyo Stock Exchange (TSE)** stocks with:
- Live price quotes via yfinance
- Latest news articles
- Trend analysis (1-month SMA 20/50)
- Market cap and volume data

## Features

### 📊 Real-Time Price Data
- Updates every 30 seconds via polling
- No WebSocket required (uses REST API)
- Japanese Yen (JPY) pricing
- Symbol format: `XXXX.T` (e.g., `7203.T` for Toyota)

### 📰 News Integration
- Latest news articles per stock (up to 3 shown)
- Direct links to source articles
- Publisher and publish date display
- Powered by yfinance news aggregation

### 📈 Trend Analysis
- 1-month price trend indicators:
  - 📈 **Strong Uptrend**: Price > SMA20 > SMA50
  - ↗️ **Uptrend**: Price > SMA20
  - ➡️ **Neutral**: Mixed signals
  - ↘️ **Downtrend**: Price < SMA20
  - 📉 **Strong Downtrend**: Price < SMA20 < SMA50
- Period change percentage (1-month)

## Default Watchlist

Pre-configured with major Japanese companies:

| Symbol | Company | Sector |
|--------|---------|--------|
| 7203.T | Toyota Motor | Automotive |
| 6758.T | Sony Group | Electronics |
| 9984.T | SoftBank Group | Telecom/Tech |
| 7974.T | Nintendo | Gaming |

## API Integrations

### yfinance (Primary Data Source)
- **Endpoints**:
  - `/dashboard/jp/quote/{symbol}` - Real-time quote
  - `/dashboard/jp/news/{symbol}` - Latest news
  - `/dashboard/jp/trend/{symbol}` - Trend analysis
- **Rate Limits**: No strict limits (uses Yahoo Finance API)
- **Free Tier**: Unlimited

### Alpha Vantage (Optional - News Sentiment)
- **Endpoint**: `/dashboard/news/sentiment`
- **Rate Limits**: 25 requests per day (free tier)
- **Setup**: Set `ALPHA_VANTAGE_API_KEY` environment variable
- **Use Case**: Advanced news sentiment analysis

## Environment Variables

Add to your `.env` file:

```bash
# Optional: Alpha Vantage for advanced news sentiment
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Finnhub (not used for JP stocks but can coexist)
FINNHUB_API_KEY=your_finnhub_key_here
```

## Search Japanese Stocks

Search accepts:
- **Company names**: "Toyota", "Sony", "Nintendo"
- **Ticker symbols**: "7203", "6758", "7974"
- **Full symbols**: "7203.T", "6758.T"

The search uses Finnhub's symbol search API, so you can find any TSE-listed company.

## Adding More Stocks

Popular Japanese stocks you can add:

### Blue Chips (Nikkei 225)
- `8306.T` - Mitsubishi UFJ (Banking)
- `6861.T` - Keyence (Industrial)
- `9433.T` - KDDI (Telecom)
- `6902.T` - Denso (Auto Parts)
- `4063.T` - Shin-Etsu Chemical
- `6367.T` - Daikin (HVAC)

### Technology
- `6503.T` - Mitsubishi Electric
- `6971.T` - Kyocera
- `6701.T` - NEC Corporation
- `4755.T` - Rakuten

### Consumer
- `2502.T` - Asahi Group (Beverages)
- `4911.T` - Shiseido (Cosmetics)
- `7201.T` - Nissan Motor
- `9020.T` - JR East (Rail)

## Performance Optimization

### Polling Strategy
- **Interval**: 30 seconds (configurable in `Dashboard.tsx`)
- **Batch fetching**: Quotes, news, and trends fetched sequentially
- **Delay between stocks**: 500ms to avoid overwhelming API

### Data Caching
- Frontend caches quotes in React state
- Backend can implement caching (not required for yfinance)
- News/trends remain static between polls

### Rate Limiting
yfinance has no strict rate limits, but best practices:
- Keep polling interval ≥ 30 seconds
- Limit watchlist to 10-20 stocks
- Add delays between sequential requests (500ms)

## Troubleshooting

### No data showing
1. Check browser console for errors
2. Verify backend is running (`http://127.0.0.1:8000/healthz`)
3. Test JP endpoint directly: `curl http://127.0.0.1:8000/dashboard/jp/quote/7203.T`
4. Check authentication token is valid

### News not loading
- News availability varies by stock
- Some stocks may have limited English news
- Japanese news may not be translated
- Try major stocks like Toyota (7203.T) or Sony (6758.T)

### Trends showing "neutral"
- Requires at least 50 days of historical data
- Newly listed stocks may not have enough data
- Check if symbol is correct (must include `.T` suffix)

## Future Enhancements

- [ ] Japanese news translation (Google Translate API)
- [ ] Nikkei 225 index chart
- [ ] Sector performance comparison
- [ ] Alert notifications for price changes
- [ ] Candlestick charts for technical analysis
- [ ] Dividend yield information
- [ ] Financial ratios (P/E, P/B, ROE)

## Development

### Frontend Changes
File: `frontend/src/components/Dashboard.tsx`
- Default watchlist: Lines 66-71
- API endpoints: `fetchInitialQuotes()` function
- UI cards: Render section with news/trend display

### Backend Services
Files:
- `app/services/yfinance_service.py` - yfinance integration
- `app/services/alphavantage_service.py` - Alpha Vantage integration
- `app/routers/dashboard.py` - API endpoints

### Testing
```bash
# Test yfinance quote
curl http://127.0.0.1:8000/dashboard/jp/quote/7203.T \
  -H "Authorization: Bearer YOUR_TOKEN"

# Test news
curl http://127.0.0.1:8000/dashboard/jp/news/7203.T?limit=5 \
  -H "Authorization: Bearer YOUR_TOKEN"

# Test trend
curl http://127.0.0.1:8000/dashboard/jp/trend/7203.T?period=1mo \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## References

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)
- [Tokyo Stock Exchange](https://www.jpx.co.jp/english/)
- [Nikkei 225 Index](https://indexes.nikkei.co.jp/en/nkave)
