# Real-Time Stock Dashboard

## Overview

The Real-Time Stock Dashboard provides live stock price updates using Finnhub's free-tier API with WebSocket streaming. This feature allows users to monitor multiple stocks simultaneously with automatic updates every 2 seconds.

## Features

### ✅ Real-Time Price Updates
- **WebSocket streaming** from Finnhub for live data
- **2-second refresh rate** (optimized for free tier limits)
- Automatic reconnection on connection loss
- Connection status indicator (Connected/Connecting/Disconnected)

### ✅ Watchlist Management
- Add/remove stocks from your personalized watchlist
- Default watchlist: AAPL, MSFT, GOOGL, TSLA
- Search functionality to find and add new stocks
- Persistent subscriptions across sessions

### ✅ Comprehensive Stock Data
Each stock card displays:
- **Current Price** with real-time updates
- **Price Change** (absolute $ and %)
- **Open/Previous Close** prices
- **Day High/Low** prices
- **Company Logo** (when available)
- **Company Name** and basic profile
- **Market Capitalization**

### ✅ Symbol Search
- Search by company name or ticker symbol
- Results show: Symbol, Description, Type (Stock/ETF/etc.)
- One-click add to watchlist

## Setup

### 1. Get Finnhub API Key (Free)

1. Visit [Finnhub.io](https://finnhub.io/)
2. Click "Get free API key"
3. Sign up (email confirmation required)
4. Copy your API key from the dashboard

**Free Tier Limits:**
- 60 API calls per minute
- Real-time data for US stocks
- Company profiles and news
- No credit card required

### 2. Configure Environment Variable

Add to your `.env` file:

```bash
FINNHUB_API_KEY=your_api_key_here
```

### 3. Install Dependencies

Backend (already in requirements.txt):
```bash
pip install finnhub-python websocket-client
```

Frontend (no additional dependencies needed):
```bash
cd frontend
npm install
```

### 4. Run the Application

```bash
# Backend
uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend
npm run dev
```

## Usage

### Accessing the Dashboard

1. Login to the application
2. Click the **"Dashboard"** tab in the navigation bar
3. The dashboard will automatically connect via WebSocket

### Managing Your Watchlist

**Add a Stock:**
1. Type company name or ticker in the search bar
2. Press Enter or click "Search"
3. Click on a search result to add it to your watchlist

**Remove a Stock:**
- Click the "×" button in the top-right corner of any stock card

**Default Stocks:**
- The dashboard loads with AAPL, MSFT, GOOGL, and TSLA
- You can remove these and add your own

### Understanding the Display

**Price Colors:**
- 🟢 **Green**: Price increased (positive change)
- 🔴 **Red**: Price decreased (negative change)
- ⚪ **Gray**: No change or loading

**Connection Status:**
- 🟢 **Green dot**: Live updates active
- 🟡 **Yellow dot**: Connecting to server
- 🔴 **Red dot**: Disconnected (will auto-reconnect)

## API Endpoints

### WebSocket
```
WS /dashboard/ws
```

**Client → Server Messages:**
```json
// Subscribe to symbols
{"action": "subscribe", "symbols": ["AAPL", "MSFT"]}

// Unsubscribe from symbols
{"action": "unsubscribe", "symbols": ["AAPL"]}

// Ping (health check)
{"action": "ping"}
```

**Server → Client Messages:**
```json
// Quote update
{
  "type": "quote_update",
  "data": {
    "symbol": "AAPL",
    "current_price": 175.43,
    "change": 2.15,
    "percent_change": 1.24,
    "high": 176.50,
    "low": 174.00,
    "open": 174.50,
    "previous_close": 173.28,
    "timestamp": 1696176000
  }
}

// Subscription confirmation
{"type": "subscribed", "symbols": ["AAPL"]}

// Error message
{"type": "error", "message": "..."}
```

### REST Endpoints

**Get Quote:**
```http
GET /dashboard/quote/{symbol}
Authorization: Bearer <token>
```

**Get Company Profile:**
```http
GET /dashboard/profile/{symbol}
Authorization: Bearer <token>
```

**Search Symbols:**
```http
GET /dashboard/search?q={query}
Authorization: Bearer <token>
```

**Health Check:**
```http
GET /dashboard/health
```

## Technical Architecture

### Backend (FastAPI)

**Components:**
1. **`app/services/finnhub_service.py`**
   - Finnhub client wrapper
   - Quote, profile, news fetching
   - Symbol search

2. **`app/routers/dashboard.py`**
   - WebSocket endpoint for real-time updates
   - REST endpoints for quotes and search
   - Connection manager for WebSocket subscriptions

3. **`app/core/config.py`**
   - `FINNHUB_API_KEY` configuration

### Frontend (React + TypeScript)

**Component:**
- **`frontend/src/components/Dashboard.tsx`**
  - WebSocket client management
  - Watchlist state management
  - Real-time UI updates
  - Search interface

**Key Features:**
- Automatic WebSocket reconnection
- Efficient state updates (per-symbol)
- Responsive grid layout (1-4 columns)
- Error handling and loading states

## Rate Limiting

Finnhub free tier allows:
- **60 API calls per minute**
- **Unlimited WebSocket connections**

Our implementation:
- Uses **WebSocket** for real-time data (no API call limit)
- Updates every **2 seconds** per symbol
- Efficient subscription management
- No unnecessary API calls

**Example:** Monitoring 10 stocks = 10 updates every 2 seconds = 300 updates/minute via WebSocket (free!)

## Troubleshooting

### Dashboard shows "Disconnected"

**Causes:**
1. Backend not running
2. Finnhub API key not configured
3. Network connection issues

**Solutions:**
1. Check backend is running: `http://localhost:8000/dashboard/health`
2. Verify `FINNHUB_API_KEY` in `.env`
3. Check browser console for WebSocket errors

### Prices not updating

**Check:**
1. Connection status indicator (should be green)
2. Stocks are added to watchlist
3. Browser console for errors
4. Finnhub API key is valid

### Search returns no results

**Possible reasons:**
1. Ticker/company name not found
2. Typo in search query
3. API rate limit reached (wait 1 minute)

**Tips:**
- Use ticker symbols for exact matches (e.g., "AAPL")
- Try company name (e.g., "Apple")
- Check [Finnhub Stock Screener](https://finnhub.io/stock-screener) for valid symbols

### "Error fetching quote for {symbol}"

**Common causes:**
1. Invalid ticker symbol
2. Market closed (historical data still shows)
3. Symbol not available on Finnhub
4. API rate limit exceeded

**Note:** Free tier provides real-time data for **US stocks only**. International stocks may have delayed data.

## Future Enhancements

Planned features:
- [ ] Historical price charts (line/candlestick)
- [ ] Portfolio tracking (track holdings and P&L)
- [ ] Price alerts (email/push notifications)
- [ ] Customizable refresh rate
- [ ] Export watchlist to CSV
- [ ] News feed per stock
- [ ] Technical indicators (RSI, MACD, etc.)
- [ ] Comparison view (multiple stocks side-by-side)

## Comparison with Existing Features

| Feature | Chat Assistant | Dashboard |
|---------|---------------|-----------|
| Data Source | yfinance (delayed) | Finnhub (real-time) |
| Update Frequency | On request | Every 2 seconds |
| Multiple Stocks | Sequential queries | Parallel display |
| Analysis | AI-powered | Visual cards |
| Best For | Deep analysis, Q&A | Quick monitoring |

## Performance Notes

- **WebSocket** is more efficient than polling REST APIs
- Dashboard loads **company profiles once** per symbol
- **Real-time quotes** update automatically
- Minimal CPU/network usage
- Scales to 50+ symbols per watchlist

## Security

- WebSocket requires JWT authentication
- API key stored server-side only
- Never exposed to client
- Per-user session management
- Automatic cleanup on disconnect

## Credits

- **Finnhub API**: Real-time stock data provider
- **WebSocket**: Efficient real-time communication protocol
- **React**: Frontend UI framework
- **FastAPI**: Backend WebSocket support

---

**Need help?** Open an issue on GitHub or check the main [README.md](README.md) for general setup instructions.
