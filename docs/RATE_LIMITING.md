# Finnhub Rate Limiting - Implementation Guide

## ⚠️ Problem: API Rate Limit Exceeded

**Error:** `FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0`

### Finnhub Free Tier Limits
- **60 API calls per minute** (strict limit)
- **1 call per second** (60 calls / 60 seconds)
- If exceeded: HTTP 429 error, must wait for minute to reset

## ✅ Solutions Implemented

### 1. Rate Limiter Class (Backend)
**Location:** `app/services/finnhub_service.py`

```python
class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        # Tracks calls per minute
        # Enforces minimum interval between calls
        # Auto-resets every 60 seconds
```

**Features:**
- ✅ Tracks API calls per minute
- ✅ Enforces minimum 1.1-second interval (55 calls/min for safety margin)
- ✅ Automatically resets counter every 60 seconds
- ✅ Blocks/sleeps if limit reached

### 2. Sequential Updates with Delays
**Location:** `app/routers/dashboard.py`

**Before (WRONG):**
```python
# Updated all symbols simultaneously
for symbol in symbols:
    get_quote(symbol)  # 4 symbols = 4 calls instantly
await asyncio.sleep(2)  # Only 2 second delay
# = 120 calls/minute ❌ EXCEEDS LIMIT
```

**After (CORRECT):**
```python
# Update symbols one at a time
for symbol in symbols:
    get_quote(symbol)
    await asyncio.sleep(1.5)  # 1.5 seconds between each call
# 4 symbols × 1.5 sec = 6 seconds per cycle
# = 40 calls/minute ✅ SAFE
```

### 3. Error Handling for 429
**Location:** `app/services/finnhub_service.py`

```python
try:
    self.rate_limiter.wait_if_needed()
    quote = self.client.quote(symbol)
except finnhub.FinnhubAPIException as e:
    if e.status_code == 429:
        return {
            "error": "Rate limit exceeded. Please wait...",
            "rate_limited": True
        }
```

### 4. Automatic Pause on Rate Limit
**Location:** `app/routers/dashboard.py`

```python
if quote.get("rate_limited"):
    # Notify client
    await websocket.send_json({
        "type": "rate_limit_warning",
        "message": "Rate limit reached. Updates paused for 60 seconds."
    })
    # Wait full minute before trying again
    await asyncio.sleep(60)
    break  # Stop update cycle
```

### 5. Frontend Error Display
**Location:** `frontend/src/components/Dashboard.tsx`

- ✅ Shows rate limit warning banner
- ✅ Displays "last updated" timestamp per stock
- ✅ Shows update cycle time based on watchlist size
- ✅ Warns if too many stocks (>20)

## 📊 Rate Limit Calculator

### Update Frequency by Watchlist Size

| Stocks | Delay Between Calls | Cycle Time | Updates/Min/Stock |
|--------|---------------------|------------|-------------------|
| 4      | 1.5s               | 6s         | 10 times         |
| 10     | 1.5s               | 15s        | 4 times          |
| 20     | 1.5s               | 30s        | 2 times          |
| 40     | 1.5s               | 60s        | 1 time           |

**Formula:**
- Cycle Time = `Stocks × 1.5 seconds`
- Updates per Stock per Minute = `60 / Cycle Time`

### Recommended Watchlist Sizes

| Use Case | Recommended Stocks | Update Frequency |
|----------|-------------------|------------------|
| Active Trading | 5-10 stocks | Every 7-15 seconds |
| Portfolio Monitoring | 10-20 stocks | Every 15-30 seconds |
| Research/Tracking | 20-40 stocks | Every 30-60 seconds |

**Warning:** More than 40 stocks means each stock updates less than once per minute!

## 🔧 Configuration Options

### Backend Rate Limit Settings

**File:** `app/services/finnhub_service.py`

```python
# Adjust rate limit (default: 55 calls/min for safety)
self.rate_limiter = RateLimiter(calls_per_minute=55)
```

**File:** `app/routers/dashboard.py`

```python
# Adjust delay between calls (default: 1.5 seconds)
await asyncio.sleep(1.5)  # Increase for more safety margin
```

### Aggressive (Risky) Settings
```python
RateLimiter(calls_per_minute=59)  # Close to limit
await asyncio.sleep(1.0)           # Minimum delay
```
**Risk:** May still hit rate limit during API response delays

### Conservative (Safe) Settings
```python
RateLimiter(calls_per_minute=50)  # Safe margin
await asyncio.sleep(2.0)           # Longer delay
```
**Benefit:** Never hit rate limit, but slower updates

### Current (Balanced) Settings
```python
RateLimiter(calls_per_minute=55)  # 5 call buffer
await asyncio.sleep(1.5)           # 1.5 second delay
```
**Balance:** Safe margin with reasonable update speed

## 🐛 Troubleshooting

### Still Getting 429 Errors?

1. **Check concurrent connections:**
   ```bash
   # Multiple browser tabs/windows share the limit!
   # Close other dashboard tabs
   ```

2. **Check other API usage:**
   ```bash
   # Are you calling /dashboard/quote REST endpoint separately?
   # This counts toward the same limit!
   ```

3. **Increase delay:**
   ```python
   # In dashboard.py
   await asyncio.sleep(2.0)  # Was 1.5, now 2.0
   ```

4. **Reduce watchlist:**
   - Remove stocks you're not actively monitoring
   - Recommended: 10-15 stocks max

### Rate Limit Not Resetting?

Finnhub resets every 60 seconds from first call in the window.

**Example:**
- 10:00:00 - First call (counter = 1)
- 10:00:59 - 60th call (counter = 60, limit reached)
- 10:01:00 - Counter resets (can make calls again)

**Our implementation tracks this automatically!**

## 📈 Optimization Tips

### 1. Prioritize Important Stocks
Keep frequently-traded stocks at the top of watchlist (they update first in cycle).

### 2. Use Separate Watchlists
For different purposes:
- **Active Trading:** 5 stocks, rapid updates
- **Long-term Holdings:** 20 stocks, slow updates

### 3. Cache Profile Data
Company profiles rarely change - we already cache these!
```python
# Profiles fetched once on mount, not in update loop
```

### 4. Consider Upgrading
Finnhub paid tiers offer:
- **Starter ($59/mo):** 300 calls/minute
- **Grow ($149/mo):** 600 calls/minute
- **Business ($399/mo):** 1,200 calls/minute

With 300 calls/min, you could:
- Monitor 100 stocks with 2-second updates
- Monitor 200 stocks with 4-second updates

## 🎯 Best Practices

### DO ✅
- Keep watchlist under 15 stocks for good performance
- Monitor backend logs for rate limit warnings
- Use the info banner to understand update frequency
- Let the rate limiter handle delays automatically

### DON'T ❌
- Don't manually call REST endpoints while WebSocket is running
- Don't open multiple dashboard tabs (they share the rate limit)
- Don't reduce delay below 1.5 seconds
- Don't ignore 429 errors (they compound quickly)

## 📝 Summary

**Current Implementation:**
- ✅ Rate limiter with 55 calls/min safety margin
- ✅ Sequential updates with 1.5s delay
- ✅ Automatic 60s pause if rate limit hit
- ✅ Error handling for 429 responses
- ✅ UI warnings and info banners
- ✅ "Last updated" timestamps per stock

**Result:**
- **Safe:** Won't exceed rate limit under normal use
- **Transparent:** Users see update frequency
- **Graceful:** Automatic recovery from rate limit errors
- **Optimal:** Best balance of speed and safety

---

**Rate limit issues resolved!** The dashboard now respects Finnhub's 60 calls/minute limit with built-in safety margins.
