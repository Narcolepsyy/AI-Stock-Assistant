# Dashboard Troubleshooting Guide

## Issue: "Error fetching profile: SyntaxError: Unexpected token '<', "<!DOCTYPE "... is not valid JSON"

### Problem
The frontend is receiving HTML instead of JSON from the API. This happens when:
1. The API endpoint doesn't exist (404 HTML page returned)
2. CORS is blocking the request
3. The backend server isn't running
4. Wrong API URL configuration

### ✅ Solutions Applied

#### 1. Fixed API Base URL
**Changed:** All fetch calls now use `apiBase()` helper
```typescript
// Before (WRONG)
fetch('/dashboard/profile/AAPL')

// After (CORRECT)
fetch(`${apiBase()}/dashboard/profile/AAPL`)
```

#### 2. Fixed WebSocket URL
**Changed:** WebSocket now uses correct base URL parsing
```typescript
const base = apiBase()  // e.g., 'http://127.0.0.1:8000'
const protocol = base.startsWith('https') ? 'wss:' : 'ws:'
const host = base.replace(/^https?:\/\//, '')
const wsUrl = `${protocol}//${host}/dashboard/ws`
```

#### 3. Added Test Endpoint
**Added:** `/dashboard/test` endpoint for connectivity testing
```bash
# Test backend connectivity
curl http://127.0.0.1:8000/dashboard/test
```

#### 4. Added Error Display
**Added:** Error message banner in UI to show configuration issues

### 🔍 How to Verify the Fix

1. **Check Backend is Running:**
```bash
# Should return JSON
curl http://127.0.0.1:8000/dashboard/test

# Expected response:
{
  "message": "Dashboard API is working",
  "timestamp": "2025-10-01T...",
  "finnhub_configured": true
}
```

2. **Check Profile Endpoint:**
```bash
# Get your access token from browser localStorage or login
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://127.0.0.1:8000/dashboard/profile/AAPL
```

3. **Check Frontend API Base:**
Open browser console on the dashboard page:
```javascript
// Should log the API base URL
console.log(window.location.origin)
// Development: http://localhost:5173
// The apiBase() function returns: http://127.0.0.1:8000
```

### 🚀 Quick Test Checklist

Run these commands to verify everything works:

```bash
# 1. Test backend health
curl http://127.0.0.1:8000/dashboard/test

# 2. Test with authentication (replace TOKEN)
TOKEN="your_access_token_here"
curl -H "Authorization: Bearer $TOKEN" \
     http://127.0.0.1:8000/dashboard/profile/AAPL

# 3. Check backend logs
# Look for any errors in the terminal running `uvicorn main:app --reload`

# 4. Check frontend console
# Open browser DevTools → Console tab
# Look for any red errors or failed requests
```

### 📋 Common Issues & Fixes

#### Issue: Still getting HTML response
**Check:**
- Is backend running on `http://127.0.0.1:8000`?
- Is frontend dev server on `http://localhost:5173`?
- Check `frontend/src/lib/api.ts` - `apiBase()` should return correct URL

**Fix:**
```bash
# Restart backend
uvicorn main:app --reload

# Restart frontend (new terminal)
cd frontend
npm run dev
```

#### Issue: "finnhub_configured": false
**Fix:**
```bash
# Add to .env file
echo "FINNHUB_API_KEY=your_api_key_here" >> .env

# Restart backend
# (uvicorn will auto-reload)
```

#### Issue: CORS errors in browser console
**Check:** Backend CORS configuration in `main.py`

**Verify CORS is working:**
```bash
curl -X OPTIONS \
     -H "Origin: http://localhost:5173" \
     -H "Access-Control-Request-Method: GET" \
     http://127.0.0.1:8000/dashboard/test
```

#### Issue: WebSocket not connecting
**Check:**
- Browser console for WebSocket errors
- Backend logs for connection attempts
- Firewall/antivirus blocking WebSocket connections

**Test WebSocket manually:**
```javascript
// In browser console
const ws = new WebSocket('ws://127.0.0.1:8000/dashboard/ws')
ws.onopen = () => console.log('Connected!')
ws.onerror = (e) => console.error('Error:', e)
```

### 🎯 Expected Behavior

When working correctly:
1. Dashboard loads with default watchlist (AAPL, MSFT, GOOGL, TSLA)
2. Company logos appear (fetched from profiles)
3. Prices update every 2 seconds
4. Green/red indicators show price changes
5. Connection status shows "Live" (green dot)
6. No errors in browser console

### 📝 Debug Mode

Enable verbose logging:

**Frontend** (browser console):
```javascript
localStorage.setItem('debug', 'true')
// Reload page
```

**Backend** (set in .env):
```bash
LOG_LEVEL=DEBUG
```

### 🆘 Still Having Issues?

1. Check `test_dashboard_setup.py` output:
```bash
python test_dashboard_setup.py
```

2. Verify all files have correct content:
- `app/routers/dashboard.py` - Dashboard router registered
- `main.py` - `dashboard_router` imported and included
- `app/core/config.py` - `FINNHUB_API_KEY` defined

3. Check browser Network tab:
- Look for failed requests (red status codes)
- Check request URL (should be `http://127.0.0.1:8000/dashboard/...`)
- Check response (should be JSON, not HTML)

4. Run setup script:
```bash
./setup_dashboard.sh
```

### ✅ Verification Script

Run this in browser console on the dashboard page:

```javascript
// Test API connectivity
fetch('http://127.0.0.1:8000/dashboard/test')
  .then(r => r.json())
  .then(d => console.log('✅ API Test:', d))
  .catch(e => console.error('❌ API Error:', e))

// Test authenticated endpoint
const token = localStorage.getItem('auth_token')
fetch('http://127.0.0.1:8000/dashboard/profile/AAPL', {
  headers: { 'Authorization': `Bearer ${token}` }
})
  .then(r => r.json())
  .then(d => console.log('✅ Profile:', d))
  .catch(e => console.error('❌ Profile Error:', e))
```

---

**Status:** ✅ All fixes applied. The dashboard should now load profiles correctly without HTML parsing errors.
