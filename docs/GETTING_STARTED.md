# Getting Started Guide

This guide walks you through using the Azure-OpenAI Stock Analysis Tool after successfully running `main.py`.

## Table of Contents
1. [Starting the Application](#starting-the-application)
2. [Accessing the Interface](#accessing-the-interface)
3. [First Steps](#first-steps)
4. [Feature Walkthrough](#feature-walkthrough)
5. [API Usage](#api-usage)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Starting the Application

### 1. Start LocalStack (for AWS features)
```bash
# Start LocalStack with all AWS services
docker compose up -d localstack

# Verify it's running
curl http://localhost:4566/_localstack/health
```

### 2. Start the Backend
```bash
# From project root
python main.py

# Or with auto-reload for development
RELOAD=true python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:app:Starting AI Stocks Assistant API...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 3. Start the Frontend (Optional - Development Mode)
```bash
# In a new terminal
cd frontend
npm install  # First time only
npm run dev
```

**Expected output:**
```
VITE v5.x.x  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### 4. Access the Application

**Option A: Production Build (Recommended)**
- Open: http://localhost:8000/app
- Frontend is served from `/frontend/dist` (pre-built)

**Option B: Development Mode**
- Backend API: http://localhost:8000
- Frontend Dev Server: http://localhost:5173
- Hot reload enabled for both

---

## Accessing the Interface

### Main Endpoints

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:8000` | API root (info page) |
| `http://localhost:8000/app` | Main web application |
| `http://localhost:8000/docs` | Interactive API documentation (Swagger UI) |
| `http://localhost:8000/redoc` | Alternative API documentation (ReDoc) |
| `http://localhost:8000/healthz` | Health check (liveness probe) |
| `http://localhost:8000/readyz` | Readiness check (system status) |

### API Documentation

Visit **http://localhost:8000/docs** for interactive API documentation:
- Test all endpoints directly in your browser
- See request/response schemas
- Try authentication workflows
- Experiment with chat queries

---

## First Steps

### 1. Check System Status

```bash
# Check if backend is ready
curl http://localhost:8000/readyz | jq
```

**Expected response:**
```json
{
  "status": "ready",
  "provider": "openai",
  "openai_configured": true,
  "azure_configured": true,
  "rag_enabled": true,
  "knowledge_dir_exists": true
}
```

### 2. Create Your First User Account

**Option A: Using the Web UI**
1. Open http://localhost:8000/app
2. Click "Sign Up" or "Register"
3. Fill in username and password
4. Click "Register"

**Option B: Using API**
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "SecurePass123!"
  }'
```

**Response:**
```json
{
  "username": "testuser",
  "is_admin": false,
  "created_at": "2025-10-02T13:30:00Z"
}
```

### 3. Login and Get Access Token

**Option A: Using the Web UI**
1. Enter username and password
2. Click "Login"
3. Token is stored automatically in browser

**Option B: Using API**
```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "SecurePass123!"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Save this token** - you'll need it for authenticated API requests!

### 4. Your First Chat Query

**Option A: Using the Web UI**
1. Navigate to Chat tab
2. Type: "What is the current price of AAPL?"
3. Press Enter or click Send
4. Watch the AI use tools to fetch real-time data

**Option B: Using API**
```bash
# Set your token from login
TOKEN="your_access_token_here"

# Make a chat request
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the current price of AAPL?"
  }'
```

**What happens:**
1. AI analyzes your query
2. Calls `get_stock_quote("AAPL")` tool
3. Fetches real-time data from Yahoo Finance
4. Formats a natural language response

---

## Feature Walkthrough

### 1. Stock Price Queries

Try these example queries:

**Basic Price Check:**
```
"What's the current stock price of Apple?"
"How much is Microsoft stock worth?"
"Get me the latest price for TSLA"
```

**Multiple Stocks:**
```
"Compare the stock prices of AAPL, MSFT, and GOOGL"
"Show me Tesla, Amazon, and Netflix stock prices"
```

**Historical Data:**
```
"What was Apple's stock price 30 days ago?"
"Show me Tesla's stock performance over the last week"
```

### 2. Company Information

**Company Details:**
```
"Tell me about Apple Inc."
"What does Microsoft do?"
"Give me information about Tesla"
```

**Financial Metrics:**
```
"What's Apple's market cap?"
"Show me Microsoft's P/E ratio"
"What's Tesla's revenue?"
```

### 3. News and Analysis

**Latest News:**
```
"What's the latest news about Apple?"
"Show me recent news for TSLA"
"Any news on tech stocks?"
```

**News with Sentiment:**
```
"Get me news about Apple with sentiment analysis"
"Show me positive news about Tesla"
```

### 4. Risk Analysis

**Risk Metrics:**
```
"What's the risk profile of AAPL?"
"Calculate risk metrics for Tesla"
"Show me volatility for MSFT"
```

**The AI will provide:**
- Annual Volatility
- Sharpe Ratio
- Maximum Drawdown
- Value at Risk (VaR)
- Beta (vs benchmark)

### 5. Real-Time Dashboard

Access the dashboard at: http://localhost:8000/app (Dashboard tab)

**Features:**
- Real-time stock prices via Finnhub WebSocket
- Updates every 2 seconds
- Multiple stock watchlist
- Price change indicators
- Chart visualization

**Add stocks to watchlist:**
1. Click "Add Stock" 
2. Enter ticker symbol (e.g., AAPL)
3. Click "Add"
4. Watch real-time updates

### 6. Multi-Turn Conversations

The AI maintains context across multiple messages:

```
You: "What's the price of Apple stock?"
AI: "Apple (AAPL) is currently trading at $175.43..."

You: "How about Microsoft?"
AI: "Microsoft (MSFT) is currently trading at $338.11..."

You: "Compare their P/E ratios"
AI: "Based on the previous stocks mentioned:
     - Apple (AAPL): P/E ratio of 28.5
     - Microsoft (MSFT): P/E ratio of 35.2..."
```

### 7. Web Search Integration

Enhanced queries with Brave Search:

```
"Search the web for recent AI news"
"What are people saying about the stock market crash?"
"Find information about cryptocurrency regulations"
```

**Features:**
- Perplexity-style answers with citations
- High-quality sources from Brave Search
- Automatic relevance ranking
- Source attribution

### 8. RAG Knowledge Base

Query your custom knowledge base:

```
"What information do you have about stock analysis?"
"Search the knowledge base for risk management"
"What does the documentation say about portfolio optimization?"
```

**Add documents to knowledge base:**
1. Place files in `knowledge/` directory
2. Supported formats: .txt, .md, .pdf, .docx
3. Reindex: `POST /admin/rag/reindex` (admin only)

---

## API Usage

### Authentication Flow

```bash
# 1. Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# 2. Login
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}' \
  | jq -r '.access_token')

# 3. Use token for authenticated requests
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your question here"}'
```

### Chat Endpoints

#### Basic Chat
```bash
POST /chat
{
  "prompt": "What's the price of AAPL?",
  "system_prompt": "You are a helpful stock assistant",  # Optional
  "deployment": "gpt-4o-mini",  # Optional
  "conversation_id": "conv-123"  # Optional (for multi-turn)
}
```

#### Streaming Chat
```bash
POST /chat/stream
{
  "prompt": "Tell me about Apple stock",
  "stream": true
}
```

**Response:** Server-Sent Events (SSE) stream

#### Clear Conversation
```bash
POST /chat/clear
{
  "conversation_id": "conv-123"
}
```

### Stock Endpoints

#### Get Stock Quote
```bash
GET /stock/AAPL
```

#### Get Stock News
```bash
GET /news/AAPL?limit=10
```

#### Web Search
```bash
POST /api/search
{
  "query": "AI trends 2025",
  "max_results": 5
}
```

### RAG Endpoints

#### Search Knowledge Base
```bash
GET /api/rag/search?query=portfolio&top_k=5
Authorization: Bearer <token>
```

#### Reindex Knowledge Base (Admin)
```bash
POST /admin/rag/reindex
Authorization: Bearer <admin_token>
```

### Admin Endpoints

#### Get All Chat Logs (Admin)
```bash
GET /admin/logs
Authorization: Bearer <admin_token>
```

#### Get Chat History (Admin)
```bash
GET /admin/chats?user_id=1&limit=10
Authorization: Bearer <admin_token>
```

---

## Advanced Features

### 1. Custom Models

Select different AI models:

```bash
# Get available models
curl http://localhost:8000/models

# Use specific model
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt": "Your question",
    "deployment": "gpt-oss-120b"
  }'
```

**Available models:**
- `gpt-4o-mini` - Fast, cost-effective (default)
- `gpt-4.1` - High accuracy
- `gpt-4.1-mini` - Lightweight
- `gpt-oss-120b` - Open source large model
- `o3` - Reasoning model
- `gpt-5` - Next generation (if available)

### 2. System Prompts

Customize AI behavior:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt": "What is Apple?",
    "system_prompt": "You are a financial analyst. Always provide detailed analysis with numbers and metrics."
  }'
```

### 3. Conversation Management

```bash
# Start a conversation
CONV_ID=$(curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"prompt": "Hello"}' \
  | jq -r '.conversation_id')

# Continue the conversation
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"prompt\": \"Tell me more\", \"conversation_id\": \"$CONV_ID\"}"

# Clear conversation
curl -X POST http://localhost:8000/chat/clear \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"conversation_id\": \"$CONV_ID\"}"
```

### 4. AWS Integration (LocalStack)

When `USE_LOCALSTACK=true`:

**S3 Storage:**
- Knowledge base files stored in `stocktool-knowledge` bucket
- Automatic versioning and metadata

**DynamoDB:**
- Conversations cached with 24h TTL
- Stock data cached with 5min TTL

**SQS Queue:**
- Async analysis tasks
- Background processing

**CloudWatch:**
- API latency metrics
- Tool execution metrics
- Cache performance metrics

---

## Troubleshooting

### Backend Won't Start

**Problem:** `ModuleNotFoundError: No module named 'app'`

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

**Problem:** `Port 8000 already in use`

**Solution:**
```bash
# Find process using port
sudo lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
PORT=8001 python main.py
```

---

### Frontend Issues

**Problem:** Frontend shows blank page

**Solution:**
```bash
# Build frontend
cd frontend
npm run build

# Or use dev server
npm run dev
```

---

**Problem:** CORS errors in browser console

**Solution:**
Check `.env` has:
```bash
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:8000
```

---

### Authentication Issues

**Problem:** "Invalid credentials"

**Solution:**
```bash
# Check user exists
curl http://localhost:8000/auth/users

# Register new user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

---

**Problem:** "Token expired"

**Solution:**
```bash
# Login again to get new token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

---

### AI/Model Issues

**Problem:** "Provider not configured"

**Solution:**
Check `.env` has at least one of:
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Or Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

---

**Problem:** Tool calls not working

**Solution:**
1. Check API keys are valid
2. Verify stock ticker symbols are correct
3. Check logs: `docker compose logs -f`
4. Try simpler query: "Get stock price for AAPL"

---

### LocalStack Issues

**Problem:** AWS services not available

**Solution:**
```bash
# Check LocalStack status
curl http://localhost:4566/_localstack/health

# Restart LocalStack
docker compose restart localstack

# Verify resources
python scripts/verify_aws_resources.py
```

---

**Problem:** "License not activated"

**Solution:**
1. Check token in `.env`: `LOCALSTACK_AUTH_TOKEN=ls-...`
2. Verify using Pro image: `image: localstack/localstack-pro:latest`
3. Restart: `docker compose down && docker compose up -d`

---

### Performance Issues

**Problem:** Slow responses

**Solutions:**
1. **Enable caching**: Check `NEWS_CACHE_SIZE` and `NEWS_TTL_SECONDS` in `.env`
2. **Use faster model**: Try `gpt-4o-mini` instead of `gpt-4.1`
3. **Reduce context**: Clear old conversations
4. **Check network**: Slow internet affects API calls
5. **Scale resources**: Increase Docker memory to 4GB+

---

## Next Steps

### For Development
1. ✅ Explore API documentation: http://localhost:8000/docs
2. ✅ Read architecture guide: `docs/ARCHITECTURE.md`
3. ✅ Run tests: `python tests/test_aws_integration.py`
4. ✅ Check performance guide: `docs/PERFORMANCE_OPTIMIZATIONS.md`

### For Portfolio
1. ✅ Customize the UI (frontend styling)
2. ✅ Add new tools to the chat system
3. ✅ Deploy Lambda functions (see `docs/AWS_INTEGRATION.md`)
4. ✅ Create CloudWatch dashboards
5. ✅ Add screenshots/demos to your GitHub README
6. ✅ Document your AWS architecture

### For Production
1. ✅ Set up real AWS resources (not LocalStack)
2. ✅ Configure environment variables for production
3. ✅ Set up CI/CD pipeline
4. ✅ Enable monitoring and alerts
5. ✅ Implement rate limiting
6. ✅ Add authentication/authorization middleware
7. ✅ Set up HTTPS with SSL certificates

---

## Quick Reference Card

### Essential Commands

```bash
# Start everything
docker compose up -d localstack
python main.py

# Check status
curl http://localhost:8000/readyz

# Get token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"prompt": "What is AAPL stock price?"}'

# Stop everything
docker compose down
# Ctrl+C to stop main.py
```

### Essential URLs

- **Application**: http://localhost:8000/app
- **API Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000
- **Health**: http://localhost:8000/healthz
- **LocalStack**: http://localhost:4566

### Essential Files

- **Configuration**: `.env`
- **Main app**: `main.py`
- **Database**: `app.db`
- **Knowledge**: `knowledge/`
- **Logs**: `docker compose logs`

---

## Support

### Documentation
- Architecture: `docs/ARCHITECTURE.md`
- AWS Integration: `docs/AWS_INTEGRATION.md`
- LocalStack: `docs/LOCALSTACK_SETUP_GUIDE.md`
- Performance: `docs/PERFORMANCE_OPTIMIZATIONS.md`
- Tasks/Roadmap: `docs/TASKS.md`

### Community
- GitHub Issues: Report bugs and feature requests
- GitHub Discussions: Ask questions and share ideas

### Resources
- FastAPI: https://fastapi.tiangolo.com/
- OpenAI API: https://platform.openai.com/docs
- LocalStack: https://docs.localstack.cloud/
- yfinance: https://github.com/ranaroussi/yfinance

---

**Happy coding! 🚀** Your AI Stock Assistant is ready to use!
