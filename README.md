# AI Stocks Assistant

**A modern, AI-powered stock analysis platform combining FastAPI backend with React frontend, featuring conversational AI, real-time market data, and web search capabilities.**

[![Demo Video](https://img.shields.io/badge/📹-Watch%20Demo-red?style=for-the-badge)](https://youtu.be/J3EujYV7LqE)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)

## 🎯 Overview

AI Stocks Assistant is an enterprise-grade financial analysis platform that combines:
- **🤖 Conversational AI** - GPT-powered chat with function calling for stock analysis
- **📊 Real-time Market Data** - Live stock prices, company profiles, and financial metrics
- **🔍 Web Search Integration** - Perplexity-style web search with citations
- **📚 RAG System** - Knowledge base search with ChromaDB and LangChain
- **📈 Risk Assessment** - Comprehensive risk metrics (volatility, Sharpe ratio, VaR, beta)
- **🌐 Real-time Dashboard** - WebSocket-powered live stock monitoring with watchlist
- **☁️ AWS Integration** - Production-ready AWS services (S3, DynamoDB, Lambda, CloudWatch)

## ✨ Key Features

### 🤖 AI-Powered Analysis
- **Multi-turn Conversations** - Context-aware chat with conversation history
- **Function Calling** - Automatic tool selection for stock data, news, and risk analysis
- **Multiple AI Models** - Support for GPT-4.1, GPT-5, O3, and custom deployments
- **Smart Query Routing** - Optimized model selection based on query complexity

### 📊 Financial Data
- **Real-time Stock Quotes** - Current prices with currency and timezone information
- **Historical Data** - Price history, technical indicators, and trends
- **Company Profiles** - Financials, market cap, sector information
- **News Aggregation** - Multi-source news with sentiment analysis
- **Risk Metrics** - Volatility, Sharpe ratio, maximum drawdown, VaR, beta

### 🔍 Advanced Search
- **Perplexity-Style Web Search** - AI-synthesized answers with citations
- **Brave Search Integration** - High-quality search results from trusted sources
- **RAG Knowledge Base** - Semantic search over curated financial documents
- **Hybrid Search** - Combined web search + knowledge base results

### 📈 Live Dashboard
- **WebSocket Streaming** - Real-time price updates every 2 seconds
- **Watchlist Management** - Track multiple stocks simultaneously
- **Market Summary** - Free RSS feeds from Yahoo Finance, CNBC, CoinTelegraph
- **Market Sentiment** - AI-powered market sentiment indicator

### 🏗️ Enterprise Architecture
- **Modular Design** - Clean separation of concerns (auth, services, routers, utils)
- **Performance Optimized** - Multi-layer caching, connection pooling, parallel execution
- **AWS Integration** - Optional S3, DynamoDB, SQS, SNS, Lambda, CloudWatch
- **LocalStack Support** - Local AWS development environment
- **Docker Ready** - Full containerization with docker-compose

## 📁 Project Structure

```
Azure-OpenAI_StockTool/
├── app/                    # Backend application
│   ├── auth/              # JWT authentication & user management
│   ├── core/              # Configuration & environment settings
│   ├── models/            # SQLAlchemy models & Pydantic schemas
│   ├── routers/           # API endpoints (chat, auth, admin, rag, dashboard)
│   ├── services/          # Business logic & external integrations
│   │   ├── aws/          # AWS service clients (S3, DynamoDB, SQS, CloudWatch)
│   │   ├── stock_service.py     # Stock data & financial analysis
│   │   ├── perplexity_web_search.py  # Web search with AI synthesis
│   │   ├── free_news_service.py      # RSS news aggregation
│   │   └── enhanced_rag_service.py   # RAG knowledge base
│   └── utils/             # Tool registry, caching, conversation management
├── frontend/              # React + TypeScript + TailwindCSS
│   ├── src/
│   │   ├── components/   # Reusable React components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── context/      # React Context providers
│   │   └── utils/        # Frontend utilities
├── docs/                  # 📚 40+ documentation files
├── tests/                 # 🧪 60+ test files
├── scripts/               # 🔧 Setup & utility scripts
├── demos/                 # 🎬 Demo scripts
├── knowledge/             # RAG knowledge base files
├── lambda_functions/      # AWS Lambda functions
├── localstack/            # LocalStack initialization
├── main.py               # FastAPI application entry point
├── docker-compose.yml    # Docker orchestration
└── .env.example          # Environment variable template
```

**Key Documentation:**
- 📚 `docs/ARCHITECTURE.md` - System architecture and design patterns
- 🚀 `docs/GETTING_STARTED.md` - Comprehensive setup guide
- ☁️ `docs/AWS_INTEGRATION.md` - AWS services integration
- ⚡ `docs/PERFORMANCE_OPTIMIZATIONS.md` - Performance tuning guide
- 🔍 `docs/WEB_SEARCH_INTEGRATION.md` - Search capabilities overview


## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** - Backend runtime
- **Node.js 18+** - Frontend development
- **Docker & Docker Compose** - For LocalStack (optional)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Narcolepsyy/Azure-OpenAI_StockTool.git
cd Azure-OpenAI_StockTool
```

### 2️⃣ Setup Backend
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys (use your favorite editor)
nano .env
```

**Minimum Required Configuration:**
```bash
# Option 1: Azure OpenAI (Recommended)
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_OSS_120B=your_deployment_name

# Option 2: Standard OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# Optional but Recommended
BRAVE_API_KEY=your_brave_search_key_here  # High-quality web search
FINNHUB_API_KEY=your_finnhub_key_here     # Real-time stock data
```

### 4️⃣ Start Backend
```bash
# Run FastAPI server
uvicorn main:app --reload

# Server will start at http://127.0.0.1:8000
```

### 5️⃣ Setup Frontend (Development)
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Frontend will be available at http://localhost:5173
```

### 6️⃣ Access Application
- **Web Interface**: http://localhost:5173 (dev) or http://localhost:8000/app (production)
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/healthz

### 7️⃣ Production Build
```bash
cd frontend

# Build for production
npm run build

# Serve built files from FastAPI at /app
```

### 📖 Detailed Setup Guides
- **Complete Walkthrough**: `docs/GETTING_STARTED.md`
- **AWS Integration**: `docs/AWS_INTEGRATION.md`
- **LocalStack Setup**: `docs/LOCALSTACK_SETUP_GUIDE.md`
- **Performance Tuning**: `docs/PERFORMANCE_OPTIMIZATIONS.md`

---

## 🤖 Supported AI Models

The application supports multiple AI models through flexible provider routing:

| Model | Provider | Description | Environment Variable |
|-------|----------|-------------|---------------------|
| **gpt-oss-120b** | Azure | Large open-source model (default) | `AZURE_OPENAI_DEPLOYMENT_OSS_120B` |
| **gpt-4.1** | Azure/OpenAI | Enhanced GPT-4.1 | `AZURE_OPENAI_DEPLOYMENT_4_1` |
| **gpt-4.1-mini** | Azure/OpenAI | Lightweight, fast variant | `AZURE_OPENAI_DEPLOYMENT_4_1_MINI` |
| **gpt-4o** | OpenAI | GPT-4 Omni multimodal | `OPENAI_API_KEY` |
| **gpt-4o-mini** | OpenAI | Cost-effective variant | `OPENAI_API_KEY` |
| **o3** | Azure/OpenAI | Advanced reasoning model | `AZURE_OPENAI_DEPLOYMENT_O3` |
| **gpt-5** | Azure/OpenAI | Next-generation model | `AZURE_OPENAI_DEPLOYMENT_GPT5` |

**Model Selection Logic:**
1. Standard OpenAI key present → OpenAI provider
2. Otherwise → Azure OpenAI provider
3. Azure uses model aliases → deployment name mapping
4. Fallback to `AZURE_OPENAI_DEPLOYMENT` or `OPENAI_MODEL`

**API Endpoint:** `GET /models` returns all available models with their configurations.

## 📡 API Endpoints

### Chat & Conversation
- **POST** `/chat` - AI-powered chat with function calling
  - Request: `{ prompt, system_prompt?, model?, conversation_id?, reset? }`
  - Response: `{ content, tool_calls?, conversation_id, sources? }`
- **POST** `/chat/stream` - Streaming chat responses
- **POST** `/chat/clear` - Clear conversation history
  - Request: `{ conversation_id }`
  - Response: `{ conversation_id, cleared: true }`
- **GET** `/models` - List available AI models

### Stock Data
- **GET** `/stock/{symbol}` - Quick stock quote
- **GET** `/stock/{symbol}/profile` - Company profile
- **GET** `/stock/{symbol}/historical` - Historical prices
- **GET** `/stock/{symbol}/risk` - Risk assessment metrics

### News & Search
- **GET** `/news/{symbol}` - Stock-related news
- **POST** `/search/web` - Web search with citations
- **POST** `/search/rag` - Knowledge base search

### Dashboard
- **GET** `/dashboard/watchlist` - User's watchlist
- **POST** `/dashboard/watchlist/{symbol}` - Add to watchlist
- **DELETE** `/dashboard/watchlist/{symbol}` - Remove from watchlist
- **GET** `/dashboard/market/summary` - Market summary (RSS feeds)
- **WebSocket** `/ws/stocks` - Real-time price updates

### Authentication
- **POST** `/auth/register` - Register new user
- **POST** `/auth/login` - User login (returns JWT)
- **POST** `/auth/refresh` - Refresh access token
- **POST** `/auth/logout` - User logout

### Admin (requires admin role)
- **GET** `/admin/logs` - View chat logs
- **POST** `/admin/rag/reindex` - Reindex knowledge base
- **GET** `/admin/users` - List all users

**Full API Documentation:** http://localhost:8000/docs

## 🌐 AWS Integration (Optional)

The application includes enterprise-grade AWS integrations for production deployment:

### Supported Services
- **S3** - Document storage for RAG knowledge base
- **DynamoDB** - Distributed conversation & cache storage with TTL
- **SQS** - Asynchronous task queue for background jobs
- **SNS** - Notification system for alerts
- **Lambda** - Scheduled stock data updates
- **CloudWatch** - Metrics, logs, and monitoring dashboards

### LocalStack Development
Test AWS features locally without incurring costs:

```bash
# 1. Get free LocalStack auth token
# Sign up: https://app.localstack.cloud/
# Get token: https://app.localstack.cloud/workspace/auth-token

# 2. Add to .env
LOCALSTACK_AUTH_TOKEN=ls-your-token-here

# 3. Start LocalStack
docker compose up -d localstack

# 4. Setup AWS resources
./scripts/setup_localstack.sh

# 5. Verify
python scripts/verify_aws_resources.py

# 6. Run tests
python tests/test_aws_integration.py
```

### Production Deployment
```bash
# Deploy Lambda functions
./scripts/deploy_lambda.sh

# Create CloudWatch dashboard
./scripts/create_cloudwatch_dashboard.sh

# Test Lambda
python scripts/test_lambda.py
```

**📖 Complete AWS Guides:**
- `docs/AWS_INTEGRATION.md` - Comprehensive AWS setup
- `docs/LOCALSTACK_SETUP_GUIDE.md` - LocalStack development guide
- `docs/LAMBDA_AND_CLOUDWATCH_GUIDE.md` - Lambda & monitoring setup

## 🏗️ Architecture Overview

The application follows a modular architecture with clear separation of concerns:

### Backend Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                        │
├─────────────────────────────────────────────────────────────┤
│  Routers (API Endpoints)                                     │
│  ├── /chat      - AI conversations                          │
│  ├── /auth      - Authentication                            │
│  ├── /dashboard - Real-time dashboard                       │
│  └── /admin     - Admin operations                          │
├─────────────────────────────────────────────────────────────┤
│  Services (Business Logic)                                   │
│  ├── openai_client     - Multi-provider AI client           │
│  ├── stock_service     - Financial data & analysis          │
│  ├── perplexity_web_search - Web search with synthesis      │
│  ├── free_news_service - RSS news aggregation               │
│  └── enhanced_rag_service - Knowledge base search           │
├─────────────────────────────────────────────────────────────┤
│  Utils (Cross-cutting Concerns)                             │
│  ├── Tool Registry     - Function calling tools             │
│  ├── Conversation Mgmt - History & context                  │
│  ├── Cache Manager     - Multi-layer caching                │
│  └── Connection Pool   - HTTP client pooling                │
├─────────────────────────────────────────────────────────────┤
│  External Integrations                                       │
│  ├── OpenAI/Azure      - GPT models                         │
│  ├── yfinance          - Stock data                         │
│  ├── Brave Search      - Web search                         │
│  ├── Finnhub           - Real-time quotes                   │
│  └── ChromaDB          - Vector storage                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns
- **Tool Function Calling** - Auto-injected tools based on query analysis
- **Multi-Provider Routing** - Seamless switching between Azure and OpenAI
- **Citation Preservation** - Perplexity-style citations with smart truncation
- **Performance Optimization** - TTL caching, connection pooling, parallel execution
- **Circuit Breakers** - Fault tolerance for external API failures

**📖 Architecture Documentation:** `docs/ARCHITECTURE.md`

## ⚡ Performance Features

The application is optimized for production workloads:

### Caching Strategy
- **Multi-layer caching** - Short (2h), medium (24h), long-term (7d)
- **TTL-based expiration** - Automatic cache invalidation
- **Smart cache keys** - Per-symbol, per-query granularity

### Parallel Execution
- **ThreadPoolExecutor** - 8 workers for tool calls
- **asyncio.gather()** - Concurrent embeddings generation
- **Parallel web search** - BM25 + semantic scoring

### Optimized Timeouts
- **Model timeouts** - 25-45s (not 60-120s)
- **Token limits** - 500-800 tokens for speed
- **Web search** - <3s target (was 15-30s)

### Smart Query Routing
- **Simple query cache** - 60s TTL for basic queries
- **Fast model selection** - gpt-4o-mini for simple queries
- **Tool optimization** - Selective tool injection

**📖 Performance Guide:** `docs/PERFORMANCE_OPTIMIZATIONS.md`

## 🔒 Security & Authentication

### JWT-Based Authentication
- **Access tokens** - 120-minute expiration
- **Refresh tokens** - 7-day expiration with HTTP-only cookies
- **Role-based access** - User and admin roles
- **Secure cookies** - Production-ready configuration

### Best Practices
- Store API keys in `.env` (never commit to git)
- Use strong `JWT_SECRET` in production
- Enable `COOKIE_SECURE=true` for HTTPS
- Specify allowed `FRONTEND_ORIGINS`
- Rotate API keys regularly

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/

# Specific test categories
python tests/test_enhanced_search.py
python tests/test_brave_search_integration.py
python tests/test_aws_integration.py
python tests/test_citation_system.py

# Performance benchmarks
python demos/performance_benchmark.py

# Quick validation
./scripts/test_quick_start.sh
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework
- **OpenAI/Azure OpenAI** - AI capabilities
- **yfinance** - Stock data API
- **ChromaDB** - Vector database
- **Brave Search** - High-quality web search
- **LocalStack** - Local AWS development

---

**Built with ❤️ for financial analysis and AI-powered insights**
