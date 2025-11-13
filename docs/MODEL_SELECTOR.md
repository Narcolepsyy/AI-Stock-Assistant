# Enhanced Model Selector Documentation

## Overview
The application now supports an advanced model selection system that allows you to choose between your Azure-deployed GPT OSS 120B model and other OpenAI models (GPT-5, O3, GPT-4.1) seamlessly.

## Configuration

### Environment Variables
Set these environment variables to configure your models:

```bash
# Azure OpenAI Configuration (for your GPT OSS 120B deployment)
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_OSS_120B=your-gpt-oss-120b-deployment-name

# Standard OpenAI Configuration (for GPT-5, O3, GPT-4.1)
OPENAI_API_KEY=your_openai_api_key
```

## Available Models

The system is configured with the following models:

### Azure Models (Your Deployment)
- **gpt-oss-120b** (Default) - Your Azure-deployed GPT OSS 120B model
  - Provider: Azure OpenAI
  - Deployment: Uses `AZURE_OPENAI_DEPLOYMENT_OSS_120B`
  - Description: Large open-source model deployed on Azure

### OpenAI Models
- **gpt-5** - Latest GPT-5 model from OpenAI
- **o3** - OpenAI O3 reasoning model  
- **gpt-4.1** - Enhanced GPT-4.1 model
- **gpt-4o** - GPT-4 Omni multimodal model
- **gpt-4o-mini** - Faster, cost-effective GPT-4o variant

## API Endpoints

### 1. List Available Models
```http
GET /chat/models
```

**Response:**
```json
{
  "default": "gpt-oss-120b",
  "available": [
    {
      "id": "gpt-oss-120b",
      "name": "GPT OSS 120B (Azure)",
      "description": "Large open-source model deployed on Azure",
      "provider": "azure",
      "available": true,
      "default": true
    },
    {
      "id": "gpt-5",
      "name": "GPT-5",
      "description": "Latest GPT-5 model from OpenAI",
      "provider": "openai",
      "available": true,
      "default": false
    }
  ],
  "total_count": 6,
  "available_count": 2
}
```

### 2. Chat with Model Selection
```http
POST /chat
```

**Request Body:**
```json
{
  "prompt": "What's the current price of AAPL?",
  "deployment": "gpt-oss-120b",
  "system_prompt": "You are a helpful financial assistant.",
  "conversation_id": "optional-conversation-id"
}
```

**Available deployment values:**
- `gpt-oss-120b` (your Azure deployment - default)
- `gpt-5` (OpenAI)
- `o3` (OpenAI)  
- `gpt-4.1` (OpenAI)
- `gpt-4o` (OpenAI)
- `gpt-4o-mini` (OpenAI)

### 3. Model Status and Debug Info
```http
GET /api/models/status
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "azure_client_available": true,
    "openai_client_available": true,
    "configured_models": {
      "gpt-oss-120b": {
        "provider": "azure",
        "deployment": "your-deployment-name",
        "configured": true,
        "provider_available": true
      }
    }
  }
}
```

## How It Works

### 1. Smart Provider Selection
- The system automatically chooses the appropriate client (Azure or OpenAI) based on the selected model
- Your GPT OSS 120B model uses Azure OpenAI client
- GPT-5, O3, GPT-4.1 use standard OpenAI client

### 2. Fallback Mechanism
- If Azure client is unavailable, OpenAI models are used as fallback
- If OpenAI client is unavailable, Azure deployment is used as fallback
- Ensures high availability and seamless user experience

### 3. Cost Optimization
- Conversation management with token limits
- Tool result truncation to control costs
- Smart context summarization for long conversations

## Usage Examples

### Frontend Integration
```javascript
// Get available models
const models = await fetch('/chat/models').then(r => r.json());

// Use your Azure GPT OSS 120B (default)
const response1 = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "Analyze TSLA stock performance",
    deployment: "gpt-oss-120b"
  })
});

// Use OpenAI GPT-5
const response2 = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "Explain portfolio diversification",
    deployment: "gpt-5"
  })
});
```

### Python Client
```python
import requests

# List available models
models = requests.get('http://localhost:8000/chat/models').json()
print(f"Default model: {models['default']}")

# Chat with your Azure GPT OSS 120B
response = requests.post('http://localhost:8000/chat', json={
    'prompt': 'What are the key financial ratios for stock analysis?',
    'deployment': 'gpt-oss-120b'
})

# Chat with OpenAI O3
response = requests.post('http://localhost:8000/chat', json={
    'prompt': 'Solve this complex financial calculation...',
    'deployment': 'o3'
})
```

## Benefits

### 1. **Flexibility**
- Choose the best model for each specific task
- Your powerful GPT OSS 120B for complex analysis
- OpenAI models for different reasoning patterns

### 2. **Performance**
- Azure deployment for low-latency access to your custom model
- OpenAI's optimized infrastructure for their models
- Automatic failover between providers

### 3. **Cost Control**
- Use your Azure deployment credits efficiently
- Switch to OpenAI models when needed
- Token optimization across all models

### 4. **Seamless Integration**
- Same API interface for all models
- No code changes needed in your frontend
- Backward compatible with existing implementations

## Troubleshooting

### Model Not Available
If a model shows `"available": false`:

1. **For Azure models**: Check `AZURE_OPENAI_DEPLOYMENT_OSS_120B` environment variable
2. **For OpenAI models**: Verify `OPENAI_API_KEY` is set correctly
3. **For all models**: Ensure the respective API credentials are valid

### Deployment Issues
- Use `/api/models/status` endpoint to debug configuration
- Check application logs for detailed error messages
- Verify environment variables are properly set

Your enhanced model selector is now fully functional and ready for production use! 🚀
