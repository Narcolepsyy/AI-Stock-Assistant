# LocalStack Setup Guide

This guide walks you through setting up LocalStack with authentication for the Azure-OpenAI Stock Analysis Tool.

## Table of Contents
- [What is LocalStack?](#what-is-localstack)
- [Getting Your Auth Token](#getting-your-auth-token)
- [Configuration](#configuration)
- [Running LocalStack](#running-localstack)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## What is LocalStack?

LocalStack is a cloud service emulator that runs AWS services locally on your machine. This allows you to:
- Develop and test AWS integrations without incurring AWS costs
- Work offline without internet connectivity
- Speed up development with instant service availability
- Showcase AWS skills in your portfolio

## Getting Your Auth Token

### Option 1: Free Tier (Recommended for Development)

1. **Sign Up for LocalStack Account**
   - Go to: https://app.localstack.cloud/
   - Click "Sign up" (no credit card required)
   - Create account with email or GitHub

2. **Get Your Auth Token**
   - After login, go to: https://app.localstack.cloud/workspace/auth-token
   - Copy your auth token (looks like: `ls-1234abcd-5678-efgh-9012-ijklmnop3456`)
   - This token is **free** and includes:
     - Core AWS services (S3, DynamoDB, SQS, SNS, Lambda, CloudWatch)
     - Persistence across restarts
     - Basic debugging features

3. **Add Token to .env File**
   ```bash
   # Open your .env file
   nano .env
   
   # Add your token
   LOCALSTACK_AUTH_TOKEN="ls-your-token-here"
   ```

### Option 2: Community Edition (No Auth Token)

If you prefer not to sign up, you can use LocalStack Community Edition with limited features:

1. **Leave Token Empty**
   ```bash
   LOCALSTACK_AUTH_TOKEN=""
   ```

2. **Limitations**:
   - No persistence (data lost on restart)
   - Limited Lambda support
   - No advanced debugging features
   - Some services may not work fully

**Note**: For this project, we **recommend Option 1** (Free Tier with auth token) for full functionality.

## Configuration

### 1. Environment Variables

Your `.env` file should include:

```bash
# LocalStack Configuration
LOCALSTACK_AUTH_TOKEN="ls-your-token-here"  # Get from https://app.localstack.cloud/

# AWS Configuration (for LocalStack)
USE_LOCALSTACK=true
AWS_ENDPOINT_URL=http://localhost:4566
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test

# Resource Names
S3_BUCKET_NAME=stocktool-knowledge
DYNAMODB_TABLE_CONVERSATIONS=stocktool-conversations
DYNAMODB_TABLE_CACHE=stocktool-stock-cache
SQS_QUEUE_ANALYSIS=stocktool-analysis-queue
SNS_TOPIC_NOTIFICATIONS=stocktool-notifications
```

### 2. Docker Compose Configuration

The `docker-compose.yml` file is already configured with:

```yaml
environment:
  - LOCALSTACK_AUTH_TOKEN=${LOCALSTACK_AUTH_TOKEN:-}
  - SERVICES=s3,dynamodb,sqs,sns,lambda,cloudwatch,secretsmanager,sts
  - PERSISTENCE=1
  - DEBUG=1
```

## Running LocalStack

### Method 1: Using Setup Script (Recommended)

```bash
# Run the automated setup script
./scripts/setup_localstack.sh
```

This script will:
1. Check for Docker and Docker Compose
2. Start LocalStack container
3. Wait for services to be ready
4. Initialize all AWS resources
5. Verify everything is working

### Method 2: Manual Setup

```bash
# Start LocalStack
docker-compose up -d localstack

# Check status
docker-compose ps

# View logs
docker-compose logs -f localstack

# Initialize resources (wait ~30 seconds after start)
docker-compose exec localstack bash -c "
  awslocal s3 mb s3://stocktool-knowledge
  awslocal dynamodb create-table --table-name stocktool-conversations ...
  # ... etc
"
```

### Method 3: Using Docker Directly

```bash
# Pull LocalStack image
docker pull localstack/localstack:latest

# Run LocalStack with auth token
docker run -d \
  --name stocktool-localstack \
  -p 4566:4566 \
  -e LOCALSTACK_AUTH_TOKEN=your-token-here \
  -e SERVICES=s3,dynamodb,sqs,sns,lambda,cloudwatch \
  -e PERSISTENCE=1 \
  -v localstack-data:/var/lib/localstack \
  localstack/localstack:latest
```

## Verification

### 1. Check LocalStack Health

```bash
# Check if LocalStack is running
curl http://localhost:4566/_localstack/health | jq

# Expected output:
# {
#   "services": {
#     "s3": "running",
#     "dynamodb": "running",
#     "sqs": "running",
#     ...
#   }
# }
```

### 2. Verify AWS Resources

```bash
# Run verification script
python scripts/verify_aws_resources.py
```

Expected output:
```
🔍 Verifying AWS Resources in LocalStack...
============================================================
📦 Checking S3...
  ✓ Bucket 'stocktool-knowledge' exists
✅ S3 verification passed!

🗄️  Checking DynamoDB...
  ✓ Table 'stocktool-conversations' exists
  ✓ Table 'stocktool-stock-cache' exists
✅ DynamoDB verification passed!

📬 Checking SQS...
  ✓ Queue 'stocktool-analysis-queue' exists
✅ SQS verification passed!

📢 Checking SNS...
  ✓ Topic 'stocktool-notifications' exists
✅ SNS verification passed!

🎉 All AWS resources verified successfully!
```

### 3. Test Integration

```bash
# Run integration tests
python tests/test_aws_integration.py
```

## Troubleshooting

### Issue: "Auth token required" Error

**Symptom**: LocalStack logs show authentication errors

**Solution**:
1. Verify token in `.env` file is correct
2. Check token at: https://app.localstack.cloud/workspace/auth-token
3. Ensure no extra spaces or quotes in token
4. Restart LocalStack: `docker-compose restart localstack`

### Issue: Services Not Starting

**Symptom**: Health check shows services as "unavailable"

**Solution**:
```bash
# Check logs for errors
docker-compose logs localstack | tail -50

# Common fixes:
# 1. Increase Docker memory (4GB+ recommended)
# 2. Wait longer (can take 30-60 seconds)
# 3. Restart LocalStack
docker-compose down && docker-compose up -d
```

### Issue: Port 4566 Already in Use

**Symptom**: "port is already allocated" error

**Solution**:
```bash
# Check what's using the port
sudo lsof -i :4566

# Stop any existing LocalStack
docker stop $(docker ps -q --filter "ancestor=localstack/localstack")

# Or change port in docker-compose.yml
ports:
  - "4567:4566"  # Use different host port
```

### Issue: Resources Not Persisting

**Symptom**: Data lost after restart

**Solution**:
1. Verify `LOCALSTACK_AUTH_TOKEN` is set (persistence requires auth)
2. Check `PERSISTENCE=1` in docker-compose.yml
3. Verify volume is mounted correctly:
   ```bash
   docker volume ls | grep localstack
   docker volume inspect stocktool_localstack-data
   ```

### Issue: Lambda Functions Not Working

**Symptom**: Lambda invocation errors

**Solution**:
1. Ensure Docker socket is mounted:
   ```yaml
   volumes:
     - "/var/run/docker.sock:/var/run/docker.sock"
   ```
2. Check Lambda executor mode:
   ```yaml
   environment:
     - LAMBDA_EXECUTOR=docker-reuse
   ```
3. Verify function code is in correct location:
   ```bash
   ls -la lambda_functions/stock_updater/
   ```

## Advanced Configuration

### Enable Additional Services

Edit `docker-compose.yml`:

```yaml
environment:
  - SERVICES=s3,dynamodb,sqs,sns,lambda,cloudwatch,secretsmanager,sts,apigateway,kinesis
```

### Increase Debug Logging

```yaml
environment:
  - DEBUG=1
  - LS_LOG=debug
```

### Configure Custom Endpoints

For running LocalStack on different host/port:

```bash
# In .env
AWS_ENDPOINT_URL=http://192.168.1.100:4566

# In docker-compose.yml
ports:
  - "4566:4566"
```

## Production Deployment

When moving to production AWS:

1. **Update Environment Variables**:
   ```bash
   USE_LOCALSTACK=false
   AWS_ENDPOINT_URL=  # Leave empty for real AWS
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-real-key
   AWS_SECRET_ACCESS_KEY=your-real-secret
   ```

2. **No Code Changes Required**: The application automatically detects `USE_LOCALSTACK` and switches to real AWS services.

3. **Create Real AWS Resources**: Use CloudFormation or Terraform to create resources matching LocalStack setup.

## Resources

- **LocalStack Docs**: https://docs.localstack.cloud/
- **LocalStack Dashboard**: https://app.localstack.cloud/
- **Getting Started**: https://docs.localstack.cloud/getting-started/
- **AWS Service Coverage**: https://docs.localstack.cloud/user-guide/aws/feature-coverage/
- **LocalStack GitHub**: https://github.com/localstack/localstack
- **Community Support**: https://discuss.localstack.cloud/

## Next Steps

After successful LocalStack setup:

1. ✅ Run the application: `python main.py`
2. ✅ Test AWS integrations: `python tests/test_aws_integration.py`
3. ✅ Deploy Lambda functions: See `docs/AWS_INTEGRATION.md`
4. ✅ Set up monitoring: Configure CloudWatch dashboards
5. ✅ Review AWS architecture: See `docs/AWS_ARCHITECTURE_DIAGRAM.md`

## Support

If you encounter issues not covered here:
1. Check LocalStack logs: `docker-compose logs localstack`
2. Review LocalStack status page: https://status.localstack.cloud/
3. Search community forum: https://discuss.localstack.cloud/
4. Check project issues on GitHub
