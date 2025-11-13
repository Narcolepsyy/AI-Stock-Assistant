# LocalStack with GitHub Student Plan - Quick Reference

## ✅ Your Setup is Complete!

Your LocalStack Pro is successfully activated with your GitHub Education Student Plan.

## Configuration

### Docker Compose Image
```yaml
image: localstack/localstack-pro:latest  # Pro image (not community)
```

### Auth Token
```bash
# In .env file
LOCALSTACK_AUTH_TOKEN="ls-your-token-here"
```

## Verification Commands

### Check LocalStack Status
```bash
# Check if running
docker compose ps

# Check health
curl http://localhost:4566/_localstack/health | jq

# Check license activation
curl http://localhost:4566/_localstack/info | jq
```

**Expected Output:**
```json
{
  "edition": "pro",
  "is_license_activated": true
}
```

### Verify AWS Resources
```bash
python scripts/verify_aws_resources.py
```

### Run Integration Tests
```bash
PYTHONPATH=$(pwd) python tests/test_aws_integration.py
```

## What's Included in Your GitHub Student Plan

### ✅ Pro Features
- **Persistence** - Data survives restarts
- **CloudWatch Logs** - Full Lambda logging
- **100+ AWS Services** - Extended service support
- **IAM/Security** - Advanced authentication
- **Advanced Lambda** - Docker executor, layers, etc.
- **Better Debugging** - Enhanced error messages
- **Priority Support** - Access to support channels

### 🆓 Free Tier Limitations
Your GitHub Student Plan gives you **FREE access** to Pro features while you're a student!

## Common Commands

### Start LocalStack
```bash
docker compose up -d localstack
```

### Stop LocalStack
```bash
docker compose down
```

### View Logs
```bash
docker compose logs -f localstack
```

### Restart LocalStack
```bash
docker compose restart localstack
```

### Reset All Data (Nuclear Option)
```bash
docker compose down -v  # Remove volumes
docker compose up -d localstack
```

## Troubleshooting

### License Not Activating?

1. **Check token is loaded:**
   ```bash
   docker exec stocktool-localstack env | grep LOCALSTACK_AUTH_TOKEN
   ```

2. **Verify using Pro image:**
   ```bash
   docker inspect stocktool-localstack | grep Image
   # Should show: localstack/localstack-pro
   ```

3. **Check logs for errors:**
   ```bash
   docker logs stocktool-localstack | grep -i auth
   ```

### Services Not Running?

```bash
# Check health endpoint
curl http://localhost:4566/_localstack/health

# Enable service in docker-compose.yml
environment:
  - SERVICES=s3,dynamodb,sqs,sns,lambda,cloudwatch,logs,...
```

### Port Already in Use?

```bash
# Check what's using port 4566
sudo lsof -i :4566

# Stop conflicting container
docker stop $(docker ps -q --filter "ancestor=localstack/localstack")
```

## AWS Service Endpoints

All services use the same endpoint:
```
http://localhost:4566
```

### Example boto3 Configuration
```python
import boto3

# LocalStack client
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)
```

## Resources Created in This Project

### S3 Buckets
- `stocktool-knowledge` - RAG knowledge base storage

### DynamoDB Tables
- `stocktool-conversations` - Chat conversation history (24h TTL)
- `stocktool-stock-cache` - Stock data cache (5min TTL)

### SQS Queues
- `stocktool-analysis-queue` - Async analysis tasks

### SNS Topics
- `stocktool-notifications` - System notifications

### CloudWatch Metrics
Namespace: `StockTool`
- API latency
- Tool execution time
- Cache hit rate
- Model token usage
- Model latency

## Production Deployment

When moving to real AWS:

1. **Update .env:**
   ```bash
   USE_LOCALSTACK=false
   AWS_ENDPOINT_URL=  # Leave empty
   AWS_ACCESS_KEY_ID=your-real-key
   AWS_SECRET_ACCESS_KEY=your-real-secret
   ```

2. **No code changes needed** - Application automatically switches to real AWS

3. **Create resources in AWS:**
   - Use CloudFormation or Terraform
   - Match the resource names from LocalStack

## Getting Help

### LocalStack Documentation
- Website: https://docs.localstack.cloud/
- Pro Features: https://docs.localstack.cloud/user-guide/aws/feature-coverage/
- GitHub: https://github.com/localstack/localstack

### GitHub Education
- Dashboard: https://education.github.com/
- Benefits: https://education.github.com/benefits

### Your Auth Token
- Manage at: https://app.localstack.cloud/workspace/auth-token
- Valid while you have GitHub Student Pack access

## Test Results (Latest Run)

```
============================================================
Test Summary
============================================================
S3              ✅ PASSED
DynamoDB        ✅ PASSED
SQS             ✅ PASSED
CloudWatch      ✅ PASSED

🎉 All tests passed! AWS integration is working correctly.
```

## Pro Tips

1. **Use Persistence** - Your data survives restarts (enabled by default in your setup)
2. **Check Logs** - Pro edition has better error messages
3. **Monitor Usage** - Check LocalStack dashboard for API call metrics
4. **Test Locally** - Develop everything locally before deploying to real AWS
5. **No AWS Costs** - Perfect for learning and portfolio projects!

## Next Steps

✅ LocalStack Pro is running  
✅ All services verified  
✅ Integration tests passing  

**You're ready to:**
1. Start developing with AWS services
2. Run the application: `python main.py`
3. Deploy Lambda functions
4. Create CloudWatch dashboards
5. Build your portfolio with real AWS skills

## Showcase This in Your Portfolio

This setup demonstrates:
- ✅ AWS service expertise (S3, DynamoDB, SQS, SNS, Lambda, CloudWatch)
- ✅ Infrastructure as Code (Docker Compose)
- ✅ DevOps best practices (local development environment)
- ✅ Cost optimization (no AWS bills during development)
- ✅ Testing strategy (comprehensive integration tests)
- ✅ Professional tooling (LocalStack Pro with GitHub Student benefits)

Perfect for interviews at companies looking for cloud engineers! 🚀
