# AWS + LocalStack Integration Quick Reference

## 🚀 Quick Start

### 1. Start LocalStack
```bash
# Run the automated setup script
./setup_localstack.sh

# Or manually:
docker-compose up -d localstack
```

### 2. Verify Setup
```bash
# Check LocalStack health
curl http://localhost:4566/_localstack/health

# List all resources
awslocal s3 ls                          # S3 buckets
awslocal dynamodb list-tables           # DynamoDB tables
awslocal sqs list-queues                # SQS queues
awslocal sns list-topics                # SNS topics
awslocal cloudwatch list-metrics        # CloudWatch metrics
```

### 3. Configure Application
Add to `.env`:
```bash
USE_LOCALSTACK=true
AWS_ENDPOINT_URL=http://localhost:4566
S3_ENABLED=true
DYNAMODB_ENABLED=true
SQS_ENABLED=true
CLOUDWATCH_ENABLED=true
```

### 4. Run Application
```bash
python main.py
# Or with Docker:
docker-compose up
```

---

## 📋 AWS Services Overview

| Service | Purpose | Resource Name | Port |
|---------|---------|---------------|------|
| **S3** | Knowledge base storage | `stocktool-knowledge` | 4566 |
| **DynamoDB** | Conversations & cache | `stocktool-conversations`, `stocktool-stock-cache` | 4566 |
| **SQS** | Async task queue | `stocktool-analysis-queue` | 4566 |
| **SNS** | Notifications | `stocktool-notifications` | 4566 |
| **CloudWatch** | Metrics & monitoring | `StockTool` namespace | 4566 |
| **Lambda** | Scheduled tasks | `stock-data-updater` | 4566 |

---

## 🔧 Common Operations

### S3 Operations
```bash
# Upload file
awslocal s3 cp knowledge/file.md s3://stocktool-knowledge/

# Download file
awslocal s3 cp s3://stocktool-knowledge/file.md ./

# List files
awslocal s3 ls s3://stocktool-knowledge/ --recursive

# Sync directory
awslocal s3 sync knowledge/ s3://stocktool-knowledge/
```

### DynamoDB Operations
```bash
# Scan conversations
awslocal dynamodb scan \
  --table-name stocktool-conversations \
  --limit 10

# Get specific item
awslocal dynamodb get-item \
  --table-name stocktool-conversations \
  --key '{"conversation_id": {"S": "abc123"}, "timestamp": {"N": "1696262400000"}}'

# Delete item
awslocal dynamodb delete-item \
  --table-name stocktool-conversations \
  --key '{"conversation_id": {"S": "abc123"}, "timestamp": {"N": "1696262400000"}}'
```

### SQS Operations
```bash
# Send message
awslocal sqs send-message \
  --queue-url http://localhost:4566/000000000000/stocktool-analysis-queue \
  --message-body '{"task": "analyze", "symbol": "AAPL"}'

# Receive messages
awslocal sqs receive-message \
  --queue-url http://localhost:4566/000000000000/stocktool-analysis-queue \
  --max-number-of-messages 10

# Purge queue
awslocal sqs purge-queue \
  --queue-url http://localhost:4566/000000000000/stocktool-analysis-queue
```

### CloudWatch Operations
```bash
# List metrics
awslocal cloudwatch list-metrics --namespace StockTool

# Get metric statistics
awslocal cloudwatch get-metric-statistics \
  --namespace StockTool \
  --metric-name APILatency \
  --start-time $(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S') \
  --end-time $(date -u '+%Y-%m-%dT%H:%M:%S') \
  --period 300 \
  --statistics Average,Maximum,Minimum

# View dashboard
awslocal cloudwatch get-dashboard --dashboard-name stocktool-metrics
```

---

## 🐍 Python Code Examples

### Using S3
```python
from app.services.aws import get_s3_service

s3 = get_s3_service()

# Upload
s3.upload_file('knowledge/doc.pdf', 'docs/doc.pdf')

# Download
content = s3.download_fileobj('docs/doc.pdf')

# List
files = s3.list_files(prefix='docs/')

# Sync directory
stats = s3.sync_directory('knowledge/', 'knowledge/')
```

### Using DynamoDB
```python
from app.services.aws import get_dynamodb_service

# Conversations
db = get_dynamodb_service('conversation')
db.put_conversation(
    'conv123',
    messages=[{'role': 'user', 'content': 'Hello'}],
    user_id='user456'
)
messages = db.get_conversation('conv123')

# Cache
cache = get_dynamodb_service('cache')
cache.put_cache_item('AAPL', stock_data, ttl_seconds=300)
data = cache.get_cache_item('AAPL')
```

### Using SQS
```python
from app.services.aws import get_sqs_service

sqs = get_sqs_service()

# Send task
sqs.send_message({
    'task': 'portfolio_analysis',
    'symbols': ['AAPL', 'GOOGL'],
    'user_id': 'user456'
})

# Process messages
def handle_task(message):
    print(f"Processing: {message}")
    return True

stats = sqs.process_messages(handle_task, max_messages=10)
```

### Using CloudWatch
```python
from app.services.aws import get_cloudwatch_service

cw = get_cloudwatch_service()

# Track metrics
cw.track_api_latency('/chat', latency_ms=250, status_code=200)
cw.track_tool_execution('get_stock_quote', execution_time_ms=120)
cw.track_cache_hit_rate('stock_cache', hit=True)

# Get summary
summary = cw.get_performance_summary(hours=1)
```

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_aws_services.py -v
```

### Integration Tests
```bash
# Start LocalStack first
docker-compose up -d localstack

# Run tests
pytest tests/integration/test_aws_integration.py -v
```

### Manual Testing
```bash
# Test S3
python -c "from app.services.aws import get_s3_service; \
  s3 = get_s3_service(); \
  s3.upload_file('README.md', 'test.md'); \
  print(s3.list_files())"

# Test DynamoDB
python -c "from app.services.aws import get_dynamodb_service; \
  db = get_dynamodb_service('conversation'); \
  db.put_conversation('test', [{'role': 'user', 'content': 'hi'}]); \
  print(db.get_conversation('test'))"
```

---

## 🔍 Monitoring & Debugging

### View Logs
```bash
# Application logs
docker-compose logs -f app

# LocalStack logs
docker-compose logs -f localstack

# Lambda logs
awslocal logs tail /aws/lambda/stock-data-updater --follow
```

### Debug Issues
```bash
# Check LocalStack services
curl http://localhost:4566/_localstack/health | jq

# Test connectivity
curl http://localhost:4566/_localstack/init

# View LocalStack diagnostics
docker-compose exec localstack curl localhost:4566/_localstack/diagnostics
```

---

## 📊 Performance Metrics

Access metrics via CloudWatch:
```bash
# API latency (last hour)
awslocal cloudwatch get-metric-statistics \
  --namespace StockTool \
  --metric-name APILatency \
  --start-time $(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S') \
  --end-time $(date -u '+%Y-%m-%dT%H:%M:%S') \
  --period 300 \
  --statistics Average

# Cache hit rate
awslocal cloudwatch get-metric-statistics \
  --namespace StockTool \
  --metric-name CacheHitRate \
  --start-time $(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S') \
  --end-time $(date -u '+%Y-%m-%dT%H:%M:%S') \
  --period 300 \
  --statistics Average
```

---

## 🚨 Troubleshooting

### LocalStack won't start
```bash
# Check Docker is running
docker ps

# Check ports
netstat -an | grep 4566

# Restart
docker-compose restart localstack

# View detailed logs
docker-compose logs localstack | tail -100
```

### Services not initialized
```bash
# Re-run init script
docker-compose exec localstack bash /etc/localstack/init/ready.d/01-init-resources.sh

# Or restart with fresh state
docker-compose down -v
docker-compose up -d
```

### Connection errors
```bash
# Verify endpoint
echo $AWS_ENDPOINT_URL

# Test connectivity
curl http://localhost:4566/_localstack/health

# Check firewall
sudo ufw status
```

---

## 🎓 Learning Resources

### AWS Services
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/)
- [SQS Developer Guide](https://docs.aws.amazon.com/sqs/)
- [CloudWatch User Guide](https://docs.aws.amazon.com/cloudwatch/)

### LocalStack
- [LocalStack Docs](https://docs.localstack.cloud/)
- [LocalStack Pro Features](https://docs.localstack.cloud/references/coverage/)
- [LocalStack CLI Reference](https://docs.localstack.cloud/references/cli/)

---

## 💡 Tips & Best Practices

1. **Use LocalStack for Development**
   - No AWS costs during development
   - Fast iteration without cloud latency
   - Easy reset: `docker-compose down -v && docker-compose up`

2. **Monitor Resource Usage**
   - Check CloudWatch metrics regularly
   - Set up alarms for critical metrics
   - Review DynamoDB TTL effectiveness

3. **Optimize Costs (Production)**
   - Use DynamoDB on-demand pricing
   - Enable S3 Intelligent-Tiering
   - Set appropriate TTL values
   - Use SQS batching

4. **Security Best Practices**
   - Use IAM roles in production
   - Enable encryption at rest
   - Use VPC endpoints
   - Rotate credentials regularly

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **LocalStack**: Check [LocalStack Discuss](https://discuss.localstack.cloud/)
- **AWS SDK**: [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---

**Ready to show off your AWS skills!** 🚀
