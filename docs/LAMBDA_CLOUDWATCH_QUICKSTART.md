# Lambda & CloudWatch Quick Start

## 🚀 Deploy Lambda Functions

Deploy AWS Lambda functions to LocalStack for automated stock data updates.

### Quick Deploy

```bash
./scripts/deploy_lambda.sh
```

**What it does**:
- ✅ Deploys `stocktool-stock-updater` Lambda function
- ✅ Creates EventBridge schedule (daily at 4:30 PM ET)
- ✅ Configures Lambda permissions
- ✅ Tests invocation with sample data

**Output**:
```
╔════════════════════════════════════════════════╗
║     Deploy Lambda Functions to LocalStack     ║
╚════════════════════════════════════════════════╝

✓ LocalStack is running
✓ Lambda deployed: stocktool-stock-updater
✓ EventBridge schedule created
✓ Lambda invocation successful

🎉 Lambda deployment complete!
```

---

## 📊 Create CloudWatch Dashboard

Build comprehensive monitoring dashboard with metrics and alarms.

### Quick Create

```bash
./scripts/create_cloudwatch_dashboard.sh
```

**Dashboard includes**:
1. API Latency (average & p99)
2. Tool Execution Time
3. Cache Hit Rate
4. Model Token Usage
5. Model Latency
6. Lambda Metrics
7. Lambda Duration
8. SQS Queue Metrics
9. DynamoDB Capacity
10. Recent Errors/Warnings

**Alarms configured**:
- ⚠️ High API Latency (> 2s)
- ⚠️ Low Cache Hit Rate (< 50%)
- ⚠️ Lambda Errors (> 5 in 5min)

---

## 🧪 Test Lambda Functions

Test Lambda functions locally before production.

### Quick Test

```bash
python scripts/test_lambda.py
```

**What it does**:
- Lists all Lambda functions
- Invokes with test data
- Shows response and logs
- Validates execution

---

## 📖 Full Documentation

See **`docs/LAMBDA_AND_CLOUDWATCH_GUIDE.md`** for complete guide including:
- Lambda function details
- Manual deployment steps
- Dashboard customization
- Monitoring and alerts
- Troubleshooting
- Production deployment

---

## 🛠️ Manual Commands

### Invoke Lambda

```bash
aws lambda invoke \
    --function-name stocktool-stock-updater \
    --endpoint-url http://localhost:4566 \
    output.json

cat output.json | jq '.'
```

### View Logs

```bash
aws logs tail /aws/lambda/stocktool-stock-updater \
    --follow \
    --endpoint-url http://localhost:4566
```

### List Dashboards

```bash
aws cloudwatch list-dashboards \
    --endpoint-url http://localhost:4566
```

### Check Alarms

```bash
aws cloudwatch describe-alarms \
    --endpoint-url http://localhost:4566
```

---

## ✅ Prerequisites

1. **LocalStack Running**:
   ```bash
   docker compose up -d localstack
   ```

2. **AWS CLI Installed**:
   ```bash
   pip install awscli
   ```

3. **Dependencies Installed**:
   ```bash
   pip install boto3 yfinance
   ```

---

## 🎯 Next Steps

After deployment:

1. **Run application** to generate metrics:
   ```bash
   python main.py
   ```

2. **Make API calls** to populate dashboard

3. **View metrics** in CloudWatch

4. **Set up SNS** for alarm notifications

5. **Monitor Lambda** executions

---

## 📁 File Locations

- **Lambda Code**: `lambda_functions/stock_updater/handler.py`
- **Deploy Script**: `scripts/deploy_lambda.sh`
- **Dashboard Script**: `scripts/create_cloudwatch_dashboard.sh`
- **Test Script**: `scripts/test_lambda.py`
- **Full Guide**: `docs/LAMBDA_AND_CLOUDWATCH_GUIDE.md`

---

## 🎉 Ready!

Your Lambda functions and CloudWatch monitoring are ready to deploy!

**Start with**: `./scripts/deploy_lambda.sh`
