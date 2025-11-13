# Lambda Functions and CloudWatch Dashboards Guide

Complete guide for deploying AWS Lambda functions and building CloudWatch dashboards for monitoring.

## Table of Contents
1. [Lambda Functions Overview](#lambda-functions-overview)
2. [Deploying Lambda Functions](#deploying-lambda-functions)
3. [CloudWatch Dashboards](#cloudwatch-dashboards)
4. [Monitoring and Alerts](#monitoring-and-alerts)
5. [Testing and Debugging](#testing-and-debugging)
6. [Production Deployment](#production-deployment)

---

## Lambda Functions Overview

### Stock Updater Lambda

**Function**: `stocktool-stock-updater`

**Purpose**: Automatically updates stock data daily at market close.

**Features**:
- Fetches real-time stock quotes for popular symbols
- Stores data in DynamoDB with TTL
- Sends SNS notifications with update summary
- Handles errors gracefully with detailed logging

**Trigger**: EventBridge (CloudWatch Events) - Daily at 4:30 PM ET

**Configuration**:
- Runtime: Python 3.11
- Memory: 256 MB
- Timeout: 60 seconds
- Environment Variables:
  - `AWS_ENDPOINT_URL` - LocalStack endpoint
  - `DYNAMODB_TABLE_CACHE` - Cache table name
  - `SNS_TOPIC_ARN` - Notification topic ARN

**Default Symbols**:
```
AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, JPM, V, JNJ
^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
```

---

## Deploying Lambda Functions

### Prerequisites

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

### Method 1: Automated Deployment (Recommended)

```bash
# Deploy all Lambda functions
./scripts/deploy_lambda.sh
```

**What it does**:
1. ✅ Checks LocalStack is running
2. ✅ Creates deployment package with dependencies
3. ✅ Deploys Lambda function to LocalStack
4. ✅ Creates EventBridge schedule (daily at 4:30 PM ET)
5. ✅ Configures Lambda permissions
6. ✅ Tests invocation with sample data

**Output**:
```
╔════════════════════════════════════════════════╗
║     Deploy Lambda Functions to LocalStack     ║
╚════════════════════════════════════════════════╝

▶ Checking LocalStack...
✓ LocalStack is running

▶ Deploying Lambda: stocktool-stock-updater
✓ Lambda deployed: stocktool-stock-updater

▶ Creating EventBridge schedule...
✓ EventBridge schedule created

▶ Testing Lambda invocation...
✓ Lambda invocation successful

╔════════════════════════════════════════════════╗
║            Deployment Summary                  ║
╚════════════════════════════════════════════════╝

✓ Lambda Function: stocktool-stock-updater
✓ EventBridge Rule: stocktool-daily-update
✓ Schedule: Daily at 4:30 PM ET (market close)

🎉 Lambda deployment complete!
```

### Method 2: Manual Deployment

#### Step 1: Create Deployment Package

```bash
# Navigate to Lambda function directory
cd lambda_functions/stock_updater

# Create temporary directory
mkdir -p /tmp/lambda-package

# Copy function code
cp handler.py /tmp/lambda-package/

# Install dependencies
pip install -r requirements.txt -t /tmp/lambda-package/

# Create ZIP package
cd /tmp/lambda-package
zip -r /tmp/stock-updater.zip .
```

#### Step 2: Deploy to LocalStack

```bash
# Create Lambda function
aws lambda create-function \
    --function-name stocktool-stock-updater \
    --runtime python3.11 \
    --role arn:aws:iam::000000000000:role/lambda-role \
    --handler handler.lambda_handler \
    --zip-file fileb:///tmp/stock-updater.zip \
    --timeout 60 \
    --memory-size 256 \
    --endpoint-url http://localhost:4566 \
    --environment "Variables={
        AWS_ENDPOINT_URL=http://localhost:4566,
        DYNAMODB_TABLE_CACHE=stocktool-stock-cache,
        SNS_TOPIC_ARN=arn:aws:sns:us-east-1:000000000000:stocktool-notifications
    }"
```

#### Step 3: Create EventBridge Schedule

```bash
# Create rule
aws events put-rule \
    --name stocktool-daily-update \
    --schedule-expression "cron(30 20 * * ? *)" \
    --endpoint-url http://localhost:4566

# Add Lambda permission
aws lambda add-permission \
    --function-name stocktool-stock-updater \
    --statement-id AllowEventBridgeInvoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --endpoint-url http://localhost:4566

# Add target
aws events put-targets \
    --rule stocktool-daily-update \
    --targets "Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:stocktool-stock-updater" \
    --endpoint-url http://localhost:4566
```

### Method 3: Python Test Script

```bash
# Test Lambda locally before deployment
python scripts/test_lambda.py
```

---

## CloudWatch Dashboards

### Creating Dashboards

#### Automated Creation (Recommended)

```bash
# Create dashboard with all widgets and alarms
./scripts/create_cloudwatch_dashboard.sh
```

**Dashboard includes**:
1. **API Latency** - Average and p99 response times
2. **Tool Execution Time** - Tool call performance metrics
3. **Cache Hit Rate** - Caching efficiency percentage
4. **Model Token Usage** - Total tokens consumed by AI models
5. **Model Latency** - AI model response times
6. **Lambda Metrics** - Invocations, errors, throttles
7. **Lambda Duration** - Function execution times
8. **SQS Queue Metrics** - Message processing stats
9. **DynamoDB Capacity** - Read/write capacity usage
10. **Recent Errors/Warnings** - Log query for issues

#### Manual Dashboard Creation

```bash
# Get dashboard JSON template
aws cloudwatch get-dashboard \
    --dashboard-name StockTool-Monitoring \
    --endpoint-url http://localhost:4566 \
    > dashboard.json

# Modify dashboard.json as needed

# Update dashboard
aws cloudwatch put-dashboard \
    --dashboard-name StockTool-Monitoring \
    --dashboard-body file://dashboard.json \
    --endpoint-url http://localhost:4566
```

### Dashboard Widgets

#### Widget 1: API Latency

```json
{
  "type": "metric",
  "properties": {
    "metrics": [
      [ "StockTool", "APILatency", { "stat": "Average" } ],
      [ "...", { "stat": "p99" } ]
    ],
    "period": 300,
    "stat": "Average",
    "title": "API Latency",
    "yAxis": {
      "left": { "label": "Milliseconds" }
    }
  }
}
```

#### Widget 2: Cache Performance

```json
{
  "type": "metric",
  "properties": {
    "metrics": [
      [ "StockTool", "CacheHitRate", { "stat": "Average" } ]
    ],
    "period": 300,
    "title": "Cache Hit Rate",
    "yAxis": {
      "left": {
        "label": "Percentage",
        "min": 0,
        "max": 100
      }
    }
  }
}
```

#### Widget 3: Lambda Performance

```json
{
  "type": "metric",
  "properties": {
    "metrics": [
      [ "AWS/Lambda", "Invocations", { "stat": "Sum" } ],
      [ ".", "Errors", { "stat": "Sum" } ],
      [ ".", "Duration", { "stat": "Average" } ]
    ],
    "period": 300,
    "title": "Lambda Metrics"
  }
}
```

---

## Monitoring and Alerts

### CloudWatch Alarms

Three critical alarms are configured automatically:

#### 1. High API Latency

```bash
# Alarm when API latency > 2 seconds
aws cloudwatch put-metric-alarm \
    --alarm-name "StockTool-HighAPILatency" \
    --metric-name "APILatency" \
    --namespace "StockTool" \
    --statistic "Average" \
    --threshold 2000 \
    --comparison-operator "GreaterThanThreshold" \
    --evaluation-periods 2 \
    --endpoint-url http://localhost:4566
```

**Action**: Investigate slow endpoints, check external API latency

#### 2. Low Cache Hit Rate

```bash
# Alarm when cache hit rate < 50%
aws cloudwatch put-metric-alarm \
    --alarm-name "StockTool-LowCacheHitRate" \
    --metric-name "CacheHitRate" \
    --namespace "StockTool" \
    --statistic "Average" \
    --threshold 50 \
    --comparison-operator "LessThanThreshold" \
    --evaluation-periods 2 \
    --endpoint-url http://localhost:4566
```

**Action**: Review TTL settings, increase cache size

#### 3. Lambda Errors

```bash
# Alarm when Lambda has > 5 errors in 5 minutes
aws cloudwatch put-metric-alarm \
    --alarm-name "StockTool-LambdaErrors" \
    --metric-name "Errors" \
    --namespace "AWS/Lambda" \
    --statistic "Sum" \
    --threshold 5 \
    --comparison-operator "GreaterThanThreshold" \
    --evaluation-periods 1 \
    --endpoint-url http://localhost:4566
```

**Action**: Check Lambda logs, verify stock data sources

### SNS Notifications

Configure SNS topic for alarm notifications:

```bash
# Subscribe email to notifications topic
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:000000000000:stocktool-notifications \
    --protocol email \
    --notification-endpoint your-email@example.com \
    --endpoint-url http://localhost:4566
```

---

## Testing and Debugging

### Manual Lambda Invocation

```bash
# Invoke with default symbols
aws lambda invoke \
    --function-name stocktool-stock-updater \
    --endpoint-url http://localhost:4566 \
    output.json

# View response
cat output.json | jq '.'
```

### Invoke with Custom Event

```bash
# Create test event
cat > test-event.json <<EOF
{
  "symbols": ["AAPL", "GOOGL", "MSFT"]
}
EOF

# Invoke
aws lambda invoke \
    --function-name stocktool-stock-updater \
    --payload file://test-event.json \
    --endpoint-url http://localhost:4566 \
    output.json
```

### View Lambda Logs

```bash
# Tail logs in real-time
aws logs tail /aws/lambda/stocktool-stock-updater \
    --follow \
    --endpoint-url http://localhost:4566

# Get recent logs
aws logs tail /aws/lambda/stocktool-stock-updater \
    --since 1h \
    --endpoint-url http://localhost:4566
```

### Test with Python Script

```bash
# Comprehensive Lambda testing
python scripts/test_lambda.py
```

**Output**:
```
============================================================
  Lambda Function Testing Tool
============================================================

✓ Connected to LocalStack

============================================================
  Available Lambda Functions
============================================================

  • stocktool-stock-updater
    Runtime: python3.11
    Handler: handler.lambda_handler
    Memory: 256 MB
    Timeout: 60 seconds

============================================================
  Running Lambda Tests
============================================================

▶ Invoking Lambda: stocktool-stock-updater
  ✓ Status Code: 200
  ✓ Response:
  {
    "statusCode": 200,
    "body": {
      "total": 3,
      "successful": 3,
      "failed": 0,
      "errors": []
    }
  }
```

### Local Testing (Without Deployment)

```bash
# Test Lambda handler locally
cd lambda_functions/stock_updater
python handler.py
```

---

## Production Deployment

### Step 1: Prepare for Production

1. **Update Environment Variables**:
   ```bash
   # Remove LocalStack endpoint
   unset AWS_ENDPOINT_URL
   
   # Set real AWS credentials
   export AWS_ACCESS_KEY_ID=your-real-key
   export AWS_SECRET_ACCESS_KEY=your-real-secret
   export AWS_DEFAULT_REGION=us-east-1
   ```

2. **Update Function Code**:
   Remove LocalStack-specific configurations from `handler.py`

3. **Create IAM Role**:
   ```bash
   aws iam create-role \
       --role-name StockToolLambdaRole \
       --assume-role-policy-document file://lambda-trust-policy.json
   
   aws iam attach-role-policy \
       --role-name StockToolLambdaRole \
       --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
   ```

### Step 2: Deploy to AWS

```bash
# Package and deploy
aws lambda create-function \
    --function-name stocktool-stock-updater \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/StockToolLambdaRole \
    --handler handler.lambda_handler \
    --zip-file fileb://stock-updater.zip \
    --timeout 60 \
    --memory-size 256 \
    --environment "Variables={
        DYNAMODB_TABLE_CACHE=stocktool-stock-cache,
        SNS_TOPIC_ARN=arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:stocktool-notifications
    }"
```

### Step 3: Configure EventBridge

```bash
# Create schedule in production
aws events put-rule \
    --name stocktool-daily-update \
    --schedule-expression "cron(30 20 * * ? *)"

# Add permissions
aws lambda add-permission \
    --function-name stocktool-stock-updater \
    --statement-id AllowEventBridgeInvoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com

# Add target
aws events put-targets \
    --rule stocktool-daily-update \
    --targets "Id=1,Arn=arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:stocktool-stock-updater"
```

### Step 4: Create Production Dashboard

```bash
# Modify dashboard JSON for production
# Remove LocalStack references
# Update ARNs and region

aws cloudwatch put-dashboard \
    --dashboard-name StockTool-Production \
    --dashboard-body file://production-dashboard.json
```

---

## Troubleshooting

### Lambda Not Deploying

**Problem**: `ResourceNotFoundException` when deploying

**Solution**:
```bash
# Verify LocalStack is running
curl http://localhost:4566/_localstack/health

# Check Lambda service is enabled
docker compose logs localstack | grep lambda

# Restart LocalStack
docker compose restart localstack
```

### Lambda Timing Out

**Problem**: Function exceeds timeout

**Solutions**:
1. Increase timeout:
   ```bash
   aws lambda update-function-configuration \
       --function-name stocktool-stock-updater \
       --timeout 120 \
       --endpoint-url http://localhost:4566
   ```

2. Optimize code - fetch fewer symbols per invocation
3. Use concurrent Lambda invocations

### EventBridge Not Triggering

**Problem**: Scheduled Lambda not executing

**Solutions**:
```bash
# Verify rule exists
aws events list-rules --endpoint-url http://localhost:4566

# Check rule is enabled
aws events describe-rule --name stocktool-daily-update --endpoint-url http://localhost:4566

# Verify target is set
aws events list-targets-by-rule --rule stocktool-daily-update --endpoint-url http://localhost:4566

# Enable rule if disabled
aws events enable-rule --name stocktool-daily-update --endpoint-url http://localhost:4566
```

### Dashboard Not Showing Data

**Problem**: Metrics not appearing in dashboard

**Solutions**:
1. Generate metrics by running application:
   ```bash
   python main.py
   # Make API calls to generate metrics
   ```

2. Verify metrics exist:
   ```bash
   aws cloudwatch list-metrics \
       --namespace StockTool \
       --endpoint-url http://localhost:4566
   ```

3. Check metric period (may need to wait for data points)

---

## Quick Reference

### Essential Commands

```bash
# Deploy Lambda
./scripts/deploy_lambda.sh

# Create Dashboard
./scripts/create_cloudwatch_dashboard.sh

# Test Lambda
python scripts/test_lambda.py

# Invoke Lambda
aws lambda invoke --function-name stocktool-stock-updater --endpoint-url http://localhost:4566 output.json

# View Logs
aws logs tail /aws/lambda/stocktool-stock-updater --follow --endpoint-url http://localhost:4566

# List Dashboards
aws cloudwatch list-dashboards --endpoint-url http://localhost:4566

# Describe Alarms
aws cloudwatch describe-alarms --endpoint-url http://localhost:4566
```

### File Locations

- Lambda Code: `lambda_functions/stock_updater/handler.py`
- Deploy Script: `scripts/deploy_lambda.sh`
- Dashboard Script: `scripts/create_cloudwatch_dashboard.sh`
- Test Script: `scripts/test_lambda.py`

---

## Next Steps

1. ✅ Deploy Lambda functions
2. ✅ Create CloudWatch dashboard
3. ✅ Set up alarms and notifications
4. ✅ Test end-to-end workflow
5. ✅ Monitor performance metrics
6. ✅ Optimize based on dashboard data
7. ✅ Prepare for production deployment

**For Production**: See production deployment section above

**For Development**: Continue testing with LocalStack

🎉 **You're all set!** Your Lambda functions and monitoring are ready!
