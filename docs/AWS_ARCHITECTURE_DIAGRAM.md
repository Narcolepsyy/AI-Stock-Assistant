# AWS Cloud Architecture - Visual Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                                     │
│                    React Frontend (Port 5173)                                 │
│                    - Portfolio Dashboard                                      │
│                    - Chat Interface                                           │
│                    - Real-time Updates                                        │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ HTTPS/WebSocket
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (Port 8000)                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬─────────────┐│
│  │   Chat       │   Admin      │    RAG       │   Stock      │    Auth     ││
│  │   Router     │   Router     │   Router     │   Service    │   Service   ││
│  └──────────────┴──────────────┴──────────────┴──────────────┴─────────────┘│
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┤
│  │                      AWS SERVICE CLIENTS                                   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  │    S3    │  │ DynamoDB │  │   SQS    │  │   SNS    │  │CloudWatch│  │
│  │  │  Client  │  │  Client  │  │  Client  │  │  Client  │  │  Client  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│  └───────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────┘
│          │             │             │             │             │
└──────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────────┘
           │             │             │             │             │
           │             │             │             │             │
           ▼             ▼             ▼             ▼             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          LOCALSTACK / AWS CLOUD                               │
│                         (Port 4566 - All Services)                            │
│                                                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐ │
│  │    Amazon S3        │  │   Amazon DynamoDB   │  │    Amazon SQS        │ │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌────────────────┐ │ │
│  │  │ stocktool-    │  │  │  │ Conversations │  │  │  │ analysis-queue │ │ │
│  │  │ knowledge     │  │  │  │   Table       │  │  │  │                │ │ │
│  │  │               │  │  │  │ - conv_id (PK)│  │  │  │ - Task Queue   │ │ │
│  │  │ /knowledge/   │  │  │  │ - timestamp   │  │  │  │ - Dead Letter  │ │ │
│  │  │ /docs/        │  │  │  │ - messages    │  │  │  │ - Batch Ops    │ │ │
│  │  │ /reports/     │  │  │  │ - user_id     │  │  │  └────────────────┘ │ │
│  │  └───────────────┘  │  │  │ - TTL (24h)   │  │  │                      │ │
│  │                     │  │  └───────────────┘  │  └──────────────────────┘ │
│  │  Features:          │  │                     │                            │
│  │  • Versioning       │  │  ┌───────────────┐  │  ┌──────────────────────┐ │
│  │  • Lifecycle        │  │  │  Stock Cache  │  │  │    Amazon SNS        │ │
│  │  • Encryption       │  │  │    Table      │  │  │  ┌────────────────┐ │ │
│  │  • Presigned URLs   │  │  │ - symbol (PK) │  │  │  │ notifications  │ │ │
│  └─────────────────────┘  │  │ - data_type   │  │  │  │     Topic      │ │ │
│                            │  │ - data        │  │  │  │                │ │ │
│  ┌─────────────────────┐  │  │ - TTL (5min)  │  │  │  │ Subscribers:   │ │ │
│  │   AWS CloudWatch    │  │  └───────────────┘  │  │  │ - SQS          │ │ │
│  │  ┌───────────────┐  │  │                     │  │  │ - Email        │ │ │
│  │  │   Metrics     │  │  │  Features:          │  │  │ - Webhooks     │ │ │
│  │  │               │  │  │  • Auto-scaling     │  │  └────────────────┘ │ │
│  │  │ - APILatency  │  │  │  • Point-in-time    │  └──────────────────────┘ │
│  │  │ - ToolExecTime│  │  │  • Global Tables    │                            │
│  │  │ - CacheHitRate│  │  │  • Streams          │  ┌──────────────────────┐ │
│  │  │ - ModelTokens │  │  └─────────────────────┘  │   AWS Lambda         │ │
│  │  │ - QueueDepth  │  │                            │  ┌────────────────┐ │ │
│  │  └───────────────┘  │                            │  │ stock-updater  │ │ │
│  │                     │                            │  │                │ │ │
│  │  ┌───────────────┐  │                            │  │ Triggers:      │ │ │
│  │  │  Log Groups   │  │                            │  │ - EventBridge  │ │ │
│  │  │               │  │                            │  │ - Cron         │ │ │
│  │  │ /aws/lambda/  │  │                            │  │ - S3 Events    │ │ │
│  │  │ /stocktool/   │  │                            │  └────────────────┘ │ │
│  │  └───────────────┘  │                            └──────────────────────┘ │
│  └─────────────────────┘                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
           ▲                                                          │
           │                                                          │
           └──────────────────────────────────────────────────────────┘
                              Monitoring & Alerts


┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW EXAMPLES                                  │
└──────────────────────────────────────────────────────────────────────────────┘

1. CHAT REQUEST (Synchronous)
   User → FastAPI → OpenAI → Tools → Stock API → Cache (DynamoDB) → Response
                                                 ↓
                                           CloudWatch Metrics

2. DOCUMENT UPLOAD (Async)
   Admin → FastAPI → S3 Upload → SNS Event → SQS → RAG Indexer → ChromaDB
                        ↓
                  CloudWatch Logs

3. PORTFOLIO ANALYSIS (Background)
   User → FastAPI → SQS Message → Background Worker → Analysis → DynamoDB
                                                                      ↓
                                                              SNS Notification

4. SCHEDULED UPDATES (Lambda)
   EventBridge Cron → Lambda → yfinance API → DynamoDB Cache → SNS Summary
                                                    ↓
                                            CloudWatch Metrics


┌──────────────────────────────────────────────────────────────────────────────┐
│                        SCALABILITY PATTERNS                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ API Server 1│────┐    │ API Server 2│────┐    │ API Server N│
└─────────────┘    │    └─────────────┘    │    └─────────────┘
                   │                       │
                   ▼                       ▼
            ┌──────────────────────────────────────┐
            │         Application Load Balancer     │
            └──────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   SQS Queue      │
                    └──────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Worker 1    │  │  Worker 2    │  │  Worker N    │
    └──────────────┘  └──────────────┘  └──────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   DynamoDB       │
                    │  (Auto-scaling)  │
                    └──────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                         COST OPTIMIZATION                                     │
└──────────────────────────────────────────────────────────────────────────────┘

1. S3 Lifecycle Policies
   - Archive old docs to Glacier after 90 days
   - Delete temporary files after 7 days

2. DynamoDB TTL
   - Auto-delete conversations after 24 hours
   - Auto-delete cache after 5 minutes
   - Saves: ~$50/month in storage

3. SQS Batching
   - Batch up to 10 messages per API call
   - Reduces API costs by 90%

4. CloudWatch Log Retention
   - Keep logs for 7 days only
   - Saves: ~$20/month

5. Lambda Provisioned Concurrency
   - Only for high-traffic functions
   - Use on-demand for scheduled tasks


┌──────────────────────────────────────────────────────────────────────────────┐
│                      SECURITY ARCHITECTURE                                    │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────┐
│    User    │
└─────┬──────┘
      │ HTTPS (TLS 1.3)
      ▼
┌────────────────────────────────────────┐
│        CloudFront (CDN)                │
│  - DDoS Protection                     │
│  - Web Application Firewall (WAF)      │
└─────┬──────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────┐
│     Application Load Balancer          │
│  - SSL Termination                     │
│  - Rate Limiting                       │
└─────┬──────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────┐
│         VPC (Virtual Private Cloud)    │
│  ┌──────────────────────────────────┐  │
│  │    Public Subnet                 │  │
│  │  - NAT Gateway                   │  │
│  │  - Bastion Host                  │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │    Private Subnet                │  │
│  │  - API Servers                   │  │
│  │  - Worker Instances              │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 │  IAM Roles            │
│                 │  Security Groups      │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │    Data Subnet                   │  │
│  │  - DynamoDB VPC Endpoint         │  │
│  │  - S3 VPC Endpoint               │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

Security Features:
✓ Encryption at rest (S3, DynamoDB)
✓ Encryption in transit (TLS 1.3)
✓ IAM roles (no credentials in code)
✓ Secrets Manager (API keys)
✓ VPC isolation (private subnets)
✓ Security Groups (firewall rules)
✓ CloudTrail (audit logging)
✓ GuardDuty (threat detection)
```
