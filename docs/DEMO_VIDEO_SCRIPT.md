# 🎬 Demo Video Script - MarketScope AI

## Video Details
- **Duration**: 2-3 minutes
- **Format**: 1920x1080 (1080p)
- **Audio**: Clear voiceover with background music (optional)
- **Style**: Technical showcase for recruiters/employers

---

## 📋 Pre-Recording Checklist

### Setup (10 minutes before recording)

1. **Clean Desktop**:
   ```bash
   # Close unnecessary applications
   # Organize windows for screen recording
   # Set wallpaper to professional/clean background
   ```

2. **Prepare Terminal**:
   ```bash
   # Open terminal in project directory
   cd /home/khaitran/PycharmProjects/Azure-OpenAI_StockTool
   
   # Clear terminal
   clear
   
   # Set larger font size for readability (Ctrl+Shift++)
   # Use high-contrast theme (white text on dark background)
   ```

3. **Start LocalStack**:
   ```bash
   docker compose up -d localstack
   
   # Wait 10 seconds for initialization
   sleep 10
   
   # Verify it's running
   docker compose ps
   ```

4. **Prepare Browser**:
   - Open tabs in this order:
     1. GitHub repository: https://github.com/Narcolepsyy/Azure-OpenAI_StockTool
     2. AWS_SHOWCASE.md (local file or GitHub)
     3. Architecture diagram (if created)
     4. http://localhost:8000/docs (API docs - start backend first)

5. **Test Run**:
   ```bash
   # Do a complete test run WITHOUT recording
   # Practice the script 2-3 times
   # Time yourself - aim for 2:00 - 2:30
   ```

---

## 🎥 Recording Script

### Scene 1: Introduction (0:00 - 0:20) - 20 seconds

**Visual**: Face camera or screen with GitHub repo homepage

**Script**:
```
Hi, I'm [Your Name]. 

Today I'll show you MarketScope AI - an enterprise-grade 
AI-powered stock analysis platform I built to demonstrate 
production-ready AWS architecture.

This project integrates 10 AWS services including Lambda, 
CloudWatch, EventBridge, and DynamoDB - all running locally 
with zero development costs using LocalStack Pro.

Let's dive in!
```

**On-Screen Text** (optional overlay):
```
MarketScope AI
10 AWS Services | Serverless Architecture | $0 Dev Costs
```

---

### Scene 2: Architecture Overview (0:20 - 0:45) - 25 seconds

**Visual**: Show architecture diagram or AWS_SHOWCASE.md

**Script**:
```
The system architecture has three main layers:

First, a React frontend for the user interface.

Second, a FastAPI backend with OpenAI GPT-4 integration, 
a RAG system using ChromaDB, and Perplexity-style web search.

Third, the AWS services layer - this is what makes it 
production-ready: Lambda for serverless computing, 
CloudWatch for monitoring, EventBridge for event-driven 
scheduling, DynamoDB for distributed storage, S3 for 
document storage, plus SQS, SNS, and Secrets Manager.

All of this runs locally using LocalStack Pro, which is 
free with GitHub Student Plan.
```

**On-Screen Text**:
```
Frontend: React + TypeScript
Backend: FastAPI + OpenAI GPT-4
AWS: 10 Services (Lambda, CloudWatch, EventBridge, DynamoDB, S3, SQS, SNS, Secrets, Logs, STS)
Dev Environment: LocalStack Pro (Free)
```

**Visual Actions**:
- Scroll through AWS_SHOWCASE.md showing service list
- Highlight architecture diagram sections as you mention them

---

### Scene 3: LocalStack Setup (0:45 - 1:00) - 15 seconds

**Visual**: Terminal showing docker compose commands

**Script**:
```
Let's verify LocalStack is running. 

As you can see, LocalStack Pro is active with all services 
enabled. This gives me a complete AWS environment on my 
laptop without any cloud costs.
```

**Terminal Commands** (pre-run, just show output):
```bash
# Show docker compose status
docker compose ps

# Show LocalStack health
curl http://localhost:4566/_localstack/health | jq

# Or use AWS CLI
aws --endpoint-url=http://localhost:4566 s3 ls
```

**Expected Output**:
```json
{
  "services": {
    "cloudwatch": "running",
    "dynamodb": "running",
    "events": "running",
    "lambda": "running",
    "s3": "running",
    ...
  },
  "edition": "pro",
  "version": "4.0.0"
}
```

---

### Scene 4: Lambda Deployment (1:00 - 1:25) - 25 seconds

**Visual**: Terminal running deployment script with output

**Script**:
```
Now let's deploy a Lambda function. I've created an 
automated deployment script that packages the function, 
creates it in LocalStack, and sets up an EventBridge 
schedule to trigger it daily at 4:30 PM Eastern Time.

[Run script]

Perfect! The Lambda function is deployed. It will 
automatically update stock data every day, publish 
metrics to CloudWatch, and send notifications via SNS.

This demonstrates serverless architecture and 
Infrastructure as Code principles.
```

**Terminal Commands**:
```bash
# Run deployment script
./scripts/deploy_lambda.sh

# Let output scroll naturally - don't rush
# Highlight key messages:
# ✓ Lambda deployed: stocktool-stock-updater
# ✓ EventBridge schedule created
# ✓ Lambda invocation successful
```

**On-Screen Text** (show during deployment):
```
Lambda Function: stocktool-stock-updater
Runtime: Python 3.11
Schedule: Daily at 4:30 PM ET (EventBridge)
Features: Stock updates + CloudWatch metrics + SNS notifications
```

---

### Scene 5: Lambda Testing (1:25 - 1:45) - 20 seconds

**Visual**: Terminal running test script

**Script**:
```
Let's test the Lambda function to make sure it works.

[Run test]

Excellent! The function successfully processed 10 stock 
symbols, returned a 200 status code, and completed in 
about 7 seconds. The logs show successful execution with 
all stocks updated.

This is production-ready code with proper error handling, 
logging, and monitoring.
```

**Terminal Commands**:
```bash
# Run Lambda test
python scripts/test_lambda.py

# Highlight results:
# ✓ Status Code: 200
# ✓ Response: {"total": 10, "successful": 10, "failed": 0}
# ✓ Duration: ~7 seconds
# ✓ Memory: 128 MB
```

---

### Scene 6: CloudWatch Dashboard (1:45 - 2:05) - 20 seconds

**Visual**: Terminal showing dashboard creation + AWS CLI output

**Script**:
```
For monitoring, I've created a CloudWatch dashboard with 
10 metric widgets tracking API latency, cache hit rate, 
Lambda execution, DynamoDB operations, and more.

[Show dashboard creation output]

I've also configured 3 alarms: one for high API latency, 
one for low cache hit rate, and one for Lambda errors. 
These would send SNS notifications in production.

This demonstrates comprehensive observability and 
proactive monitoring.
```

**Terminal Commands**:
```bash
# Create dashboard (if not already created)
./scripts/create_cloudwatch_dashboard.sh

# List dashboards
aws --endpoint-url=http://localhost:4566 cloudwatch list-dashboards

# Describe alarms
aws --endpoint-url=http://localhost:4566 cloudwatch describe-alarms
```

**On-Screen Text**:
```
CloudWatch Dashboard: StockTool-Monitoring
- 10 Metric Widgets (API latency, cache hit rate, Lambda execution, etc.)
- 3 Alarms (High latency, low cache, Lambda errors)
- Real-time monitoring + Historical trends
```

---

### Scene 7: Additional AWS Services (2:05 - 2:20) - 15 seconds

**Visual**: Terminal showing resource verification

**Script**:
```
Beyond Lambda and CloudWatch, the system uses DynamoDB 
for conversation storage with automatic TTL expiration, 
S3 for knowledge base documents, SQS for asynchronous 
task queues, and SNS for notifications.

Let's verify all resources are operational.

[Run verification script]

Perfect - all 10 AWS services are configured and tested.
```

**Terminal Commands**:
```bash
# Verify all AWS resources
python scripts/verify_aws_resources.py

# Show output:
# ✅ S3 bucket: stocktool-knowledge-base
# ✅ DynamoDB tables: conversations, user_sessions
# ✅ SQS queue: stocktool-tasks
# ✅ SNS topic: stocktool-alerts
# ✅ CloudWatch metrics published
```

---

### Scene 8: Code Quality & Documentation (2:20 - 2:35) - 15 seconds

**Visual**: VS Code or GitHub showing project structure

**Script**:
```
The project follows best practices with organized structure: 
separate directories for docs, tests, scripts, and demos.

I've written over 5,000 lines of comprehensive documentation 
covering getting started guides, architecture explanations, 
Lambda deployment, LocalStack setup, and troubleshooting.

All tests pass with 100% success rate, and deployment is 
fully automated with bash scripts.
```

**Visual Actions**:
- Show project directory tree
- Open docs/ folder showing multiple .md files
- Show tests/ folder showing test files
- Open TEST_RESULTS.md showing passing tests

**On-Screen Text**:
```
Documentation: 8+ comprehensive guides (5,000+ lines)
Tests: 60+ test files (100% passing)
Automation: 7 deployment scripts
Code Organization: Modular architecture
```

---

### Scene 9: Technology Stack (2:35 - 2:50) - 15 seconds

**Visual**: GitHub README or split screen showing frontend + backend

**Script**:
```
The technology stack includes:

Backend: Python 3.11 with FastAPI for high-performance APIs.
Frontend: React with TypeScript and Vite for modern UI.
AI: OpenAI GPT-4 with function calling and RAG system using ChromaDB.
Cloud: 10 AWS services orchestrated with LocalStack Pro.
DevOps: Docker, automated deployment, comprehensive monitoring.

This demonstrates full-stack development with cloud-native architecture.
```

**On-Screen Text**:
```
Tech Stack:
Backend: Python 3.11, FastAPI, OpenAI GPT-4
Frontend: React, TypeScript, Vite, TailwindCSS
AI/ML: RAG (ChromaDB + LangChain), Embeddings
Cloud: AWS (Lambda, CloudWatch, DynamoDB, S3, SQS, SNS, EventBridge)
DevOps: Docker, LocalStack Pro, Automated Deployment
Database: DynamoDB, SQLAlchemy, PostgreSQL/SQLite
```

---

### Scene 10: Closing & Call to Action (2:50 - 3:00) - 10 seconds

**Visual**: GitHub repository page or your professional photo

**Script**:
```
This project demonstrates enterprise-grade AWS skills, 
serverless architecture, AI integration, and DevOps 
automation - all with zero development costs.

Full code, documentation, and setup instructions are 
available on GitHub. Link in the description.

Thanks for watching!
```

**On-Screen Text**:
```
GitHub: github.com/Narcolepsyy/Azure-OpenAI_StockTool
LinkedIn: [your profile]
Portfolio: [your website]

Skills Demonstrated:
✅ 10 AWS Services
✅ Serverless Architecture  
✅ AI/ML Integration
✅ DevOps Automation
✅ Full-Stack Development
```

**Visual Actions**:
- Show GitHub repo with star button visible
- Show README with badges and project description
- Fade to end screen with contact info

---

## 🎬 Recording Setup

### Recommended Software (All Free)

1. **Screen Recording**:
   - **OBS Studio** (Linux/Windows/Mac): https://obsproject.com/
   - **SimpleScreenRecorder** (Linux): In package manager
   - **Kazam** (Linux): `sudo apt install kazam`

2. **Audio Recording**:
   - **Audacity** (noise removal): https://www.audacityteam.org/
   - Built-in microphone (test first!)
   - Consider USB microphone for better quality

3. **Video Editing**:
   - **DaVinci Resolve** (free version): https://www.blackmagicdesign.com/
   - **Kdenlive** (Linux): `sudo apt install kdenlive`
   - **OpenShot** (simple): https://www.openshot.org/

4. **On-Screen Text/Annotations**:
   - Add in video editor
   - Use simple animations (fade in/out)
   - Keep text on screen for 3-5 seconds

---

## 🎙️ Audio Recording Tips

### Voice Recording Best Practices

1. **Environment**:
   - Record in quiet room (close windows, turn off AC/fans)
   - Use blankets/pillows to dampen echo
   - Record during quiet hours

2. **Microphone**:
   - Position 6-8 inches from mouth
   - Slightly off to side (reduces plosives)
   - Test levels before full recording

3. **Delivery**:
   - Speak clearly and confidently (not too fast!)
   - Pause between scenes (easier to edit)
   - Smile while talking (sounds more energetic)
   - Re-record sentences if needed

4. **Post-Processing in Audacity**:
   ```
   1. Effect → Noise Reduction
      - Profile noise from silent section
      - Apply with 12 dB reduction
   
   2. Effect → Normalize
      - Peak amplitude: -1.0 dB
   
   3. Effect → Compressor
      - Threshold: -12 dB
      - Ratio: 3:1
   
   4. Export as WAV or high-quality MP3
   ```

---

## 🎨 Visual Recording Tips

### Screen Recording Settings (OBS Studio)

1. **Resolution**: 1920x1080 (1080p)
2. **Frame Rate**: 30 FPS
3. **Bitrate**: 6000-10000 kbps
4. **Encoder**: x264 or hardware (NVENC/AMD)

### Terminal Settings for Recording

```bash
# Make text readable
- Font size: 14-16pt
- Font: Monospace or JetBrains Mono
- Theme: High contrast (white on black)
- Terminal size: 120x30 characters

# Add to ~/.bashrc for colorful output
export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
```

### Browser Settings

```
- Zoom: 125-150% (more readable)
- Hide bookmarks bar (cleaner)
- Use private/incognito window (no history)
- Close unnecessary tabs
```

---

## 📝 Recording Workflow

### Step-by-Step Process

**1. Record Video (Screen + Audio)**:
```bash
# Option A: OBS Studio
# - Add "Display Capture" source
# - Add "Audio Input Capture" (microphone)
# - Click "Start Recording"
# - Follow script
# - Click "Stop Recording"

# Option B: SimpleScreenRecorder (Linux)
simplescreenrecorder
# - Select area: Full screen or window
# - Audio: PulseAudio
# - Start recording
```

**2. Review Raw Recording**:
- Watch full video
- Note timestamps of mistakes
- Check audio quality (volume, clarity)
- Check video quality (readability, smoothness)

**3. Edit Video** (DaVinci Resolve or Kdenlive):
```
Timeline:
├─ Video Track: Screen recording
├─ Audio Track 1: Voiceover
├─ Audio Track 2: Background music (optional, low volume)
└─ Text Track: On-screen annotations

Editing Steps:
1. Import raw recording
2. Cut mistakes/pauses (use "Ripple Delete")
3. Add text overlays at key moments
4. Add transitions between scenes (simple cuts or fades)
5. Add intro title (2 seconds)
6. Add outro with links (5 seconds)
7. Color correction (optional, boost contrast slightly)
8. Audio: Fade in/out, normalize levels
```

**4. Add Background Music** (Optional):
```
Free Music Sources:
- YouTube Audio Library (no attribution needed)
- Incompetech (Kevin MacLeod)
- Bensound (with attribution)

Settings:
- Volume: -20 to -30 dB (very quiet, background only)
- Fade in/out at start/end
- Use instrumental only (no lyrics)
```

**5. Export Video**:
```
Format: MP4 (H.264)
Resolution: 1920x1080
Frame Rate: 30 FPS
Bitrate: 8000-12000 kbps (high quality)
Audio: AAC, 192 kbps, 48 kHz
```

---

## 📤 Publishing Checklist

### YouTube Upload

**Video Details**:
```
Title: MarketScope AI - Enterprise AWS Integration Showcase | Serverless Architecture Demo

Description:
MarketScope AI is an enterprise-grade AI-powered stock analysis platform demonstrating production-ready AWS architecture with 10 integrated services.

🎯 Project Highlights:
• 10 AWS Services: Lambda, CloudWatch, EventBridge, DynamoDB, S3, SQS, SNS, Secrets Manager, CloudWatch Logs, STS
• Serverless Architecture: Event-driven design with automated scheduling
• AI Integration: OpenAI GPT-4 with RAG system (ChromaDB + LangChain)
• Zero Development Costs: LocalStack Pro with GitHub Student Plan
• 100% Test Coverage: All services verified and operational

⚙️ Technology Stack:
• Backend: Python 3.11, FastAPI
• Frontend: React, TypeScript, Vite
• AI/ML: OpenAI GPT-4, ChromaDB, LangChain
• Cloud: AWS services via LocalStack Pro
• DevOps: Docker, automated deployment scripts

📚 Resources:
GitHub Repository: https://github.com/Narcolepsyy/Azure-OpenAI_StockTool
Documentation: Full setup guides and architecture docs
LinkedIn: [your profile]

⏱️ Timestamps:
0:00 Introduction
0:20 Architecture Overview
0:45 LocalStack Setup
1:00 Lambda Deployment
1:25 Lambda Testing
1:45 CloudWatch Dashboard
2:05 Additional AWS Services
2:20 Code Quality & Documentation
2:35 Technology Stack
2:50 Closing

#AWS #Serverless #Python #FastAPI #React #CloudComputing #DevOps #AI #MachineLearning #Portfolio

Tags: aws, serverless, lambda, cloudwatch, dynamodb, python, fastapi, react, openai, localstack, devops, cloud computing, portfolio project, software engineering, backend development, full stack, event driven architecture
```

**Settings**:
- Visibility: Public (or Unlisted for resume only)
- Category: Science & Technology
- Allow comments
- Add to playlist: "Portfolio Projects"
- Thumbnail: Create custom thumbnail with project logo + "AWS Integration"

### LinkedIn Video Post

**Post Text**:
```
🚀 Excited to share my latest project: MarketScope AI

I built an enterprise-grade AI-powered stock analysis platform to demonstrate production-ready AWS architecture and serverless design patterns.

Key achievements:
✅ Integrated 10 AWS services (Lambda, CloudWatch, EventBridge, DynamoDB, S3, and more)
✅ Implemented serverless architecture with automated scheduling
✅ Created comprehensive monitoring with 10 metrics + 3 alarms
✅ Achieved 100% test coverage across all services
✅ Zero development costs using LocalStack Pro (GitHub Student Plan)

Tech stack: Python, FastAPI, React, OpenAI GPT-4, AWS services, Docker

This project showcases skills in cloud architecture, serverless computing, AI/ML integration, and DevOps automation.

Full code and documentation available on GitHub: [link]
Demo video: [YouTube link]

#AWS #Serverless #CloudComputing #Python #React #DevOps #SoftwareEngineering #AI #MachineLearning #PortfolioProject

Open to opportunities in cloud engineering, backend development, and AI/ML!
```

### Dev.to Article (Optional)

Create a detailed blog post walking through the architecture:
```
Title: Building a Production-Ready AWS Application with $0 Development Costs

Introduction
- Problem statement
- Why LocalStack?

Architecture Overview
- System design
- AWS services chosen

Implementation Details
- Lambda functions
- CloudWatch monitoring
- EventBridge scheduling

Lessons Learned
- Challenges faced
- Solutions implemented

Conclusion
- Skills demonstrated
- Next steps

[Include code snippets, diagrams, screenshots]
```

---

## ✅ Final Checklist

Before publishing, verify:

- [ ] Video is 2-3 minutes long
- [ ] Audio is clear and audible
- [ ] Text overlays are readable
- [ ] All terminal commands work as shown
- [ ] No sensitive information visible (API keys, passwords)
- [ ] GitHub link is correct in description
- [ ] LinkedIn profile link is correct
- [ ] Video thumbnail is professional
- [ ] Closed captions/subtitles added (YouTube auto-generates)
- [ ] Video tested on mobile device (readability)

---

## 🎯 Alternative: Quick Loom Video

If you want something faster (30 minutes instead of 3 hours):

**Tool**: Loom (https://www.loom.com/) - Free for videos up to 5 minutes

**Benefits**:
- Records screen + webcam bubble automatically
- No editing needed
- Instant sharing link
- Good for quick demonstrations

**Drawbacks**:
- Less polished than edited video
- 5-minute limit on free plan
- Can't add fancy graphics/text overlays

**Recommendation**: 
- Use Loom for quick demos to friends/family
- Use OBS + editing for portfolio/LinkedIn video

---

## 📊 Success Metrics

After publishing, track:

**YouTube**:
- Views (target: 100+ in first month)
- Watch time (aim for >60% retention)
- Likes/comments
- Subscribers gained

**LinkedIn**:
- Post views (target: 500+)
- Engagement (likes, comments, shares)
- Profile views spike
- Connection requests from recruiters

**GitHub**:
- Stars (target: 10+ in first month)
- Forks
- Issues/questions
- Traffic spike after video release

---

**Good luck with your demo video! 🎬**

You've got this! The project is impressive - now it's time to show it off.
