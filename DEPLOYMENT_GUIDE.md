# 🚀 Trading Bot Deployment Guide

This guide provides step-by-step instructions for deploying the trading bot to various free and paid platforms.

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Python 3.8+** installed locally
2. **Git** repository with your trading bot code
3. **MT5 Terminal** installed (for local testing)
4. **XM Trading Account** (demo or live)

## 🆓 Free Deployment Platforms

### 1. Railway (Recommended for Free Tier)

**Pros**: Free tier available, easy deployment, automatic HTTPS
**Cons**: Limited free tier resources

#### Setup Steps:

1. **Create Railway Account**
   ```bash
   # Visit https://railway.app and sign up with GitHub
   ```

2. **Connect Repository**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your trading bot repository

3. **Configure Environment Variables**
   ```bash
   # In Railway dashboard, go to Variables tab
   SECRET_KEY=your-secret-key-here
   API_KEY=your-api-key-here
   LOG_LEVEL=INFO
   MT5_CONNECTION_TIMEOUT=30
   MT5_RETRY_ATTEMPTS=3
   ```

4. **Deploy**
   - Railway will automatically deploy on every git push
   - Access your app via the provided URL

### 2. Render

**Pros**: Free tier, good Python support, automatic deployments
**Cons**: Free tier has cold starts

#### Setup Steps:

1. **Create Render Account**
   ```bash
   # Visit https://render.com and sign up
   ```

2. **Create Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**
   ```bash
   Name: trading-bot
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python app.py
   ```

4. **Set Environment Variables**
   ```bash
   SECRET_KEY=your-secret-key-here
   API_KEY=your-api-key-here
   LOG_LEVEL=INFO
   PORT=5000
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy automatically

### 3. PythonAnywhere

**Pros**: Python-focused, free tier available
**Cons**: Limited resources on free tier

#### Setup Steps:

1. **Create PythonAnywhere Account**
   ```bash
   # Visit https://pythonanywhere.com and sign up
   ```

2. **Upload Code**
   ```bash
   # In Files tab, upload your code or clone from GitHub
   git clone https://github.com/your-username/trading-bot.git
   ```

3. **Install Dependencies**
   ```bash
   # In Bash console
   cd trading-bot
   pip install -r requirements.txt
   ```

4. **Configure Web App**
   - Go to Web tab
   - Click "Add a new web app"
   - Choose "Flask" and Python 3.9
   - Set source code directory to `/home/yourusername/trading-bot`

5. **Set Environment Variables**
   ```bash
   # In Web app configuration
   SECRET_KEY=your-secret-key-here
   API_KEY=your-api-key-here
   LOG_LEVEL=INFO
   ```

6. **Deploy**
   - Click "Reload" to deploy your changes

## 💰 Paid Deployment Platforms

### 4. Heroku

**Pros**: Excellent Python support, extensive add-ons
**Cons**: No free tier anymore

#### Setup Steps:

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-trading-bot-name
   ```

3. **Add Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set SECRET_KEY=your-secret-key-here
   heroku config:set API_KEY=your-api-key-here
   heroku config:set LOG_LEVEL=INFO
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

### 5. Google Cloud Platform

**Pros**: Free tier credits, highly scalable
**Cons**: Complex setup, requires credit card

#### Setup Steps:

1. **Create GCP Account**
   ```bash
   # Visit https://cloud.google.com and sign up
   ```

2. **Install Google Cloud CLI**
   ```bash
   # Download from https://cloud.google.com/sdk/docs/install
   ```

3. **Enable Cloud Run API**
   ```bash
   gcloud services enable run.googleapis.com
   ```

4. **Build and Deploy**
   ```bash
   # Build container
   gcloud builds submit --tag gcr.io/PROJECT_ID/trading-bot
   
   # Deploy to Cloud Run
   gcloud run deploy trading-bot \
     --image gcr.io/PROJECT_ID/trading-bot \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## 🔧 Configuration

### Environment Variables

Set these environment variables in your deployment platform:

```bash
# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Application
LOG_LEVEL=INFO
PORT=5000
HOST=0.0.0.0

# MT5 Settings
MT5_CONNECTION_TIMEOUT=30
MT5_RETRY_ATTEMPTS=3

# Performance
ENABLE_CACHING=true
CACHE_TIMEOUT=300
MAX_WORKERS=4

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

### Security Considerations

1. **Never commit credentials** to your repository
2. **Use environment variables** for all sensitive data
3. **Enable HTTPS** on all communications
4. **Implement rate limiting** to prevent abuse
5. **Use strong API keys** and rotate them regularly

## 🚀 Deployment Files

The repository includes these deployment files:

- `Procfile` - Heroku deployment configuration
- `runtime.txt` - Python version specification
- `Dockerfile` - Container deployment configuration
- `.dockerignore` - Docker build exclusions
- `deployment_config.py` - Platform-specific configurations

## 📊 Monitoring

### Health Checks

Your deployed app includes health check endpoints:

```bash
# Check app status
GET /api/status

# Check MT5 connection
POST /api/connect

# Check trading bot status
GET /api/enhanced/status
```

### Logging

Configure logging based on your platform:

```python
# For Railway/Render/Heroku (console logging)
LOG_LEVEL=INFO
LOG_TO_FILE=false
LOG_TO_CONSOLE=true

# For PythonAnywhere (file logging)
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_TO_CONSOLE=false
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check if port is available
   lsof -i :5000
   
   # Use different port
   PORT=5001 python app.py
   ```

2. **MT5 Connection Issues**
   ```bash
   # Test MT5 connection locally first
   python start_unified_bot.py test
   ```

3. **Import Errors**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Check Python version
   python --version
   ```

4. **Memory Issues**
   ```bash
   # Reduce ML model complexity
   USE_ML=false
   
   # Limit concurrent workers
   MAX_WORKERS=2
   ```

### Platform-Specific Issues

#### Railway
- **Cold starts**: Add health check endpoint
- **Memory limits**: Optimize ML model loading
- **Timeout issues**: Increase connection timeouts

#### Render
- **Build failures**: Check Python version compatibility
- **Cold starts**: Use health checks to keep app warm
- **Environment variables**: Ensure all required vars are set

#### PythonAnywhere
- **File permissions**: Check file ownership
- **Memory limits**: Optimize for free tier constraints
- **WSGI configuration**: Verify WSGI file setup

## 📈 Performance Optimization

### For Production

1. **Enable Caching**
   ```python
   ENABLE_CACHING=true
   CACHE_TIMEOUT=300
   ```

2. **Optimize ML Models**
   ```python
   # Use smaller models for production
   ML_MODEL_COMPLEXITY=medium
   ```

3. **Database Optimization**
   ```python
   # Use connection pooling
   DATABASE_POOL_SIZE=10
   ```

4. **Rate Limiting**
   ```python
   RATE_LIMIT_ENABLED=true
   RATE_LIMIT_REQUESTS=100
   ```

## 🔄 Continuous Deployment

### GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Railway

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Railway
      uses: railway/deploy@v1
      with:
        railway_token: ${{ secrets.RAILWAY_TOKEN }}
```

## 📞 Support

For deployment issues:

1. Check platform-specific documentation
2. Review application logs
3. Test locally before deploying
4. Use demo accounts for testing
5. Monitor resource usage

## ⚠️ Important Notes

- **MT5 Connection**: May require VPN or proxy in cloud environments
- **File Storage**: Use cloud storage for logs and models in production
- **Security**: Never expose MT5 credentials in logs or responses
- **Testing**: Always test with demo accounts first
- **Backup**: Implement proper backup strategies for models and data

## 🎯 Next Steps

After successful deployment:

1. Test all API endpoints
2. Configure monitoring and alerts
3. Set up proper logging
4. Implement backup strategies
5. Monitor performance metrics
6. Plan for scaling

---

**Happy Trading! 🤖📈**
