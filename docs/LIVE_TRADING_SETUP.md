# Live Trading Setup Guide

## Overview

This guide explains how to enable live trading for your trading bot. Live trading requires MetaTrader5 to be available, which is only possible on Windows-based platforms.

## Prerequisites

### 1. Windows Environment
- **Local Development**: Windows 10/11 with Python 3.10+
- **Cloud Deployment**: Windows Server (Azure, AWS EC2, etc.)
- **VPS**: Windows-based VPS provider

### 2. MetaTrader5 Terminal
- Download and install MetaTrader5 from your broker
- Configure with your trading account credentials
- Enable AutoTrading in MT5

### 3. Trading Account
- Demo or Live account with your preferred broker
- Account credentials (login, password, server)

## Setup Options

### Option 1: Local Development (Recommended for Testing)

1. **Install MetaTrader5**
   ```bash
   # Download from your broker's website
   # Install and configure with your account
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   ```bash
   # Create .env file
   XM_ACCOUNT_NUMBER=your_account_number
   XM_PASSWORD=your_password
   XM_SERVER=XMGlobal-Demo  # or XMGlobal-Live
   AUTO_TRADE=true
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

### Option 2: Windows VPS/Cloud Deployment

1. **Choose a Windows-based platform:**
   - **Azure**: Windows Server VM
   - **AWS EC2**: Windows Server instance
   - **DigitalOcean**: Windows Droplet
   - **Vultr**: Windows VPS

2. **Install MetaTrader5 on the server**
   ```powershell
   # Download and install MT5
   # Configure with your account
   # Enable AutoTrading
   ```

3. **Deploy using Windows requirements**
   ```bash
   # Use the Windows-specific requirements
   pip install -r requirements-windows.txt
   ```

4. **Configure environment variables**
   ```bash
   # Set in system environment or .env file
   XM_ACCOUNT_NUMBER=your_account_number
   XM_PASSWORD=your_password
   XM_SERVER=XMGlobal-Demo
   AUTO_TRADE=true
   FLASK_ENV=production
   ```

### Option 3: Docker on Windows

1. **Use Windows Dockerfile**
   ```bash
   docker build -f Dockerfile.windows -t trading-bot-live .
   ```

2. **Run with MT5 access**
   ```bash
   docker run -p 5000:5000 --env-file .env trading-bot-live
   ```

## Configuration

### Environment Variables

```bash
# Required for live trading
XM_ACCOUNT_NUMBER=12345678
XM_PASSWORD=your_secure_password
XM_SERVER=XMGlobal-Demo

# Optional settings
AUTO_TRADE=true
LOG_LEVEL=INFO
API_KEY=your_api_key
FLASK_ENV=production
```

### Trading Parameters

Configure these in the web interface or via API:

```json
{
  "symbol": "EURUSD",
  "timeframe": "1h",
  "risk_per_trade": 0.02,
  "auto_trade": true
}
```

## Verification

### 1. Check Connection Status
```bash
curl http://localhost:5000/api/status
```

Expected response:
```json
{
  "deployment_info": {
    "mode": "live",
    "limitations": [],
    "mt5_available": true
  },
  "connected": true
}
```

### 2. Test MT5 Connection
```bash
python -c "
from mt5_connector import MT5Connector
connector = MT5Connector()
print('Connected:', connector.connect())
print('Account:', connector.get_account_summary())
"
```

### 3. Verify AutoTrading
- Check MT5 terminal: AutoTrading button should be green
- Check terminal info: `trade_allowed` should be `true`

## Security Considerations

### 1. Account Security
- Use demo account for testing
- Never commit credentials to version control
- Use environment variables for sensitive data

### 2. Network Security
- Use HTTPS in production
- Implement API key authentication
- Restrict access to trusted IPs

### 3. Risk Management
- Start with small position sizes
- Set appropriate stop losses
- Monitor trading activity regularly

## Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   - Verify account credentials
   - Check if MT5 is running
   - Ensure AutoTrading is enabled

2. **Import Error: MetaTrader5**
   - Install on Windows environment
   - Install Visual C++ Redistributable
   - Use `requirements-windows.txt`

3. **AutoTrading Disabled**
   - Enable AutoTrading in MT5 terminal
   - Check Expert Advisors are allowed
   - Verify DLLs are allowed

4. **Order Placement Failed**
   - Check account balance
   - Verify symbol is available
   - Check market hours

### Debug Commands

```bash
# Test MT5 functionality
python -c "
from mt5_connector import MT5Connector
connector = MT5Connector()
connector.connect()
connector.test_mt5_functionality()
"

# Check symbol info
python -c "
from mt5_connector import MT5Connector
connector = MT5Connector()
connector.connect()
print(connector.get_symbol_info('EURUSD'))
"
```

## Next Steps

1. **Start with Demo Account**: Test thoroughly before live trading
2. **Monitor Performance**: Track trading results and adjust strategies
3. **Scale Gradually**: Increase position sizes as confidence grows
4. **Backup Strategy**: Keep local backup of trading bot
5. **Regular Updates**: Keep MT5 and Python packages updated

## Support

For issues with:
- **MT5 Connection**: Check broker documentation
- **Trading Bot**: Review logs in `logs/trading_bot.log`
- **Deployment**: Check platform-specific guides
