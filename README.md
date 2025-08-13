# 🤖 Unified Trading Bot System

A comprehensive trading bot system that combines manual analysis, automated trading, and MT5 integration into a unified platform.

## 🚀 What's New

### 🔄 Complete Migration
- **Enhanced Trading Bot**: The old `trading_bot.py` now supports both manual and automated trading
- **Unified Interface**: Single entry point for all trading functionality
- **Backward Compatibility**: All existing functionality preserved
- **New Capabilities**: MT5 integration, automated trading, and enhanced analysis

### 🎯 Key Features

#### 📊 Enhanced Trading Bot (`trading_bot.py`)
- **Manual Analysis**: Traditional yfinance-based analysis with trend identification
- **Automated Trading**: MT5 integration for automated trade execution
- **Multi-Market Support**: Stocks, Forex, Crypto, and Commodities
- **Risk Management**: Advanced position sizing and risk control
- **Continuous Automation**: 24/7 automated trading with periodic analysis

#### 🤖 Automated Trading Bot (`automated_trading_bot.py`)
- **Full Automation**: Completely autonomous trading system
- **Multi-Symbol Trading**: Analyzes 5 major currency pairs simultaneously
- **AI-Powered Analysis**: Machine learning models for signal generation
- **Smart Money Concepts**: Advanced market structure analysis
- **Real-time Monitoring**: Continuous position and market monitoring

#### 🌐 Web Interface (`app.py`)
- **Unified API**: Single API supporting both manual and automated trading
- **Enhanced Endpoints**: New endpoints for enhanced trading bot functionality
- **Real-time Status**: Live monitoring of bot status and positions
- **Interactive Controls**: Start/stop automation, position management

## 📋 Prerequisites

- Python 3.8 or higher
- MetaTrader 5 terminal installed and running
- XM trading account (demo or live)
- Internet connection for market data

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd trading_bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. Credentials: enter account number, password, and server via the web UI. Do not store XM credentials in `.env`.
   You may still configure non-credential settings in `.env` (e.g., `RISK_PER_TRADE`, `USE_ML`).

## 🚀 Quick Start

### Option 1: Unified Startup Script (Recommended)
   ```bash
# Test MT5 connection
python start_unified_bot.py test

# Run enhanced trading bot with automation
python start_unified_bot.py enhanced --symbol EURUSD --timeframe 5m

# Run fully automated trading bot
python start_unified_bot.py automated

# Run web interface
python start_unified_bot.py web
```

### Option 2: Individual Components
   ```bash
# Enhanced trading bot (manual + automated)
python start_automated_bot.py

# Original automated trading bot
python automated_trading_bot.py

# Web interface
   python app.py
   ```

### Option 3: Direct Python Usage
```python
from trading_bot import TradingBot

# Create enhanced trading bot
bot = TradingBot(
    symbol='EURUSD',
    period='5m',
    market_type='forex',
    enable_automation=True
)

# Connect to MT5
bot.connect_mt5()

# Run automated analysis
result = bot.run_automated_analysis_cycle()

# Start continuous automation
bot.run_continuous_automation(interval_minutes=5)
```

## 🌐 Web Interface

Once started, access the web interface at: **http://localhost:5000**

### Available Features:
- **Enhanced Trading Bot**: Manual and automated trading capabilities
- **Automated Trading Bot**: Fully automated MT5 trading
- **Connection Management**: MT5 connection controls
- **Real-time Analysis**: Live market analysis and signals
- **Position Management**: View and manage open positions
- **Configuration**: Bot settings and parameters

### API Endpoints:

#### Enhanced Trading Bot
- `POST /api/enhanced/initialize` - Initialize enhanced trading bot
- `POST /api/enhanced/connect` - Connect to MT5
- `POST /api/enhanced/analyze` - Run manual analysis
- `POST /api/enhanced/automated-analysis` - Run automated analysis cycle
- `POST /api/enhanced/execute-trade` - Execute trade
- `POST /api/enhanced/start-automation` - Start continuous automation
- `POST /api/enhanced/stop-automation` - Stop automation
- `GET /api/enhanced/status` - Get bot status
- `GET /api/enhanced/positions` - Get MT5 positions
- `POST /api/enhanced/close-position` - Close position
- `GET /api/enhanced/capabilities` - Get capabilities

#### Original Trading Bot (Still Available)
- `POST /api/connect` - Connect to MT5
- `POST /api/analyze` - Run analysis
- `POST /api/start-trading` - Start trading
- `POST /api/stop-trading` - Stop trading
- `GET /api/status` - Get status
- `GET /api/positions` - Get positions

## ⚙️ Configuration

### Environment Variables

Only non-credential variables are used from `.env`:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RISK_PER_TRADE` | Risk per trade percentage | `0.02` | `0.02` (2%) |
| `USE_ML` | Enable machine learning | `true` | `true` |
| `ENABLE_AUTOMATION` | Enable automated trading | `true` | `true` |
| `MAX_POSITIONS_PER_SYMBOL` | Max positions per symbol | `3` | `3` |
| `ANALYSIS_INTERVAL` | Analysis interval in seconds | `300` | `300` (5 min) |

### Trading Parameters

#### Enhanced Trading Bot
- **Symbol**: Any supported symbol (EURUSD, GBPUSD, etc.)
- **Timeframe**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Market Type**: stock, forex, crypto, commodities
- **Risk Management**: Automatic position sizing
- **Automation**: Optional MT5 integration

#### Automated Trading Bot
- **Symbols**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD
- **Timeframes**: 5m, 15m, 1h
- **Risk Management**: 2% risk per trade
- **Automation**: Fully automated with ML signals

## 🔧 Migration Guide

### From Old Trading Bot
The old `trading_bot.py` functionality is fully preserved and enhanced:

```python
# Old way (still works)
bot = TradingBot('EURUSD', '5m', 'forex')
bot.fetch_data()
bot.identify_higher_highs_lows()

# New way (with automation)
bot = TradingBot('EURUSD', '5m', 'forex', enable_automation=True)
bot.connect_mt5()
bot.run_continuous_automation()
```

### From Automated Trading Bot
The automated trading bot remains unchanged and can be used independently:

```python
# Still works as before
from automated_trading_bot import AutomatedTradingBot
bot = AutomatedTradingBot()
bot.run()
```

### New Unified Approach
Use the unified startup script for the best experience:

```bash
# Test everything first
python start_unified_bot.py test

# Run with your preferred mode
python start_unified_bot.py enhanced --symbol EURUSD --timeframe 5m
python start_unified_bot.py automated
python start_unified_bot.py web
```

## 📊 Features Comparison

| Feature | Enhanced Bot | Automated Bot | Web Interface |
|---------|-------------|---------------|---------------|
| Manual Analysis | ✅ | ❌ | ✅ |
| Automated Trading | ✅ | ✅ | ✅ |
| MT5 Integration | ✅ | ✅ | ✅ |
| Multi-Market | ✅ | ❌ | ✅ |
| Risk Management | ✅ | ✅ | ✅ |
| Position Management | ✅ | ✅ | ✅ |
| Real-time Monitoring | ✅ | ✅ | ✅ |
| Web API | ✅ | ✅ | ✅ |
| Continuous Automation | ✅ | ✅ | ✅ |

## 🛡️ Risk Management

### Enhanced Trading Bot
- **Position Sizing**: Automatic calculation based on account balance
- **Stop Loss**: Automatic stop loss placement
- **Take Profit**: Risk-reward ratio optimization
- **Maximum Positions**: Configurable limits per symbol
- **Risk Per Trade**: Configurable percentage (default 2%)

### Automated Trading Bot
- **Risk Per Trade**: 2% of account balance
- **Maximum Positions**: 3 positions per symbol
- **Stop Loss**: Automatic placement
- **Take Profit**: 1:3 risk-reward ratio
- **Signal Confidence**: 70% minimum confidence threshold

## 🔍 Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   ```bash
   python start_unified_bot.py test
   ```
   Check your credentials in `.env` file

2. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**
   ```bash
   python start_unified_bot.py web --port 5001
   ```

4. **Automation Not Working**
   - Check MT5 connection
   - Verify credentials
   - Check log files in `logs/` directory

### Log Files
- `logs/unified_bot.log` - Unified bot logs
- `logs/trading_bot.log` - Web interface logs
- `logs/automated_trading_bot.log` - Automated bot logs

## 📈 Performance Monitoring

### Enhanced Trading Bot
- Real-time status monitoring
- Performance metrics
- Position tracking
- Risk analysis

### Automated Trading Bot
- Signal accuracy tracking
- Profit/loss monitoring
- Position management
- Market analysis logs

## 🔮 Future Enhancements

- **Advanced ML Models**: More sophisticated prediction models
- **Portfolio Management**: Multi-symbol portfolio optimization
- **Backtesting**: Historical performance analysis
- **Mobile App**: Mobile interface for monitoring
- **Alerts**: Email/SMS notifications
- **Advanced Analytics**: Detailed performance reports

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files
3. Test with connection test mode
4. Verify configuration settings

## 📄 License

This project is for educational purposes. Use at your own risk. Trading involves substantial risk of loss.

---

**Happy Trading! 🚀**
