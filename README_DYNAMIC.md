# Dynamic Trading Bot with Vue Frontend & XM Integration

A comprehensive automated trading bot with a modern Vue.js frontend, MetaTrader 5 integration, and machine learning capabilities for XM trading accounts.

## 🚀 Features

### Core Features
- **Dynamic Symbol Selection**: Trade any of 27+ forex pairs available on XM
- **Multiple Timeframes**: Support for 1m, 5m, 15m, 30m, 1h, 4h, 1d timeframes
- **XM Account Integration**: Direct connection to XM demo and live accounts
- **Machine Learning Analysis**: AI-powered signal generation and prediction
- **Real-time Monitoring**: Live position tracking and account status
- **Risk Management**: Configurable risk per trade and position sizing

### Frontend Features
- **Modern Vue.js UI**: Beautiful, responsive interface
- **Real-time Updates**: Live status updates every 5 seconds
- **Interactive Charts**: Visual analysis results and trading signals
- **Position Management**: Close individual or all positions
- **Configuration Panel**: Easy setup for XM credentials and trading parameters

### Technical Analysis
- **Higher Highs/Lows Detection**: Advanced trend analysis
- **Pivot Point Analysis**: Support and resistance identification
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
- **ML-Enhanced Signals**: Combined traditional and AI analysis

## 📋 Prerequisites

### Required Software
- **Python 3.8+**: For the backend trading bot
- **MetaTrader 5**: Must be installed and running
- **XM Trading Account**: Demo or live account
- **Web Browser**: For the Vue.js frontend

### Python Dependencies
```bash
pip install -r requirements.txt
```

### MetaTrader 5 Setup
1. Install MetaTrader 5 from your broker
2. Enable AutoTrading:
   - Tools → Options → Expert Advisors
   - Check "Allow automated trading"
   - Check "Allow DLL imports"
   - Check "Allow WebRequest for listed URL"
3. Restart MetaTrader 5
4. Ensure the "AutoTrading" button is green (enabled)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd trading_bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure XM Account
```bash
# Copy the example configuration
cp xm_config_example.txt .env

# Edit .env with your XM credentials
nano .env
```

### 4. Start the Application
```bash
python app.py
```

### 5. Access the Frontend
Open your browser and navigate to: `http://localhost:5000`

## 🔧 Configuration

### XM Account Settings (.env file)
```env
# XM Account Credentials
XM_ACCOUNT_NUMBER=your_xm_account_number_here
XM_PASSWORD=your_xm_password_here
XM_SERVER=XMGlobal-Demo

# Trading Bot Configuration
RISK_PER_TRADE=2.0
ACCOUNT_SIZE=10000
AUTO_TRADE=false
USE_ML=true
```

### Server Options
- **XMGlobal-Demo**: For demo/training accounts
- **XMGlobal-Live**: For live trading accounts

### Available Symbols
The bot supports 27+ forex pairs including:
- **Major Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Cross Pairs**: EURGBP, EURJPY, GBPJPY, CHFJPY, AUDCAD, AUDCHF, AUDJPY
- **Exotic Pairs**: AUDNZD, CADCHF, CADJPY, EURAUD, EURCAD, EURCHF, EURNZD

### Available Timeframes
- **1m**: 1 Minute (scalping)
- **5m**: 5 Minutes (day trading)
- **15m**: 15 Minutes (swing trading)
- **30m**: 30 Minutes (swing trading)
- **1h**: 1 Hour (position trading)
- **4h**: 4 Hours (position trading)
- **1d**: 1 Day (long-term trading)

## 🎯 Usage Guide

### 1. Connect to XM Account
1. Open the web interface at `http://localhost:5000`
2. Enter your XM account credentials
3. Select Demo or Live server
4. Click "Connect to MT5"

### 2. Configure Trading Parameters
1. Select your desired trading symbol from the grid
2. Choose your preferred timeframe
3. Set risk per trade percentage (1-10%)
4. Configure account size
5. Enable/disable auto trading and ML features

### 3. Run Market Analysis
1. Click "Analyze Market" to perform technical analysis
2. Review the analysis results:
   - Uptrend confirmation
   - Higher highs/lows count
   - Trading signals
   - ML predictions

### 4. Start Automated Trading
1. Review the trading signals
2. Click "Start Trading" to begin automated trading
3. Monitor positions in real-time
4. Use "Stop Trading" to halt automated trading

### 5. Manage Positions
- View all open positions in the positions table
- Close individual positions using the "Close" button
- Close all positions using "Close All Positions"

## 📊 Analysis Features

### Technical Analysis
- **Trend Analysis**: Higher highs and lows detection
- **Pivot Points**: Support and resistance identification
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Volume Analysis**: OBV, volume ratios, volume patterns

### Machine Learning
- **Feature Engineering**: 40+ technical indicators as ML features
- **Random Forest Model**: Ensemble learning for signal prediction
- **Confidence Scoring**: Probability-based trading decisions
- **Model Persistence**: Save and load trained models

### Risk Management
- **Position Sizing**: Automatic calculation based on risk percentage
- **Stop Loss**: Dynamic stop loss placement
- **Take Profit**: Risk-reward ratio optimization
- **Account Protection**: Maximum risk per trade limits

## 🔒 Security Features

### Account Security
- **Environment Variables**: Secure credential storage
- **No Hardcoded Passwords**: All credentials in .env file
- **Connection Validation**: MT5 connection verification
- **Error Handling**: Graceful failure handling

### Trading Safety
- **Demo Account Support**: Test with demo accounts first
- **Risk Limits**: Configurable maximum risk per trade
- **Position Monitoring**: Real-time position tracking
- **Emergency Stop**: Immediate trading halt capability

## 📈 Performance Monitoring

### Real-time Metrics
- **Account Balance**: Live balance updates
- **Equity Tracking**: Real-time equity monitoring
- **Position P&L**: Individual position profit/loss
- **Trading Status**: Connection and trading state

### Historical Data
- **Trade History**: Complete trading record
- **Performance Analytics**: Win rate, profit factor
- **Risk Metrics**: Maximum drawdown, Sharpe ratio
- **ML Model Performance**: Prediction accuracy tracking

## 🚨 Important Notes

### Risk Disclaimer
- **Trading involves substantial risk of loss**
- **Past performance does not guarantee future results**
- **Always test with demo accounts first**
- **Never risk more than you can afford to lose**

### Technical Requirements
- **Stable Internet Connection**: Required for real-time data
- **MetaTrader 5 Running**: Must be active during trading
- **Sufficient Account Balance**: Minimum balance requirements
- **Market Hours**: Consider forex market hours and holidays

### Best Practices
1. **Start with Demo**: Always test with demo accounts first
2. **Small Position Sizes**: Begin with small risk percentages
3. **Monitor Regularly**: Check positions and account status
4. **Keep MT5 Running**: Ensure MetaTrader 5 stays active
5. **Backup Configuration**: Save your .env file securely

## 🐛 Troubleshooting

### Common Issues

#### Connection Problems
```
Error: Failed to connect to MT5
```
**Solutions:**
- Verify XM credentials are correct
- Ensure MetaTrader 5 is running
- Check AutoTrading is enabled in MT5
- Verify internet connection

#### Trading Errors
```
Error: Order failed
```
**Solutions:**
- Check account balance
- Verify symbol is available for trading
- Ensure stop levels meet broker requirements
- Check market hours

#### Analysis Failures
```
Error: Analysis failed
```
**Solutions:**
- Verify symbol and timeframe are valid
- Check data availability for the symbol
- Ensure sufficient historical data
- Restart the application

### Support
For technical support or questions:
1. Check the troubleshooting section
2. Review the error logs
3. Verify all prerequisites are met
4. Test with demo accounts first

## 📝 License

This project is for educational and research purposes. Use at your own risk.

## 🔄 Updates

### Version History
- **v2.0**: Added Vue.js frontend and dynamic symbol selection
- **v1.5**: Enhanced ML capabilities and risk management
- **v1.0**: Initial release with basic MT5 integration

### Future Enhancements
- **Additional Indicators**: More technical analysis tools
- **Backtesting Module**: Historical performance testing
- **Mobile App**: iOS/Android mobile interface
- **Advanced ML**: Deep learning models
- **Multi-Account Support**: Multiple XM accounts
- **Social Trading**: Copy trading features

---

**Happy Trading! 📈** 