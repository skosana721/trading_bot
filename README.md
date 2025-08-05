# Dynamic Trading Bot - XM Integration

A comprehensive automated trading bot with MetaTrader 5 integration, featuring advanced technical analysis, machine learning, and Smart Money Concepts (SMC) for enhanced trading signals.

## 🚀 Recent Improvements

### Enhanced Error Handling & Monitoring
- **Comprehensive Error Management**: Custom exception classes with severity levels and categories
- **Performance Monitoring**: Real-time tracking of execution times and system metrics
- **Centralized Logging**: Structured logging with file rotation and multiple log levels
- **Error Recovery**: Automatic retry mechanisms with exponential backoff
- **Alert System**: Configurable alerts for critical errors and performance issues

### Configuration Management
- **Centralized Configuration**: Type-safe configuration management with validation
- **Environment Variable Support**: Flexible configuration through .env files
- **Runtime Validation**: Automatic validation of trading parameters and settings
- **Configuration Hot-Reloading**: Dynamic configuration updates without restart

### Frontend Enhancements
- **Improved User Experience**: Better loading states, error handling, and feedback
- **Real-time Status Updates**: Automatic status polling with connection health monitoring
- **Enhanced Error Display**: Detailed error information with retry options
- **Responsive Design**: Better mobile compatibility and accessibility
- **Performance Metrics**: Real-time display of system performance and trading metrics

### Trading Bot Improvements
- **Robust Position Sizing**: Dynamic position sizing based on real-time account balance
- **Enhanced Risk Management**: Multi-level risk controls and position limits
- **Connection Resilience**: Automatic reconnection and connection health monitoring
- **Volume Validation**: Strict compliance with broker volume requirements
- **Margin Management**: Intelligent margin checking and position size adjustment

## 🏗️ Architecture

```
trading_bot/
├── app.py                 # Flask backend API
├── config.py              # Configuration management
├── error_handler.py       # Error handling and monitoring
├── mt5_connector.py       # MT5 integration layer
├── mt5_trading_bot.py     # Core trading logic
├── smart_money_concept.py # SMC analysis
├── templates/
│   └── index.html         # Vue.js frontend
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- MetaTrader 5 terminal
- XM trading account (demo or live)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd trading_bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp xm_config_example.txt .env
   # Edit .env with your XM credentials
   ```

4. **Run the setup script**
   ```bash
   python setup_mt5.py
   ```

5. **Start the application**
   ```bash
   python app.py
   ```

## 📊 Features

### Core Trading Features
- **Multi-Timeframe Analysis**: Support for 1m to 1d timeframes
- **Technical Analysis**: RSI, MACD, Bollinger Bands, ATR, and more
- **Machine Learning**: Random Forest classifier for price prediction
- **Smart Money Concepts**: Order blocks, fair value gaps, liquidity zones
- **Risk Management**: Configurable risk per trade and position limits
- **Auto Trading**: Fully automated trading with manual override options

### Advanced Features
- **Real-time Monitoring**: Live position tracking and P&L monitoring
- **Performance Analytics**: Detailed trading performance metrics
- **Error Recovery**: Automatic error handling and system recovery
- **Configuration Management**: Centralized settings with validation
- **Logging & Monitoring**: Comprehensive logging and performance tracking

## 🔧 Configuration

### Environment Variables
```bash
# XM Account Configuration
XM_ACCOUNT_NUMBER=your_account_number
XM_PASSWORD=your_password
XM_SERVER=XMGlobal-Demo  # or XMGlobal-Live

# Trading Parameters
TRADING_SYMBOL=EURUSD
TRADING_TIMEFRAME=5m
RISK_PER_TRADE=0.02  # 2%

# System Configuration
LOG_LEVEL=INFO
AUTO_TRADE=false
USE_ML=true
USE_SMC=true

# Risk Management
MAX_POSITIONS_PER_SYMBOL=3
MAX_SAME_DIRECTION_POSITIONS=2
MAX_DAILY_TRADES=10
MAX_DAILY_LOSS=0.05  # 5%
```

### Trading Parameters
- **Risk Per Trade**: 0.1% to 10% of account balance
- **Position Limits**: Configurable per-symbol and direction limits
- **Daily Limits**: Maximum trades and loss limits per day
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d

## 📈 Usage

### Web Interface
1. Open `http://localhost:5000` in your browser
2. Enter your XM credentials
3. Configure trading parameters
4. Connect to MT5
5. Start automated trading

### API Endpoints
- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration
- `POST /api/connect` - Connect to MT5
- `POST /api/analyze` - Perform market analysis
- `POST /api/start-trading` - Start automated trading
- `POST /api/stop-trading` - Stop automated trading
- `GET /api/status` - Get system status
- `GET /api/positions` - Get open positions

## 🔍 Error Handling

### Error Categories
- **Connection Errors**: MT5 connection issues
- **Trading Errors**: Order execution failures
- **Analysis Errors**: Market analysis problems
- **Data Errors**: Data retrieval issues
- **Configuration Errors**: Invalid settings
- **System Errors**: General system issues

### Error Severity Levels
- **LOW**: Informational messages
- **MEDIUM**: Warnings and non-critical issues
- **HIGH**: Important issues requiring attention
- **CRITICAL**: System-threatening errors

### Error Recovery
- Automatic retry with exponential backoff
- Connection health monitoring
- Graceful degradation of features
- Detailed error logging and reporting

## 📊 Performance Monitoring

### Metrics Tracked
- **Execution Times**: Function and API call performance
- **Error Rates**: Error frequency and types
- **Connection Health**: MT5 connection stability
- **Trading Performance**: Win rate, profit/loss, drawdown
- **System Resources**: Memory usage, CPU utilization

### Monitoring Features
- Real-time performance dashboards
- Historical performance analysis
- Automated alerting for issues
- Performance trend analysis

## 🛡️ Security

### Security Features
- **Credential Management**: Secure storage of trading credentials
- **Input Validation**: Comprehensive validation of all inputs
- **Error Sanitization**: Safe error message handling
- **Access Control**: API endpoint protection
- **Audit Logging**: Complete audit trail of all actions

### Best Practices
- Use demo accounts for testing
- Regularly update credentials
- Monitor system logs
- Implement proper backup procedures
- Follow risk management guidelines

## 🚨 Risk Warning

**Trading involves substantial risk of loss and is not suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade, you should carefully consider your investment objectives, level of experience, and risk appetite.**

### Risk Management Features
- Configurable position sizing
- Stop-loss and take-profit orders
- Maximum position limits
- Daily loss limits
- Automatic position monitoring

## 🔧 Development

### Code Quality
- Type hints and documentation
- Comprehensive error handling
- Unit tests and integration tests
- Code linting and formatting
- Performance optimization

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 Changelog

### Version 2.0.0 (Current)
- **Major**: Enhanced error handling and monitoring system
- **Major**: Centralized configuration management
- **Major**: Improved frontend with better UX
- **Major**: Robust position sizing and risk management
- **Minor**: Performance optimizations and bug fixes

### Version 1.0.0
- Initial release with basic trading functionality
- MT5 integration
- Technical analysis features
- Web interface

## 📞 Support

For support and questions:
- Check the logs in the `logs/` directory
- Review error messages in the web interface
- Consult the configuration documentation
- Open an issue on GitHub

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MetaTrader 5 for trading platform integration
- XM for broker services
- Vue.js for frontend framework
- Flask for backend API
- Technical analysis libraries (ta, pandas, numpy)
- Machine learning libraries (scikit-learn)

---

**Disclaimer**: This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.