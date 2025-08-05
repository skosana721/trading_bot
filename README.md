# Advanced Trading Bot with Smart Money Concepts & XM Integration

A comprehensive automated trading bot featuring Smart Money Concepts (SMC), modern Vue.js frontend, MetaTrader 5 integration, and advanced machine learning capabilities for professional forex trading.

## 🚀 Features

### Core Trading Features
- **Smart Money Concepts (SMC)**: Order blocks, Fair Value Gaps, BOS/ChoCH detection
- **Advanced Trend Analysis**: Higher Highs/Lows and Lower Highs/Lows detection
- **Dual-Direction Trading**: Both uptrend (BUY) and downtrend (SELL) signal generation
- **Dynamic Symbol Selection**: Trade 27+ forex pairs available on XM
- **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d timeframes
- **XM Account Integration**: Direct connection to XM demo and live accounts
- **Machine Learning Analysis**: AI-powered signal generation and prediction

### Smart Money Concepts Implementation
- **Order Blocks**: Institutional buy/sell zones detection
- **Fair Value Gaps (FVG)**: Price gap analysis and fill detection
- **Break of Structure (BOS)**: Trend continuation patterns
- **Change of Character (ChoCH)**: Trend reversal patterns
- **Liquidity Zones**: Support/resistance with stop-loss clustering
- **Market Structure Analysis**: Trend phase and volatility assessment

### Frontend Features
- **Modern Vue.js UI**: Beautiful, responsive interface
- **Real-time Updates**: Live status updates every 5 seconds
- **Configuration Persistence**: Auto-saves account credentials
- **Position Management**: Individual and bulk position closing
- **SMC Analysis Display**: Visual SMC results and confidence scoring

### Risk Management
- **Dynamic Position Sizing**: Based on account balance and risk percentage
- **SMC-Enhanced Stop Losses**: Using liquidity zones instead of fixed pips
- **Intelligent Target Placement**: At resistance/support levels
- **Confidence-Based Trading**: Signal strength filtering
- **Account Protection**: Maximum risk per trade limits

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
1. Install MetaTrader 5 from XM
2. Enable AutoTrading:
   - Tools → Options → Expert Advisors
   - Check "Allow automated trading"
   - Check "Allow DLL imports"
   - Check "Allow WebRequest for listed URL"
3. Restart MetaTrader 5
4. Ensure the "AutoTrading" button is green (enabled)

## 🛠️ Quick Start

### 1. Installation
```bash
git clone <repository-url>
cd trading_bot
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python app.py
```

### 3. Access the Interface
Open your browser and navigate to: `http://localhost:5000`

### 4. Connect to XM Account
1. Enter your XM account credentials in the web interface
2. Select Demo or Live server
3. Click "Connect to MT5"
4. Credentials are automatically saved for future use

### 5. Run SMC Analysis
1. Select trading symbol and timeframe
2. Click "Analyze Market" for SMC-enhanced analysis
3. Review comprehensive analysis results
4. Execute trades based on SMC signals

## 🧠 Smart Money Concepts Analysis

### Order Blocks Detection
```
📦 Found 3 significant Order Blocks
   - Bullish Order Block: Strong green candle after consolidation
   - Bearish Order Block: Strong red candle indicating institutional selling
   - Strength scoring based on volume expansion
```

### Fair Value Gaps Analysis
```
📊 Found 2 unfilled Fair Value Gaps
   - Bullish FVG: Price gap requiring upward fill
   - Bearish FVG: Price gap requiring downward fill
   - Fill tracking and invalidation detection
```

### Market Structure Analysis
```
🔄 Found 1 recent BOS/ChoCH patterns
   - BOS (Break of Structure): Trend continuation
   - ChoCH (Change of Character): Trend reversal
   - Swing point identification and strength scoring
```

### Liquidity Zones
```
💧 Found 5 liquidity zones
   - Support zones: Areas with clustered buy-side liquidity
   - Resistance zones: Areas with clustered sell-side liquidity
   - Multiple touch confirmation
```

## 📊 Analysis Output

### Console Output Example
```
============================================================
🔍 STARTING MARKET ANALYSIS
============================================================
📊 Request: EURUSD 1h
📈 Uptrend analysis: ✅ Found
📉 Downtrend analysis: ❌ None
📈 Using uptrend (strength: 75.2)
🧠 Running Smart Money Concepts analysis...
📦 Found 3 significant Order Blocks
📊 Found 2 unfilled Fair Value Gaps
🔄 Found 1 recent BOS/ChoCH patterns
💧 Found 5 liquidity zones
✅ SMC Analysis complete - Score: 68/100, Bias: BULLISH
🎯 Generating SMC-enhanced trading signals...
📈 SMC BUY Signal - Traditional: 75.2, SMC: 68.2, Confidence: 85
✅ SMC-enhanced trading signals generated
============================================================
🎉 ANALYSIS COMPLETED SUCCESSFULLY
============================================================
```

### API Response Format
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "1h",
  "analysis": {
    "trend_direction": "UPTREND",
    "smc_bias": "BULLISH",
    "smc_score": 68.2,
    "overall_strength": 75.2
  },
  "signals": {
    "signal_type": "BUY",
    "entry_price": 1.1585,
    "stop_loss": 1.1560,     // SMC-based (support zone)
    "target": 1.1635,        // SMC-based (resistance zone)
    "signal_confidence": 85,
    "smc_context": {
      "order_blocks": 3,
      "fair_value_gaps": 2,
      "structure_breaks": 1
    }
  },
  "smc": {
    "smc_score": 68.2,
    "current_bias": "BULLISH",
    "order_blocks": 3,
    "fair_value_gaps": 2,
    "bos_choch_patterns": 1,
    "liquidity_zones": 5,
    "market_phase": "expansion"
  }
}
```

## 🎯 Trading Strategy

### SMC-Enhanced Signals
- **Traditional Trend Analysis** + **Smart Money Concepts** = **High-Probability Setups**
- **Confluence Trading**: Multiple confirmations required
- **Institutional Levels**: Stop losses and targets at real liquidity zones
- **Confidence Scoring**: 0-100 score helps filter weak signals

### Signal Types
#### Bullish Signals (BUY)
- Uptrend confirmed + Bullish SMC bias
- Entry at bullish order blocks or FVG fills
- Stop loss below support zones
- Target at resistance zones

#### Bearish Signals (SELL)
- Downtrend confirmed + Bearish SMC bias  
- Entry at bearish order blocks or FVG fills
- Stop loss above resistance zones
- Target at support zones

## 🔧 Configuration

### Available Symbols (27+ Forex Pairs)
- **Major Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Cross Pairs**: EURGBP, EURJPY, GBPJPY, CHFJPY, AUDCAD, AUDCHF, AUDJPY
- **Exotic Pairs**: AUDNZD, CADCHF, CADJPY, EURAUD, EURCAD, EURCHF, EURNZD

### Configuration Persistence
All settings are automatically saved to `.env` file:
```env
XM_ACCOUNT_NUMBER=your_account_number
XM_PASSWORD=your_password
XM_SERVER=XMGlobal-Demo
DEFAULT_SYMBOL=EURUSD
DEFAULT_TIMEFRAME=1h
RISK_PER_TRADE=2.0
AUTO_TRADE=false
USE_ML=true
```

## 📁 Project Structure

```
trading_bot/
├── app.py                      # Main Flask application with SMC
├── mt5_trading_bot.py         # MT5-specific trading bot
├── mt5_connector.py           # MetaTrader 5 connection handler
├── trading_bot.py             # Core trading analysis engine
├── templates/
│   └── index.html             # Vue.js frontend interface
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (auto-generated)
├── .gitignore                # Git ignore patterns
└── README.md                 # This documentation
```

## 🚨 Risk Disclaimer

- **Trading involves substantial risk of loss**
- **Past performance does not guarantee future results**
- **Smart Money Concepts are educational tools, not guarantees**
- **Always test with demo accounts first**
- **Never risk more than you can afford to lose**
- **This is not financial advice**

## 🔒 Security Features

### Account Security
- **Environment Variables**: Secure credential storage
- **Auto-save Credentials**: Persistent configuration
- **Connection Validation**: MT5 connection verification
- **Graceful Error Handling**: Safe failure management

### Trading Safety
- **Demo Account Support**: Risk-free testing
- **Confidence-Based Filtering**: Only high-probability signals
- **Dynamic Risk Management**: Adaptive position sizing
- **Emergency Stop**: Immediate trading halt capability

## 🐛 Troubleshooting

### Common Issues

#### Analysis Errors
```
Error: Response formatting error
```
**Solution**: Enhanced error handling now provides detailed console output for debugging

#### Connection Problems
```
Error: Failed to connect to MT5
```
**Solutions:**
- Verify XM credentials are correct
- Ensure MetaTrader 5 is running
- Check AutoTrading is enabled in MT5
- Verify internet connection

#### SMC Analysis Issues
```
Warning: SMC Analysis failed, using traditional analysis only
```
**Solutions:**
- Ensure sufficient historical data (minimum 100 bars)
- Try different timeframes
- Check symbol availability

## 🔄 Version History

- **v3.0**: Smart Money Concepts integration, dual-direction trading
- **v2.5**: Configuration persistence, enhanced error handling  
- **v2.0**: Vue.js frontend, dynamic symbol selection
- **v1.5**: Machine learning capabilities, risk management
- **v1.0**: Basic MT5 integration

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional SMC patterns
- Advanced ML models
- Backtesting capabilities
- Mobile interface
- Multi-timeframe analysis

---

**Trade Like the Institutions with Smart Money Concepts! 🧠📈**