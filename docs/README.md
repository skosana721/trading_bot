# 🤖 Advanced Trading Bot System

A comprehensive trading bot system that combines manual analysis, automated trading, MT5 integration, and advanced strategies including Market Structure Analysis and ML Ensemble predictions.

## 🚀 Key Features

### 🎯 Core Trading System
- **MT5 Integration**: Direct connection to MetaTrader 5 for real-time trading
- **Multi-Strategy Support**: Traditional analysis, Smart Money Concepts, Market Structure, ML Ensemble
- **Risk Management**: Advanced position sizing and risk control
- **Multi-Market Support**: Forex, Stocks, Crypto, and Commodities
- **Web Interface**: Real-time monitoring and control via web API

### 🏗️ Market Structure Strategy
- **Multi-Timeframe Analysis**: D1 trend filter, H4 structure/zones, H1 entry signals
- **Market Structure Detection**: Higher Highs/Lows, Lower Highs/Lows patterns
- **Support/Resistance Zones**: Automatic zone detection with strength calculation
- **Candlestick Patterns**: Bullish/Bearish Engulfing pattern recognition
- **Advanced Trade Management**: Re-entry logic, trailing stops, risk management

### 🤖 ML Ensemble System
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, SVM, Neural Networks, LSTM
- **Ensemble Predictions**: Combines predictions from all models for better accuracy
- **Dynamic Weighting**: Automatically adjusts model weights based on performance
- **Advanced Features**: 50+ technical indicators, real-time performance tracking
- **Model Persistence**: Save and load trained models for consistent performance

### 🌐 Smart Money Concepts
- **Market Structure Analysis**: Advanced price action and structure recognition
- **Order Block Detection**: Identifies institutional order blocks
- **Fair Value Gaps**: Detects and analyzes FVG patterns
- **Liquidity Analysis**: Finds liquidity pools and sweeps

## 📋 Prerequisites

- Python 3.8 or higher
- MetaTrader 5 terminal installed and running
- XM trading account (demo or live)
- Internet connection for market data

## 📁 Project Structure

The project is organized into logical folders for better maintainability:

```
trading_bot/
├── core/                    # Core application files
│   ├── app.py              # Main Flask application
│   ├── mt5_trading_bot.py  # Main trading bot logic
│   ├── trading_bot.py      # Legacy trading bot
│   └── error_handler.py    # Error handling system
├── connectors/             # External service connectors
│   └── mt5_connector.py    # MetaTrader 5 connector
├── strategies/             # Trading strategies
│   ├── ml_ensemble.py      # ML ensemble system
│   ├── market_structure_strategy.py
│   ├── smart_money_concept.py
│   └── reinforcement_learning_trader.py
├── config/                 # Configuration files
│   ├── config.py           # Main configuration
│   ├── deployment_config.py
│   └── tensorflow_config.py
├── tests/                  # Test files
│   ├── test_deployment_fix.py
│   ├── test_live_trading.py
│   └── test_warnings_fix.py
├── scripts/                # Utility scripts
│   ├── start_unified_bot.py
│   └── startup.py
├── docs/                   # Documentation
│   ├── README.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── LIVE_TRADING_SETUP.md
├── models/                 # Trained ML models
├── logs/                   # Log files
├── templates/              # Web templates
└── utils/                  # Utility modules
    └── backoff.py
```

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

3. **Configure credentials**: Enter account number, password, and server via the web UI. Do not store XM credentials in `.env`.
   You may still configure non-credential settings in `.env` (e.g., `RISK_PER_TRADE`, `USE_ML`).

## 🚀 Quick Start

### Unified Startup Script (Recommended)
   ```bash
# Test MT5 connection
python scripts/start_unified_bot.py test

# Run enhanced trading bot with automation
python scripts/start_unified_bot.py enhanced --symbol EURUSD --timeframe 5m

# Run fully automated trading bot
python scripts/start_unified_bot.py automated

# Run web interface
python scripts/start_unified_bot.py web
```

### Individual Components
   ```bash
# Enhanced trading bot (manual + automated)
python core/mt5_trading_bot.py

# Web interface
   python core/app.py
   ```

### Easy Startup Scripts
   ```bash
# Start the web application (recommended)
python start_app.py

# Run all tests
python run_tests.py
   ```

## 📊 Available Strategies

### 1. Traditional Analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Trend Analysis**: Multiple timeframe trend identification
- **Signal Generation**: Buy/sell signals based on indicator confluence

### 2. Market Structure Strategy
- **Trend Definition**: Based on Price vs 50 EMA (D1) and HH/HL/LH/LL detection
- **Entry Conditions**: 
  - Buy: Uptrend + Support Zone + Bullish Engulfing
  - Sell: Downtrend + Resistance Zone + Bearish Engulfing
- **Trade Management**: Zone-based SL, TP multiplier, re-entry logic, trailing stops

### 3. Smart Money Concepts
- **Order Blocks**: Institutional order block detection
- **Fair Value Gaps**: FVG pattern recognition and analysis
- **Liquidity Sweeps**: Liquidity pool identification and sweep detection
- **Market Structure**: Advanced price action analysis

### 4. ML Ensemble System
- **Multiple Models**: 6 different ML algorithms working together
- **Feature Engineering**: 50+ technical indicators and time-based features
- **Ensemble Voting**: Soft voting mechanism for final predictions
- **Performance Tracking**: Real-time model performance monitoring

### 5. Reinforcement Learning System v2.0
- **Advanced State Representation**: 16-dimensional state space with volatility, volume, time features
- **Market Regime Detection**: Automatic trending vs ranging market classification
- **Experience Replay Buffer**: Stable learning with mini-batch training
- **Risk-Adjusted Rewards**: Rewards based on profit/risk ratio and risk management
- **Advanced Exploration**: Epsilon-greedy with decay and softmax action selection
- **Risk Management Integration**: Position sizing, drawdown limits, cooldown periods

## 🔧 Configuration

### Market Structure Strategy Configuration
```python
strategy_config = {
    'UsePairs': ['EURUSD', 'GBPUSD'],
    'LotSizeInitial': 0.01,
    'RiskPerTrade': 2.0,
    'RiskRewardRatio': 2.0,
    'SL_Buffer_Pips': 10,
    'TP_Multiplier': 2.0,
    'EnableTrailingStop': False,
    'TrailStartProfitPips': 50,
    'TrailStepPips': 10
}
```

### ML Ensemble Configuration
```python
ml_config = {
    'use_deep_learning': True,
    'feature_selection': True,
    'ensemble_voting': 'soft',
    'model_persistence': True
}
```

### Reinforcement Learning Configuration
```python
rl_config = {
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'epsilon': 0.3,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'use_experience_replay': True,
    'replay_buffer_size': 10000,
    'batch_size': 32,
    'use_market_regime': True,
    'max_drawdown_limit': 0.1,
    'cooldown_period': 5
}
```

## 🌐 Web API Endpoints

### Core Endpoints
```http
GET /api/status                    # Bot status and connection info
POST /api/connect                  # Connect to MT5
POST /api/config                   # Configure trading parameters
POST /api/start-automated-trading  # Start automated trading
POST /api/stop-automated-trading   # Stop automated trading
GET /api/analysis/<symbol>         # Get market analysis
```

### Strategy-Specific Endpoints
```http
GET /api/market_structure_analysis/<symbol>/<timeframe>    # Market structure analysis
GET /api/market_structure_summary/<symbol>/<timeframe>     # Strategy summary
GET /api/ml_ensemble_summary/<symbol>                      # ML ensemble summary
GET /api/rl_analysis/<symbol>/<timeframe>                  # Reinforcement learning analysis
GET /api/rl_summary/<symbol>/<timeframe>                   # RL performance metrics
GET /api/smc_analysis/<symbol>                             # Smart Money Concepts
```

### Combined Analysis
```http
GET /api/combined-analysis/<symbol>  # All strategies combined
```

## 📈 Performance Monitoring

### Real-time Metrics
- **Connection Status**: MT5 connection health
- **Active Positions**: Current open trades and P&L
- **Strategy Performance**: Individual strategy metrics
- **Risk Metrics**: Current risk exposure and limits

### Strategy Performance
- **Market Structure**: Signal accuracy and win rate
- **ML Ensemble**: Model performance and prediction confidence
- **Reinforcement Learning**: Q-table learning progress, market regime detection, risk metrics
- **Smart Money Concepts**: SMC pattern detection accuracy
- **Combined Signals**: Overall strategy performance

## 🛡️ Risk Management

### Position Sizing
- **Risk Per Trade**: Configurable percentage (default 2%)
- **Account Balance**: Automatic position size calculation
- **Maximum Positions**: Limits per symbol and total

### Stop Loss & Take Profit
- **Zone-Based SL**: Support/resistance zone-based stop losses
- **TP Multiplier**: Configurable take profit ratios
- **Trailing Stops**: Optional trailing stop functionality

### Risk Controls
- **Daily Loss Limits**: Maximum daily loss protection
- **Correlation Limits**: Avoid correlated pair exposure
- **News Filter**: Optional news event filtering

## 🔍 Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   ```bash
   python scripts/start_unified_bot.py test
   ```
   Check your credentials in the web UI

2. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**
   ```bash
   python scripts/start_unified_bot.py web --port 5001
   ```

4. **Strategy Not Working**
   - Check MT5 connection
   - Verify symbol configuration
   - Review log files in `logs/` directory

### Log Files
- `logs/unified_bot.log` - Main bot logs
- `logs/trading_bot.log` - Web interface logs

## 📚 File Structure

```
trading_bot/
├── app.py                          # Web interface and API
├── mt5_trading_bot.py              # Main trading bot
├── mt5_connector.py                # MT5 connection handler
├── start_unified_bot.py            # Unified startup script
├── market_structure_strategy.py    # Market structure strategy
├── smart_money_concept.py          # Smart Money Concepts
├── ml_ensemble.py                  # ML ensemble system
├── reinforcement_learning_trader.py # RL trading system
├── config.py                       # Configuration management
├── error_handler.py                # Error handling utilities
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── utils/                          # Utility functions
├── templates/                      # Web templates
├── analysis/                       # Analysis modules
├── logs/                           # Log files
└── models/                         # ML model storage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly on demo accounts before using with real money.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for detailed error information
3. Verify configuration parameters
4. Test with demo accounts first
