# Trading Bot - Multi-Market Uptrend Detection

A Python-based trading bot that identifies uptrends in **stocks**, **cryptocurrencies**, **forex**, and **commodities** using the **Higher Highs (HH) and Higher Lows (HL)** methodology on various timeframes.

## 🎯 Overview

This trading bot implements the classic technical analysis approach for identifying uptrends across multiple markets:

- **Uptrend Definition**: Market moves in higher highs and higher lows
- **Confirmation**: Requires at least 2 HH and 2 HL patterns to confirm uptrend
- **Analysis Method**: Uses trendlines connecting pivot points on various timeframes
- **Multi-Market Support**: Stocks, Cryptocurrencies, Forex, Commodities
- **Visualization**: Interactive charts with pivot points, trendlines, and pattern identification

## ✨ Features

- **Multi-Market Data**: Fetches live market data from Yahoo Finance
- **Pivot Point Detection**: Automatically identifies swing highs and lows
- **Pattern Recognition**: Detects Higher Highs (HH) and Higher Lows (HL) patterns
- **Trendline Analysis**: Creates trendlines connecting pivot points
- **Interactive Charts**: Beautiful Plotly-based visualizations
- **Detailed Reports**: Comprehensive analysis reports with trading recommendations
- **Multiple Timeframes**: Supports various analysis periods (1h, 1d, 1w, 1mo, 1y)
- **Auto-Detection**: Automatically detects market type based on symbol

## 📋 Requirements

- Python 3.7+
- Internet connection for data fetching
- Required packages (see `requirements.txt`)

## 🚀 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py AAPL 1y stock
   ```

## 📊 Usage

### Basic Usage

Analyze different markets:
```bash
python main.py AAPL 1y stock      # Stock analysis
python main.py BTC 1h crypto      # Cryptocurrency analysis
python main.py EURUSD 1d forex    # Forex analysis
python main.py GOLD 1w commodities # Commodity analysis
```

### Command Line Options

```bash
python main.py [SYMBOL] [PERIOD] [MARKET_TYPE]
```

**Parameters:**
- `SYMBOL`: Market symbol (e.g., AAPL, BTC, EURUSD, GOLD)
- `PERIOD`: Analysis period (1h, 1d, 1w, 1mo, 1y, 2y)
- `MARKET_TYPE`: Market type (stock, crypto, forex, commodities)

**Examples:**
```bash
python main.py AAPL 1y stock      # Analyze Apple stock for 1 year
python main.py BTC 1h crypto      # Analyze Bitcoin for 1 hour
python main.py ETH 1d crypto      # Analyze Ethereum for 1 day
python main.py EURUSD 1w forex    # Analyze EUR/USD for 1 week
python main.py GOLD 1mo commodities # Analyze Gold for 1 month
```

### Available Symbols

View all available symbols:
```bash
python main.py --symbols
```

### Multiple Market Analysis

Run analysis on multiple markets:
```bash
python main.py --examples
```

This will analyze: AAPL (stock), BTC (crypto), ETH (crypto), EURUSD (forex), GOLD (commodities)

### Programmatic Usage

```python
from trading_bot import TradingBot

# Stock analysis
bot = TradingBot("AAPL", "1y", "stock")
trend_analysis = bot.analyze_uptrend()

# Crypto analysis
bot = TradingBot("BTC", "1h", "crypto")
trend_analysis = bot.analyze_uptrend()

# Forex analysis
bot = TradingBot("EURUSD", "1d", "forex")
trend_analysis = bot.analyze_uptrend()

# Generate report and chart
bot.generate_report(trend_analysis)
bot.plot_analysis(trend_analysis)
```

## 📈 How It Works

### 1. Data Collection
- Fetches historical OHLCV data from Yahoo Finance
- Supports various time periods and intervals
- Auto-detects market type based on symbol

### 2. Pivot Point Detection
- Identifies swing highs and lows using a sliding window approach
- Configurable window size for pivot detection sensitivity

### 3. Pattern Recognition
- **Higher Highs (HH)**: Each new high is higher than the previous high
- **Higher Lows (HL)**: Each new low is higher than the previous low
- Requires minimum 2 HH and 2 HL points for uptrend confirmation

### 4. Trendline Creation
- Connects pivot points to create trendlines
- HH trendline: Connects higher highs
- HL trendline: Connects higher lows

### 5. Analysis Output
- **Console Report**: Detailed text analysis
- **Interactive Chart**: Visual representation with all elements
- **Trading Signals**: Buy/sell recommendations based on trend status

## 📊 Output Examples

### Console Output
```
============================================================
TRADING BOT ANALYSIS REPORT - BTC (CRYPTO)
============================================================
Analysis Period: 2024-01-01 to 2024-01-15
Total Data Points: 360
Pivot Points Found: 28

UPTREND ANALYSIS:
  Higher Highs (HH): 4
  Higher Lows (HL): 3
  Uptrend Confirmed: ✅ YES

✅ TRADING SIGNAL: UPTREND DETECTED
   The crypto market is showing a confirmed uptrend with:
   - At least 2 higher highs (HH)
   - At least 2 higher lows (HL)
   - Trendlines connecting pivot points

   HH Trendline: 2024-01-10 (45,200) → 2024-01-15 (46,800)
   HL Trendline: 2024-01-08 (44,100) → 2024-01-12 (44,900)
============================================================
```

### Interactive Chart Features
- **Candlestick Chart**: Price action visualization
- **Pivot Points**: Red triangles (highs), green triangles (lows)
- **Higher Highs/Lows**: Diamond markers with connecting lines
- **Trendlines**: Dashed lines connecting pivot points
- **Legend**: Interactive legend for all chart elements

## 🎯 Trading Strategy

### Uptrend Confirmed (✅)
- **Action**: Consider LONG positions
- **Entry**: Look for pullbacks to trendline support
- **Stop Loss**: Below recent higher low
- **Target**: Next resistance level

### No Uptrend (❌)
- **Action**: Avoid long positions
- **Strategy**: Wait for uptrend confirmation
- **Monitoring**: Watch for trend reversal signals

## 🔧 Configuration

### Pivot Detection Sensitivity
```python
bot.find_pivot_points(window=5)  # Adjust window size (3-10 recommended)
```

### Uptrend Confirmation Threshold
```python
trend_analysis = bot.identify_higher_highs_lows(min_points=2)  # Minimum HH/HL points
```

### Market Type Detection
```python
# Auto-detection
bot = TradingBot("BTC", "1h")  # Automatically detects as crypto

# Manual specification
bot = TradingBot("BTC", "1h", "crypto")  # Explicitly specify market type
```

## 📁 Project Structure

```
Bot/
├── trading_bot.py      # Main trading bot class
├── main.py            # Command-line interface
├── example.py         # Usage examples
├── simple_trading_bot.py # Demo version
├── setup.py           # Installation helper
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🛠️ Customization

### Adding New Market Types
```python
def add_custom_market(self):
    # Add your custom market types here
    pass
```

### Modifying Analysis Logic
```python
def custom_uptrend_logic(self):
    # Implement your custom uptrend detection logic
    pass
```

## ⚠️ Disclaimer

This trading bot is for **educational and research purposes only**. 

- **Not Financial Advice**: The analysis and recommendations are not financial advice
- **Risk Warning**: Trading involves substantial risk of loss
- **Backtesting**: Past performance does not guarantee future results
- **Due Diligence**: Always conduct your own research before making trading decisions
- **Market Volatility**: Cryptocurrency and forex markets are highly volatile

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- Additional market types

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the console output for error messages
2. Verify your internet connection for data fetching
3. Ensure all dependencies are installed correctly
4. Open an issue with detailed error information

---

**Happy Trading! 📈** 