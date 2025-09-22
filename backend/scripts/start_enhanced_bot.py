#!/usr/bin/env python3
"""
Enhanced Trading Bot Startup Script
==================================

This script demonstrates how to start and use the enhanced trading bot
with all advanced features including risk management, ML ensemble,
market regime detection, and real-time data processing.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced trading bot components
try:
    from core.enhanced_trading_bot import EnhancedTradingBot
    from config.enhanced_trading_config import ENHANCED_TRADING_CONFIG
    from analysis.advanced_backtesting import AdvancedBacktester
    ENHANCED_BOT_AVAILABLE = True
except ImportError as e:
    print(f"Error importing enhanced trading bot: {e}")
    ENHANCED_BOT_AVAILABLE = False

# Import existing components for fallback
try:
    from core.mt5_trading_bot import MT5TradingBot
    from connectors.mt5_connector import MT5Connector
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/enhanced_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    return logging.getLogger(__name__)

def generate_sample_data(symbols: list, days: int = 365) -> dict:
    """Generate sample data for testing"""
    logger = logging.getLogger(__name__)
    logger.info("Generating sample data for testing...")
    
    data = {}
    
    for symbol in symbols:
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='5min')
        
        # Generate price series with trend and volatility
        if 'USD' in symbol:
            base_price = 1.1000
            volatility = 0.0001
        elif 'XAU' in symbol:
            base_price = 1800.0
            volatility = 5.0
        else:
            base_price = 100.0
            volatility = 1.0
        
        # Generate returns with some autocorrelation
        returns = np.random.normal(0, volatility, len(dates))
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Add some momentum
        
        # Generate price series
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Ensure high >= low and high/low >= open/close
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        df.set_index('datetime', inplace=True)
        data[symbol] = df
        
        logger.info(f"Generated {len(df)} data points for {symbol}")
    
    return data

def run_backtest_demo():
    """Run backtest demonstration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting backtest demonstration...")
    
    if not ENHANCED_BOT_AVAILABLE:
        logger.error("Enhanced trading bot not available")
        return
    
    # Initialize enhanced trading bot
    bot = EnhancedTradingBot(ENHANCED_TRADING_CONFIG)
    
    # Generate sample data
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    data = generate_sample_data(symbols, days=180)  # 6 months of data
    
    # Train ML models
    logger.info("Training ML models...")
    bot.train_ml_models(data)
    
    # Run backtest
    logger.info("Running backtest...")
    backtest_results = bot.run_backtest(data)
    
    # Generate report
    report = bot.generate_trading_report(backtest_results)
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(report)
    
    # Save results
    with open('logs/backtest_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Backtest demonstration completed")

def run_walk_forward_demo():
    """Run walk-forward analysis demonstration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting walk-forward analysis demonstration...")
    
    if not ENHANCED_BOT_AVAILABLE:
        logger.error("Enhanced trading bot not available")
        return
    
    # Initialize enhanced trading bot
    bot = EnhancedTradingBot(ENHANCED_TRADING_CONFIG)
    
    # Generate sample data
    symbols = ['EURUSD', 'GBPUSD']
    data = generate_sample_data(symbols, days=365)  # 1 year of data
    
    # Run walk-forward analysis
    logger.info("Running walk-forward analysis...")
    wf_results = bot.run_walk_forward_analysis(data)
    
    # Analyze results
    for symbol, results in wf_results.items():
        if results:
            returns = [r.total_return for r in results]
            sharpes = [r.sharpe_ratio for r in results]
            
            logger.info(f"{symbol} Walk-Forward Analysis:")
            logger.info(f"  Average Return: {np.mean(returns):.2%}")
            logger.info(f"  Return Std: {np.std(returns):.2%}")
            logger.info(f"  Average Sharpe: {np.mean(sharpes):.2f}")
            logger.info(f"  Sharpe Std: {np.std(sharpes):.2f}")
            logger.info(f"  Number of periods: {len(results)}")
    
    logger.info("Walk-forward analysis demonstration completed")

def run_monte_carlo_demo():
    """Run Monte Carlo analysis demonstration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Monte Carlo analysis demonstration...")
    
    if not ENHANCED_BOT_AVAILABLE:
        logger.error("Enhanced trading bot not available")
        return
    
    # Initialize enhanced trading bot
    bot = EnhancedTradingBot(ENHANCED_TRADING_CONFIG)
    
    # Generate sample data
    symbols = ['EURUSD']
    data = generate_sample_data(symbols, days=180)
    
    # Train models and run backtest
    bot.train_ml_models(data)
    backtest_results = bot.run_backtest(data)
    
    # Run Monte Carlo analysis
    logger.info("Running Monte Carlo analysis...")
    mc_results = bot.run_monte_carlo_analysis(backtest_results, n_simulations=500)
    
    # Analyze results
    for symbol, mc_result in mc_results.items():
        if mc_result:
            stats = mc_result.get('statistics', {})
            logger.info(f"{symbol} Monte Carlo Analysis:")
            logger.info(f"  Mean Return: {stats.get('mean_return', 0):.2%}")
            logger.info(f"  Std Return: {stats.get('std_return', 0):.2%}")
            logger.info(f"  5th Percentile: {stats.get('percentile_5', 0):.2%}")
            logger.info(f"  95th Percentile: {stats.get('percentile_95', 0):.2%}")
            logger.info(f"  Probability of Loss: {mc_result.get('probability_of_loss', 0):.2%}")
            logger.info(f"  Worst Case Return: {stats.get('worst_case_return', 0):.2%}")
    
    logger.info("Monte Carlo analysis demonstration completed")

def run_live_demo():
    """Run live trading demonstration (simulation mode)"""
    logger = logging.getLogger(__name__)
    logger.info("Starting live trading demonstration...")
    
    if not ENHANCED_BOT_AVAILABLE:
        logger.error("Enhanced trading bot not available")
        return
    
    # Initialize enhanced trading bot
    config = ENHANCED_TRADING_CONFIG.copy()
    config['auto_trade'] = False  # Simulation mode
    bot = EnhancedTradingBot(config)
    
    # Start the bot
    bot.start()
    
    try:
        # Run for a short period
        import time
        logger.info("Running live simulation for 60 seconds...")
        time.sleep(60)
        
        # Get performance summary
        summary = bot.get_performance_summary()
        logger.info("Performance Summary:")
        logger.info(f"  Risk Metrics: {summary['risk_metrics']}")
        logger.info(f"  Regime Summary: {summary['regime_summary']}")
        logger.info(f"  Pipeline Metrics: {summary['pipeline_metrics']}")
        
    except KeyboardInterrupt:
        logger.info("Live demonstration interrupted by user")
    finally:
        bot.stop()
    
    logger.info("Live trading demonstration completed")

def main():
    """Main function"""
    logger = setup_logging()
    
    print("Enhanced Trading Bot Demonstration")
    print("=" * 40)
    print("1. Backtest Demo")
    print("2. Walk-Forward Analysis Demo")
    print("3. Monte Carlo Analysis Demo")
    print("4. Live Trading Demo (Simulation)")
    print("5. Run All Demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo to run (0-5): ").strip()
            
            if choice == '0':
                logger.info("Exiting...")
                break
            elif choice == '1':
                run_backtest_demo()
            elif choice == '2':
                run_walk_forward_demo()
            elif choice == '3':
                run_monte_carlo_demo()
            elif choice == '4':
                run_live_demo()
            elif choice == '5':
                logger.info("Running all demonstrations...")
                run_backtest_demo()
                print("\n" + "-"*60 + "\n")
                run_walk_forward_demo()
                print("\n" + "-"*60 + "\n")
                run_monte_carlo_demo()
                print("\n" + "-"*60 + "\n")
                run_live_demo()
            else:
                print("Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            logger.info("Exiting...")
            break
        except Exception as e:
            logger.error(f"Error running demo: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
