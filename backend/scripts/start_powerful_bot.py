#!/usr/bin/env python3
"""
Start Powerful Trading Bot
==========================

This script starts the powerful trading bot with all advanced features.
"""

import sys
import os
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Setup logging for the powerful trading bot"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/powerful_bot.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('powerful_bot')

def main():
    """Main function to start the powerful trading bot"""
    logger = setup_logging()
    
    print("🚀 Starting Powerful Trading Bot")
    print("=" * 50)
    
    try:
        # Import required modules
        from core.powerful_trading_bot import PowerfulTradingBot
        from config.powerful_trading_config import POWERFUL_TRADING_CONFIG
        
        logger.info("Powerful Trading Bot modules imported successfully")
        
        # Create configuration
        config = POWERFUL_TRADING_CONFIG.copy()
        
        # Override with command line arguments if provided
        if len(sys.argv) > 1:
            config['symbol'] = sys.argv[1]
        if len(sys.argv) > 2:
            config['timeframe'] = sys.argv[2]
        if len(sys.argv) > 3:
            config['auto_trade'] = sys.argv[3].lower() == 'true'
        
        print(f"📊 Symbol: {config['symbol']}")
        print(f"⏰ Timeframe: {config['timeframe']}")
        print(f"🤖 Auto Trade: {config['auto_trade']}")
        print(f"💰 Account Balance: ${config['account_balance']:,}")
        print(f"🎯 Risk per Trade: {config['risk_per_trade']*100:.1f}%")
        
        # Display enabled features
        print("\n🔧 Enabled Features:")
        features = [
            ('Advanced Risk Management', config['use_advanced_risk']),
            ('Market Regime Detection', config['use_regime_detection']),
            ('Advanced ML Ensemble', config['use_advanced_ml']),
            ('Real-time Data Pipeline', config['use_real_time_data']),
            ('Smart Money Concept', config['use_smart_money']),
            ('Reinforcement Learning', config['use_reinforcement_learning']),
            ('Market Structure Strategy', config['use_market_structure'])
        ]
        
        for feature, enabled in features:
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"   {feature}: {status}")
        
        # Initialize the powerful trading bot
        print("\n🔧 Initializing Powerful Trading Bot...")
        bot = PowerfulTradingBot(config)
        
        # Start the bot
        print("🚀 Starting bot...")
        bot.start()
        
        print("\n✅ Powerful Trading Bot started successfully!")
        print("📊 Bot is now running with all advanced features")
        print("🌐 Web interface available at: http://localhost:5000")
        print("📱 API endpoints available at: /api/powerful/*")
        
        # Display status
        print("\n📈 Bot Status:")
        status = bot.get_comprehensive_status()
        print(f"   Running: {status['bot_status']['is_running']}")
        print(f"   Connected: {status['bot_status']['connected']}")
        print(f"   Auto Trade: {status['bot_status']['auto_trade']}")
        
        # Keep the bot running
        try:
            print("\n⏳ Bot is running... Press Ctrl+C to stop")
            while True:
                time.sleep(10)
                
                # Display periodic status update
                if datetime.now().second % 60 == 0:  # Every minute
                    status = bot.get_comprehensive_status()
                    performance = status.get('performance', {})
                    
                    print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
                    print(f"   Total Trades: {performance.get('total_trades', 0)}")
                    print(f"   Current Positions: {performance.get('current_positions', 0)}")
                    print(f"   Account Balance: ${performance.get('account_balance', 0):,.2f}")
                    
                    # Display risk metrics if available
                    risk_metrics = performance.get('risk_metrics', {})
                    if risk_metrics:
                        portfolio_metrics = risk_metrics.get('portfolio_metrics', {})
                        print(f"   Portfolio VaR (95%): {portfolio_metrics.get('var_95', 0)*100:.2f}%")
                        print(f"   Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
                        print(f"   Max Drawdown: {portfolio_metrics.get('max_drawdown', 0)*100:.2f}%")
                    
                    # Display regime metrics if available
                    regime_metrics = performance.get('regime_metrics', {})
                    if regime_metrics:
                        print(f"   Market Regime: {regime_metrics.get('current_regime', 'unknown')}")
                        print(f"   Regime Confidence: {regime_metrics.get('confidence', 0)*100:.1f}%")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping Powerful Trading Bot...")
            bot.stop()
            print("✅ Bot stopped successfully")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements-enhanced.txt")
        return False
        
    except Exception as e:
        logger.error(f"Error starting powerful bot: {e}")
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
