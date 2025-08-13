#!/usr/bin/env python3
"""
Unified Trading Bot Startup Script
==================================

This script provides a unified interface to run both:
1. Enhanced Trading Bot (manual + automated trading)
2. Automated Trading Bot (fully automated MT5 trading)

Usage:
    python start_unified_bot.py [mode] [options]

Modes:
    enhanced    - Run enhanced trading bot with manual and automated capabilities
    automated   - Run fully automated trading bot
    web         - Run web interface with both capabilities
    test        - Run connection test

Examples:
    python start_unified_bot.py enhanced --symbol EURUSD --timeframe 5m
    python start_unified_bot.py automated
    python start_unified_bot.py web
    python start_unified_bot.py test
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging for the unified bot"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/unified_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('unified_bot')

def test_connection():
    """Test MT5 connection"""
    logger = setup_logging()
    logger.info("Testing MT5 connection...")
    
    try:
        from mt5_connector import MT5Connector
        
        logger.info("Environment credential usage disabled. Please provide credentials via the web UI.")
        # For CLI test mode, prompt or skip; here we skip
        return False
        
        connector = MT5Connector(account_number, password, server)
        success = connector.connect()
        
        if success:
            account_summary = connector.get_account_summary()
            if account_summary:
                logger.info("MT5 connection successful!")
                logger.info(f"   Account Balance: ${account_summary['balance']:,.2f}")
                logger.info(f"   Account Equity: ${account_summary['equity']:,.2f}")
            else:
                logger.info("MT5 connection successful!")
            connector.disconnect()
            return True
        else:
            logger.error("MT5 connection failed")
            return False
            
    except ImportError:
        logger.error("MT5 connector not available")
        return False
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

def run_enhanced_bot(symbol, timeframe, market_type='forex', enable_automation=True):
    """Run the enhanced trading bot"""
    logger = setup_logging()
    logger.info(f"Starting Enhanced Trading Bot for {symbol} ({timeframe})")
    
    try:
        from trading_bot import TradingBot
        
        # Create enhanced trading bot
        bot = TradingBot(
            symbol=symbol,
            period=timeframe,
            market_type=market_type,
            account_size=10000,
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.02)),
            enable_automation=enable_automation,
            mt5_config={}
        )
        
        logger.info("Enhanced Trading Bot initialized successfully")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Timeframe: {timeframe}")
        logger.info(f"   Market Type: {market_type}")
        logger.info(f"   Automation: {'Enabled' if enable_automation else 'Disabled'}")
        
        # Connect to MT5 if automation is enabled
        if enable_automation:
            logger.info("Connecting to MT5...")
            if bot.connect_mt5():
                logger.info("Connected to MT5")
                
                # Run a test analysis
                logger.info("Running test analysis...")
                analysis_result = bot.run_automated_analysis_cycle()
                
                if analysis_result:
                    logger.info("Analysis completed successfully")
                    logger.info(f"   Trend Analysis: {analysis_result.get('trend_analysis', {}).get('uptrend_confirmed', False)}")
                    logger.info(f"   Trading Signals: {'Yes' if analysis_result.get('trading_signals') else 'No'}")
                else:
                    logger.warning("⚠️  Analysis failed")
                
                # Start continuous automation
                logger.info("Starting continuous automation...")
                bot.run_continuous_automation(interval_minutes=5)
            else:
                logger.error("Failed to connect to MT5")
                return False
        else:
            # Run manual analysis
            logger.info("Running manual analysis...")
            bot.fetch_data()
            trend_analysis = bot.identify_higher_highs_lows()
            
            if trend_analysis:
                bot.generate_day_trading_report(trend_analysis)
            else:
                logger.warning("⚠️  No trend analysis available")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running enhanced bot: {e}")
        return False

def run_automated_bot():
    """Run the fully automated trading bot"""
    logger = setup_logging()
    logger.info("Starting Automated Trading Bot...")
    
    try:
        from automated_trading_bot import AutomatedTradingBot
        
        # Create and run automated trading bot
        bot = AutomatedTradingBot()
        bot.run(host='0.0.0.0', port=5000, debug=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running automated bot: {e}")
        return False

def run_web_interface():
    """Run the web interface with both capabilities"""
    logger = setup_logging()
    logger.info("Starting Web Interface...")
    
    try:
        from app import app
        
        logger.info("Web interface will be available at: http://localhost:5000")
        logger.info("Features available:")
        logger.info("   - Enhanced Trading Bot (manual + automated)")
        logger.info("   - Automated Trading Bot")
        logger.info("   - MT5 Integration")
        logger.info("   - Real-time Analysis")
        logger.info("   - Position Management")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running web interface: {e}")
        return False

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Trading Bot - Enhanced and Automated Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_unified_bot.py test
  python start_unified_bot.py enhanced --symbol EURUSD --timeframe 5m
  python start_unified_bot.py automated
  python start_unified_bot.py web
        """
    )
    
    parser.add_argument('mode', choices=['enhanced', 'automated', 'web', 'test'],
                       help='Mode to run the bot in')
    
    parser.add_argument('--symbol', '-s', default='EURUSD',
                       help='Trading symbol (default: EURUSD)')
    
    parser.add_argument('--timeframe', '-t', default='5m',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Trading timeframe (default: 5m)')
    
    parser.add_argument('--market-type', '-m', default='forex',
                       choices=['stock', 'forex', 'crypto', 'commodities'],
                       help='Market type (default: forex)')
    
    parser.add_argument('--no-automation', action='store_true',
                       help='Disable automated trading (enhanced mode only)')
    
    parser.add_argument('--port', '-p', type=int, default=5000,
                       help='Port for web interface (default: 5000)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("UNIFIED TRADING BOT STARTUP")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Run based on mode
    if args.mode == 'test':
        success = test_connection()
        sys.exit(0 if success else 1)
        
    elif args.mode == 'enhanced':
        enable_automation = not args.no_automation
        success = run_enhanced_bot(
            symbol=args.symbol,
            timeframe=args.timeframe,
            market_type=args.market_type,
            enable_automation=enable_automation
        )
        sys.exit(0 if success else 1)
        
    elif args.mode == 'automated':
        success = run_automated_bot()
        sys.exit(0 if success else 1)
        
    elif args.mode == 'web':
        success = run_web_interface()
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
