#!/usr/bin/env python3
"""
Start Trading Bot with XM Credentials
=====================================

This script starts the trading bot using the configured XM credentials.
It will start both the main trading bot and the admin portal.
"""

import sys
import os
import logging
import time
import threading
from multiprocessing import Process

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def start_admin_portal():
    """Start the admin portal"""
    try:
        from admin.app import run_admin_app
        logging.info("Starting admin portal...")
        run_admin_app()
    except Exception as e:
        logging.error(f"Error starting admin portal: {e}")

def start_trading_bot():
    """Start the main trading bot"""
    try:
        # Import and start the appropriate trading bot
        from core.app import main
        logging.info("Starting trading bot...")
        main()
    except Exception as e:
        logging.error(f"Error starting trading bot: {e}")

def test_mt5_connection():
    """Test MT5 connection before starting"""
    try:
        from connectors.mt5_connector import MT5Connector
        
        logging.info("Testing MT5 connection...")
        connector = MT5Connector()
        
        if connector.connect():
            summary = connector.get_account_summary()
            if summary:
                logging.info(f"✅ Connected to XM Account: {summary['login']} on {summary['server']}")
                logging.info(f"   Balance: ${summary['balance']:,.2f}")
                logging.info(f"   Free Margin: ${summary['margin_free']:,.2f}")
            
            connector.disconnect()
            return True
        else:
            logging.error(f"❌ MT5 connection failed: {connector.get_last_error()}")
            return False
            
    except Exception as e:
        logging.error(f"Error testing MT5 connection: {e}")
        return False

def main():
    """Main function"""
    setup_logging()
    
    logging.info("="*60)
    logging.info("STARTING TRADING BOT WITH XM CREDENTIALS")
    logging.info("="*60)
    
    # Test MT5 connection first
    if not test_mt5_connection():
        logging.error("Cannot start trading bot without MT5 connection")
        logging.error("Please check your XM credentials and MT5 setup")
        return
    
    logging.info("MT5 connection test passed!")
    
    try:
        # Start admin portal in a separate process
        admin_process = Process(target=start_admin_portal)
        admin_process.start()
        
        # Give admin portal time to start
        time.sleep(2)
        
        # Start trading bot in main process
        start_trading_bot()
        
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        if 'admin_process' in locals():
            admin_process.terminate()
            admin_process.join()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        if 'admin_process' in locals():
            admin_process.terminate()
            admin_process.join()

if __name__ == "__main__":
    main()
