#!/usr/bin/env python3
"""
Live Trading Test Script
========================

This script tests if live trading is properly configured and working.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mt5_availability():
    """Test if MetaTrader5 is available"""
    print("🔍 Testing MetaTrader5 availability...")
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 imported successfully")
        return True
    except ImportError as e:
        print(f"❌ MetaTrader5 not available: {e}")
        print("💡 Solution: Install on Windows environment with MetaTrader5 terminal")
        return False

def test_mt5_connection():
    """Test MT5 connection with credentials"""
    print("\n🔗 Testing MT5 connection...")
    
    try:
        from connectors.mt5_connector import MT5Connector
        
        # Get credentials from environment
        account = os.getenv('XM_ACCOUNT_NUMBER')
        password = os.getenv('XM_PASSWORD')
        server = os.getenv('XM_SERVER', 'XMGlobal-Demo')
        
        if not account or not password:
            print("❌ Missing credentials in environment variables")
            print("💡 Set XM_ACCOUNT_NUMBER, XM_PASSWORD, and XM_SERVER")
            return False
        
        print(f"📋 Using account: {account}")
        print(f"🌐 Server: {server}")
        
        # Create connector and test connection
        connector = MT5Connector(account, password, server)
        
        if connector.connect():
            print("✅ MT5 connection successful!")
            
            # Get account info
            account_info = connector.get_account_summary()
            if account_info:
                print(f"💰 Balance: ${account_info.get('balance', 0):,.2f}")
                print(f"📈 Equity: ${account_info.get('equity', 0):,.2f}")
                print(f"🆓 Free Margin: ${account_info.get('margin_free', 0):,.2f}")
            
            # Test symbol info
            symbol_info = connector.get_symbol_info('EURUSD')
            if symbol_info:
                print(f"📊 EURUSD available for trading")
                print(f"   Min Volume: {symbol_info.get('volume_min', 'N/A')}")
                print(f"   Max Volume: {symbol_info.get('volume_max', 'N/A')}")
            
            # Test current price
            price_info = connector.get_current_price('EURUSD')
            if price_info:
                print(f"💱 Current EURUSD Price:")
                print(f"   Bid: {price_info.get('bid', 'N/A')}")
                print(f"   Ask: {price_info.get('ask', 'N/A')}")
            
            connector.disconnect()
            return True
        else:
            print(f"❌ MT5 connection failed: {connector.get_last_error()}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_trading_bot():
    """Test trading bot initialization"""
    print("\n🤖 Testing trading bot...")
    
    try:
        from core.mt5_trading_bot import MT5TradingBot
        
        # Test bot creation
        bot = MT5TradingBot(
            symbol='EURUSD',
            timeframe='1h',
            risk_per_trade=0.02
        )
        
        print("✅ Trading bot created successfully")
        print(f"📈 Symbol: {bot.symbol}")
        print(f"⏰ Timeframe: {bot.timeframe}")
        print(f"⚠️  Risk per trade: {bot.risk_per_trade * 100}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading bot test failed: {e}")
        return False

def test_web_interface():
    """Test web interface startup"""
    print("\n🌐 Testing web interface...")
    
    try:
        from core.app import app
        
        print("✅ Flask app imported successfully")
        print("🌐 Web interface should be available at http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def main():
    """Run all live trading tests"""
    print("🚀 Live Trading Configuration Test")
    print("=" * 50)
    
    # Test MT5 availability
    mt5_available = test_mt5_availability()
    
    if not mt5_available:
        print("\n❌ Live trading not available")
        print("\n📋 To enable live trading:")
        print("1. Install on Windows environment")
        print("2. Install MetaTrader5 terminal")
        print("3. Configure trading account credentials")
        print("4. Run this test again")
        return False
    
    # Test MT5 connection
    connection_ok = test_mt5_connection()
    
    # Test trading bot
    bot_ok = test_trading_bot()
    
    # Test web interface
    web_ok = test_web_interface()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   MetaTrader5: {'✅' if mt5_available else '❌'}")
    print(f"   Connection: {'✅' if connection_ok else '❌'}")
    print(f"   Trading Bot: {'✅' if bot_ok else '❌'}")
    print(f"   Web Interface: {'✅' if web_ok else '❌'}")
    
    if all([mt5_available, connection_ok, bot_ok, web_ok]):
        print("\n🎉 Live trading is ready!")
        print("\n📋 Next steps:")
        print("1. Start the application: python app.py")
        print("2. Open web interface: http://localhost:5000")
        print("3. Configure trading parameters")
        print("4. Enable auto trading")
        print("5. Monitor trading activity")
        return True
    else:
        print("\n⚠️  Live trading needs configuration")
        print("\n🔧 Fix the issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
