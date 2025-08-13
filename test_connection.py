#!/usr/bin/env python3
"""
MT5 Connection Test Script
==========================

This script helps diagnose MT5 connection issues and provides a demo mode.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mt5_installation():
    """Test if MT5 Python package is available"""
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 Python package is installed")
        print(f"   Version: {mt5.__version__}")
        return True
    except ImportError:
        print("❌ MetaTrader5 Python package is not installed")
        print("   Install it with: pip install MetaTrader5")
        return False

def test_mt5_initialization():
    """Test if MT5 terminal can be initialized"""
    try:
        import MetaTrader5 as mt5
        
        print("\n🔄 Testing MT5 initialization...")
        result = mt5.initialize()
        
        if result:
            print("✅ MT5 initialized successfully")
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"   Terminal: {terminal_info.name}")
                print(f"   Version: {terminal_info.version}")
                print(f"   Connected: {terminal_info.connected}")
                print(f"   Trade allowed: {terminal_info.trade_allowed}")
                print(f"   Expert advisors allowed: {terminal_info.trade_allowed}")
            
            return True
        else:
            error = mt5.last_error()
            print(f"❌ MT5 initialization failed")
            print(f"   Error: {error}")
            
            if error[0] == -10005:
                print("\n💡 Solution: MetaTrader 5 terminal is not running")
                print("   Please:")
                print("   1. Install MetaTrader 5 from https://www.xm.com/mt5")
                print("   2. Launch the MT5 terminal")
                print("   3. Keep it running while using the bot")
            elif error[0] == -10004:
                print("\n💡 Solution: MetaTrader 5 terminal is not installed")
                print("   Please install MT5 from https://www.xm.com/mt5")
            
            return False
            
    except Exception as e:
        print(f"❌ Error testing MT5: {e}")
        return False

def test_xm_credentials():
    """Test if XM credentials are configured"""
    print("\n🔐 Checking XM credentials...")
    
    account_number = os.getenv('XM_ACCOUNT_NUMBER')
    password = os.getenv('XM_PASSWORD')
    server = os.getenv('XM_SERVER', 'XMGlobal-Demo')
    
    if not account_number or account_number == 'your_xm_account_number_here':
        print("❌ XM_ACCOUNT_NUMBER not configured")
        print("   Please create a .env file with your XM credentials")
        return False
    
    if not password or password == 'your_xm_password_here':
        print("❌ XM_PASSWORD not configured")
        print("   Please create a .env file with your XM credentials")
        return False
    
    print(f"✅ XM credentials found")
    print(f"   Account: {account_number}")
    print(f"   Server: {server}")
    return True

def test_mt5_login():
    """Test MT5 login with XM credentials"""
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("❌ Cannot test login - MT5 not initialized")
            return False
        
        account_number = os.getenv('XM_ACCOUNT_NUMBER')
        password = os.getenv('XM_PASSWORD')
        server = os.getenv('XM_SERVER', 'XMGlobal-Demo')
        
        print(f"\n🔐 Testing MT5 login...")
        print(f"   Account: {account_number}")
        print(f"   Server: {server}")
        
        result = mt5.login(
            login=int(account_number),
            password=password,
            server=server
        )
        
        if result:
            print("✅ MT5 login successful")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Equity: ${account_info.equity:.2f}")
                print(f"   Currency: {account_info.currency}")
            
            return True
        else:
            error = mt5.last_error()
            print(f"❌ MT5 login failed")
            print(f"   Error: {error}")
            
            if error[0] == -10004:
                print("\n💡 Solution: Check your XM credentials")
                print("   - Verify account number and password")
                print("   - Make sure you're using the correct server")
                print("   - Try logging in to MT5 manually first")
            
            return False
            
    except Exception as e:
        print(f"❌ Error testing login: {e}")
        return False

def create_demo_env():
    """Create a demo .env file"""
    print("\n📝 Creating demo .env file...")
    
    demo_content = """# XM Trading Account Configuration (DEMO)
# Replace with your actual XM credentials

# XM Account Credentials
XM_ACCOUNT_NUMBER=your_xm_account_number_here
XM_PASSWORD=your_xm_password_here
XM_SERVER=XMGlobal-Demo

# Trading Bot Configuration
RISK_PER_TRADE=2.0
MAX_POSITIONS_PER_SYMBOL=3
ANALYSIS_INTERVAL=300
USE_ML=true

# Logging
LOG_LEVEL=INFO
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(demo_content)
        print("✅ Created .env file")
        print("   Please edit it with your actual XM credentials")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def main():
    """Main test function"""
    print("🔍 MT5 Connection Diagnostic Tool")
    print("=" * 40)
    
    # Test MT5 installation
    mt5_available = test_mt5_installation()
    
    if not mt5_available:
        print("\n💡 To install MetaTrader5 Python package:")
        print("   pip install MetaTrader5")
        return
    
    # Test MT5 initialization
    mt5_initialized = test_mt5_initialization()
    
    if not mt5_initialized:
        print("\n💡 Next steps:")
        print("   1. Install MetaTrader 5 terminal from https://www.xm.com/mt5")
        print("   2. Launch MT5 and keep it running")
        print("   3. Run this test again")
        return
    
    # Test credentials
    credentials_ok = test_xm_credentials()
    
    if not credentials_ok:
        print("\n💡 To configure credentials:")
        print("   1. Create a .env file with your XM credentials")
        print("   2. Or use the web interface to enter credentials")
        create_demo_env()
        return
    
    # Test login
    login_ok = test_mt5_login()
    
    if login_ok:
        print("\n🎉 All tests passed! Your MT5 connection is working.")
        print("   You can now run the automated trading bot.")
    else:
        print("\n💡 Login failed. Please check your XM credentials.")

if __name__ == "__main__":
    main()

