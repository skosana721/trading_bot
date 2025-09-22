#!/usr/bin/env python3
"""
Verify XM Configuration
=======================

This script verifies that the XM credentials are properly configured.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_xm_config():
    """Verify XM configuration"""
    print("="*60)
    print("VERIFYING XM CONFIGURATION")
    print("="*60)
    
    try:
        # Test importing XM credentials
        from config.xm_credentials import XM_CREDENTIALS, MT5_CONFIG, TRADING_CONFIG
        print("✅ XM credentials configuration loaded successfully")
        
        # Display credentials (masked for security)
        account = XM_CREDENTIALS['account_number']
        password = XM_CREDENTIALS['password']
        server = XM_CREDENTIALS['server']
        
        print(f"\n📋 XM Account Configuration:")
        print(f"   Account Number: {account}")
        print(f"   Password: {'*' * len(password)}")
        print(f"   Server: {server}")
        
        # Test configuration manager
        from config.config import config_manager
        config = config_manager.config
        
        print(f"\n🔧 Trading Configuration:")
        print(f"   Symbol: {config.symbol}")
        print(f"   Timeframe: {config.timeframe}")
        print(f"   Risk per Trade: {config.risk_per_trade * 100}%")
        print(f"   Auto Trade: {config.auto_trade}")
        print(f"   Use ML: {config.use_ml}")
        print(f"   Use SMC: {config.use_smc}")
        
        # Test MT5 connector initialization
        from connectors.mt5_connector import MT5Connector
        connector = MT5Connector()
        
        print(f"\n🔌 MT5 Connector Configuration:")
        print(f"   Account: {connector.account_number}")
        print(f"   Server: {connector.server}")
        print(f"   Password: {'*' * len(connector.password) if connector.password else 'Not set'}")
        
        print("\n✅ Configuration verification completed successfully!")
        print("\n📝 Next Steps:")
        print("   1. Run: python scripts/test_mt5_connection.py")
        print("   2. Run: python scripts/start_with_xm_credentials.py")
        print("   3. Access admin portal at: http://localhost:5001")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing configuration: {e}")
        print("   Make sure config/xm_credentials.py exists and is properly configured")
        return False
    except Exception as e:
        print(f"❌ Error verifying configuration: {e}")
        return False

if __name__ == "__main__":
    success = verify_xm_config()
    sys.exit(0 if success else 1)
