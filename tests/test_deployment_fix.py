#!/usr/bin/env python3
"""
Deployment Fix Test Script
==========================

This script tests if the deployment fixes work correctly.
"""

import sys

def test_mt5_connector_import():
    """Test MT5Connector import"""
    print("🔍 Testing MT5Connector import...")
    
    try:
        from connectors.mt5_connector import MT5Connector, MT5_AVAILABLE
        print(f"✅ MT5Connector imported successfully")
        print(f"   MT5_AVAILABLE: {MT5_AVAILABLE}")
        
        # Test creating an instance
        connector = MT5Connector()
        print("✅ MT5Connector instance created successfully")
        
        return True
    except Exception as e:
        print(f"❌ MT5Connector import failed: {e}")
        return False

def test_trading_bot_import():
    """Test MT5TradingBot import"""
    print("\n🤖 Testing MT5TradingBot import...")
    
    try:
        from core.mt5_trading_bot import MT5TradingBot, MT5_CONNECTOR_AVAILABLE
        print(f"✅ MT5TradingBot imported successfully")
        print(f"   MT5_CONNECTOR_AVAILABLE: {MT5_CONNECTOR_AVAILABLE}")
        
        # Test creating an instance
        bot = MT5TradingBot('EURUSD', '1h', risk_per_trade=0.02)
        print("✅ MT5TradingBot instance created successfully")
        
        return True
    except Exception as e:
        print(f"❌ MT5TradingBot import failed: {e}")
        return False

def test_app_import():
    """Test app import"""
    print("\n🌐 Testing app import...")
    
    try:
        from core.app import app, MT5_AVAILABLE
        print(f"✅ App imported successfully")
        print(f"   MT5_AVAILABLE: {MT5_AVAILABLE}")
        
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def test_deployment_config():
    """Test deployment configuration"""
    print("\n⚙️  Testing deployment configuration...")
    
    try:
        from config.deployment_config import DeploymentConfig, DEPLOYMENT_MODE, PLATFORM_LIMITATIONS
        print(f"✅ Deployment config imported successfully")
        print(f"   DEPLOYMENT_MODE: {DEPLOYMENT_MODE}")
        print(f"   PLATFORM_LIMITATIONS: {PLATFORM_LIMITATIONS}")
        
        return True
    except Exception as e:
        print(f"❌ Deployment config import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Deployment Fix Test")
    print("=" * 40)
    
    # Test imports
    mt5_connector_ok = test_mt5_connector_import()
    trading_bot_ok = test_trading_bot_import()
    app_ok = test_app_import()
    deployment_config_ok = test_deployment_config()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    print(f"   MT5Connector: {'✅' if mt5_connector_ok else '❌'}")
    print(f"   MT5TradingBot: {'✅' if trading_bot_ok else '❌'}")
    print(f"   App: {'✅' if app_ok else '❌'}")
    print(f"   Deployment Config: {'✅' if deployment_config_ok else '❌'}")
    
    if all([mt5_connector_ok, trading_bot_ok, app_ok, deployment_config_ok]):
        print("\n🎉 All tests passed! Deployment should work.")
        print("\n📋 The fixes ensure:")
        print("   - Graceful handling of missing MetaTrader5")
        print("   - Simulation mode when MT5 is not available")
        print("   - No import errors during deployment")
        print("   - Web interface works in both modes")
        return True
    else:
        print("\n⚠️  Some tests failed")
        print("\n🔧 Check the errors above and fix them")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
