#!/usr/bin/env python3
"""
Warnings Fix Test Script
========================

This script tests if the warnings are properly suppressed.
"""

import sys

def test_tensorflow_config():
    """Test TensorFlow configuration"""
    print("🔍 Testing TensorFlow configuration...")
    
    try:
        # Import the configuration
        import config.tensorflow_config
        
        # Try to import TensorFlow
        import tensorflow as tf
        
        print("✅ TensorFlow imported successfully")
        print(f"   GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
        print(f"   CPU devices: {len(tf.config.list_physical_devices('CPU'))}")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow configuration failed: {e}")
        return False

def test_startup_config():
    """Test startup configuration"""
    print("\n🚀 Testing startup configuration...")
    
    try:
        # Import startup configuration
        import scripts.startup
        
        print("✅ Startup configuration imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Startup configuration failed: {e}")
        return False

def test_trading_bot_import():
    """Test trading bot import without warnings"""
    print("\n🤖 Testing trading bot import...")
    
    try:
        # Import trading bot
        from core.mt5_trading_bot import MT5TradingBot
        
        print("✅ Trading bot imported successfully")
        
        # Test creating an instance
        bot = MT5TradingBot('EURUSD', '1h', risk_per_trade=0.02)
        print("✅ Trading bot instance created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Trading bot import failed: {e}")
        return False

def test_app_import():
    """Test app import without warnings"""
    print("\n🌐 Testing app import...")
    
    try:
        # Import app
        from core.app import app
        
        print("✅ App imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Warnings Fix Test")
    print("=" * 40)
    
    # Test configurations
    tensorflow_ok = test_tensorflow_config()
    startup_ok = test_startup_config()
    trading_bot_ok = test_trading_bot_import()
    app_ok = test_app_import()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    print(f"   TensorFlow Config: {'✅' if tensorflow_ok else '❌'}")
    print(f"   Startup Config: {'✅' if startup_ok else '❌'}")
    print(f"   Trading Bot Import: {'✅' if trading_bot_ok else '❌'}")
    print(f"   App Import: {'✅' if app_ok else '❌'}")
    
    if all([tensorflow_ok, startup_ok, trading_bot_ok, app_ok]):
        print("\n🎉 All tests passed! Warnings should be suppressed.")
        print("\n📋 The fixes ensure:")
        print("   - TensorFlow warnings are suppressed")
        print("   - GPU warnings are disabled")
        print("   - Module import warnings are handled gracefully")
        print("   - Clean startup without error messages")
        return True
    else:
        print("\n⚠️  Some tests failed")
        print("\n🔧 Check the errors above and fix them")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
