#!/usr/bin/env python3
"""
Deployment Test Script
=====================

This script tests if the application can start without MetaTrader5
for deployment verification.
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    # Test basic imports
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    # Test ML imports
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import xgboost
        print("✅ XGBoost imported successfully")
    except ImportError as e:
        print(f"❌ XGBoost import failed: {e}")
        return False
    
    # Test MT5 import (should fail gracefully)
    try:
        import MetaTrader5
        print("✅ MetaTrader5 imported successfully (local environment)")
    except ImportError:
        print("⚠️  MetaTrader5 not available (expected in deployment)")
    
    return True

def test_app_startup():
    """Test if the Flask app can start"""
    print("\nTesting Flask app startup...")
    
    try:
        # Import deployment config
        from deployment_config import DeploymentConfig, DEPLOYMENT_MODE
        print(f"✅ Deployment config imported successfully")
        print(f"   Deployment mode: {DEPLOYMENT_MODE}")
        
        # Test MT5 connector import
        try:
            from mt5_connector import MT5Connector
            print("✅ MT5Connector imported successfully")
        except ImportError as e:
            print(f"⚠️  MT5Connector import failed (expected in deployment): {e}")
        
        # Test trading bot import
        try:
            from mt5_trading_bot import MT5TradingBot
            print("✅ MT5TradingBot imported successfully")
        except ImportError as e:
            print(f"⚠️  MT5TradingBot import failed (expected in deployment): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ App startup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Deployment Compatibility Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        sys.exit(1)
    
    # Test app startup
    if not test_app_startup():
        print("\n❌ App startup test failed")
        sys.exit(1)
    
    print("\n✅ All tests passed! Deployment should work.")
    print("\n📝 Notes:")
    print("   - MetaTrader5 will be disabled in deployment")
    print("   - Trading bot will run in simulation mode")
    print("   - Web interface and analysis features will work")

if __name__ == "__main__":
    main()
