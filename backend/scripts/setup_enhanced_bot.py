#!/usr/bin/env python3
"""
Enhanced Trading Bot Setup Script
================================

This script helps set up the enhanced trading bot with proper configuration
and validates the installation.
"""

import sys
import os
import subprocess
import importlib
from typing import List, Dict, Any

def check_dependency(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name: str) -> bool:
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies() -> Dict[str, bool]:
    """Check all required dependencies"""
    dependencies = {
        'scikit-learn': 'sklearn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'tensorflow': 'tensorflow',
        'textblob': 'textblob',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'websockets': 'websockets',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'MetaTrader5': 'MetaTrader5',
        'flask': 'flask',
        'flask_cors': 'flask_cors',
        'python-dotenv': 'dotenv',
        'ta': 'ta',
        'joblib': 'joblib'
    }
    
    results = {}
    for package, import_name in dependencies.items():
        results[package] = check_dependency(package, import_name)
    
    return results

def install_missing_dependencies(missing: List[str]) -> bool:
    """Install missing dependencies"""
    print(f"Installing {len(missing)} missing dependencies...")
    
    for package in missing:
        print(f"Installing {package}...")
        if not install_package(package):
            print(f"Failed to install {package}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'models',
        'data',
        'backtests',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def validate_enhanced_bot():
    """Validate that the enhanced bot can be imported"""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from core.enhanced_trading_bot import EnhancedTradingBot
        from config.enhanced_trading_config import ENHANCED_TRADING_CONFIG
        from core.advanced_risk_manager import AdvancedRiskManager
        from strategies.advanced_ml_ensemble import AdvancedMLEnsemble
        from strategies.market_regime_detector import MarketRegimeDetector
        from core.real_time_data_pipeline import RealTimeDataPipeline
        from analysis.advanced_backtesting import AdvancedBacktester
        
        print("✅ All enhanced bot components imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_basic_test():
    """Run a basic test of the enhanced bot"""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from core.enhanced_trading_bot import EnhancedTradingBot
        from config.enhanced_trading_config import ENHANCED_TRADING_CONFIG
        
        print("Running basic test...")
        
        # Create a minimal config for testing
        test_config = ENHANCED_TRADING_CONFIG.copy()
        test_config['symbols'] = ['EURUSD']  # Test with one symbol
        test_config['auto_trade'] = False
        
        # Initialize bot
        bot = EnhancedTradingBot(test_config)
        print("✅ Enhanced trading bot initialized successfully!")
        
        # Test risk manager
        risk_summary = bot.risk_manager.get_risk_summary()
        print("✅ Risk manager working!")
        
        # Test regime detector
        regime_summary = bot.regime_detector.get_regime_summary()
        print("✅ Market regime detector working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Enhanced Trading Bot Setup")
    print("=" * 40)
    
    # Check dependencies
    print("Checking dependencies...")
    dependency_status = check_dependencies()
    
    missing = [pkg for pkg, status in dependency_status.items() if not status]
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        
        response = input("Would you like to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if not install_missing_dependencies(missing):
                print("❌ Failed to install some dependencies")
                return False
        else:
            print("⚠️  Please install missing dependencies manually:")
            print(f"pip install {' '.join(missing)}")
            return False
    else:
        print("✅ All dependencies are installed!")
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Validate enhanced bot
    print("\nValidating enhanced bot...")
    if not validate_enhanced_bot():
        print("❌ Enhanced bot validation failed")
        return False
    
    # Run basic test
    print("\nRunning basic test...")
    if not run_basic_test():
        print("❌ Basic test failed")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demonstration: python scripts/start_enhanced_bot.py")
    print("2. Configure your settings in config/enhanced_trading_config.py")
    print("3. Set up your data sources (MT5, news APIs, etc.)")
    print("4. Start with backtesting before live trading")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
