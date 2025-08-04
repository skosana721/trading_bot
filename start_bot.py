#!/usr/bin/env python3
"""
Dynamic Trading Bot Startup Script
==================================

This script starts the trading bot with proper setup and error handling.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask-cors', 'pandas', 'numpy', 'yfinance', 
        'matplotlib', 'plotly', 'python-dotenv', 'MetaTrader5'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Not installed")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("   Please run: pip install -r requirements.txt")
            return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Please copy xm_config_example.txt to .env and configure your XM credentials")
        return False
    
    # Check if .env has required variables
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = ['XM_ACCOUNT_NUMBER', 'XM_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if f'{var}=' not in content or f'{var}=your_' in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing or unconfigured variables in .env: {', '.join(missing_vars)}")
        print("   Please configure your XM credentials in the .env file")
        return False
    
    print("✅ .env file configured")
    return True

def check_mt5_installation():
    """Check if MetaTrader 5 is available"""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            print("✅ MetaTrader 5 is available")
            mt5.shutdown()
            return True
        else:
            print("❌ MetaTrader 5 initialization failed")
            print("   Please ensure MetaTrader 5 is installed and running")
            return False
    except ImportError:
        print("❌ MetaTrader 5 Python package not installed")
        print("   Please install: pip install MetaTrader5")
        return False

def create_templates_directory():
    """Create templates directory if it doesn't exist"""
    templates_dir = Path('templates')
    if not templates_dir.exists():
        templates_dir.mkdir()
        print("✅ Created templates directory")

def main():
    """Main startup function"""
    print("🚀 Dynamic Trading Bot Startup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check .env file
    print("\n🔧 Checking configuration...")
    if not check_env_file():
        sys.exit(1)
    
    # Check MT5 installation
    print("\n📊 Checking MetaTrader 5...")
    if not check_mt5_installation():
        print("⚠️  MetaTrader 5 check failed, but continuing...")
        print("   You can still use the bot for analysis without trading")
    
    # Create templates directory
    print("\n📁 Setting up directories...")
    create_templates_directory()
    
    # Start the application
    print("\n🌐 Starting the trading bot...")
    print("   Web interface will be available at: http://localhost:5000")
    print("   Press Ctrl+C to stop the bot")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Trading bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting the bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 