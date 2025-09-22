#!/usr/bin/env python3
"""
Main Trading Bot Startup Script
==============================

This script starts the trading bot application with the correct PYTHONPATH.
"""

import os
import sys

def main():
    """Start the trading bot application"""
    print("🚀 Starting Trading Bot Application...")
    print("=" * 50)
    
    # Set PYTHONPATH to current directory
    os.environ['PYTHONPATH'] = '.'
    
    # Import and run the app
    try:
        from backend.core.app import app
        print("✅ Application imported successfully")
        print(f"📁 Template folder: {app.template_folder}")
        print("🌐 Starting Flask server...")
        print("📍 Web interface will be available at: http://localhost:5000")
        print("=" * 50)
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this from the project root directory")
        return 1
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
