#!/usr/bin/env python3
"""
Startup Configuration Script
============================

This script configures the environment and suppresses warnings before starting the application.
"""

import os
import sys
import warnings

def configure_environment():
    """Configure environment variables and suppress warnings"""
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Configure TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
    
    # Configure logging
    os.environ['LOG_LEVEL'] = 'INFO'
    
    # Configure Flask
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = '0'
    
    print("✅ Environment configured for deployment")

def suppress_tensorflow_warnings():
    """Suppress TensorFlow warnings"""
    try:
        import tensorflow as tf
        
        # Set TensorFlow logging level
        tf.get_logger().setLevel('ERROR')
        
        # Disable TensorFlow deprecation warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        # Configure TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        
        print("✅ TensorFlow warnings suppressed")
        
    except ImportError:
        print("⚠️  TensorFlow not available")
    except Exception as e:
        print(f"⚠️  TensorFlow configuration warning: {e}")

def check_dependencies():
    """Check if all dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = {
        'Flask': 'flask',
        'Pandas': 'pandas',
        'NumPy': 'numpy',
        'Scikit-learn': 'sklearn',
        'TensorFlow': 'tensorflow',
        'YFinance': 'yfinance',
        'TA': 'ta'
    }
    
    available = {}
    for name, module in dependencies.items():
        try:
            __import__(module)
            available[name] = True
            print(f"✅ {name}")
        except ImportError:
            available[name] = False
            print(f"❌ {name}")
    
    return available

def main():
    """Main startup function"""
    print("🚀 Trading Bot Startup Configuration")
    print("=" * 40)
    
    # Configure environment
    configure_environment()
    
    # Suppress TensorFlow warnings
    suppress_tensorflow_warnings()
    
    # Check dependencies
    dependencies = check_dependencies()
    
    print("\n" + "=" * 40)
    print("📊 Startup Summary:")
    print(f"   Environment: Configured")
    print(f"   TensorFlow: Warnings suppressed")
    print(f"   Dependencies: {sum(dependencies.values())}/{len(dependencies)} available")
    
    if dependencies.get('Flask', False):
        print("\n🎉 Ready to start Flask application")
        return True
    else:
        print("\n❌ Critical dependencies missing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
