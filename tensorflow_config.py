#!/usr/bin/env python3
"""
TensorFlow Configuration
========================

This module configures TensorFlow to suppress warnings and optimize for deployment.
"""

import os
import warnings

def configure_tensorflow():
    """Configure TensorFlow to suppress warnings and optimize for deployment"""
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
    
    # Disable GPU usage to avoid CUDA warnings
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Disable oneDNN optimizations to avoid warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Suppress TensorFlow deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Suppress specific TensorFlow warnings
    warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
    warnings.filterwarnings('ignore', message='.*Could not find cuda drivers.*')
    warnings.filterwarnings('ignore', message='.*Unable to register cuDNN factory.*')
    warnings.filterwarnings('ignore', message='.*Unable to register cuFFT factory.*')
    warnings.filterwarnings('ignore', message='.*Unable to register cuBLAS factory.*')
    warnings.filterwarnings('ignore', message='.*Could not find TensorRT.*')
    
    # Suppress protobuf warnings
    warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
    warnings.filterwarnings('ignore', message='.*runtime version.*')
    warnings.filterwarnings('ignore', message='.*compatibility violations.*')
    
    # Configure TensorFlow logging
    try:
        import tensorflow as tf
        
        # Set TensorFlow logging level
        tf.get_logger().setLevel('ERROR')
        
        # Disable TensorFlow deprecation warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        # Configure TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        
        print("✅ TensorFlow configured for CPU-only deployment")
        
    except ImportError:
        print("⚠️  TensorFlow not available")
    except Exception as e:
        print(f"⚠️  TensorFlow configuration warning: {e}")

def suppress_tensorflow_warnings():
    """Suppress TensorFlow warnings during import"""
    
    # Set environment variables before importing TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*tensorflow.*')
    warnings.filterwarnings('ignore', message='.*cuda.*')
    warnings.filterwarnings('ignore', message='.*cuDNN.*')
    warnings.filterwarnings('ignore', message='.*cuFFT.*')
    warnings.filterwarnings('ignore', message='.*cuBLAS.*')
    warnings.filterwarnings('ignore', message='.*TensorRT.*')
    warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
    warnings.filterwarnings('ignore', message='.*runtime version.*')
    warnings.filterwarnings('ignore', message='.*compatibility violations.*')

# Configure TensorFlow when this module is imported
configure_tensorflow()
