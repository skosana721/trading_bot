"""
Deployment Configuration
=======================

This module handles deployment-specific configurations and limitations.
"""

import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

class DeploymentConfig:
    """Handles deployment-specific configurations"""
    
    @staticmethod
    def is_deployment_environment():
        """Check if running in a deployment environment"""
        # Check for common deployment environment variables
        deployment_vars = [
            'RAILWAY_ENVIRONMENT',
            'RENDER',
            'HEROKU_APP_NAME',
            'PYTHONANYWHERE_SITE',
            'GOOGLE_CLOUD_PROJECT',
            'AWS_REGION'
        ]
        
        return any(os.getenv(var) for var in deployment_vars)
    
    @staticmethod
    def get_mt5_availability():
        """Check if MetaTrader5 is available"""
        try:
            import MetaTrader5 as mt5
            return True
        except ImportError:
            print("Warning: MetaTrader5 not available - running in simulation mode")
            return False
    
    @staticmethod
    def get_trading_mode():
        """Determine trading mode based on environment"""
        # Force live trading mode if MT5 is available
        if DeploymentConfig.get_mt5_availability():
            return "live"
        else:
            return "simulation"
    
    @staticmethod
    def get_platform_limitations():
        """Get list of platform limitations"""
        limitations = []
        
        if not DeploymentConfig.get_mt5_availability():
            limitations.append("MetaTrader5 integration disabled")
        
        if DeploymentConfig.is_deployment_environment():
            limitations.append("Running in deployment environment")
        
        return limitations

# Global configuration
DEPLOYMENT_MODE = DeploymentConfig.get_trading_mode()
PLATFORM_LIMITATIONS = DeploymentConfig.get_platform_limitations()

if PLATFORM_LIMITATIONS:
    logger.info(f"Platform limitations detected: {', '.join(PLATFORM_LIMITATIONS)}")
    logger.info(f"Trading mode: {DEPLOYMENT_MODE}")
