#!/usr/bin/env python3
"""
XM Trading Account Credentials Configuration
===========================================

This module contains the XM trading account credentials for the trading bot.
Update these values with your actual XM account details.
"""

# XM Trading Account Credentials
XM_CREDENTIALS = {
    'account_number': '315050186',
    'password': 'S6fAqwU$y/d5_Yi',
    'server': 'XMGlobal-MT5 7'
}

# MT5 Connection Settings
MT5_CONFIG = {
    'account': XM_CREDENTIALS['account_number'],
    'password': XM_CREDENTIALS['password'],
    'server': XM_CREDENTIALS['server'],
    'timeout': 30,
    'retries': 3
}

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'EURUSD',
    'timeframe': '5m',
    'risk_per_trade': 0.02,
    'auto_trade': False,
    'use_ml': True,
    'use_smc': True,
    'max_positions_per_symbol': 3,
    'max_same_direction_positions': 2,
    'max_daily_trades': 10,
    'max_daily_loss': 0.05
}

# Admin Portal Configuration
ADMIN_CONFIG = {
    'secret_key': 'admin-secret-key-change-in-production',
    'port': 5001,
    'debug': False
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'file': 'logs/trading_bot.log'
}

def get_xm_credentials():
    """Get XM trading account credentials"""
    return XM_CREDENTIALS.copy()

def get_mt5_config():
    """Get MT5 configuration"""
    return MT5_CONFIG.copy()

def get_trading_config():
    """Get trading configuration"""
    return TRADING_CONFIG.copy()

def get_admin_config():
    """Get admin portal configuration"""
    return ADMIN_CONFIG.copy()

def get_logging_config():
    """Get logging configuration"""
    return LOGGING_CONFIG.copy()

# Export all configurations
__all__ = [
    'XM_CREDENTIALS',
    'MT5_CONFIG', 
    'TRADING_CONFIG',
    'ADMIN_CONFIG',
    'LOGGING_CONFIG',
    'get_xm_credentials',
    'get_mt5_config',
    'get_trading_config',
    'get_admin_config',
    'get_logging_config'
]
