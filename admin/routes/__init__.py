"""
Admin routes package
"""
from .main import admin_bp
from .trading import trading_admin_bp
from .system import system_admin_bp

__all__ = ['admin_bp', 'trading_admin_bp', 'system_admin_bp']
