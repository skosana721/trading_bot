#!/usr/bin/env python3
"""
Improved Trading Configuration
=============================

Configuration settings for the enhanced trading system with better risk management
and improved win rate strategies.
"""

# Risk Management Configuration
RISK_MANAGEMENT_CONFIG = {
    # Maximum risk per trade (1% instead of 2%)
    'max_risk_per_trade': 0.01,  # 1%
    
    # Maximum daily risk (3% instead of 5%)
    'max_daily_risk': 0.03,  # 3%
    
    # Maximum portfolio risk (5% instead of 10%)
    'max_portfolio_risk': 0.05,  # 5%
    
    # Maximum concurrent trades (3 as requested)
    'max_concurrent_trades': 3,
    
    # Maximum correlated trades
    'max_correlated_trades': 1,
    
    # Maximum drawdown before stopping
    'max_drawdown': 0.08,  # 8%
    
    # Maximum daily loss
    'max_daily_loss': 0.02,  # 2%
    
    # Minimum win rate to continue trading
    'min_win_rate': 0.45,  # 45%
    
    # Position sizing adjustments
    'volatility_multiplier': 0.8,  # Reduce size in high volatility
    'max_volatility_adjustment': 0.5,  # Maximum 50% reduction
}

# Market Condition Filters
MARKET_FILTERS_CONFIG = {
    # Trend strength requirements
    'min_trend_strength': 0.4,  # Require stronger trends
    
    # Volatility limits
    'min_volatility': 0.0008,  # Higher minimum volatility
    'max_volatility': 0.004,   # Lower maximum volatility
    
    # Volume requirements
    'min_volume': 2000,  # Higher volume requirement
    
    # Spread limits
    'max_spread': 0.0002,  # Tighter spread requirement
    
    # Session preferences
    'preferred_sessions': ['london', 'new_york'],
    'avoid_news_times': True,
    'avoid_weekend': True,
}

# Signal Validation Configuration
SIGNAL_VALIDATION_CONFIG = {
    # Minimum signal strength (70% instead of 50%)
    'min_signal_strength': 0.7,
    
    # Require multiple confirmations
    'require_multiple_confirmations': False,
    
    # Minimum timeframes in agreement
    'min_timeframe_alignment': 2,
    
    # Minimum risk-reward ratio
    'min_risk_reward_ratio': 2.0,  # 1:2 minimum
    
    # Maximum signal age (in minutes)
    'max_signal_age': 15,
}

# Trading System Configuration
TRADING_SYSTEM_CONFIG = {
    # Maximum daily trades
    'max_daily_trades': 5,  # Updated to 5 as requested
    
    # Minimum confidence threshold
    'min_confidence': 0.75,  # 75% minimum confidence
    
    # Performance-based adjustments
    'performance_lookback': 20,  # Look at last 20 trades
    
    # Win rate thresholds
    'excellent_win_rate': 0.65,  # 65%+ = excellent
    'good_win_rate': 0.55,       # 55%+ = good
    'poor_win_rate': 0.40,       # <40% = poor
    
    # Position size adjustments based on performance
    'excellent_multiplier': 1.2,  # Increase size for excellent performance
    'good_multiplier': 1.0,       # Normal size for good performance
    'poor_multiplier': 0.5,       # Reduce size for poor performance
}

# Enhanced Entry/Exit Configuration
ENTRY_EXIT_CONFIG = {
    # Entry conditions
    'require_trend_alignment': True,
    'require_volume_confirmation': True,
    'require_volatility_confirmation': True,
    
    # Exit conditions
    'use_trailing_stops': True,
    'trailing_stop_distance': 0.001,  # 10 pips
    'partial_profit_taking': True,
    'profit_target_1': 0.005,  # 50% at 50 pips
    'profit_target_2': 0.010,  # 50% at 100 pips
    
    # Time-based exits
    'max_trade_duration': 24,  # 24 hours maximum
    'min_trade_duration': 5,   # 5 minutes minimum
}

# Performance Tracking Configuration
PERFORMANCE_CONFIG = {
    # Metrics to track
    'track_win_rate': True,
    'track_profit_factor': True,
    'track_sharpe_ratio': True,
    'track_max_drawdown': True,
    'track_avg_trade_duration': True,
    
    # Reporting intervals
    'daily_report': True,
    'weekly_report': True,
    'monthly_report': True,
    
    # Performance thresholds
    'warning_win_rate': 0.35,  # Warn if win rate drops below 35%
    'critical_win_rate': 0.25,  # Stop trading if win rate drops below 25%
}

# Market Session Configuration
SESSION_CONFIG = {
    # Preferred trading sessions (UTC)
    'london_session': {'start': '08:00', 'end': '17:00'},
    'new_york_session': {'start': '13:00', 'end': '22:00'},
    'overlap_session': {'start': '13:00', 'end': '17:00'},  # London-NY overlap
    
    # Avoid trading during
    'avoid_sessions': ['sydney', 'tokyo'],  # Lower liquidity
    'avoid_news_hours': True,
    'news_buffer_minutes': 30,  # 30 minutes before/after news
}

# Symbol-Specific Configuration
SYMBOL_CONFIG = {
    'EURUSD': {
        'min_volatility': 0.0008,
        'max_volatility': 0.004,
        'preferred_sessions': ['london', 'new_york'],
        'max_spread': 0.00015,
        'pip_value': 10.0,
        'pip_size': 0.0001,
    },
    'GBPUSD': {
        'min_volatility': 0.001,
        'max_volatility': 0.005,
        'preferred_sessions': ['london'],
        'max_spread': 0.0002,
        'pip_value': 10.0,
        'pip_size': 0.0001,
    },
    'USDJPY': {
        'min_volatility': 0.0008,
        'max_volatility': 0.004,
        'preferred_sessions': ['tokyo', 'london'],
        'max_spread': 0.0002,
        'pip_value': 10.0,
        'pip_size': 0.01,
    },
    'AUDUSD': {
        'min_volatility': 0.0008,
        'max_volatility': 0.004,
        'preferred_sessions': ['sydney', 'london'],
        'max_spread': 0.0002,
        'pip_value': 10.0,
        'pip_size': 0.0001,
    },
    'XAUUSD': {  # Gold
        'min_volatility': 0.5,
        'max_volatility': 3.0,
        'preferred_sessions': ['london', 'new_york'],
        'max_spread': 0.3,
        'pip_value': 10.0,
        'pip_size': 0.1,
        'trading_hours': '24/5',  # 24 hours, 5 days
    },
    'US30': {  # Dow Jones
        'min_volatility': 10.0,
        'max_volatility': 100.0,
        'preferred_sessions': ['london', 'new_york'],
        'max_spread': 2.0,
        'pip_value': 10.0,
        'pip_size': 1.0,
        'trading_hours': '24/5',  # Allow trading 24/5 like Gold
    },
    'NAS100': {  # NASDAQ 100
        'min_volatility': 5.0,
        'max_volatility': 50.0,
        'preferred_sessions': ['london', 'new_york'],
        'max_spread': 1.0,
        'pip_value': 10.0,
        'pip_size': 1.0,
        'trading_hours': '24/5',  # Allow trading 24/5 like Gold
    },
    'BTCUSD': {  # Bitcoin
        'min_volatility': 50.0,
        'max_volatility': 500.0,
        'preferred_sessions': ['london', 'new_york'],
        'max_spread': 10.0,
        'pip_value': 10.0,
        'pip_size': 1.0,
        'trading_hours': '24/7',  # 24 hours, 7 days
    },
}

# Combined Configuration
IMPROVED_TRADING_CONFIG = {
    'risk_management': RISK_MANAGEMENT_CONFIG,
    'market_filters': MARKET_FILTERS_CONFIG,
    'signal_validation': SIGNAL_VALIDATION_CONFIG,
    'trading_system': TRADING_SYSTEM_CONFIG,
    'entry_exit': ENTRY_EXIT_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'sessions': SESSION_CONFIG,
    'symbols': SYMBOL_CONFIG,
}

# Quick Start Configuration for Immediate Improvement
QUICK_IMPROVEMENT_CONFIG = {
    # Immediate risk reduction
    'risk_per_trade': 0.005,  # 0.5% per trade
    'max_daily_risk': 0.015,  # 1.5% daily
    'max_concurrent_trades': 1,  # Only 1 trade at a time
    
    # Stricter entry requirements
    'min_signal_strength': 0.8,  # 80% confidence required
    'min_risk_reward_ratio': 2.5,  # 1:2.5 minimum
    'require_multiple_confirmations': False,
    
    # Better market conditions
    'min_trend_strength': 0.5,  # Stronger trends only
    'max_spread': 0.00015,  # Tighter spreads
    'preferred_sessions_only': True,  # Only trade during London/NY
    
    # Performance protection
    'max_daily_trades': 2,  # Maximum 2 trades per day
    'stop_trading_on_loss': True,  # Stop after 2 consecutive losses
    'min_win_rate_threshold': 0.4,  # Stop if win rate drops below 40%
}

def get_config_for_symbol(symbol: str, use_quick_improvement: bool = False) -> dict:
    """
    Get configuration for a specific symbol
    
    Args:
        symbol: Trading symbol
        use_quick_improvement: Whether to use quick improvement settings
        
    Returns:
        Configuration dictionary
    """
    if use_quick_improvement:
        config = QUICK_IMPROVEMENT_CONFIG.copy()
    else:
        config = IMPROVED_TRADING_CONFIG.copy()
    
    # Add symbol-specific settings
    if symbol in SYMBOL_CONFIG:
        symbol_config = SYMBOL_CONFIG[symbol]
        config.update(symbol_config)
    
    return config

def get_risk_reduction_config() -> dict:
    """
    Get configuration focused on risk reduction
    
    Returns:
        Risk-focused configuration
    """
    return {
        'risk_per_trade': 0.005,  # 0.5%
        'max_daily_risk': 0.01,   # 1%
        'max_concurrent_trades': 1,
        'min_signal_strength': 0.8,
        'min_risk_reward_ratio': 3.0,
        'max_daily_trades': 1,
        'require_multiple_confirmations': False,
        'preferred_sessions_only': True,
    }

def get_performance_improvement_config() -> dict:
    """
    Get configuration focused on performance improvement
    
    Returns:
        Performance-focused configuration
    """
    return {
        'min_signal_strength': 0.75,
        'min_risk_reward_ratio': 2.0,
        'min_trend_strength': 0.4,
        'max_spread': 0.0002,
        'require_volume_confirmation': True,
        'use_trailing_stops': True,
        'partial_profit_taking': True,
        'max_trade_duration': 12,  # 12 hours max
    }
