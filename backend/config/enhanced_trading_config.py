#!/usr/bin/env python3
"""
Enhanced Trading Bot Configuration
=================================

Configuration file for the enhanced trading bot with all advanced features.
"""

# Enhanced Trading Bot Configuration
ENHANCED_TRADING_CONFIG = {
    # Basic trading parameters
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD', 'US30', 'NAS100'],
    'timeframes': ['5m', '15m', '1h', '4h'],
    'auto_trade': False,  # Set to True for live trading
    'account_balance': 10000,
    
    # Risk Management Configuration
    'risk_management': {
        'max_portfolio_risk': 0.02,  # 2% max portfolio risk
        'max_position_risk': 0.005,  # 0.5% max position risk
        'max_correlation': 0.7,  # Max correlation between positions
        'var_confidence': 0.95,  # VaR confidence level
        'lookback_period': 252,  # Trading days for calculations
        'min_kelly_fraction': 0.01,  # Minimum Kelly fraction
        'max_kelly_fraction': 0.25,  # Maximum Kelly fraction
        'max_daily_loss': 0.01,  # 1% max daily loss
        'max_consecutive_losses': 5,
        'max_drawdown_limit': 0.05,  # 5% max drawdown
        'account_balance': 10000,
        'risk_per_trade': 0.01,  # 1% risk per trade
        'max_position_size': 1.0
    },
    
    # ML Ensemble Configuration
    'ml_ensemble': {
        'technical_indicators': [
            'rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'williams_r',
            'cci', 'adx', 'obv', 'volume_sma', 'price_momentum', 'volatility'
        ],
        'lookback_periods': [5, 10, 20, 50],
        'feature_lags': [1, 2, 3, 5],
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'use_voting': True,
        'use_stacking': True,
        'use_blending': True,
        'ensemble_weights': None
    },
    
    # Market Regime Detection Configuration
    'regime_detection': {
        'lookback_period': 100,
        'volatility_window': 20,
        'trend_window': 50,
        'regime_threshold': 0.7,
        'vol_low_threshold': 0.5,
        'vol_high_threshold': 2.0,
        'vol_extreme_threshold': 3.0,
        'trend_strength_threshold': 0.6,
        'range_threshold': 0.02  # 2% range
    },
    
    # Real-Time Data Pipeline Configuration
    'data_pipeline': {
        'price_sources': {
            'primary': 'mt5',
            'backup': 'yahoo_finance'
        },
        'news_sources': {
            'primary': 'news_api',
            'backup': 'rss_feeds'
        },
        'economic_calendar': {
            'source': 'economic_calendar_api',
            'update_frequency': 300  # 5 minutes
        },
        'social_media': {
            'platforms': ['twitter', 'reddit'],
            'update_frequency': 30  # 30 seconds
        },
        'update_frequency': 1.0,  # seconds
        'buffer_size': 1000,
        'max_latency': 0.1  # seconds
    },
    
    # Backtesting Configuration
    'backtesting': {
        'initial_capital': 10000,
        'commission_rate': 0.001,  # 0.1%
        'slippage_rate': 0.0005,  # 0.05%
        'risk_free_rate': 0.02,  # 2% annual
        'max_positions': 5,
        'position_sizing': 'kelly',  # 'fixed', 'kelly', 'volatility'
        'risk_per_trade': 0.02  # 2%
    },
    
    # Trading Strategy Configuration
    'strategy': {
        'min_signal_confidence': 0.6,
        'min_regime_confidence': 0.7,
        'min_ml_confidence': 0.5,
        'max_positions_per_symbol': 1,
        'position_hold_time': 24,  # hours
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'trailing_stop': True,
        'trailing_stop_atr_multiplier': 1.5
    },
    
    # Performance Monitoring
    'monitoring': {
        'performance_update_frequency': 60,  # seconds
        'risk_check_frequency': 30,  # seconds
        'regime_update_frequency': 300,  # 5 minutes
        'log_level': 'INFO',
        'save_trades': True,
        'save_performance': True
    },
    
    # Data Storage
    'storage': {
        'save_models': True,
        'model_save_path': 'models/',
        'data_save_path': 'data/',
        'logs_path': 'logs/',
        'backup_frequency': 3600  # 1 hour
    }
}

# Symbol-specific configurations
SYMBOL_CONFIGS = {
    'EURUSD': {
        'max_spread': 0.00015,
        'pip_size': 0.0001,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.0,
        'position_size_multiplier': 1.0
    },
    'GBPUSD': {
        'max_spread': 0.0002,
        'pip_size': 0.0001,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.2,
        'position_size_multiplier': 0.8
    },
    'USDJPY': {
        'max_spread': 0.0003,
        'pip_size': 0.01,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.1,
        'position_size_multiplier': 0.9
    },
    'AUDUSD': {
        'max_spread': 0.0002,
        'pip_size': 0.0001,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.3,
        'position_size_multiplier': 0.7
    },
    'XAUUSD': {
        'max_spread': 0.3,
        'pip_size': 0.1,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.5,
        'position_size_multiplier': 0.5
    },
    'US30': {
        'max_spread': 2.0,
        'pip_size': 1.0,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.4,
        'position_size_multiplier': 0.6
    },
    'NAS100': {
        'max_spread': 1.0,
        'pip_size': 1.0,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.6,
        'position_size_multiplier': 0.4
    }
}

# Market session configurations
TRADING_SESSIONS = {
    'asian': {
        'start_hour': 0,
        'end_hour': 8,
        'volatility_multiplier': 0.8,
        'preferred_symbols': ['USDJPY', 'AUDUSD']
    },
    'london': {
        'start_hour': 8,
        'end_hour': 16,
        'volatility_multiplier': 1.2,
        'preferred_symbols': ['EURUSD', 'GBPUSD']
    },
    'new_york': {
        'start_hour': 16,
        'end_hour': 24,
        'volatility_multiplier': 1.1,
        'preferred_symbols': ['EURUSD', 'USDJPY', 'US30', 'NAS100']
    },
    'overlap': {
        'start_hour': 8,
        'end_hour': 16,
        'volatility_multiplier': 1.5,
        'preferred_symbols': ['EURUSD', 'GBPUSD', 'USDJPY']
    }
}

# Economic calendar high-impact events
HIGH_IMPACT_EVENTS = [
    'Non-Farm Payrolls',
    'Federal Funds Rate',
    'GDP Growth Rate',
    'Consumer Price Index',
    'Unemployment Rate',
    'Retail Sales',
    'Industrial Production',
    'Trade Balance',
    'Current Account',
    'Manufacturing PMI',
    'Services PMI',
    'Consumer Confidence',
    'Business Confidence',
    'Housing Starts',
    'Building Permits'
]

# News sentiment keywords
SENTIMENT_KEYWORDS = {
    'positive': [
        'growth', 'increase', 'rise', 'surge', 'boost', 'strong', 'robust',
        'expansion', 'recovery', 'improvement', 'gain', 'profit', 'success'
    ],
    'negative': [
        'decline', 'fall', 'drop', 'crash', 'recession', 'crisis', 'weak',
        'contraction', 'loss', 'decrease', 'downturn', 'slump', 'failure'
    ],
    'neutral': [
        'stable', 'unchanged', 'flat', 'steady', 'maintain', 'hold', 'consistent'
    ]
}

# Risk management rules
RISK_RULES = {
    'max_daily_trades': 10,
    'max_weekly_trades': 50,
    'max_monthly_trades': 200,
    'min_time_between_trades': 300,  # 5 minutes
    'max_correlation_exposure': 0.7,
    'volatility_adjustment_threshold': 2.0,
    'news_impact_reduction': 0.5,  # Reduce position size by 50% during high impact news
    'weekend_risk_reduction': 0.3,  # Reduce risk by 30% on Fridays
    'holiday_risk_reduction': 0.5   # Reduce risk by 50% during holidays
}

# Performance targets
PERFORMANCE_TARGETS = {
    'min_annual_return': 0.15,  # 15%
    'max_annual_drawdown': 0.10,  # 10%
    'min_sharpe_ratio': 1.5,
    'min_win_rate': 0.55,  # 55%
    'min_profit_factor': 1.5,
    'max_consecutive_losses': 5,
    'min_trades_per_month': 20
}

# Model training configuration
MODEL_TRAINING = {
    'train_test_split': 0.8,
    'cross_validation_folds': 5,
    'feature_selection_method': 'selectkbest',
    'max_features': 50,
    'retrain_frequency': 7,  # days
    'min_training_samples': 1000,
    'validation_split': 0.2
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': True,
    'max_file_size': 10485760,  # 10MB
    'backup_count': 5,
    'console_output': True,
    'file_output': True
}
