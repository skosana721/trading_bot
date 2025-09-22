#!/usr/bin/env python3
"""
Powerful Trading Bot Configuration
=================================

Unified configuration for the powerful trading bot with all advanced features.
"""

# Powerful Trading Bot Configuration
POWERFUL_TRADING_CONFIG = {
    # Basic Trading Parameters
    'symbol': 'EURUSD',
    'timeframe': '5m',
    'risk_per_trade': 0.01,  # 1% risk per trade
    'auto_trade': False,  # Set to True for live trading
    'account_balance': 10000,
    
    # Feature Flags - Enable/Disable Advanced Features
    'use_advanced_risk': True,
    'use_regime_detection': True,
    'use_advanced_ml': True,
    'use_real_time_data': True,
    'use_smart_money': True,
    'use_reinforcement_learning': True,
    'use_market_structure': True,
    
    # Advanced Risk Management Configuration
    'advanced_risk_management': {
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
        'risk_per_trade': 0.01
    },
    
    # Advanced ML Ensemble Configuration
    'advanced_ml_ensemble': {
        'technical_indicators': [
            'rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'williams_r',
            'cci', 'adx', 'obv', 'volume_sma', 'price_momentum', 'volatility',
            'ema_angle', 'macd_slope', 'volume_ratio', 'market_depth'
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
        'ensemble_weights': None,
        'feature_selection_method': 'selectkbest',
        'max_features': 50,
        'retrain_frequency': 7,  # days
        'min_training_samples': 1000
    },
    
    # Market Regime Detection Configuration
    'market_regime_detection': {
        'lookback_period': 100,
        'volatility_window': 20,
        'trend_window': 50,
        'regime_threshold': 0.7,
        'vol_low_threshold': 0.5,
        'vol_high_threshold': 2.0,
        'vol_extreme_threshold': 3.0,
        'trend_strength_threshold': 0.6,
        'range_threshold': 0.02,  # 2% range
        'transition_probability_threshold': 0.3
    },
    
    # Real-Time Data Pipeline Configuration
    'real_time_data_pipeline': {
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
        'max_latency': 0.1,  # seconds
        'sentiment_analysis': True,
        'economic_impact_filtering': True
    },
    
    # Smart Money Concept Configuration
    'smart_money_concept': {
        'order_block_lookback': 20,
        'fair_value_gap_threshold': 0.0005,
        'liquidity_threshold': 0.001,
        'premium_discount_zones': True,
        'institutional_order_blocks': True,
        'mitigation_retest_patterns': True
    },
    
    # Reinforcement Learning Configuration
    'reinforcement_learning': {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 0.3,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'memory_size': 10000,
        'batch_size': 32,
        'target_update_frequency': 100,
        'model_save_frequency': 1000
    },
    
    # Market Structure Strategy Configuration
    'market_structure_strategy': {
        'use_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'lot_size_initial': 0.01,
        'lot_size_reentry': 0.01,
        'risk_per_trade': 2.0,
        'risk_reward_ratio': 2.0,
        'sl_buffer_pips': 10,
        'tp_multiplier': 2.0,
        'enable_trailing_stop': False,
        'trail_start_profit_pips': 50,
        'trail_step_pips': 10
    },
    
    # Trading Strategy Configuration
    'trading_strategy': {
        'min_signal_confidence': 0.6,
        'min_regime_confidence': 0.7,
        'min_ml_confidence': 0.5,
        'max_positions_per_symbol': 1,
        'position_hold_time': 24,  # hours
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'trailing_stop': True,
        'trailing_stop_atr_multiplier': 1.5,
        'signal_weights': {
            'regime': 0.3,
            'ml': 0.25,
            'smc': 0.2,
            'structure': 0.15,
            'rl': 0.1
        }
    },
    
    # Performance Monitoring
    'performance_monitoring': {
        'update_frequency': 60,  # seconds
        'risk_check_frequency': 30,  # seconds
        'regime_update_frequency': 300,  # 5 minutes
        'log_level': 'INFO',
        'save_trades': True,
        'save_performance': True,
        'performance_targets': {
            'min_annual_return': 0.15,  # 15%
            'max_annual_drawdown': 0.10,  # 10%
            'min_sharpe_ratio': 1.5,
            'min_win_rate': 0.55,  # 55%
            'min_profit_factor': 1.5
        }
    },
    
    # Data Storage and Persistence
    'data_storage': {
        'save_models': True,
        'model_save_path': 'models/',
        'data_save_path': 'data/',
        'logs_path': 'logs/',
        'backup_frequency': 3600,  # 1 hour
        'max_log_files': 10,
        'max_log_size': 10485760  # 10MB
    },
    
    # MT5 Configuration
    'mt5_config': {
        'server': 'XM.COM-Demo',
        'login': None,  # Set in environment variables
        'password': None,  # Set in environment variables
        'timeout': 60000,
        'portable': False,
        'path': None  # Auto-detect
    }
}

# Multi-Symbol Configuration
MULTI_SYMBOL_CONFIG = {
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD', 'US30', 'NAS100'],
    'timeframes': ['5m', '15m', '1h', '4h'],
    'max_concurrent_symbols': 5,
    'symbol_rotation': True,
    'correlation_limits': {
        'max_correlation': 0.7,
        'correlation_window': 20
    }
}

# Symbol-Specific Settings
SYMBOL_SPECIFIC_CONFIG = {
    'EURUSD': {
        'max_spread': 0.00015,
        'pip_size': 0.0001,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.0,
        'position_size_multiplier': 1.0,
        'preferred_timeframes': ['5m', '15m', '1h']
    },
    'GBPUSD': {
        'max_spread': 0.0002,
        'pip_size': 0.0001,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.2,
        'position_size_multiplier': 0.8,
        'preferred_timeframes': ['5m', '15m', '1h']
    },
    'USDJPY': {
        'max_spread': 0.0003,
        'pip_size': 0.01,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.1,
        'position_size_multiplier': 0.9,
        'preferred_timeframes': ['5m', '15m', '1h']
    },
    'AUDUSD': {
        'max_spread': 0.0002,
        'pip_size': 0.0001,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.3,
        'position_size_multiplier': 0.7,
        'preferred_timeframes': ['5m', '15m', '1h']
    },
    'XAUUSD': {
        'max_spread': 0.3,
        'pip_size': 0.1,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.5,
        'position_size_multiplier': 0.5,
        'preferred_timeframes': ['15m', '1h', '4h']
    },
    'US30': {
        'max_spread': 2.0,
        'pip_size': 1.0,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.4,
        'position_size_multiplier': 0.6,
        'preferred_timeframes': ['15m', '1h', '4h']
    },
    'NAS100': {
        'max_spread': 1.0,
        'pip_size': 1.0,
        'trading_hours': '24/5',
        'volatility_multiplier': 1.6,
        'position_size_multiplier': 0.4,
        'preferred_timeframes': ['15m', '1h', '4h']
    }
}

# Market Session Configuration
MARKET_SESSION_CONFIG = {
    'asian': {
        'start_hour': 0,
        'end_hour': 8,
        'volatility_multiplier': 0.8,
        'preferred_symbols': ['USDJPY', 'AUDUSD'],
        'risk_multiplier': 0.8
    },
    'london': {
        'start_hour': 8,
        'end_hour': 16,
        'volatility_multiplier': 1.2,
        'preferred_symbols': ['EURUSD', 'GBPUSD'],
        'risk_multiplier': 1.0
    },
    'new_york': {
        'start_hour': 16,
        'end_hour': 24,
        'volatility_multiplier': 1.1,
        'preferred_symbols': ['EURUSD', 'USDJPY', 'US30', 'NAS100'],
        'risk_multiplier': 1.0
    },
    'overlap': {
        'start_hour': 8,
        'end_hour': 16,
        'volatility_multiplier': 1.5,
        'preferred_symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'risk_multiplier': 1.2
    }
}

# Economic Calendar High-Impact Events
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

# News Sentiment Keywords
SENTIMENT_KEYWORDS = {
    'positive': [
        'growth', 'increase', 'rise', 'surge', 'boost', 'strong', 'robust',
        'expansion', 'recovery', 'improvement', 'gain', 'profit', 'success',
        'bullish', 'optimistic', 'positive', 'upward', 'momentum'
    ],
    'negative': [
        'decline', 'fall', 'drop', 'crash', 'recession', 'crisis', 'weak',
        'contraction', 'loss', 'decrease', 'downturn', 'slump', 'failure',
        'bearish', 'pessimistic', 'negative', 'downward', 'volatility'
    ],
    'neutral': [
        'stable', 'unchanged', 'flat', 'steady', 'maintain', 'hold', 'consistent',
        'neutral', 'mixed', 'uncertain', 'wait', 'observe'
    ]
}

# Risk Management Rules
RISK_MANAGEMENT_RULES = {
    'max_daily_trades': 10,
    'max_weekly_trades': 50,
    'max_monthly_trades': 200,
    'min_time_between_trades': 300,  # 5 minutes
    'max_correlation_exposure': 0.7,
    'volatility_adjustment_threshold': 2.0,
    'news_impact_reduction': 0.5,  # Reduce position size by 50% during high impact news
    'weekend_risk_reduction': 0.3,  # Reduce risk by 30% on Fridays
    'holiday_risk_reduction': 0.5,  # Reduce risk by 50% during holidays
    'drawdown_reduction': {
        '2_percent': 0.8,  # Reduce position size by 20% at 2% drawdown
        '3_percent': 0.6,  # Reduce position size by 40% at 3% drawdown
        '4_percent': 0.4,  # Reduce position size by 60% at 4% drawdown
        '5_percent': 0.0   # Stop trading at 5% drawdown
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': True,
    'max_file_size': 10485760,  # 10MB
    'backup_count': 5,
    'console_output': True,
    'file_output': True,
    'loggers': {
        'powerful_trading_bot': 'INFO',
        'core.advanced_risk_manager': 'INFO',
        'strategies.advanced_ml_ensemble': 'INFO',
        'strategies.market_regime_detector': 'INFO',
        'core.real_time_data_pipeline': 'INFO'
    }
}
