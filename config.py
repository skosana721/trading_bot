#!/usr/bin/env python3
"""
Configuration Management for Trading Bot
=======================================

Centralized configuration management with environment variable support,
validation, and type safety.
"""

import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Trading bot configuration with validation"""
    
    # MT5 Connection Settings
    account_number: Optional[str] = None
    password: Optional[str] = None
    server: str = 'XMGlobal-Demo'
    
    # Trading Parameters
    symbol: str = 'EURUSD'
    timeframe: str = '5m'
    risk_per_trade: float = 0.02  # 2%
    auto_trade: bool = False
    use_ml: bool = True
    use_smc: bool = True
    
    # Risk Management
    max_positions_per_symbol: int = 3
    max_same_direction_positions: int = 2
    max_daily_trades: int = 10
    max_daily_loss: float = 0.05  # 5%
    
    # Technical Analysis
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    
    # Machine Learning
    ml_confidence_threshold: float = 0.65
    ml_lookforward_periods: int = 5
    ml_min_training_data: int = 1000
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'logs/trading_bot.log'
    
    # Performance
    analysis_interval_minutes: int = 5
    max_retries: int = 3
    connection_timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate risk per trade
        if not 0.001 <= self.risk_per_trade <= 0.1:
            raise ValueError("Risk per trade must be between 0.1% and 10%")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.timeframe}")
        
        # Validate ML confidence threshold
        if not 0.5 <= self.ml_confidence_threshold <= 0.95:
            raise ValueError("ML confidence threshold must be between 0.5 and 0.95")
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Create configuration from environment variables"""
        return cls(
            account_number=os.getenv('XM_ACCOUNT_NUMBER'),
            password=os.getenv('XM_PASSWORD'),
            server=os.getenv('XM_SERVER', 'XMGlobal-Demo'),
            symbol=os.getenv('TRADING_SYMBOL', 'EURUSD'),
            timeframe=os.getenv('TRADING_TIMEFRAME', '5m'),
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
            auto_trade=os.getenv('AUTO_TRADE', 'false').lower() == 'true',
            use_ml=os.getenv('USE_ML', 'true').lower() == 'true',
            use_smc=os.getenv('USE_SMC', 'true').lower() == 'true',
            max_positions_per_symbol=int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3')),
            max_same_direction_positions=int(os.getenv('MAX_SAME_DIRECTION_POSITIONS', '2')),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', '10')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            analysis_interval_minutes=int(os.getenv('ANALYSIS_INTERVAL_MINUTES', '5'))
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'account_number': self.account_number,
            'password': self.password,
            'server': self.server,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'risk_per_trade': self.risk_per_trade,
            'auto_trade': self.auto_trade,
            'use_ml': self.use_ml,
            'use_smc': self.use_smc,
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'max_same_direction_positions': self.max_same_direction_positions,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_loss': self.max_daily_loss
        }
    
    def update(self, updates: Dict):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._validate_config()

@dataclass
class AvailableSymbols:
    """Available trading symbols configuration"""
    
    forex_majors: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
    ])
    
    forex_minors: List[str] = field(default_factory=lambda: [
        'EURGBP', 'EURJPY', 'GBPJPY', 'CHFJPY', 'AUDCAD', 'AUDCHF', 'AUDJPY',
        'AUDNZD', 'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY'
    ])
    
    @property
    def all_symbols(self) -> List[str]:
        """Get all available symbols"""
        return self.forex_majors + self.forex_minors
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol in self.all_symbols

@dataclass
class AvailableTimeframes:
    """Available timeframes configuration"""
    
    timeframes: List[Dict[str, str]] = field(default_factory=lambda: [
        {'value': '1m', 'label': '1 Minute'},
        {'value': '5m', 'label': '5 Minutes'},
        {'value': '15m', 'label': '15 Minutes'},
        {'value': '30m', 'label': '30 Minutes'},
        {'value': '1h', 'label': '1 Hour'},
        {'value': '4h', 'label': '4 Hours'},
        {'value': '1d', 'label': '1 Day'}
    ])
    
    @property
    def values(self) -> List[str]:
        """Get all timeframe values"""
        return [tf['value'] for tf in self.timeframes]
    
    def is_valid_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is valid"""
        return timeframe in self.values

class ConfigManager:
    """Configuration manager singleton"""
    
    _instance = None
    _config: Optional[TradingConfig] = None
    _symbols: Optional[AvailableSymbols] = None
    _timeframes: Optional[AvailableTimeframes] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = TradingConfig.from_env()
            self._symbols = AvailableSymbols()
            self._timeframes = AvailableTimeframes()
    
    @property
    def config(self) -> TradingConfig:
        """Get trading configuration"""
        return self._config
    
    @property
    def symbols(self) -> AvailableSymbols:
        """Get available symbols"""
        return self._symbols
    
    @property
    def timeframes(self) -> AvailableTimeframes:
        """Get available timeframes"""
        return self._timeframes
    
    def update_config(self, updates: Dict):
        """Update configuration"""
        self._config.update(updates)
    
    def get_config_dict(self) -> Dict:
        """Get configuration as dictionary"""
        return self._config.to_dict()
    
    def validate_trading_params(self, symbol: str, timeframe: str, risk_per_trade: float) -> bool:
        """Validate trading parameters"""
        try:
            if not self._symbols.is_valid_symbol(symbol):
                return False
            if not self._timeframes.is_valid_timeframe(timeframe):
                return False
            if not 0.001 <= risk_per_trade <= 0.1:
                return False
            return True
        except Exception:
            return False

# Global configuration instance
config_manager = ConfigManager() 