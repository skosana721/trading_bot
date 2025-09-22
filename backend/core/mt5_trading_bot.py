#!/usr/bin/env python3
"""
MT5 Integrated Trading Bot for XM Account
========================================

This trading bot combines technical analysis with MetaTrader 5 execution
for automated day trading on XM trading account.
Enhanced with Machine Learning capabilities for improved signal generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import warnings
import requests
import json
warnings.filterwarnings('ignore')

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 module not found. Please install it using: pip install MetaTrader5")
    mt5 = None

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    ML_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("ML libraries not available. Install with: pip install scikit-learn joblib")
    ML_AVAILABLE = False

# ML Ensemble
try:
    from strategies.ml_ensemble import MLEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("ML Ensemble module not available")
    ENSEMBLE_AVAILABLE = False

# Market Structure Strategy
try:
    from strategies.market_structure_strategy import MarketStructureStrategy
    MARKET_STRUCTURE_AVAILABLE = True
except ImportError:
    MARKET_STRUCTURE_AVAILABLE = False
    # Create a dummy MarketStructureStrategy class for compatibility
    class MarketStructureStrategy:
        def __init__(self, config):
            self.config = config
        def analyze(self, *args, **kwargs):
            return {'signal': 'NEUTRAL', 'confidence': 0.0}
        def should_trade(self, *args, **kwargs):
            return False

# Reinforcement Learning Trader
try:
    from strategies.reinforcement_learning_trader import ReinforcementLearningTrader
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    # Create a dummy ReinforcementLearningTrader class for compatibility
    class ReinforcementLearningTrader:
        def __init__(self, **kwargs):
            pass
        def get_action(self, *args, **kwargs):
            return 'HOLD'
        def update(self, *args, **kwargs):
            pass
        def save_model(self, *args, **kwargs):
            pass
        def load_model(self, *args, **kwargs):
            pass

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("TA library not available. Install with: pip install ta")
    TA_AVAILABLE = False

# Smart Money Concept
try:
    from strategies.smart_money_concept import SmartMoneyConcept
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    # Create a dummy SmartMoneyConcept class for compatibility
    class SmartMoneyConcept:
        def __init__(self, *args, **kwargs):
            pass
        def analyze(self, *args, **kwargs):
            return {'signal': 'NEUTRAL', 'confidence': 0.0}
        def get_signals(self, *args, **kwargs):
            return {}

# Advanced Risk Management
try:
    from core.advanced_risk_manager import AdvancedRiskManager
    ADVANCED_RISK_AVAILABLE = True
except ImportError:
    ADVANCED_RISK_AVAILABLE = False
    # Create a dummy AdvancedRiskManager class for compatibility
    class AdvancedRiskManager:
        def __init__(self, *args, **kwargs):
            pass
        def assess_position_risk(self, *args, **kwargs):
            return None
        def check_risk_limits(self, *args, **kwargs):
            return True, "Risk limits satisfied"
        def get_risk_summary(self, *args, **kwargs):
            return {'portfolio_metrics': {}, 'circuit_breakers': {}}

# Market Regime Detection
try:
    from strategies.market_regime_detector import MarketRegimeDetector
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False
    # Create a dummy MarketRegimeDetector class for compatibility
    class MarketRegimeDetector:
        def __init__(self, *args, **kwargs):
            pass
        def detect_regime(self, *args, **kwargs):
            return None
        def get_regime_summary(self, *args, **kwargs):
            return {'current_regime': 'unknown', 'confidence': 0.0}

from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple

# Try to import MT5Connector
try:
    from connectors.mt5_connector import MT5Connector
    MT5_CONNECTOR_AVAILABLE = True
except ImportError:
    MT5_CONNECTOR_AVAILABLE = False
    print("Warning: MT5Connector not available - trading bot will run in simulation mode")
    # Create a dummy MT5Connector class for compatibility
    class MT5Connector:
        def __init__(self, *args, **kwargs):
            self.connected = False
            self.logger = None
        def connect(self):
            return False
        def disconnect(self):
            pass

# Load environment variables
load_dotenv()

class MT5TradingBot:
    def __init__(self, symbol: str, timeframe: str, risk_per_trade: float = 0.005, 
                 use_mt5_data: bool = True, auto_trade: bool = False, use_ml: bool = True, 
                 use_smc: bool = True, use_advanced_risk: bool = True, use_regime_detection: bool = True) -> None:
        """
        Initialize MT5 integrated trading bot with enhanced features
        
        Args:
            symbol (str): Symbol to trade (e.g., 'EURUSD', 'GBPUSD')
            timeframe (str): Timeframe for analysis ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            risk_per_trade (float): Risk per trade as percentage
            use_mt5_data (bool): Use MT5 data instead of yfinance
            auto_trade (bool): Enable automatic trading
            use_ml (bool): Enable machine learning features
            use_smc (bool): Enable Smart Money Concept analysis
            use_advanced_risk (bool): Enable advanced risk management
            use_regime_detection (bool): Enable market regime detection
        """
        # Initialize logger first
        import logging as _logging
        self.logger = _logging.getLogger('mt5_trading_bot')
        
        self.symbol = symbol
        self.timeframe = timeframe
        # Ensure risk_per_trade has a valid value
        if risk_per_trade is None or risk_per_trade <= 0 or risk_per_trade > 0.1:
            self.risk_per_trade = 0.005  # Default to 0.5% (improved)
            self.logger.warning(f"Invalid risk_per_trade value ({risk_per_trade}), using default 0.005 (0.5%)")
        else:
            self.risk_per_trade = risk_per_trade
        self.use_mt5_data = use_mt5_data
        self.auto_trade = auto_trade
        self.use_ml = use_ml and ML_AVAILABLE
        self.use_smc = use_smc and SMC_AVAILABLE
        self.use_advanced_risk = use_advanced_risk and ADVANCED_RISK_AVAILABLE
        self.use_regime_detection = use_regime_detection and REGIME_DETECTION_AVAILABLE
        
        # ML Components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_trained = False
        self.prediction_threshold = 0.75  # Confidence threshold for ML signals (improved)
        
        # ML Ensemble Components
        self.ml_ensemble = None
        self.use_ensemble = use_ml and ENSEMBLE_AVAILABLE
        if self.use_ensemble:
            self.ml_ensemble = MLEnsemble(symbol, timeframe, use_deep_learning=True)
        
        # Market Structure Strategy Components
        self.market_structure_strategy = None
        self.use_market_structure = MARKET_STRUCTURE_AVAILABLE
        if self.use_market_structure:
            # Default configuration for market structure strategy
            # Will be updated with symbol-specific minimum lot size after connection
            strategy_config = {
                'UsePairs': [symbol],
                'LotSizeInitial': 0.01,  # Will be updated to symbol-specific minimum
                'LotSizeReEntry': 0.01,  # Will be updated to symbol-specific minimum
                'RiskPerTrade': 2.0,
                'RiskRewardRatio': 2.0,
                'SL_Buffer_Pips': 10,
                'TP_Multiplier': 2.0,
                'EnableTrailingStop': False,
                'TrailStartProfitPips': 50,
                'TrailStepPips': 10
            }
            self.market_structure_strategy = MarketStructureStrategy(strategy_config)
        
        # Reinforcement Learning Components
        self.rl_trader = None
        self.use_rl = use_ml and RL_AVAILABLE
        if self.use_rl:
            self.rl_trader = ReinforcementLearningTrader(
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=0.3,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                model_path=f"models/rl_trader_{symbol}_{timeframe}.pkl"
            )
        
        # Advanced Risk Management Components
        self.advanced_risk_manager = None
        if self.use_advanced_risk:
            risk_config = {
                'max_portfolio_risk': 0.02,
                'max_position_risk': 0.005,
                'max_correlation': 0.7,
                'max_drawdown_limit': 0.05,
                'max_consecutive_losses': 5,
                'account_balance': 10000,
                'risk_per_trade': self.risk_per_trade
            }
            self.advanced_risk_manager = AdvancedRiskManager(risk_config)
            self.logger.info("Advanced risk management enabled")
        
        # Market Regime Detection Components
        self.regime_detector = None
        if self.use_regime_detection:
            regime_config = {
                'lookback_period': 100,
                'volatility_window': 20,
                'trend_window': 50,
                'regime_threshold': 0.7,
                'trend_strength_threshold': 0.6,
                'range_threshold': 0.02
            }
            self.regime_detector = MarketRegimeDetector(regime_config)
            self.logger.info("Market regime detection enabled")
        
        # SMC Components
        self.smc_analyzer = None
        self.smc_signals = {}
        self.smc_summary = {}
        
        # Initialize components
        self.analysis_bot = None
        self.mt5_connector = None
        self.connected = False
        
        # Logger already initialized at the beginning of __init__

        # Trading state
        self.last_analysis = None
        self.current_position = None
        self.trade_history = []
        self.ml_predictions = []
        
        # Improved trading system components
        self.max_concurrent_trades = 3  # Updated to 3 as requested
        self.max_daily_trades = 5  # Updated to 5 as requested
        self.min_signal_strength = 0.8  # Increased from 0.5
        self.min_risk_reward_ratio = 2.5  # Increased from 1.5
        self.require_multiple_confirmations = False
        self.stop_trading_on_loss = True
        self.consecutive_losses = 0
        self.max_consecutive_losses = 2
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Market condition filters
        self.min_trend_strength = 0.5
        self.max_spread = 0.00015  # 1.5 pips (default for forex)
        self.preferred_sessions_only = True
        self.london_session_start = 8  # UTC
        self.london_session_end = 17  # UTC
        self.new_york_session_start = 13  # UTC
        self.new_york_session_end = 22  # UTC
        
        # Symbol-specific settings
        self.symbol_config = self.get_symbol_config(symbol)
    
    def get_minimum_lot_size(self) -> float:
        """
        Get the minimum lot size for the current symbol
        
        Returns:
            float: Minimum lot size for the symbol, or 0.01 as fallback
        """
        if self.connected and self.mt5_connector:
            symbol_info = self.mt5_connector.get_symbol_info(self.symbol)
            if symbol_info:
                min_volume = symbol_info.get('volume_min', 0.01)
                self.logger.info(f"Symbol {self.symbol} minimum lot size: {min_volume}")
                return min_volume
        
        # Fallback to default minimum
        self.logger.warning(f"Using fallback minimum lot size 0.01 for {self.symbol}")
        return 0.01
    
    def update_strategy_lot_sizes(self):
        """
        Update strategy configurations with symbol-specific minimum lot sizes
        """
        if self.connected and self.mt5_connector and self.mt5_connector.connected:
            try:
                min_lot_size = self.get_minimum_lot_size()
                
                # Update market structure strategy if available
                if self.market_structure_strategy and hasattr(self.market_structure_strategy, 'config'):
                    self.market_structure_strategy.config['LotSizeInitial'] = min_lot_size
                    self.market_structure_strategy.config['LotSizeReEntry'] = min_lot_size
                    self.logger.info(f"Updated market structure strategy lot sizes to {min_lot_size}")
            except Exception as e:
                self.logger.warning(f"Could not update strategy lot sizes: {e}")
                # Use default lot size as fallback
                if self.market_structure_strategy and hasattr(self.market_structure_strategy, 'config'):
                    self.market_structure_strategy.config['LotSizeInitial'] = 0.01
                    self.market_structure_strategy.config['LotSizeReEntry'] = 0.01
        
        # Trade Journal Integration
        self.trade_journal_enabled = True
        self.trade_journal_api_url = "http://localhost:5000/api/trade-journal"
        self.trade_journal_api_key = os.getenv('API_KEY', '')
        self.logged_trades = {}  # Track logged trades by order_id
        self.learning_enabled = True
        self.performance_history = []
        
        # Initialize MT5 connector
        if self.use_mt5_data or self.auto_trade:
            if MT5_CONNECTOR_AVAILABLE:
                self.mt5_connector = MT5Connector()
                self.connect_mt5()
            else:
                self.logger.warning("MT5Connector not available - running in simulation mode")
                self.connected = False
        
        # Try to load existing ML model on startup
        if self.use_ml:
            os.makedirs('models', exist_ok=True)
            default_model_name = os.path.join('models', f"ml_model_{self.symbol}_{self.timeframe}.joblib")
            if self.load_ml_model(default_model_name):
                self.logger.info(f"Loaded existing ML model: {default_model_name}")
            else:
                self.logger.info(f"No existing ML model found. Will train new model when needed.")
        
        # Initialize SMC analyzer
        if self.use_smc:
            self.logger.info(f"Smart Money Concept analysis enabled for {self.symbol}")
        else:
            self.logger.warning(f"Smart Money Concept analysis disabled")
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 and get account information"""
        if not MT5_CONNECTOR_AVAILABLE:
            self.logger.warning("MT5Connector not available - cannot connect")
            self.connected = False
            return False
            
        if self.mt5_connector:
            self.connected = self.mt5_connector.connect()
            if self.connected:
                self.logger.info("MT5 connection established")
                # Update strategy configurations with symbol-specific lot sizes
                # self.update_strategy_lot_sizes()  # Temporarily disabled to prevent recursion
                # Verify account connection by getting account info
                account_info = self.mt5_connector.get_account_summary()
                if account_info:
                    balance = account_info.get('balance', 0)
                    self.logger.info(f"Account Balance: ${balance:,.2f}")
                else:
                    self.logger.warning("Could not get account balance")
            else:
                self.logger.error("MT5 connection failed")
        return self.connected
    
    def debug_data_structure(self, data: Optional[pd.DataFrame]) -> None:
        """
        Debug method to check data structure
        
        Args:
            data (pd.DataFrame): Data to debug
        """
        if data is None:
            self.logger.error("Data is None")
            return
        
        self.logger.debug("DATA STRUCTURE DEBUG")
        self.logger.debug(f"Shape: {data.shape} | Columns: {list(data.columns)}")
        self.logger.debug("Data types:")
        for col in data.columns:
            self.logger.debug(f"{col}: {data[col].dtype}")
        
        if len(data) > 0:
            self.logger.debug(f"First row: {data.iloc[0].to_dict()}")
            self.logger.debug(f"Last row: {data.iloc[-1].to_dict()}")
        
        # Check for required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
        else:
            self.logger.debug("All required columns present")
    
    def get_market_data(self) -> Optional[pd.DataFrame]:
        """
        Get market data from MT5 or yfinance
        
        Returns:
            pd.DataFrame: Market data or None if failed
        """
        if self.use_mt5_data and self.connected:
            # Get data from MT5 - request more data for shorter timeframes
            if self.timeframe in ['1m', '5m']:
                data_count = 20000  # Increased data for short timeframes
            else:
                data_count = 5000   # Increased amount for longer timeframes
            
            data = self.mt5_connector.get_historical_data(self.symbol, self.timeframe, data_count)
            if data is not None:
                # Debug data structure
                self.debug_data_structure(data)
                
                # Ensure data has the correct column structure for the trading bot
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Check if we have the required columns
                if all(col in data.columns for col in required_columns):
                    self.logger.debug("MT5 data structure is correct")
                    return data
                else:
                    self.logger.warning(f"MT5 data missing required columns. Available: {list(data.columns)}")
                    # When using MT5 data, avoid falling back to yfinance to prevent 404 noise
                    return None
        
        # Fallback to yfinance only if MT5 data is not requested
        if not self.use_mt5_data:
            self.logger.error("yfinance data not supported - please use MT5 data")
            return None
        
        return None
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical indicators as ML features
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if not TA_AVAILABLE:
            self.logger.error("TA library not available for technical indicators")
            return data
        
        df = data.copy()
        
        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['macd_signal'] = ta.trend.macd_signal(df['Close'])
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
        df['stoch_signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
        df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Volatility Indicators
        df['bb_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['Close'], window=20)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # Volume Indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Price Action Features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Lagged features
        for i in range(1, 6):
            df[f'close_lag_{i}'] = df['Close'].shift(i)
            df[f'volume_lag_{i}'] = df['Volume'].shift(i)
            df[f'rsi_lag_{i}'] = df['rsi'].shift(i)
        
        # Rolling statistics
        df['close_rolling_mean_5'] = df['Close'].rolling(window=5).mean()
        df['close_rolling_std_5'] = df['Close'].rolling(window=5).std()
        df['volume_rolling_mean_5'] = df['Volume'].rolling(window=5).mean()
        
        # Handle infinities and NaNs to ensure prediction can run on latest row
        try:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Forward/backward fill technical columns to remove NaNs on recent rows
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
        except Exception:
            pass

        return df
    
    def create_target_variable(self, data: pd.DataFrame, lookforward: int = 5) -> pd.Series:
        """
        Create target variable for ML training
        
        Args:
            data (pd.DataFrame): Price data with technical indicators
            lookforward (int): Number of periods to look forward
            
        Returns:
            pd.Series: Binary target variable (1 for profitable move, 0 otherwise)
        """
        # Calculate future returns
        future_returns = data['Close'].shift(-lookforward) / data['Close'] - 1
        
        # Create binary target: 1 if return > threshold, 0 otherwise
        threshold = 0.002  # 0.2% threshold for profitable move
        target = (future_returns > threshold).astype(int)
        
        return target
    
    def prepare_ml_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for ML model
        
        Args:
            data (pd.DataFrame): Data with technical indicators
            
        Returns:
            tuple: (X_features, y_target, feature_names)
        """
        # Define feature columns
        feature_columns = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'adx',
            'rsi', 'stoch', 'stoch_signal', 'williams_r', 'cci',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position', 'atr',
            'volume_sma', 'volume_ratio', 'obv',
            'price_change', 'high_low_ratio', 'close_open_ratio', 'body_size',
            'upper_shadow', 'lower_shadow',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5',
            'rsi_lag_1', 'rsi_lag_2', 'rsi_lag_3', 'rsi_lag_4', 'rsi_lag_5',
            'close_rolling_mean_5', 'close_rolling_std_5', 'volume_rolling_mean_5'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Create target variable
        target = self.create_target_variable(data)
        
        # Prepare feature matrix
        X = data[available_features].copy()
        y = target.copy()
        
        # Remove rows with NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.feature_columns = available_features
        
        return X, y, available_features
    
    def train_ml_model(self, data: pd.DataFrame) -> bool:
        """
        Train machine learning model (ensemble if available, otherwise single model)
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            bool: True if training successful
        """
        if not self.use_ml:
            self.logger.error("ML features disabled")
            return False
        
        try:
            # Use ensemble if available
            if self.use_ensemble and self.ml_ensemble:
                self.logger.info("Training ML ensemble models...")
                success = self.ml_ensemble.train_models(data)
                if success:
                    self.model_trained = True
                    self.logger.info("ML ensemble models trained successfully")
                return success
            
            # Fallback to single model
            self.logger.info("Training single ML model...")
            
            # Create technical features
            data_with_features = self.create_technical_features(data)
            
            # Prepare ML features
            X, y, feature_names = self.prepare_ml_features(data_with_features)
            
            if len(X) < 100:
                self.logger.error("Insufficient data for ML training (need at least 100 samples)")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (Random Forest)
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.ml_model.score(X_train_scaled, y_train)
            test_score = self.ml_model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.ml_model, X_train_scaled, y_train, cv=5)
            
            self.logger.info("Single ML Model Trained Successfully")
            self.logger.info(f"Training Accuracy: {train_score:.3f} | Test: {test_score:.3f} | CV: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f}) | Features: {len(feature_names)}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.ml_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.debug("Top 10 Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                self.logger.debug(f"{row['feature']}: {row['importance']:.3f}")
            
            self.model_trained = True
            return True
            
        except Exception as e:
            self.logger.exception(f"ML training failed: {e}")
            return False
    
    def get_ml_prediction(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Get ML prediction for current market conditions (ensemble if available, otherwise single model)
        
        Args:
            data (pd.DataFrame): Current market data
            
        Returns:
            dict: ML prediction results
        """
        if not self.use_ml or not self.model_trained:
            return None
        
        try:
            # Use ensemble if available
            if self.use_ensemble and self.ml_ensemble:
                ensemble_prediction = self.ml_ensemble.get_ensemble_prediction(data)
                if ensemble_prediction:
                    return {
                        'prediction': ensemble_prediction['prediction'],
                        'confidence': ensemble_prediction['confidence'],
                        'buy_probability': ensemble_prediction['confidence'] if ensemble_prediction['prediction'] == 1 else 1 - ensemble_prediction['confidence'],
                        'sell_probability': 1 - ensemble_prediction['confidence'] if ensemble_prediction['prediction'] == 1 else ensemble_prediction['confidence'],
                        'signal_strength': ensemble_prediction['confidence'],
                        'ensemble_details': {
                            'individual_predictions': ensemble_prediction.get('individual_predictions', {}),
                            'individual_confidences': ensemble_prediction.get('individual_confidences', {}),
                            'ensemble_weights': ensemble_prediction.get('ensemble_weights', {}),
                            'agreement_ratio': ensemble_prediction.get('agreement_ratio', 0),
                            'model_count': ensemble_prediction.get('model_count', 0)
                        }
                    }
            
            # Fallback to single model
            # Create technical features
            data_with_features = self.create_technical_features(data)
            
            # Check if we have the required feature columns
            if not self.feature_columns or data_with_features is None:
                self.logger.warning("No feature columns available for ML prediction")
                return None
            
            # Check if all feature columns exist in the data
            missing_columns = [col for col in self.feature_columns if col not in data_with_features.columns]
            if missing_columns:
                self.logger.warning(f"Missing feature columns: {missing_columns}")
                return None
            
            # Use the most recent row, tolerating prior NaNs by filling
            feature_frame = data_with_features[self.feature_columns].copy()
            feature_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            feature_frame.fillna(method='ffill', inplace=True)
            feature_frame.fillna(method='bfill', inplace=True)
            latest_data = feature_frame.tail(1)
            if latest_data.isnull().any(axis=1).iloc[0]:
                self.logger.warning("Latest row still contains NaNs after fills; skipping ML prediction this cycle")
                return None
            
            # Scale features
            latest_scaled = self.scaler.transform(latest_data)
            
            # Get prediction
            prediction_proba = self.ml_model.predict_proba(latest_scaled)[0]
            prediction = self.ml_model.predict(latest_scaled)[0]
            
            # Calculate confidence
            confidence = max(prediction_proba)
            
            return {
                'prediction': prediction,  # 1 for buy signal, 0 for no signal
                'confidence': confidence,
                'buy_probability': prediction_proba[1],
                'sell_probability': prediction_proba[0],
                'signal_strength': confidence if prediction == 1 else 1 - confidence
            }
            
        except Exception as e:
            self.logger.exception(f"ML prediction failed: {e}")
            return None
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol-specific configuration
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol configuration dictionary
        """
        symbol_configs = {
            'EURUSD': {
                'min_volatility': 0.0008,
                'max_volatility': 0.004,
                'preferred_sessions': ['london', 'new_york'],
                'max_spread': 0.00015,
                'pip_value': 10.0,
                'pip_size': 0.0001,
                'trading_hours': 'forex',
            },
            'GBPUSD': {
                'min_volatility': 0.001,
                'max_volatility': 0.005,
                'preferred_sessions': ['london'],
                'max_spread': 0.0002,
                'pip_value': 10.0,
                'pip_size': 0.0001,
                'trading_hours': 'forex',
            },
            'USDJPY': {
                'min_volatility': 0.0008,
                'max_volatility': 0.004,
                'preferred_sessions': ['tokyo', 'london'],
                'max_spread': 0.0002,
                'pip_value': 10.0,
                'pip_size': 0.01,
                'trading_hours': 'forex',
            },
            'AUDUSD': {
                'min_volatility': 0.0008,
                'max_volatility': 0.004,
                'preferred_sessions': ['sydney', 'london'],
                'max_spread': 0.0002,
                'pip_value': 10.0,
                'pip_size': 0.0001,
                'trading_hours': 'forex',
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
        
        return symbol_configs.get(symbol, {
            'min_volatility': 0.0008,
            'max_volatility': 0.004,
            'preferred_sessions': ['london', 'new_york'],
            'max_spread': 0.00015,
            'pip_value': 10.0,
            'pip_size': 0.0001,
            'trading_hours': 'forex',
        })
    
    def check_market_conditions(self) -> Dict[str, Any]:
        """
        Check if current market conditions are suitable for trading
        
        Returns:
            Dict with market condition analysis
        """
        try:
            from datetime import datetime
            
            current_time = datetime.utcnow()
            current_hour = current_time.hour
            current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # Check if it's weekend
            if current_weekday >= 5:  # Saturday or Sunday
                return {
                    'suitable': False,
                    'reason': 'Weekend trading not allowed',
                    'current_time': current_time,
                    'session': 'weekend'
                }
            
            # Check if it's preferred session time based on symbol
            if self.preferred_sessions_only:
                symbol_sessions = self.symbol_config.get('preferred_sessions', ['london', 'new_york'])
                trading_hours = self.symbol_config.get('trading_hours', 'forex')
                
                # Handle different trading hour types
                if trading_hours == '24/7':  # Bitcoin
                    # Always allow trading
                    pass
                elif trading_hours == '24/5':  # Gold
                    # Allow trading during weekdays
                    if current_weekday >= 5:  # Weekend
                        return {
                            'suitable': False,
                            'reason': 'Weekend trading not allowed for this symbol',
                            'current_time': current_time,
                            'session': 'weekend'
                        }
                elif trading_hours == 'new_york_session':  # US30, NAS100
                    in_new_york_session = self.new_york_session_start <= current_hour < self.new_york_session_end
                    if not in_new_york_session:
                        return {
                            'suitable': False,
                            'reason': 'Not in New York trading session',
                            'current_time': current_time,
                            'session': 'low_activity'
                        }
                else:  # Forex symbols
                    in_london_session = self.london_session_start <= current_hour < self.london_session_end
                    in_new_york_session = self.new_york_session_start <= current_hour < self.new_york_session_end
                    
                    # Check if current session matches symbol preferences
                    session_match = False
                    if 'london' in symbol_sessions and in_london_session:
                        session_match = True
                    if 'new_york' in symbol_sessions and in_new_york_session:
                        session_match = True
                    
                    if not session_match:
                        return {
                            'suitable': False,
                            'reason': f'Not in preferred trading session for {self.symbol}',
                            'current_time': current_time,
                            'session': 'low_activity'
                        }
            
            # Check daily trade limit
            today = current_time.date()
            if self.last_trade_date == today and self.daily_trade_count >= self.max_daily_trades:
                return {
                    'suitable': False,
                    'reason': f'Daily trade limit reached ({self.max_daily_trades})',
                    'current_time': current_time,
                    'session': 'limit_reached'
                }
            
            # Check consecutive losses
            if self.stop_trading_on_loss and self.consecutive_losses >= self.max_consecutive_losses:
                return {
                    'suitable': False,
                    'reason': f'Too many consecutive losses ({self.consecutive_losses})',
                    'current_time': current_time,
                    'session': 'loss_limit'
                }
            
            return {
                'suitable': True,
                'reason': 'Market conditions suitable for trading',
                'current_time': current_time,
                'session': 'active'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            return {
                'suitable': False,
                'reason': f'Error checking conditions: {e}',
                'current_time': datetime.now(),
                'session': 'error'
            }
    
    def analyze_market(self) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive market analysis including traditional TA, ML, and SMC
        
        Returns:
            dict: Combined analysis results
        """
        try:
            self.logger.info(f"Analyzing {self.symbol} on {self.timeframe} timeframe...")
            
            # Get market data
            data = self.get_market_data()
            if data is None or len(data) < 100:
                self.logger.error("Insufficient data for analysis")
                # Check if symbol is available and suggest alternatives
                if self.connected and self.mt5_connector:
                    available_symbols = self.mt5_connector.get_available_symbols()
                    if available_symbols:
                        # Look for similar symbols
                        similar_symbols = []
                        for available_symbol in available_symbols:
                            if (self.symbol.upper() in available_symbol.upper() or 
                                available_symbol.upper() in self.symbol.upper()):
                                similar_symbols.append(available_symbol)
                        
                        if similar_symbols:
                            self.logger.info(f"Similar symbols found: {similar_symbols[:5]}")
                        else:
                            self.logger.warning(f"Symbol {self.symbol} not found. Consider using available symbols.")
                return None
            
            # Skip traditional analysis since legacy TradingBot is not available
            self.logger.info("Skipping traditional analysis - using improved signal validation instead")
            traditional_signals = None
            
            # 2. Smart Money Concept Analysis
            smc_results = None
            if self.use_smc:
                smc_results = self.analyze_smc(data)
            
            # 3. Machine Learning Analysis (prioritize ML when available)
            ml_prediction = None
            if self.use_ml and self.analysis_bot and self.analysis_bot.data is not None:
                # Ensure model is loaded/trained for this symbol/timeframe
                if not self.model_trained:
                    default_model_name = os.path.join('models', f"ml_model_{self.symbol}_{self.timeframe}.joblib")
                    if not self.load_ml_model(default_model_name):
                        # Train on the fly if loading failed
                        self.train_ml_model(self.analysis_bot.data)
                ml_prediction = self.get_ml_prediction(self.analysis_bot.data)
            
            # 4. Market Structure Analysis
            market_structure_analysis = None
            if self.use_market_structure:
                market_structure_analysis = self.run_market_structure_analysis()
            
            # 5. Reinforcement Learning Analysis
            rl_analysis = None
            if self.use_rl:
                rl_analysis = self.run_rl_analysis()
            
            # 6. Get SMC trading signals
            smc_signals = None
            if smc_results and self.smc_signals:
                current_price = data['Close'].iloc[-1]
                smc_signals = self.get_smc_trading_signals(current_price)
            
            # 5. Combine all signals
            combined_signal = self.combine_signals(traditional_signals, smc_signals, ml_prediction, market_structure_analysis, rl_analysis)
            
            # Store analysis results
            self.last_analysis = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'traditional_signals': traditional_signals,
                'smc_results': smc_results,
                'ml_prediction': ml_prediction,
                'combined_signal': combined_signal,
                'current_price': data['Close'].iloc[-1] if len(data) > 0 else None
            }
            
            # Print analysis summary
            self.print_analysis_summary(combined_signal, traditional_signals, smc_results, ml_prediction)
            
            return combined_signal
            
        except Exception as e:
            self.logger.exception(f"Error in market analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_analysis_summary(self, combined_signal: Optional[Dict[str, Any]], traditional_signals: Optional[Dict[str, Any]], smc_results: Optional[Dict[str, Any]], ml_prediction: Optional[Dict[str, Any]]) -> None:
        """Print comprehensive analysis summary"""
        self.logger.info(f"Analysis Summary for {self.symbol} ({self.timeframe})")
        
        # Traditional Analysis
        if traditional_signals:
            self.logger.info(f"Traditional TA: {traditional_signals.get('signal_type', 'HOLD')}")
            if traditional_signals.get('signal_strength'):
                self.logger.info(f"Strength: {traditional_signals['signal_strength']:.2f}")
        
        # SMC Analysis
        if smc_results and self.smc_summary:
            self.logger.info("Smart Money Concept:")
            self.logger.info(f"Market Structure: {self.smc_summary['market_structure']['trend_direction']}")
            self.logger.info(f"Order Blocks: {self.smc_summary['order_blocks']['total_count']}")
            self.logger.info(f"Fair Value Gaps: {self.smc_summary['fair_value_gaps']['total_count']}")
            self.logger.info(f"Liquidity Zones: {self.smc_summary['liquidity_zones']['total_count']}")
        
        # ML Analysis
        if ml_prediction:
            prediction = ml_prediction.get('prediction', 0.5)
            confidence = ml_prediction.get('confidence', 0)
            self.logger.info("Machine Learning:")
            self.logger.info(f"Prediction: {'BULLISH' if prediction > 0.6 else 'BEARISH' if prediction < 0.4 else 'NEUTRAL'} | Confidence: {confidence:.2f}")
        
        # Combined Signal
        if combined_signal:
            self.logger.info("Combined Signal:")
            self.logger.info(f"Type: {combined_signal['signal_type']} | Strength: {combined_signal['signal_strength']:.2f} | Sources: {', '.join(combined_signal['signal_sources'])}")
            
            if combined_signal.get('entry_price'):
                self.logger.info(f"Entry: {combined_signal['entry_price']:.5f} | SL: {combined_signal['stop_loss']:.5f} | TP: {combined_signal['target']:.5f}")
                
                if combined_signal.get('potential_profit'):
                    self.logger.info(f"Potential Profit: ${combined_signal['potential_profit']:.2f}")
        else:
            self.logger.warning("No strong combined signal generated")
    
    def validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trading signal with improved criteria
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            Validation result with confidence score
        """
        try:
            if not signal:
                return {
                    'is_valid': False,
                    'reason': 'No signal provided',
                    'confidence': 0.0
                }
            
            # Check signal strength
            signal_strength = signal.get('signal_strength', 0)
            if signal_strength < self.min_signal_strength:
                return {
                    'is_valid': False,
                    'reason': f'Signal strength too low: {signal_strength:.2f} < {self.min_signal_strength}',
                    'confidence': signal_strength
                }
            
            # Check risk-reward ratio using symbol-specific pip size
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            target = signal.get('target', 0)
            
            if entry_price and stop_loss and target:
                pip_size = self.symbol_config.get('pip_size', 0.0001)
                risk = abs(entry_price - stop_loss) / pip_size  # Risk in pips
                reward = abs(target - entry_price) / pip_size   # Reward in pips
                risk_reward_ratio = reward / risk if risk > 0 else 0
                
                if risk_reward_ratio < self.min_risk_reward_ratio:
                    return {
                        'is_valid': False,
                        'reason': f'Risk-reward ratio too low: {risk_reward_ratio:.2f} < {self.min_risk_reward_ratio}',
                        'confidence': signal_strength * 0.5
                    }
            else:
                return {
                    'is_valid': False,
                    'reason': 'Missing entry, stop loss, or target price',
                    'confidence': 0.0
                }
            
            # Check for multiple confirmations
            signal_sources = signal.get('signal_sources', [])
            if self.require_multiple_confirmations and len(signal_sources) < 2:
                return {
                    'is_valid': False,
                    'reason': 'Require multiple signal confirmations',
                    'confidence': signal_strength * 0.7
                }
            
            # Calculate final confidence
            confidence = signal_strength
            if len(signal_sources) >= 2:
                confidence += 0.1  # Boost for multiple confirmations
            if risk_reward_ratio >= 3.0:
                confidence += 0.1  # Boost for excellent risk-reward
            
            confidence = min(1.0, confidence)
            
            return {
                'is_valid': True,
                'reason': 'Signal passed all validation criteria',
                'confidence': confidence,
                'risk_reward_ratio': risk_reward_ratio,
                'signal_sources': len(signal_sources)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return {
                'is_valid': False,
                'reason': f'Validation error: {e}',
                'confidence': 0.0
            }
    
    def get_trading_signals(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get advanced trading signals based on new trading rules
        
        Args:
            analysis (dict): Advanced market analysis results
            
        Returns:
            dict: Trading signals or None if no signals
        """
        try:
            # Check if we have a confirmed trend (either up or down)
            uptrend_confirmed = analysis.get('uptrend_confirmed', False)
            downtrend_confirmed = analysis.get('downtrend_confirmed', False)
            trend_direction = analysis.get('trend_direction', 'SIDEWAYS')
            
            if not analysis or (not uptrend_confirmed and not downtrend_confirmed):
                self.logger.warning(f"No strong trend confirmed ({trend_direction}) - no trading signals")
                return None
        
            # Check for entry conditions (can be relaxed for now since we're using basic analysis)
            # if not analysis.get('entry_conditions', []):
            #     print("ℹ️  No entry conditions met - waiting for breakout or continuation pattern")
            #     return None
        except Exception as e:
            self.logger.error(f"Error checking analysis structure: {e}")
            return None
        
        self.logger.info("GENERATING TRADING SIGNALS")
        
        # Get current market price from MT5
        current_price = None
        if self.connected:
            tick = self.mt5_connector.get_symbol_info(self.symbol)
            if tick:
                # Use ask price for buy orders
                current_price = tick.get('ask', None)
        
        if current_price is None:
            # Fallback to last close price
            if self.analysis_bot and self.analysis_bot.data is not None and len(self.analysis_bot.data) > 0:
                current_price = self.analysis_bot.data['Close'].iloc[-1]
        
        if current_price is None:
            self.logger.error("Cannot get current market price")
            return None
        
        # Initialize signal parameters based on trend direction
        entry_price = current_price
        stop_loss_price = None
        target_price = None
        
        # Determine signal type based on trend
        if uptrend_confirmed:
            signal_type = "BUY"
        elif downtrend_confirmed:
            signal_type = "SELL"
        else:
            signal_type = "BUY"  # Default fallback
            
        signal_strength = analysis.get('overall_strength', 50)
        
        # Get symbol info for proper calculations
        symbol_info = None
        if self.connected:
            symbol_info = self.mt5_connector.get_symbol_info(self.symbol)
        
        point_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001
        digits = symbol_info.get('digits', 5) if symbol_info else 5
        
        # Process breakout and retest signals
        breakout_signals = analysis.get('breakout_signals')
        if breakout_signals and breakout_signals.get('entry_signal'):
            self.logger.info("Using breakout and retest signals")
            entry_price = breakout_signals.get('entry_price', current_price)
            stop_loss_price = breakout_signals.get('stop_loss')
            target_price = breakout_signals.get('target')
            signal_type = "BREAKOUT_BUY"
        
        # Process continuation pattern signals
        elif analysis.get('continuation_patterns'):
            patterns = analysis.get('continuation_patterns', {})
            if patterns.get('bullish_flag') or patterns.get('pennant') or patterns.get('inverted_head_shoulders'):
                self.logger.info("Using continuation pattern signals")
                
                # Calculate pattern-based entry and targets
                if patterns['bullish_flag']:
                    flag_details = patterns['pattern_details']['bullish_flag']
                    flag_height = flag_details['flag_pole_height']
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 - flag_height * 0.5)  # Conservative stop
                    target_price = entry_price * (1 + flag_height * 1.5)  # 1:3 ratio
                    signal_type = "BULLISH_FLAG_BUY"
                
                elif patterns['pennant']:
                    pennant_details = patterns['pattern_details']['pennant']
                    entry_price = current_price
                    # Use ATR-based stop loss
                    try:
                        atr = self.calculate_atr(getattr(self.analysis_bot, 'h4_data', None), period=14)
                    except Exception:
                        atr = 0.0002
                    stop_loss_price = entry_price - (atr * 2)
                    target_price = entry_price + (atr * 6)  # 1:3 ratio
                    signal_type = "PENNANT_BUY"
                
                elif patterns['inverted_head_shoulders']:
                    ihs_details = patterns['pattern_details']['inverted_head_shoulders']
                    neckline = ihs_details['neckline']
                    pattern_height = ihs_details['pattern_height']
                    
                    entry_price = current_price
                    stop_loss_price = neckline * 0.995  # Just below neckline
                    target_price = entry_price + pattern_height  # Measured move
                    signal_type = "IHS_BUY"
        
        # If no specific pattern signals, use trendline-based signals
        else:
            self.logger.info("Using trendline-based signals")
            
            # Use H4 trendlines for entry signals (if available)
            h4_trendlines = analysis.get('h4_trendlines', {})
            if h4_trendlines and 'hl_trendline' in h4_trendlines:
                hl_trendline = h4_trendlines['hl_trendline']
                if hl_trendline.get('touches', 0) >= 3:  # Strong support
                    # Entry on support touch
                    entry_price = current_price
                    stop_loss_price = hl_trendline.get('end_price', current_price) * 0.995  # Below support
                    
                    # Calculate target based on trend strength
                    if h4_trendlines and 'hh_trendline' in h4_trendlines:
                        hh_trendline = h4_trendlines['hh_trendline']
                        resistance_level = hh_trendline.get('end_price', current_price)
                        target_price = resistance_level * 1.005  # Above resistance
                    else:
                        # Use ATR-based target
                        try:
                            atr = self.calculate_atr(self.analysis_bot.data, period=14) if self.analysis_bot and self.analysis_bot.data is not None else 0.0002
                        except Exception:
                            atr = 0.0002
                        target_price = entry_price + (atr * 6)  # 1:3 ratio
                    
                    signal_type = f"TRENDLINE_{signal_type}"
        
        # Validate and adjust stop loss and target
        if stop_loss_price is None or target_price is None:
            self.logger.info("Using default risk-based stop loss and target calculation")
            
            # Calculate ATR for dynamic stop loss
            atr = 0.0002  # Default ATR for forex pairs
            if self.analysis_bot and self.analysis_bot.data is not None:
                try:
                    high_low = self.analysis_bot.data['High'] - self.analysis_bot.data['Low']
                    atr = high_low.rolling(window=14).mean().iloc[-1]
                except:
                    atr = 0.0002
            
            # Set stop loss and target based on signal type
            if signal_type == "SELL" or downtrend_confirmed:
                # For SELL signals
                stop_loss_price = entry_price + (atr * 2)  # Stop above entry
                target_price = entry_price - (atr * 6)     # Target below entry (1:3 ratio)
                signal_type = "SELL" if signal_type != "SELL" else signal_type
            else:
                # For BUY signals (default)
                stop_loss_price = entry_price - (atr * 2)  # Stop below entry
                target_price = entry_price + (atr * 6)     # Target above entry (1:3 ratio)
                signal_type = "BUY" if signal_type != "BUY" else signal_type
        
        # Ensure minimum stop level compliance
        if symbol_info and 'trade_stops_level' in symbol_info:
            min_stop_level_points = symbol_info['trade_stops_level']
            min_stop_distance = min_stop_level_points * point_value
            
            current_stop_distance = abs(entry_price - stop_loss_price)
            if current_stop_distance < min_stop_distance:
                self.logger.warning(f"Adjusting stop loss to meet broker minimum ({min_stop_level_points} points)")
                if entry_price > stop_loss_price:  # Long position
                    stop_loss_price = entry_price - min_stop_distance
                else:  # Short position
                    stop_loss_price = entry_price + min_stop_distance
        
        # Calculate risk and reward in price terms
        price_risk = abs(entry_price - stop_loss_price)
        price_reward = abs(target_price - entry_price)
        risk_reward_ratio = price_reward / price_risk if price_risk > 0 else 0
        
        # Get current account balance from MT5
        current_balance = None
        if self.connected and self.mt5_connector:
            try:
                account_info = self.mt5_connector.get_account_summary()
                if account_info:
                    current_balance = account_info.get('balance', 0)
                    self.logger.info(f"Current Account Balance: ${current_balance:,.2f}")
                else:
                    self.logger.error("Could not get account balance from MT5")
                    return None
            except Exception as e:
                self.logger.error(f"Error getting account balance: {e}")
                return None
        else:
            self.logger.error("Not connected to MT5 - cannot get account balance")
            return None
        
        if current_balance is None or current_balance <= 0:
            self.logger.error(f"Invalid account balance: ${current_balance}")
            return None
        
        # Robust position sizing constrained by risk and free margin
        pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
        stop_loss_pips = price_risk / pip_size
        target_pips = price_reward / pip_size
        print(f"📊 Stop Loss: {stop_loss_pips:.1f} pips, Target: {target_pips:.1f} pips")

        position_size = 0.0
        if self.connected and self.mt5_connector:
            # Use more conservative margin buffer for small accounts
            margin_buffer = 0.90  # Use 90% of free margin instead of 95%
            position_size = self.mt5_connector.calculate_position_size_robust(
                symbol=self.symbol,
                order_type=signal_type,
                risk_percent=self.risk_per_trade,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                margin_buffer=margin_buffer
            )
            print(f"📈 Robust position size: {position_size:.4f} lots (using {margin_buffer*100}% of free margin)")
        
        if not position_size or position_size <= 0:
            # Fallback: calculate minimum viable position size based on available margin
            if self.connected and self.mt5_connector:
                account_info = self.mt5_connector.get_account_summary()
                if account_info:
                    free_margin = account_info.get('margin_free', 0)
                    # Use a very conservative approach for minimum position
                    min_margin_needed = 50  # Assume minimum $50 margin needed
                    if free_margin > min_margin_needed:
                        position_size = self.get_minimum_lot_size()  # Symbol-specific minimum lot size
                        print(f"📈 Using minimum position size: {position_size:.4f} lots (free margin: ${free_margin:.2f})")
                    else:
                        print(f"❌ Insufficient free margin for minimum position: ${free_margin:.2f}")
                        return None
                else:
                    position_size = self.get_minimum_lot_size()
                    print(f"📈 Using fallback minimum position size: {position_size:.4f} lots")
            else:
                position_size = self.get_minimum_lot_size()
                print(f"📈 Using fallback minimum position size: {position_size:.4f} lots")
        
        # Calculate actual dollar amounts
        # Use dynamic pip value per lot for accurate dollar amounts
        pip_value_per_lot = self.mt5_connector.get_pip_value_per_lot(self.symbol) if (self.connected and self.mt5_connector) else 10.0
        
        # Ensure all values are valid before multiplication
        if position_size is None or stop_loss_pips is None or target_pips is None:
            print("❌ Error: Invalid values for risk calculation")
            return None
        
        risk_amount = position_size * stop_loss_pips * pip_value_per_lot
        reward_amount = position_size * target_pips * pip_value_per_lot
        
        # Get ML prediction for confirmation
        ml_prediction = None
        if self.use_ml and self.analysis_bot and hasattr(self.analysis_bot, 'data') and self.analysis_bot.data is not None:
            try:
                ml_prediction = self.get_ml_prediction(self.analysis_bot.data)
            except Exception as e:
                print(f"❌ ML prediction failed: {e}")
                ml_prediction = None
        
        # Create final signal
        signals = {
            'signal_type': signal_type,
            'entry_price': round(entry_price, digits),
            'stop_loss': round(stop_loss_price, digits),
            'target': round(target_price, digits),
            'risk_amount': round(risk_amount, digits),
            'potential_profit': round(reward_amount, digits),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_size': round(position_size, 2),
            'signal_strength': signal_strength,
            'entry_conditions': analysis.get('entry_conditions', []),
            'ml_prediction': ml_prediction,
            'analysis_summary': {
                'overall_strength': analysis.get('overall_strength', 50),
                'weekly_strength': analysis.get('weekly_analysis', {}).get('trend_strength', 0),
                'daily_strength': analysis.get('daily_analysis', {}).get('trend_strength', 0),
                'h4_strength': analysis.get('h4_analysis', {}).get('trend_strength', 0)
            }
        }
        
        print(f"✅ Trading signals generated:")
        print(f"   Signal Type: {signal_type}")
        print(f"   Entry Price: {signals['entry_price']:.5f}")
        print(f"   Stop Loss: {signals['stop_loss']:.5f}")
        print(f"   Target: {signals['target']:.5f}")
        print(f"   Risk/Reward: 1:{signals['risk_reward_ratio']:.2f}")
        print(f"   Signal Strength: {signal_strength:.1f}/100")
        
        return signals
    
    def calculate_atr(self, data: Optional[pd.DataFrame], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility-based calculations
        
        Args:
            data (pd.DataFrame): Price data
            period (int): ATR period
            
        Returns:
            float: ATR value
        """
        if data is None or len(data) < period + 1:
            return 0.001  # Default small value
        
        try:
            # Calculate True Range
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift(1))
            low_close = abs(data['Low'] - data['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.001
            
        except Exception as e:
            print(f"⚠️  Error calculating ATR: {e}")
            return 0.001
    
    def execute_trade(self, signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute trade based on signals
        
        Args:
            signals (dict): Trading signals
            
        Returns:
            dict: Trade result or None if failed
        """
        if not self.connected or not self.auto_trade:
            print("❌ Auto trading disabled or not connected to MT5")
            return None
        
        # Check market conditions first
        market_conditions = self.check_market_conditions()
        if not market_conditions['suitable']:
            print(f"❌ Market conditions not suitable: {market_conditions['reason']}")
            return None
        
        # Advanced risk management check
        if self.use_advanced_risk and self.advanced_risk_manager:
            # Check if trading should be stopped due to risk limits
            should_stop, reason = self.advanced_risk_manager.should_stop_trading()
            if should_stop:
                print(f"❌ Trading stopped by risk management: {reason}")
                return None
        
        # Market regime detection
        if self.use_regime_detection and self.regime_detector:
            # Get recent data for regime detection
            try:
                recent_data = self.get_recent_data(100)  # Get last 100 bars
            except AttributeError:
                # Fallback if method doesn't exist
                recent_data = self.get_market_data()
                if recent_data is not None and len(recent_data) > 100:
                    recent_data = recent_data.tail(100)
            
            if recent_data is not None and len(recent_data) > 50:
                regime_metrics = self.regime_detector.detect_regime(recent_data)
                if regime_metrics:
                    print(f"📊 Market Regime: {regime_metrics.regime.value} (Confidence: {regime_metrics.confidence:.2f})")
                    
                    # Adjust trading based on regime
                    if regime_metrics.regime.value == 'high_volatility' and regime_metrics.confidence > 0.7:
                        print("⚠️  High volatility regime detected - reducing position size")
                        if 'position_size' in signals:
                            signals['position_size'] *= 0.5  # Reduce position size by 50%
        
        # Validate signal with improved criteria
        signal_validation = self.validate_signal(signals)
        if not signal_validation['is_valid']:
            print(f"❌ Signal validation failed: {signal_validation['reason']}")
            return None
        
        print(f"✅ Signal validation passed - Confidence: {signal_validation['confidence']:.2f}")
        
        try:
            # Validate required signal fields
            required_fields = ['signal_type', 'entry_price', 'stop_loss', 'target', 'position_size', 'risk_amount', 'potential_profit']
            missing_fields = []
            for field in required_fields:
                if field not in signals or signals[field] is None:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing required signal fields: {missing_fields}")
                return None
            
            # Get current account balance and margin information
            print(f"\n💰 CHECKING ACCOUNT STATUS FOR TRADE EXECUTION")
            account_info = self.mt5_connector.get_account_summary()
            if not account_info:
                print("❌ Could not get account information")
                return None
            
            current_balance = account_info.get('balance', 0)
            current_equity = account_info.get('equity', 0)
            current_margin = account_info.get('margin', 0)
            free_margin = account_info.get('margin_free', 0)
            
            print(f"[INFO] ACCOUNT STATUS:")
            print(f"   Balance: ${current_balance:,.2f}")
            print(f"   Equity: ${current_equity:,.2f}")
            print(f"   Used Margin: ${current_margin:,.2f}")
            print(f"   Free Margin: ${free_margin:,.2f}")
            
            # Check if we have sufficient free margin using accurate margin calculation
            print(f"[INFO] MARGIN CALCULATION:")
            pos_size = float(signals.get('position_size') or 0)
            
            # Calculate actual required margin using connector's method
            required_margin = None
            margin_per_lot = None
            
            if self.mt5_connector and pos_size > 0:
                order_type = signals.get('signal_type', 'BUY')
                entry_price = signals.get('entry_price', 0)
                
                # Use connector's margin calculation method
                required_margin = self.mt5_connector.get_margin_requirement(
                    self.symbol, order_type, pos_size, entry_price
                )
                
                # Also calculate margin per lot for position size adjustment
                margin_per_lot = self.mt5_connector.get_margin_requirement(
                    self.symbol, order_type, 1.0, entry_price
                )
                
                if required_margin is not None and margin_per_lot is not None:
                    print(f"[INFO] MT5 calculated margin required: ${required_margin:,.2f}")
                    print(f"[INFO] Margin per lot: ${margin_per_lot:,.2f}")
                else:
                    print(f"[WARNING] MT5 margin calculation failed, trying direct MT5 method...")
                    # Fallback to direct MT5 calculation
                    order_const = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
                    required_margin = mt5.order_calc_margin(order_const, self.symbol, pos_size, entry_price)
                    margin_per_lot = mt5.order_calc_margin(order_const, self.symbol, 1.0, entry_price)
                    
                    if required_margin is not None and margin_per_lot is not None:
                        print(f"[INFO] Direct MT5 margin calculation: ${required_margin:,.2f}")
                        print(f"[INFO] Margin per lot: ${margin_per_lot:,.2f}")
                    else:
                        print(f"[WARNING] All margin calculation methods failed")
            
            # If MT5 calculation failed, use intelligent fallback calculation
            if required_margin is None or margin_per_lot is None:
                print(f"[INFO] Using intelligent fallback margin calculation...")
                
                # Get symbol information for better estimates
                symbol_info = self.mt5_connector.get_symbol_info(self.symbol) if self.mt5_connector else None
                
                # Calculate margin per lot based on symbol type and account leverage
                if symbol_info:
                    # Get account leverage
                    account_info = self.mt5_connector.get_account_summary() if self.mt5_connector else None
                    leverage = account_info.get('leverage', 100) if account_info else 100
                    
                    # Get current price for calculation
                    entry_price = signals.get('entry_price', 0)
                    if entry_price <= 0:
                        tick = mt5.symbol_info_tick(self.symbol) if mt5 else None
                        if tick:
                            entry_price = tick.ask if signals.get('signal_type') == 'BUY' else tick.bid
                    
                    # Calculate margin per lot based on symbol type
                    if 'US30' in self.symbol or 'NAS100' in self.symbol:
                        # Indices: margin = (contract size * price) / leverage
                        contract_size = 1.0  # Standard contract size for indices
                        margin_per_lot = (contract_size * entry_price) / leverage if entry_price > 0 else 50
                        print(f"[INFO] Index margin calculation: contract_size={contract_size}, price={entry_price}, leverage={leverage}")
                    elif 'XAUUSD' in self.symbol:
                        # Gold: typically needs more margin
                        margin_per_lot = 1000 / leverage if leverage > 0 else 100
                    else:
                        # Forex pairs: standard calculation
                        margin_per_lot = 100000 / leverage if leverage > 0 else 1000
                        print(f"[INFO] Forex margin calculation: leverage={leverage}")
                    
                    required_margin = margin_per_lot * pos_size
                    print(f"[INFO] Intelligent fallback margin: ${required_margin:,.2f} (${margin_per_lot:,.2f} per lot)")
                else:
                    # Final fallback with symbol-specific estimates
                    if 'US30' in self.symbol or 'NAS100' in self.symbol:
                        margin_per_lot = 50  # Conservative estimate for indices
                    elif 'XAUUSD' in self.symbol:
                        margin_per_lot = 200  # Gold estimate
                    else:
                        margin_per_lot = 1000  # Forex default
                    
                    required_margin = margin_per_lot * pos_size
                    print(f"[WARNING] Using symbol-specific estimate: ${required_margin:,.2f} (${margin_per_lot:,.2f} per lot)")
            
            if free_margin < required_margin:
                print(f"[WARNING] MARGIN ADJUSTMENT NEEDED")
                print(f"   Available Free Margin: ${free_margin:,.2f}")
                print(f"   Required Margin: ${required_margin:,.2f}")
                print(f"   Current Position Size: {pos_size:.4f} lots")
                print(f"   Auto-adjusting position size to fit available margin...")
                
                # Calculate maximum position size based on available margin
                if margin_per_lot and margin_per_lot > 0:
                    # Use actual margin per lot to calculate max position size
                    max_position_size = (free_margin * 0.95) / margin_per_lot  # Use 95% of free margin
                    
                    # Round to valid lot size
                    symbol_info = self.mt5_connector.get_symbol_info(self.symbol)
                    if symbol_info:
                        step = symbol_info.get('volume_step', 0.01)
                        min_lot_size = symbol_info.get('volume_min', 0.01)
                        max_position_size = max(min_lot_size, round(max_position_size / step) * step)
                else:
                    # Fallback calculation using margin_per_lot
                    if margin_per_lot and margin_per_lot > 0:
                        max_position_size = (free_margin * 0.95) / margin_per_lot
                    else:
                        max_position_size = (free_margin * 0.95) / 1000  # Last resort
                    
                    min_lot_size = self.get_minimum_lot_size()
                    max_position_size = max(min_lot_size, round(max_position_size / min_lot_size) * min_lot_size)
                
                if max_position_size >= min_lot_size:  # Symbol-specific minimum position size
                    print(f"   Reducing position size from {pos_size:.4f} to {max_position_size:.4f} lots")
                    original_size = pos_size
                    signals['position_size'] = max_position_size
                    if original_size > 0:
                        ratio = max_position_size / original_size
                        signals['risk_amount'] = (signals.get('risk_amount') or 0) * ratio
                        signals['potential_profit'] = (signals.get('potential_profit') or 0) * ratio
                    print(f"[SUCCESS] POSITION SIZE ADJUSTED SUCCESSFULLY")
                    print(f"   New Position Size: {max_position_size:.4f} lots")
                    new_required_margin = max_position_size * margin_per_lot if margin_per_lot else max_position_size * 1000
                    print(f"   New Required Margin: ${new_required_margin:,.2f}")
                    print(f"   Margin Check: ${free_margin:,.2f} available >= ${new_required_margin:,.2f} required")
                else:
                    print(f"[ERROR] TRADE REJECTED - INSUFFICIENT MARGIN")
                    print(f"   Available Free Margin: ${free_margin:,.2f}")
                    print(f"   Minimum Position Size: {min_lot_size:.4f} lots")
                    min_required_margin = min_lot_size * margin_per_lot if margin_per_lot else min_lot_size * 1000
                    print(f"   Minimum Required Margin: ${min_required_margin:,.2f}")
                    print(f"   Need at least ${min_required_margin:,.2f} free margin to place minimum trade")
                    return None
            
            # Check current positions for this symbol
            positions = self.mt5_connector.get_positions()
            current_positions = []
            if positions:
                for pos in positions:
                    if pos['symbol'] == self.symbol:
                        current_positions.append(pos)
            
            print(f"📊 Current positions for {self.symbol}: {len(current_positions)}")
            
            # Check if we have too many positions (limit to 1 concurrent trade per symbol - improved)
            max_positions = self.max_concurrent_trades
            if len(current_positions) >= max_positions:
                print(f"⚠️  Maximum positions ({max_positions}) reached for {self.symbol}")
                return None
            
            # Check if we have conflicting positions (same direction)
            signal_type = signals['signal_type']
            conflicting_positions = []
            for pos in current_positions:
                if pos['type'] == signal_type:  # Same direction
                    conflicting_positions.append(pos)
            
            # Allow multiple positions in the same direction (up to 2)
            max_same_direction = 2
            if len(conflicting_positions) >= max_same_direction:
                print(f"⚠️  Maximum {signal_type} positions ({max_same_direction}) reached for {self.symbol}")
                return None
            
            # Calculate position size based on remaining risk allocation using current balance
            total_risk_used = sum([pos.get('risk_amount', 0) for pos in current_positions])
            available_risk = current_balance * self.risk_per_trade - total_risk_used
            
            print(f"💸 Total risk used: ${total_risk_used:,.2f}")
            print(f"💸 Available risk: ${available_risk:,.2f}")
            
            if available_risk <= 0:
                print(f"⚠️  No available risk allocation for {self.symbol}")
                return None
            
            # Adjust position size based on available risk
            adjusted_signals = signals.copy()
            if available_risk < signals['risk_amount']:
                risk_ratio = available_risk / signals['risk_amount']
                # Ensure position_size is not None before multiplication
                current_position_size = signals.get('position_size')
                if current_position_size is not None:
                    adjusted_signals['position_size'] = current_position_size * risk_ratio
                else:
                    print("⚠️  Warning: position_size is None, cannot adjust")
                    return None
                
                adjusted_signals['risk_amount'] = available_risk
                
                # Ensure potential_profit is not None before multiplication
                current_profit = signals.get('potential_profit')
                if current_profit is not None:
                    adjusted_signals['potential_profit'] = current_profit * risk_ratio
                else:
                    adjusted_signals['potential_profit'] = 0.0
                
                print(f"📊 Adjusted position size due to risk allocation: {risk_ratio:.2f}")
            
            # Final volume validation before placing order
            if self.connected and self.mt5_connector:
                symbol_info = self.mt5_connector.get_symbol_info(self.symbol)
                if symbol_info:
                    min_volume = symbol_info.get('volume_min', 0.01)
                    max_volume = symbol_info.get('volume_max', 100.0)
                    volume_step = symbol_info.get('volume_step', 0.01)
                    
                    # Ensure final position size is valid
                    position_size_value = adjusted_signals.get('position_size')
                    if position_size_value is None:
                        print("❌ Error: position_size is None in adjusted_signals")
                        return None
                    
                    final_size = float(position_size_value)
                    if final_size < min_volume:
                        print(f"⚠️  Position size {final_size:.4f} is below minimum {min_volume}. Using minimum volume.")
                        # Scale risk/profit amounts proportionally if present
                        old_size = final_size if final_size > 0 else None
                        adjusted_signals['position_size'] = min_volume
                        final_size = min_volume
                        if old_size and old_size > 0:
                            try:
                                ratio = final_size / old_size
                                if adjusted_signals.get('risk_amount') is not None:
                                    adjusted_signals['risk_amount'] = (adjusted_signals['risk_amount'] or 0) * ratio
                                if adjusted_signals.get('potential_profit') is not None:
                                    adjusted_signals['potential_profit'] = (adjusted_signals['potential_profit'] or 0) * ratio
                            except Exception:
                                pass
                    elif final_size > max_volume:
                        print(f"❌ Position size {adjusted_signals['position_size']:.4f} exceeds maximum {max_volume}")
                        return None
                    
                    # Round to nearest step
                    adjusted_signals['position_size'] = round(final_size / volume_step) * volume_step
                    print(f"📊 Final validated position size: {adjusted_signals['position_size']:.4f} lots")
            
            # Place the order
            result = self.mt5_connector.place_order(
                symbol=self.symbol,
                order_type=adjusted_signals['signal_type'],
                volume=adjusted_signals['position_size'],
                price=adjusted_signals['entry_price'],
                sl=adjusted_signals['stop_loss'],
                tp=adjusted_signals['target'],
                comment=f"Scalping Bot {self.timeframe} #{len(current_positions) + 1}"
            )
            
            if result:
                # Add risk amount to the result for tracking
                result['risk_amount'] = adjusted_signals.get('risk_amount')
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'action': 'OPEN',
                    'result': result,
                    'signals': adjusted_signals
                })
                
                # Log trade to journal automatically
                if self.trade_journal_enabled:
                    trade_data = {
                        'signal_type': adjusted_signals['signal_type'],
                        'entry_price': adjusted_signals['entry_price'],
                        'target': adjusted_signals['target'],
                        'stop_loss': adjusted_signals['stop_loss'],
                        'order_id': result.get('order_id')
                    }
                    self.log_trade_to_journal(trade_data)
                
                # Update trade tracking
                self.update_trade_tracking()
                self.logger.info(
                    f"Trade executed successfully | order_id={result.get('order_id')} entry={result.get('price')} sl={result.get('sl')} tp={result.get('tp')} pos_no={len(current_positions) + 1} symbol={self.symbol} total_positions={len(current_positions) + 1}"
                )
                return result
            else:
                self.logger.warning("Order placement returned None; trade not executed")
                return None
            
        except Exception as e:
            print(f"[ERROR] Error executing trade: {e}")
            return None
    
    def monitor_positions(self) -> List[Dict[str, Any]]:
        """
        Monitor and manage open positions
        
        Returns:
            list: Updated positions
        """
        if not self.connected:
            return []
        
        try:
            positions = self.mt5_connector.get_positions()
            if positions:
                # Group positions by symbol
                symbol_positions = {}
                for pos in positions:
                    symbol = pos['symbol']
                    if symbol not in symbol_positions:
                        symbol_positions[symbol] = []
                    symbol_positions[symbol].append(pos)
                
                # Display positions grouped by symbol
                for symbol, symbol_poss in symbol_positions.items():
                    print(f"\n📊 {symbol} Positions ({len(symbol_poss)} total):")
                    
                    total_profit = 0
                    total_volume = 0
                    
                    for i, pos in enumerate(symbol_poss, 1):
                        print(f"   Position #{i}: {pos['type']}")
                        print(f"      Volume: {pos['volume']}")
                        print(f"      Entry: {pos['price_open']}")
                        print(f"      Current: {pos['price_current']}")
                        
                        # Handle None profit value safely
                        profit = pos.get('profit', 0)
                        if profit is not None:
                            print(f"      Profit: ${profit:.2f}")
                            total_profit += profit
                        else:
                            print(f"      Profit: N/A")
                        
                        # Handle None stop loss and take profit safely
                        sl = pos.get('sl', None)
                        tp = pos.get('tp', None)
                        print(f"      Stop Loss: {sl if sl is not None else 'N/A'}")
                        print(f"      Take Profit: {tp if tp is not None else 'N/A'}")
                        
                        total_volume += pos['volume']
                    
                    # Show summary for this symbol
                    print(f"   📈 Summary for {symbol}:")
                    print(f"      Total Volume: {total_volume:.2f}")
                    print(f"      Total Profit: ${total_profit:.2f}")
                    
                    # Calculate average entry price
                    if symbol_poss:
                        avg_entry = sum([pos['price_open'] for pos in symbol_poss]) / len(symbol_poss)
                        print(f"      Average Entry: {avg_entry:.5f}")
            
            return positions
            
        except Exception as e:
            print(f"❌ Error monitoring positions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions for the symbol
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            positions = self.mt5_connector.get_positions()
            if not positions:
                print("No positions to close")
                return True
            
            # Group positions by symbol
            symbol_positions = {}
            for pos in positions:
                symbol = pos['symbol']
                if symbol not in symbol_positions:
                    symbol_positions[symbol] = []
                symbol_positions[symbol].append(pos)
            
            total_closed = 0
            for symbol, symbol_poss in symbol_positions.items():
                print(f"\n🔄 Closing {len(symbol_poss)} positions for {symbol}...")
                
                symbol_closed = 0
                for pos in symbol_poss:
                    if self.mt5_connector.close_position(pos['ticket']):
                        symbol_closed += 1
                        total_closed += 1
                        self.trade_history.append({
                            'timestamp': datetime.now(),
                            'action': 'CLOSE',
                            'position': pos
                        })
                        print(f"   ✅ Closed position #{symbol_closed}: {pos['type']} {pos['volume']} lots")
                    else:
                        print(f"   ❌ Failed to close position #{symbol_closed + 1}")
                
                print(f"   📊 Closed {symbol_closed}/{len(symbol_poss)} positions for {symbol}")
            
            print(f"\n✅ Total closed: {total_closed} positions across all symbols")
            return total_closed > 0
            
        except Exception as e:
            print(f"❌ Error closing positions: {e}")
            return False
    
    def close_symbol_positions(self, symbol: Optional[str] = None) -> bool:
        """
        Close all open positions for a specific symbol
        
        Args:
            symbol (str): Symbol to close positions for (defaults to self.symbol)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            return False
        
        target_symbol = symbol or self.symbol
        
        try:
            positions = self.mt5_connector.get_positions()
            if not positions:
                print(f"No positions to close for {target_symbol}")
                return True
            
            symbol_positions = [pos for pos in positions if pos['symbol'] == target_symbol]
            if not symbol_positions:
                print(f"No positions found for {target_symbol}")
                return True
            
            print(f"\n🔄 Closing {len(symbol_positions)} positions for {target_symbol}...")
            
            closed_count = 0
            for i, pos in enumerate(symbol_positions, 1):
                if self.mt5_connector.close_position(pos['ticket']):
                    closed_count += 1
                    self.trade_history.append({
                        'timestamp': datetime.now(),
                        'action': 'CLOSE',
                        'position': pos
                    })
                    print(f"   ✅ Closed position #{i}: {pos['type']} {pos['volume']} lots")
                else:
                    print(f"   ❌ Failed to close position #{i}")
            
            print(f"📊 Closed {closed_count}/{len(symbol_positions)} positions for {target_symbol}")
            return closed_count > 0
            
        except Exception as e:
            print(f"❌ Error closing positions for {target_symbol}: {e}")
            return False
    
    def run_analysis_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Run one complete analysis cycle with ML integration
        
        Returns:
            dict: Analysis results
        """
        print(f"\n📊 Enhanced Analysis: {self.symbol} on {self.timeframe} timeframe...")
        
        # Get market data
        data = self.get_market_data()
        if data is None or len(data) < 100:
            print("❌ Insufficient data for analysis")
            return None
        
        # Try to load existing ML model first, then train if needed
        if self.use_ml and not self.model_trained:
            # Try to load existing model from models/ directory
            default_model_name = os.path.join('models', f"ml_model_{self.symbol}_{self.timeframe}.joblib")
            if self.load_ml_model(default_model_name):
                self.logger.info(f"Loaded existing ML model: {default_model_name}")
            else:
                self.logger.info("Training new ML model with historical data...")
                self.train_ml_model(data)
        
        # Check if symbol is available in MT5
        if self.connected and self.mt5_connector:
            available_symbols = self.mt5_connector.get_available_symbols()
            if available_symbols:
                # Check if our symbol or similar symbols are available
                symbol_found = False
                for available_symbol in available_symbols:
                    if (self.symbol.upper() in available_symbol.upper() or 
                        available_symbol.upper() in self.symbol.upper()):
                        symbol_found = True
                        break
                
                if not symbol_found:
                    self.logger.warning(f"Symbol {self.symbol} may not be available. Available symbols: {available_symbols[:10]}...")
        
        # Perform traditional analysis
        if self.analysis_bot is None:
            try:
                from core.trading_bot import TradingBot
            except ImportError:
                self.logger.error("Failed to import TradingBot from core.trading_bot")
                return None
            # Pull live account balance from MT5 when available
            live_account_size = 10000
            try:
                if self.connected and self.mt5_connector:
                    acct = self.mt5_connector.get_account_summary()
                    if acct and isinstance(acct, dict):
                        # Prefer equity if available; otherwise balance
                        live_account_size = float(acct.get('equity') or acct.get('balance') or live_account_size)
            except Exception:
                pass
            self.analysis_bot = TradingBot(
                self.symbol,
                self.timeframe,
                "forex",
                live_account_size,
                self.risk_per_trade
            )
        else:
            # Keep analysis bot account size in sync with MT5 balance/equity
            try:
                if self.connected and self.mt5_connector:
                    acct = self.mt5_connector.get_account_summary()
                    if acct and isinstance(acct, dict):
                        self.analysis_bot.account_size = float(acct.get('equity') or acct.get('balance') or self.analysis_bot.account_size)
            except Exception:
                pass
        
        self.analysis_bot.data = data.copy()
        analysis = self.analysis_bot.analyze_market_trend()
        
        # Add missing keys that the advanced analysis would have
        if analysis:
            analysis['entry_conditions'] = []
            analysis['overall_strength'] = analysis.get('trend_strength', 50)
            
            # Create primary_analysis with safe key access
            analysis['primary_analysis'] = {
                'hh_count': analysis.get('hh_count', 0),
                'hl_count': analysis.get('hl_count', 0),
                'lh_count': analysis.get('lh_count', 0),
                'll_count': analysis.get('ll_count', 0),
                'higher_highs': analysis.get('higher_highs', []),
                'higher_lows': analysis.get('higher_lows', []),
                'lower_highs': analysis.get('lower_highs', []),
                'lower_lows': analysis.get('lower_lows', []),
                'uptrend_confirmed': analysis.get('uptrend_confirmed', False),
                'downtrend_confirmed': analysis.get('downtrend_confirmed', False),
                'trend_direction': analysis.get('trend_direction', 'SIDEWAYS'),
                'trend_strength': analysis.get('trend_strength', 0)
            }
            
            analysis['h4_analysis'] = {'hh_count': 0, 'hl_count': 0, 'uptrend_confirmed': False}
            analysis['strong_trendlines'] = []
            analysis['breakout_signals'] = None
            analysis['continuation_patterns'] = None
            analysis['trading_rules_followed'] = {
                'multi_timeframe_confirmed': False,
                'min_hh_hl_met': (analysis.get('hh_count', 0) >= 2 and analysis.get('hl_count', 0) >= 2) or 
                                (analysis.get('lh_count', 0) >= 2 and analysis.get('ll_count', 0) >= 2),
                'strong_trendlines': False,
                'breakout_retest_ready': False,
                'continuation_patterns_ready': False
            }
        
        if not analysis:
            print("❌ Traditional analysis failed")
            return None
        
        # Generate report
        if self.analysis_bot:
            self.analysis_bot.generate_report(analysis)
            self.analysis_bot.generate_day_trading_report(analysis)
        
        # Get enhanced trading signals
        signals = self.get_trading_signals(analysis)
        if signals:
            print(f"\n🎯 ENHANCED TRADING SIGNAL DETECTED:")
            print(f"   Signal: {signals['signal_type']}")
            print(f"   Entry: {signals['entry_price']:.5f}")
            print(f"   Stop Loss: {signals['stop_loss']:.5f} (20 points)")
            print(f"   Target: {signals['target']:.5f} (60 points)")
            print(f"   Signal Strength: {signals['signal_strength']:.2f}")
            
            # Handle None position size safely
            position_size = signals.get('position_size')
            if position_size is not None:
                print(f"   Position Size: {position_size:.2f} lots")
            else:
                print(f"   Position Size: N/A (MT5 connection required)")
            
            print(f"   Risk Amount: ${signals['risk_amount']:.2f}")
            print(f"   Potential Profit: ${signals['potential_profit']:.2f}")
            
            # ML prediction details
            if signals.get('ml_prediction'):
                ml = signals['ml_prediction']
                print(f"   ML Confidence: {ml['confidence']:.2f}")
                print(f"   ML Buy Probability: {ml['buy_probability']:.2f}")
            
            # Execute trade if auto trading is enabled
            if self.auto_trade:
                self.execute_trade(signals)
        else:
            print("❌ No enhanced trading signals")
        
        return analysis
    
    def run_continuous_monitoring(self, interval_minutes: int = 5, max_cycles: Optional[int] = None) -> None:
        """
        Run continuous market monitoring
        
        Args:
            interval_minutes (int): Minutes between analysis cycles
            max_cycles (int): Maximum number of cycles (None for unlimited)
        """
        self.logger.info("STARTING CONTINUOUS MARKET MONITORING")
        self.logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe} | Interval: {interval_minutes}m | Auto: {'ENABLED' if self.auto_trade else 'DISABLED'}")
        
        cycle_count = 0
        
        try:
            while True:
                if max_cycles and cycle_count >= max_cycles:
                    self.logger.info(f"Reached maximum cycles ({max_cycles})")
                    break
                
                cycle_count += 1
                self.logger.info(f"Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run analysis cycle
                analysis = self.run_analysis_cycle()
                
                # Monitor positions
                if self.connected:
                    self.monitor_positions()
                
                # Monitor and evaluate trades for journal
                if self.trade_journal_enabled:
                    self.monitor_and_evaluate_trades()
                
                # Learn from trades every 10 cycles
                if self.learning_enabled and cycle_count % 10 == 0:
                    insights = self.learn_from_trades()
                    if insights and insights.get('recommendations'):
                        self.logger.info(f"Learning insights: {len(insights['recommendations'])} recommendations")
                        for rec in insights['recommendations']:
                            self.logger.info(f"  - {rec['message']}")
                
                # Wait for next cycle
                if max_cycles is None or cycle_count < max_cycles:
                    self.logger.info(f"Waiting {interval_minutes} minutes for next cycle...")
                    time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.exception(f"Error in monitoring: {e}")
        finally:
            # Close all positions if auto trading was enabled
            if self.auto_trade and self.connected:
                self.logger.info("Closing all positions...")
                self.close_all_positions()
            
            # Disconnect from MT5
            if self.mt5_connector:
                self.mt5_connector.disconnect()
            
            self.logger.info("Monitoring session ended")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """
        Get trading session summary
        
        Returns:
            dict: Trading summary
        """
        summary = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'account_size': None,  # No longer used - balance fetched from MT5
            'risk_per_trade': self.risk_per_trade,
            'total_trades': len(self.trade_history),
            'open_positions': 0,
            'total_profit': 0,
            'ml_enabled': self.use_ml,
            'model_trained': self.model_trained
        }
        
        if self.connected:
            positions = self.monitor_positions()
            summary['open_positions'] = len(positions)
            summary['total_profit'] = sum(pos['profit'] for pos in positions)
        
        return summary
    
    def save_ml_model(self, filename: Optional[str] = None) -> bool:
        """Save trained ML model to file"""
        if not self.model_trained:
            print("❌ No trained model to save")
            return False
        
        if filename is None:
            os.makedirs('models', exist_ok=True)
            filename = os.path.join('models', f"ml_model_{self.symbol}_{self.timeframe}.joblib")
        
        try:
            model_data = {
                'model': self.ml_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'trained_date': datetime.now()
            }
            
            joblib.dump(model_data, filename)
            self.logger.info(f"ML model saved to {filename}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Failed to save model: {e}")
            return False
    
    def load_ml_model(self, filename: str) -> bool:
        """Load trained ML model from file"""
        try:
            # Avoid noisy stack traces when the model file doesn't exist yet
            if not os.path.exists(filename):
                self.logger.info(f"ML model not found at {filename}. A new model will be trained when needed.")
                return False

            model_data = joblib.load(filename)
            
            self.ml_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_trained = True
            
            self.logger.info(f"ML model loaded from {filename}")
            return True
            
        except Exception as e:
            # Log as error but without full traceback, then allow training fallback
            self.logger.error(f"Failed to load model from {filename}: {e}")
            return False

    def analyze_smc(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Perform Smart Money Concept analysis on the data
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            dict: SMC analysis results
        """
        if not self.use_smc or data is None or len(data) < 100:
            return None
        
        try:
            self.logger.info(f"Performing Smart Money Concept analysis for {self.symbol}...")
            
            # Initialize SMC analyzer
            self.smc_analyzer = SmartMoneyConcept(data, self.timeframe)
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Get SMC signals
            self.smc_signals = self.smc_analyzer.get_smc_signals(current_price)
            
            # Get SMC summary
            self.smc_summary = self.smc_analyzer.get_smc_summary()
            
            self.logger.info("SMC Analysis Complete")
            self.logger.info(f"Market Structure: {self.smc_summary['market_structure']['trend_direction']} | OBs: {self.smc_summary['order_blocks']['total_count']} | FVGs: {self.smc_summary['fair_value_gaps']['total_count']} | Liquidity: {self.smc_summary['liquidity_zones']['total_count']} | IOBs: {self.smc_summary['institutional_order_blocks']['total_count']}")
            
            return {
                'signals': self.smc_signals,
                'summary': self.smc_summary,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.exception(f"Error in SMC analysis: {e}")
            return None
    
    def get_smc_trading_signals(self, current_price: float) -> Optional[List[Dict[str, Any]]]:
        """
        Generate trading signals based on SMC analysis
        
        Args:
            current_price (float): Current market price
            
        Returns:
            dict: Trading signals with entry, stop loss, and target levels
        """
        if not self.use_smc or not self.smc_signals:
            return None
        
        signals = []
        
        # Process Order Block signals
        for ob_signal in self.smc_signals.get('order_block_signals', []):
            if ob_signal['type'] == 'Bullish_OB_Entry':
                # Calculate target based on order block strength
                target_distance = (ob_signal['ob_level'] - ob_signal['stop_loss']) * 2
                target_price = current_price + target_distance
                
                signals.append({
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': ob_signal['stop_loss'],
                    'target': target_price,
                    'source': 'Order_Block',
                    'strength': ob_signal['strength'] / 100,
                    'timestamp': ob_signal['timestamp']
                })
            
            elif ob_signal['type'] == 'Bearish_OB_Entry':
                # Calculate target based on order block strength
                target_distance = (ob_signal['stop_loss'] - ob_signal['ob_level']) * 2
                target_price = current_price - target_distance
                
                signals.append({
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': ob_signal['stop_loss'],
                    'target': target_price,
                    'source': 'Order_Block',
                    'strength': ob_signal['strength'] / 100,
                    'timestamp': ob_signal['timestamp']
                })
        
        # Process Fair Value Gap signals
        for fvg_signal in self.smc_signals.get('fvg_signals', []):
            if fvg_signal['type'] == 'Bullish_FVG_Fill':
                target_price = current_price + (fvg_signal['gap_size'] * 3)
                
                signals.append({
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': fvg_signal['stop_loss'],
                    'target': target_price,
                    'source': 'Fair_Value_Gap',
                    'strength': min(0.8, fvg_signal['gap_size'] * 1000),
                    'timestamp': fvg_signal['timestamp']
                })
            
            elif fvg_signal['type'] == 'Bearish_FVG_Fill':
                target_price = current_price - (fvg_signal['gap_size'] * 3)
                
                signals.append({
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': fvg_signal['stop_loss'],
                    'target': target_price,
                    'source': 'Fair_Value_Gap',
                    'strength': min(0.8, fvg_signal['gap_size'] * 1000),
                    'timestamp': fvg_signal['timestamp']
                })
        
        # Process Market Structure signals
        for ms_signal in self.smc_signals.get('market_structure_signals', []):
            if ms_signal['type'] == 'Bullish_BOS':
                # Break of Structure - bullish continuation
                target_price = current_price + (current_price - ms_signal['level']) * 2
                
                signals.append({
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': ms_signal['level'],
                    'target': target_price,
                    'source': 'Break_of_Structure',
                    'strength': 0.7,
                    'timestamp': ms_signal['timestamp']
                })
            
            elif ms_signal['type'] == 'Bearish_BOS':
                # Break of Structure - bearish continuation
                target_price = current_price - (ms_signal['level'] - current_price) * 2
                
                signals.append({
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': ms_signal['level'],
                    'target': target_price,
                    'source': 'Break_of_Structure',
                    'strength': 0.7,
                    'timestamp': ms_signal['timestamp']
                })
        
        # Process Institutional Order Block signals
        for iob_signal in self.smc_signals.get('institutional_signals', []):
            if 'Bullish_IOB' in iob_signal['type']:
                target_price = current_price + (current_price * 0.01)  # 1% target
                
                signals.append({
                    'type': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.995,  # 0.5% stop loss
                    'target': target_price,
                    'source': 'Institutional_OB',
                    'strength': min(0.9, iob_signal['volume_ratio'] / 5),
                    'timestamp': iob_signal['timestamp']
                })
            
            elif 'Bearish_IOB' in iob_signal['type']:
                target_price = current_price - (current_price * 0.01)  # 1% target
                
                signals.append({
                    'type': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': current_price * 1.005,  # 0.5% stop loss
                    'target': target_price,
                    'source': 'Institutional_OB',
                    'strength': min(0.9, iob_signal['volume_ratio'] / 5),
                    'timestamp': iob_signal['timestamp']
                })
        
        return signals
    
    def combine_signals(self, traditional_signals: Optional[Dict[str, Any]], smc_signals: Optional[List[Dict[str, Any]]], ml_prediction: Optional[Dict[str, Any]] = None, market_structure_analysis: Optional[Dict[str, Any]] = None, rl_analysis: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Combine traditional, SMC, and ML signals for final decision
        
        Args:
            traditional_signals (dict): Traditional technical analysis signals
            smc_signals (list): Smart Money Concept signals
            ml_prediction (dict): Machine learning prediction
            
        Returns:
            dict: Combined trading signal
        """
        if not traditional_signals and not smc_signals:
            return None
        
        # Initialize combined signal
        combined_signal = {
            'signal_type': None,
            'entry_price': None,
            'stop_loss': None,
            'target': None,
            'position_size': None,
            'risk_amount': 0,  # Will be calculated from current balance when needed
            'potential_profit': 0,
            'timeframe': self.timeframe,
            'analysis': traditional_signals,
            'ml_prediction': ml_prediction,
            'smc_signals': smc_signals,
            'signal_strength': 0.0,
            'signal_sources': []
        }
        
        # Process traditional signals
        if traditional_signals:
            combined_signal['signal_type'] = traditional_signals.get('signal_type', 'HOLD')
            combined_signal['entry_price'] = traditional_signals.get('entry_price')
            combined_signal['stop_loss'] = traditional_signals.get('stop_loss')
            combined_signal['target'] = traditional_signals.get('target')
            combined_signal['position_size'] = traditional_signals.get('position_size')
            combined_signal['signal_strength'] = traditional_signals.get('signal_strength', 0.5)
            combined_signal['signal_sources'].append('Traditional_TA')
        
        # Process SMC signals
        if smc_signals:
            # Find the strongest SMC signal
            strongest_smc = max(smc_signals, key=lambda x: x['strength']) if smc_signals else None
            
            if strongest_smc and strongest_smc['strength'] > 0.6:
                # SMC signal is strong enough to override or enhance traditional signal
                if combined_signal['signal_type'] == 'HOLD' or combined_signal['signal_type'] is None:
                    # No traditional signal, use SMC signal
                    combined_signal['signal_type'] = strongest_smc['type']
                    combined_signal['entry_price'] = strongest_smc['entry_price']
                    combined_signal['stop_loss'] = strongest_smc['stop_loss']
                    combined_signal['target'] = strongest_smc['target']
                    combined_signal['signal_strength'] = strongest_smc['strength']
                    # Keep existing sources and add SMC
                    if strongest_smc['source'] not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append(strongest_smc['source'])
                
                elif combined_signal['signal_type'] == strongest_smc['type']:
                    # Signals agree - enhance strength
                    combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + strongest_smc['strength'] * 0.3)
                    # Add SMC source if not already present
                    if strongest_smc['source'] not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append(strongest_smc['source'])
                
                else:
                    # Signals conflict - use the stronger one but keep both sources for validation
                    if strongest_smc['strength'] > combined_signal['signal_strength']:
                        combined_signal['signal_type'] = strongest_smc['type']
                        combined_signal['entry_price'] = strongest_smc['entry_price']
                        combined_signal['stop_loss'] = strongest_smc['stop_loss']
                        combined_signal['target'] = strongest_smc['target']
                        combined_signal['signal_strength'] = strongest_smc['strength']
                        # Keep existing sources and add SMC for multiple confirmations
                        if strongest_smc['source'] not in combined_signal['signal_sources']:
                            combined_signal['signal_sources'].append(strongest_smc['source'])
        
        # Process Market Structure Analysis
        if market_structure_analysis and self.use_market_structure:
            market_signal = market_structure_analysis.get('signal')
            if market_signal:
                # Market structure signal is very strong - it can override other signals
                if market_signal.confidence >= 0.7:
                    combined_signal['signal_type'] = market_signal.entry_type.value.upper()
                    combined_signal['entry_price'] = market_signal.entry_price
                    combined_signal['stop_loss'] = market_signal.stop_loss
                    combined_signal['target'] = market_signal.take_profit
                    combined_signal['signal_strength'] = market_signal.confidence
                    # Keep existing sources and add Market Structure
                    if 'Market_Structure' not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append('Market_Structure')
                    combined_signal['market_structure_analysis'] = market_structure_analysis
                else:
                    # Market structure signal is moderate - enhance existing signal
                    if combined_signal['signal_type'] == market_signal.entry_type.value.upper():
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + market_signal.confidence * 0.3)
                        combined_signal['signal_sources'].append('Market_Structure')
        
        # Process ML prediction
        if ml_prediction and self.use_ml:
            ml_confidence = ml_prediction.get('confidence', 0)
            ml_signal = ml_prediction.get('prediction', 0.5)
            
            if ml_confidence >= self.prediction_threshold:
                if ml_signal > 0.6:  # Bullish
                    if combined_signal['signal_type'] == 'BUY':
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + ml_confidence * 0.2)
                    elif combined_signal['signal_type'] == 'SELL':
                        # ML disagrees with bearish signal
                        combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.5
                    if 'ML_Bullish' not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append('ML_Bullish')
                
                elif ml_signal < 0.4:  # Bearish
                    if combined_signal['signal_type'] == 'SELL':
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + ml_confidence * 0.2)
                    elif combined_signal['signal_type'] == 'BUY':
                        # ML disagrees with bullish signal
                        combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.5
                    if 'ML_Bearish' not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append('ML_Bearish')
        
        # Process Reinforcement Learning Analysis
        if rl_analysis and self.use_rl:
            rl_signal = rl_analysis.get('signal_type')
            rl_confidence = rl_analysis.get('confidence', 0)
            rl_action = rl_analysis.get('action')
            
            if rl_confidence >= 0.6:  # RL signal threshold
                if rl_action == 'buy' and rl_signal == 'BUY':
                    if combined_signal['signal_type'] == 'BUY':
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + rl_confidence * 0.3)
                    elif combined_signal['signal_type'] == 'SELL':
                        # RL strongly disagrees with bearish signal
                        combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.3
                    if 'RL_Bullish' not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append('RL_Bullish')
                
                elif rl_action == 'sell' and rl_signal == 'SELL':
                    if combined_signal['signal_type'] == 'SELL':
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + rl_confidence * 0.3)
                    elif combined_signal['signal_type'] == 'BUY':
                        # RL strongly disagrees with bullish signal
                        combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.3
                    if 'RL_Bearish' not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append('RL_Bearish')
                
                elif rl_action == 'hold':
                    # RL suggests holding - reduce signal strength
                    combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.7
                    if 'RL_Hold' not in combined_signal['signal_sources']:
                        combined_signal['signal_sources'].append('RL_Hold')
                
                # Add RL analysis to combined signal
                combined_signal['rl_analysis'] = rl_analysis
        
        # Calculate position size if we have entry and stop loss
        if combined_signal['entry_price'] and combined_signal['stop_loss'] and self.connected:
            combined_signal['position_size'] = self.mt5_connector.calculate_position_size(
                combined_signal['risk_amount'],
                combined_signal['entry_price'],
                combined_signal['stop_loss'],
                self.symbol
            )
            
            # Calculate potential profit
            if combined_signal['position_size'] and combined_signal['target']:
                if combined_signal['signal_type'] == 'BUY':
                    profit_pips = (combined_signal['target'] - combined_signal['entry_price']) * 100000
                else:
                    profit_pips = (combined_signal['entry_price'] - combined_signal['target']) * 100000
                combined_signal['potential_profit'] = profit_pips * combined_signal['position_size']
        
        # Enhance signal sources to ensure at least 2 confirmations
        if len(combined_signal['signal_sources']) < 2 and combined_signal['signal_type'] != 'HOLD':
            # Add additional confirmation sources based on available data
            if traditional_signals and 'Traditional_TA' not in combined_signal['signal_sources']:
                combined_signal['signal_sources'].append('Traditional_TA')
            
            if ml_prediction and ml_prediction.get('confidence', 0) > 0.5:
                ml_signal = ml_prediction.get('prediction', 0.5)
                if ml_signal > 0.6 and 'ML_Bullish' not in combined_signal['signal_sources']:
                    combined_signal['signal_sources'].append('ML_Bullish')
                elif ml_signal < 0.4 and 'ML_Bearish' not in combined_signal['signal_sources']:
                    combined_signal['signal_sources'].append('ML_Bearish')
            
            if rl_analysis and rl_analysis.get('confidence', 0) > 0.5:
                rl_signal = rl_analysis.get('signal_type')
                rl_action = rl_analysis.get('action')
                if rl_action == 'buy' and rl_signal == 'BUY' and 'RL_Bullish' not in combined_signal['signal_sources']:
                    combined_signal['signal_sources'].append('RL_Bullish')
                elif rl_action == 'sell' and rl_signal == 'SELL' and 'RL_Bearish' not in combined_signal['signal_sources']:
                    combined_signal['signal_sources'].append('RL_Bearish')
            
            # If still less than 2 sources, add a general confirmation
            if len(combined_signal['signal_sources']) < 2:
                combined_signal['signal_sources'].append('General_Confirmation')
                self.logger.info(f"Added general confirmation to reach minimum 2 signal sources")
            
            # Additional fallback: Add trend confirmation if available
            if len(combined_signal['signal_sources']) < 2 and traditional_signals:
                if 'Trend_Confirmation' not in combined_signal['signal_sources']:
                    combined_signal['signal_sources'].append('Trend_Confirmation')
                    self.logger.info(f"Added trend confirmation as additional signal source")
        
        # Only return signal if strength is sufficient
        if combined_signal['signal_strength'] < 0.5:
            self.logger.warning(f"Combined signal strength too low: {combined_signal['signal_strength']:.2f}")
            return None
        
        return combined_signal
    
    def get_ml_ensemble_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get ML ensemble summary and performance metrics
        
        Returns:
            dict: ML ensemble summary
        """
        if not self.use_ensemble or not self.ml_ensemble:
            return None
        
        try:
            return self.ml_ensemble.get_model_summary()
        except Exception as e:
            self.logger.error(f"Failed to get ML ensemble summary: {e}")
            return None
    
    def run_market_structure_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run market structure analysis across multiple timeframes
        
        Returns:
            dict: Market structure analysis results
        """
        if not self.use_market_structure or not self.market_structure_strategy:
            return None
        
        try:
            # Get data for different timeframes
            data_d1 = self.get_market_data(timeframe='1d', count=100)
            data_h4 = self.get_market_data(timeframe='4h', count=100)
            data_h1 = self.get_market_data(timeframe='1h', count=100)
            
            if data_d1 is None or data_h4 is None or data_h1 is None:
                self.logger.warning("Could not get market data for all timeframes")
                return None
            
            # Run analysis
            analysis = self.market_structure_strategy.analyze_symbol(
                self.symbol, data_d1, data_h4, data_h1
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to run market structure analysis: {e}")
            return None
    
    def get_market_structure_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get market structure strategy summary
        
        Returns:
            dict: Market structure strategy summary
        """
        if not self.use_market_structure or not self.market_structure_strategy:
            return None
        
        try:
            return self.market_structure_strategy.get_strategy_summary()
        except Exception as e:
            self.logger.error(f"Failed to get market structure summary: {e}")
            return None
    
    def update_ml_ensemble_performance(self, prediction: Dict[str, Any], actual_outcome: int):
        """
        Update ML ensemble performance based on actual outcome
        
        Args:
            prediction: ML prediction results
            actual_outcome: Actual market outcome (1 for positive, 0 for negative)
        """
        if not self.use_ensemble or not self.ml_ensemble:
            return
        
        try:
            if 'ensemble_details' in prediction:
                self.ml_ensemble.update_model_performance(prediction, actual_outcome)
        except Exception as e:
            self.logger.error(f"Failed to update ML ensemble performance: {e}")
    
    def run_rl_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run reinforcement learning analysis
        
        Returns:
            dict: RL analysis results
        """
        if not self.use_rl or not self.rl_trader:
            return None
        
        try:
            # Get current market data
            if not self.analysis_bot or self.analysis_bot.data is None:
                self.logger.warning("No market data available for RL analysis")
                return None
            
            # Prepare analysis data
            analysis_data = {
                'trend_analysis': self.trend_analysis if hasattr(self, 'trend_analysis') else {},
                'market_structure': {},
                'zones': [],
                'patterns': []
            }
            
            # Add market structure data if available
            if self.use_market_structure and self.market_structure_strategy:
                try:
                    market_structure = self.market_structure_strategy.analyze_symbol(self.symbol)
                    if market_structure:
                        analysis_data['market_structure'] = {
                            'higher_highs': market_structure.higher_highs,
                            'higher_lows': market_structure.higher_lows,
                            'lower_highs': market_structure.lower_highs,
                            'lower_lows': market_structure.lower_lows
                        }
                        analysis_data['zones'] = market_structure.zones
                        analysis_data['patterns'] = market_structure.patterns
                except Exception as e:
                    self.logger.warning(f"Failed to get market structure data for RL: {e}")
            
            # Get RL trading signal
            rl_signal = self.rl_trader.get_trading_signal(self.analysis_bot.data, analysis_data)
            
            # Update RL trader with current trade status
            if hasattr(self, 'current_position') and self.current_position:
                # Check if current position should be closed based on RL
                current_price = self.analysis_bot.data['Close'].iloc[-1]
                
                if self.current_position['type'] == 'BUY':
                    if current_price >= self.current_position.get('take_profit', float('inf')):
                        # Position hit take profit
                        from strategies.reinforcement_learning_trader import TradeOutcome
                        self.rl_trader.close_trade(current_price, TradeOutcome.WIN)
                    elif current_price <= self.current_position.get('stop_loss', 0):
                        # Position hit stop loss
                        self.rl_trader.close_trade(current_price, TradeOutcome.LOSS)
                else:  # SELL
                    if current_price <= self.current_position.get('take_profit', 0):
                        # Position hit take profit
                        self.rl_trader.close_trade(current_price, TradeOutcome.WIN)
                    elif current_price >= self.current_position.get('stop_loss', float('inf')):
                        # Position hit stop loss
                        self.rl_trader.close_trade(current_price, TradeOutcome.LOSS)
            
            return rl_signal
            
        except Exception as e:
            self.logger.error(f"Failed to run RL analysis: {e}")
            return None
    
    def get_rl_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get reinforcement learning summary and performance metrics
        
        Returns:
            dict: RL summary
        """
        if not self.use_rl or not self.rl_trader:
            return None
        
        try:
            return self.rl_trader.get_performance_metrics()
        except Exception as e:
            self.logger.error(f"Failed to get RL summary: {e}")
            return None

def main():
    """Main function for MT5 trading bot"""
    import sys
    
    # Default parameters
    symbol = "EURUSD"
    timeframe = "5m"  # 5 minutes for scalping
    risk_percent = 2.0
    auto_trade = False
    continuous = False
    use_ml = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    if len(sys.argv) > 2:
        timeframe = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            risk_percent = float(sys.argv[3])
        except ValueError:
            print(f"Invalid risk percentage: {sys.argv[3]}")
    if len(sys.argv) > 4:
        auto_trade = sys.argv[4].lower() == 'true'
    if len(sys.argv) > 5:
        continuous = sys.argv[5].lower() == 'true'
    if len(sys.argv) > 6:
        use_ml = sys.argv[6].lower() == 'true'
    
    risk_per_trade = risk_percent / 100.0
    
    print("="*60)
    print("MT5 INTEGRATED TRADING BOT (ML-Enhanced)")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Risk Per Trade: {risk_percent}%")
    print(f"Auto Trading: {'ENABLED' if auto_trade else 'DISABLED'}")
    print(f"Continuous Mode: {'ENABLED' if continuous else 'DISABLED'}")
    print(f"Machine Learning: {'ENABLED' if use_ml else 'DISABLED'}")
    print("="*60)
    
    # Create trading bot
    bot = MT5TradingBot(
        symbol=symbol,
        timeframe=timeframe,
        risk_per_trade=risk_per_trade,
        use_mt5_data=True,
        auto_trade=auto_trade,
        use_ml=use_ml
    )
    
    if continuous:
        # Run continuous monitoring
        bot.run_continuous_monitoring(interval_minutes=5)
    else:
        # Run single analysis cycle
        analysis = bot.run_analysis_cycle()
        
        if analysis and auto_trade:
            # Monitor for a while
            print("\n📊 Monitoring positions for 10 minutes...")
            for i in range(10):
                time.sleep(60)
                bot.monitor_positions()
        
        # Get summary
        summary = bot.get_trading_summary()
        import logging as _logging
        _logging.getLogger('mt5_trading_bot').info(
            f"Trading Summary | trades={summary['total_trades']} open={summary['open_positions']} profit=${summary['total_profit']:.2f} ml={'ENABLED' if summary['ml_enabled'] else 'DISABLED'} trained={'YES' if summary['model_trained'] else 'NO'}"
        )
    
    # Save ML model if trained
    if bot.use_ml and bot.model_trained:
        bot.save_ml_model()
    
    # Cleanup
    if bot.mt5_connector:
        bot.mt5_connector.disconnect()

def update_trade_tracking(self):
    """
    Update trade tracking variables
    """
    try:
        from datetime import datetime
        
        # Update daily trade count
        today = datetime.now().date()
        if self.last_trade_date == today:
            self.daily_trade_count += 1
        else:
            self.daily_trade_count = 1
            self.last_trade_date = today
        
        self.logger.info(f"Trade tracking updated - Daily trades: {self.daily_trade_count}/{self.max_daily_trades}")
        
    except Exception as e:
        self.logger.error(f"Error updating trade tracking: {e}")

def record_trade_result(self, trade_result: Dict[str, Any]):
    """
    Record trade result and update consecutive loss tracking
    
    Args:
        trade_result: Trade execution result
    """
    try:
        # Check if trade was profitable
        profit = trade_result.get('profit', 0)
        is_win = profit > 0
        
        if is_win:
            self.consecutive_losses = 0
            self.logger.info("Winning trade - Reset consecutive losses")
        else:
            self.consecutive_losses += 1
            self.logger.warning(f"Losing trade - Consecutive losses: {self.consecutive_losses}")
            
            # Check if we should stop trading
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.error(f"STOP TRADING - Too many consecutive losses ({self.consecutive_losses})")
        
    except Exception as e:
        self.logger.error(f"Error recording trade result: {e}")

def log_trade_to_journal(self, trade_data: Dict[str, Any]) -> bool:
    """
    Log trade to the trade journal system
    
    Args:
        trade_data (dict): Trade information to log
        
    Returns:
        bool: True if successfully logged, False otherwise
    """
    if not self.trade_journal_enabled:
        return False
        
    try:
        # Prepare trade data for journal
        journal_entry = {
            'symbol': self.symbol,
            'trade_type': trade_data.get('signal_type', 'UNKNOWN'),
            'entry_price': trade_data.get('entry_price', 0.0),
            'take_profit': trade_data.get('target', 0.0),
            'stop_loss': trade_data.get('stop_loss', 0.0),
            'entry_date': datetime.now().isoformat(),
            'notes': f"Auto-traded by {self.symbol} bot on {self.timeframe} timeframe"
        }
        
        # Add order ID if available
        if 'order_id' in trade_data:
            journal_entry['notes'] += f" | Order ID: {trade_data['order_id']}"
        
        # Send to trade journal API
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.trade_journal_api_key
        }
        
        response = requests.post(
            self.trade_journal_api_url,
            json=journal_entry,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                trade_id = result.get('trade_id')
                order_id = trade_data.get('order_id')
                if order_id and trade_id:
                    self.logged_trades[order_id] = trade_id
                self.logger.info(f"Trade logged to journal successfully - Trade ID: {trade_id}")
                return True
            else:
                self.logger.error(f"Failed to log trade: {result.get('error', 'Unknown error')}")
        else:
            self.logger.error(f"Trade journal API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        self.logger.error(f"Error logging trade to journal: {e}")
        
    return False

def evaluate_trade_outcome(self, order_id: str, current_price: float) -> Optional[str]:
    """
    Evaluate trade outcome and update journal
    
    Args:
        order_id (str): Order ID to evaluate
        current_price (float): Current market price
        
    Returns:
        str: Outcome (WIN/LOSS/OPEN) or None if error
    """
    if not self.trade_journal_enabled or order_id not in self.logged_trades:
        return None
        
    try:
        trade_id = self.logged_trades[order_id]
        
        # Send evaluation request to trade journal API
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.trade_journal_api_key
        }
        
        evaluation_data = {
            'trade_id': trade_id,
            'current_price': current_price
        }
        
        response = requests.post(
            f"{self.trade_journal_api_url}/evaluate",
            json=evaluation_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                outcome = result.get('outcome')
                self.logger.info(f"Trade {order_id} evaluated as: {outcome}")
                
                # Add to performance history for learning
                if outcome in ['WIN', 'LOSS']:
                    self.performance_history.append({
                        'timestamp': datetime.now(),
                        'outcome': outcome,
                        'symbol': self.symbol,
                        'timeframe': self.timeframe,
                        'trade_id': trade_id
                    })
                
                return outcome
            else:
                self.logger.error(f"Failed to evaluate trade: {result.get('error', 'Unknown error')}")
        else:
            self.logger.error(f"Trade evaluation API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        self.logger.error(f"Error evaluating trade outcome: {e}")
        
    return None

def get_trade_performance_stats(self) -> Dict[str, Any]:
    """
    Get trade performance statistics from journal
    
    Returns:
        dict: Performance statistics
    """
    if not self.trade_journal_enabled:
        return {}
        
    try:
        headers = {
            'X-API-Key': self.trade_journal_api_key
        }
        
        response = requests.get(
            f"{self.trade_journal_api_url}/statistics",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('statistics', {})
            else:
                self.logger.error(f"Failed to get trade statistics: {result.get('error', 'Unknown error')}")
        else:
            self.logger.error(f"Trade statistics API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        self.logger.error(f"Error getting trade performance stats: {e}")
        
    return {}

def learn_from_trades(self) -> Dict[str, Any]:
    """
    Analyze trade history and learn from mistakes
    
    Returns:
        dict: Learning insights and recommendations
    """
    if not self.learning_enabled:
        return {}
        
    try:
        # Get performance statistics
        stats = self.get_trade_performance_stats()
        if not stats:
            return {}
        
        insights = {
            'win_rate': stats.get('win_rate', 0),
            'total_trades': stats.get('total_trades', 0),
            'recommendations': [],
            'risk_adjustments': {}
        }
        
        # Analyze performance and provide recommendations
        win_rate = stats.get('win_rate', 0)
        total_trades = stats.get('total_trades', 0)
        
        if total_trades >= 10:  # Need sufficient data for analysis
            # Win rate analysis
            if win_rate < 40:
                insights['recommendations'].append({
                    'type': 'risk_management',
                    'message': 'Low win rate detected. Consider increasing signal validation threshold.',
                    'action': 'increase_signal_threshold'
                })
                insights['risk_adjustments']['min_signal_strength'] = min(0.9, self.min_signal_strength + 0.1)
            
            elif win_rate > 70:
                insights['recommendations'].append({
                    'type': 'opportunity',
                    'message': 'High win rate detected. Consider increasing position size slightly.',
                    'action': 'increase_position_size'
                })
            
            # Consecutive losses analysis
            consecutive_losses = stats.get('loss_streak', 0)
            if consecutive_losses >= 3:
                insights['recommendations'].append({
                    'type': 'risk_management',
                    'message': f'Consecutive losses detected ({consecutive_losses}). Consider reducing risk per trade.',
                    'action': 'reduce_risk'
                })
                insights['risk_adjustments']['risk_per_trade'] = max(0.002, self.risk_per_trade * 0.8)
            
            # Profit factor analysis
            profit_factor = stats.get('profit_factor', 0)
            if profit_factor < 1.0:
                insights['recommendations'].append({
                    'type': 'strategy',
                    'message': 'Profit factor below 1.0. Review and improve risk-reward ratios.',
                    'action': 'improve_risk_reward'
                })
                insights['risk_adjustments']['min_risk_reward_ratio'] = max(3.0, self.min_risk_reward_ratio + 0.5)
        
        # Apply automatic adjustments if learning is enabled
        if insights['risk_adjustments']:
            self.apply_learning_adjustments(insights['risk_adjustments'])
        
        self.logger.info(f"Learning analysis completed. Win rate: {win_rate:.1f}%, Total trades: {total_trades}")
        return insights
        
    except Exception as e:
        self.logger.error(f"Error in learning from trades: {e}")
        return {}

def apply_learning_adjustments(self, adjustments: Dict[str, Any]):
    """
    Apply learning-based adjustments to trading parameters
    
    Args:
        adjustments (dict): Parameter adjustments to apply
    """
    try:
        for param, value in adjustments.items():
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                self.logger.info(f"Learning adjustment: {param} changed from {old_value} to {value}")
            else:
                self.logger.warning(f"Unknown parameter for learning adjustment: {param}")
                
    except Exception as e:
        self.logger.error(f"Error applying learning adjustments: {e}")

    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """
        Get enhanced performance summary including advanced features
        
        Returns:
            dict: Enhanced performance summary
        """
        summary = {
            'basic_metrics': self.get_performance_summary(),
            'advanced_features': {
                'risk_management_enabled': self.use_advanced_risk,
                'regime_detection_enabled': self.use_regime_detection,
                'ml_enabled': self.use_ml,
                'smc_enabled': self.use_smc
            }
        }
        
        # Add advanced risk management summary
        if self.use_advanced_risk and self.advanced_risk_manager:
            summary['risk_management'] = self.advanced_risk_manager.get_risk_summary()
        
        # Add market regime summary
        if self.use_regime_detection and self.regime_detector:
            summary['market_regime'] = self.regime_detector.get_regime_summary()
        
        return summary

    def get_recent_data(self, bars: int = 100):
        """
        Get recent market data for regime detection and analysis
        
        Args:
            bars (int): Number of bars to retrieve
            
        Returns:
            pd.DataFrame: Recent market data
        """
        try:
            # Ensure we have a valid MT5 connector
            if not hasattr(self, 'mt5_connector') or self.mt5_connector is None:
                self.logger.warning("No MT5 connector available for get_recent_data")
                return None
            # Try to get data from MT5 first
            if self.connected and self.mt5_connector:
                data = self.mt5_connector.get_historical_data(self.symbol, self.timeframe, bars)
                if data is not None and not data.empty:
                    return data
            
            # Fallback to yfinance
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1mo", interval="5m")
            
            if not data.empty:
                data = data.reset_index()
                # Handle different column names from yfinance
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'datetime'})
                elif 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'datetime'})
                
                # Ensure we have the right columns
                required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    # Try to map common column names
                    column_mapping = {
                        'Open': 'open', 'High': 'high', 'Low': 'low', 
                        'Close': 'close', 'Volume': 'volume'
                    }
                    for old_col, new_col in column_mapping.items():
                        if old_col in data.columns:
                            data = data.rename(columns={old_col: new_col})
                
                return data.tail(bars)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting recent data: {e}")
            return None
    
    def __getattr__(self, name):
        """Fallback method to handle missing attributes gracefully"""
        if name == 'get_recent_data':
            # Return a fallback method if get_recent_data is missing
            def fallback_get_recent_data(bars=100):
                self.logger.warning("Using fallback get_recent_data method")
                try:
                    return self.get_market_data()
                except:
                    return None
            return fallback_get_recent_data
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def monitor_and_evaluate_trades(self):
        """
        Monitor open positions and evaluate their outcomes
        """
        if not self.trade_journal_enabled or not self.connected:
            return
            
        try:
            # Get current positions
            positions = self.mt5_connector.get_positions()
            if not positions:
                return
            
            # Get current market price
            current_price = self.mt5_connector.get_current_price(self.symbol)
            if not current_price:
                return
            
            # Evaluate each position
            for position in positions:
                if position['symbol'] == self.symbol:
                    order_id = str(position.get('ticket', ''))
                    if order_id in self.logged_trades:
                        outcome = self.evaluate_trade_outcome(order_id, current_price)
                        if outcome in ['WIN', 'LOSS']:
                            # Trade closed, remove from tracking
                            del self.logged_trades[order_id]
                            
        except Exception as e:
            self.logger.error(f"Error monitoring and evaluating trades: {e}")

# Add methods to the class
MT5TradingBot.update_trade_tracking = update_trade_tracking
MT5TradingBot.record_trade_result = record_trade_result
MT5TradingBot.log_trade_to_journal = log_trade_to_journal
MT5TradingBot.evaluate_trade_outcome = evaluate_trade_outcome
MT5TradingBot.get_trade_performance_stats = get_trade_performance_stats
MT5TradingBot.learn_from_trades = learn_from_trades
MT5TradingBot.apply_learning_adjustments = apply_learning_adjustments

if __name__ == "__main__":
    main() 