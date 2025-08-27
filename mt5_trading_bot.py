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
warnings.filterwarnings('ignore')

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
    from ml_ensemble import MLEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("ML Ensemble module not available")
    ENSEMBLE_AVAILABLE = False

# Market Structure Strategy
try:
    from market_structure_strategy import MarketStructureStrategy
    MARKET_STRUCTURE_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("Market Structure Strategy module not available")
    MARKET_STRUCTURE_AVAILABLE = False

# Reinforcement Learning Trader
try:
    from reinforcement_learning_trader import ReinforcementLearningTrader
    RL_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("Reinforcement Learning Trader module not available")
    RL_AVAILABLE = False

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
    from smart_money_concept import SmartMoneyConcept
    SMC_AVAILABLE = True
except ImportError:
    import logging as _logging
    _logging.getLogger('mt5_trading_bot').warning("Smart Money Concept module not available")
    SMC_AVAILABLE = False

from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from mt5_connector import MT5Connector

# Load environment variables
load_dotenv()

class MT5TradingBot:
    def __init__(self, symbol: str, timeframe: str, risk_per_trade: float = 0.02, 
                 use_mt5_data: bool = True, auto_trade: bool = False, use_ml: bool = True, use_smc: bool = True) -> None:
        """
        Initialize MT5 integrated trading bot
        
        Args:
            symbol (str): Symbol to trade (e.g., 'EURUSD', 'GBPUSD')
            timeframe (str): Timeframe for analysis ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            risk_per_trade (float): Risk per trade as percentage
            use_mt5_data (bool): Use MT5 data instead of yfinance
            auto_trade (bool): Enable automatic trading
            use_ml (bool): Enable machine learning features
            use_smc (bool): Enable Smart Money Concept analysis
        """
        self.symbol = symbol
        self.timeframe = timeframe
        # Ensure risk_per_trade has a valid value
        if risk_per_trade is None or risk_per_trade <= 0:
            self.risk_per_trade = 0.02  # Default to 2%
            self.logger.warning(f"Invalid risk_per_trade value ({risk_per_trade}), using default 0.02 (2%)")
        else:
            self.risk_per_trade = risk_per_trade
        self.use_mt5_data = use_mt5_data
        self.auto_trade = auto_trade
        self.use_ml = use_ml and ML_AVAILABLE
        self.use_smc = use_smc and SMC_AVAILABLE
        
        # ML Components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_trained = False
        self.prediction_threshold = 0.65  # Confidence threshold for ML signals
        
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
            strategy_config = {
                'UsePairs': [symbol],
                'LotSizeInitial': 0.01,
                'LotSizeReEntry': 0.01,
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
        
        # SMC Components
        self.smc_analyzer = None
        self.smc_signals = {}
        self.smc_summary = {}
        
        # Initialize components
        self.analysis_bot = None
        self.mt5_connector = None
        self.connected = False
        
        # Logger
        import logging as _logging
        self.logger = _logging.getLogger('mt5_trading_bot')

        # Trading state
        self.last_analysis = None
        self.current_position = None
        self.trade_history = []
        self.ml_predictions = []
        
        # Initialize MT5 connector
        if self.use_mt5_data or self.auto_trade:
            self.mt5_connector = MT5Connector()
            self.connect_mt5()
        
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
        if self.mt5_connector:
            self.connected = self.mt5_connector.connect()
            if self.connected:
                self.logger.info("MT5 connection established")
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
            try:
                from trading_bot import TradingBot
                default_account_size = 10000
                self.analysis_bot = TradingBot(self.symbol, self.timeframe, "forex", 
                                             default_account_size, self.risk_per_trade)
                if self.analysis_bot.fetch_data():
                    data = self.analysis_bot.data
                    self.logger.info("Using yfinance data")
                    self.debug_data_structure(data)
                    return data
            except Exception as e:
                self.logger.exception(f"Error getting market data: {e}")
        
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
                return None
            
            # Initialize analysis bot if needed
            if self.analysis_bot is None:
                from trading_bot import TradingBot
                self.analysis_bot = TradingBot(self.symbol, self.timeframe)
            
            # Update analysis bot data
            self.analysis_bot.data = data
            
            # 1. Traditional Technical Analysis
            self.logger.info("Performing Traditional Technical Analysis...")
            traditional_signals = self.analysis_bot.analyze_market_trend()
            
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
            position_size = self.mt5_connector.calculate_position_size_robust(
                symbol=self.symbol,
                order_type=signal_type,
                risk_percent=self.risk_per_trade,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                margin_buffer=0.95
            )
            print(f"📈 Robust position size: {position_size:.4f} lots")
        if not position_size or position_size <= 0:
            # Fallback minimum if calculations fail
            position_size = 0.01
            print(f"📈 Using minimum position size: {position_size:.4f} lots")
        
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
            
            print(f"💰 Account Balance: ${current_balance:,.2f}")
            print(f"💰 Account Equity: ${current_equity:,.2f}")
            print(f"💰 Used Margin: ${current_margin:,.2f}")
            print(f"💰 Free Margin: ${free_margin:,.2f}")
            
            # Check if we have sufficient free margin
            pos_size = float(signals.get('position_size') or 0)
            required_margin = pos_size * 1000  # Rough estimate: 1 lot = $1000 margin
            if free_margin < required_margin:
                print(f"❌ Insufficient free margin: ${free_margin:,.2f} available, ${required_margin:,.2f} required")
                print(f"   Position size: {pos_size:.4f} lots")
                
                # Try to reduce position size to fit available margin
                max_position_size = free_margin / 1000  # Maximum position size based on available margin
                if max_position_size >= 0.01:  # Minimum position size
                    print(f"🔄 Attempting to reduce position size to {max_position_size:.4f} lots")
                    original_size = pos_size
                    signals['position_size'] = max_position_size
                    if original_size > 0:
                        ratio = max_position_size / original_size
                        signals['risk_amount'] = (signals.get('risk_amount') or 0) * ratio
                        signals['potential_profit'] = (signals.get('potential_profit') or 0) * ratio
                    print(f"✅ Adjusted position size to {max_position_size:.4f} lots")
                else:
                    print(f"❌ Cannot place trade - insufficient margin even for minimum position size")
                    return None
            
            # Check current positions for this symbol
            positions = self.mt5_connector.get_positions()
            current_positions = []
            if positions:
                for pos in positions:
                    if pos['symbol'] == self.symbol:
                        current_positions.append(pos)
            
            print(f"📊 Current positions for {self.symbol}: {len(current_positions)}")
            
            # Check if we have too many positions (limit to 3 concurrent trades per symbol)
            max_positions = 3
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
                self.logger.info(
                    f"Trade executed successfully | order_id={result.get('order_id')} entry={result.get('price')} sl={result.get('sl')} tp={result.get('tp')} pos_no={len(current_positions) + 1} symbol={self.symbol} total_positions={len(current_positions) + 1}"
                )
                return result
            else:
                self.logger.warning("Order placement returned None; trade not executed")
                return None
            
        except Exception as e:
            print(f"❌ Error executing trade: {e}")
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
        
        # Perform traditional analysis
        if self.analysis_bot is None:
            from trading_bot import TradingBot
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
                    combined_signal['signal_sources'] = [strongest_smc['source']]
                
                elif combined_signal['signal_type'] == strongest_smc['type']:
                    # Signals agree - enhance strength
                    combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + strongest_smc['strength'] * 0.3)
                    combined_signal['signal_sources'].append(strongest_smc['source'])
                
                else:
                    # Signals conflict - use the stronger one
                    if strongest_smc['strength'] > combined_signal['signal_strength']:
                        combined_signal['signal_type'] = strongest_smc['type']
                        combined_signal['entry_price'] = strongest_smc['entry_price']
                        combined_signal['stop_loss'] = strongest_smc['stop_loss']
                        combined_signal['target'] = strongest_smc['target']
                        combined_signal['signal_strength'] = strongest_smc['strength']
                        combined_signal['signal_sources'] = [strongest_smc['source']]
        
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
                    combined_signal['signal_sources'] = ['Market_Structure']
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
                    combined_signal['signal_sources'].append('ML_Bullish')
                
                elif ml_signal < 0.4:  # Bearish
                    if combined_signal['signal_type'] == 'SELL':
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + ml_confidence * 0.2)
                    elif combined_signal['signal_type'] == 'BUY':
                        # ML disagrees with bullish signal
                        combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.5
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
                    combined_signal['signal_sources'].append('RL_Bullish')
                
                elif rl_action == 'sell' and rl_signal == 'SELL':
                    if combined_signal['signal_type'] == 'SELL':
                        combined_signal['signal_strength'] = min(1.0, combined_signal['signal_strength'] + rl_confidence * 0.3)
                    elif combined_signal['signal_type'] == 'BUY':
                        # RL strongly disagrees with bullish signal
                        combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.3
                    combined_signal['signal_sources'].append('RL_Bearish')
                
                elif rl_action == 'hold':
                    # RL suggests holding - reduce signal strength
                    combined_signal['signal_strength'] = combined_signal['signal_strength'] * 0.7
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
                        from reinforcement_learning_trader import TradeOutcome
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
    print(f"Auto Trading: {'✅ ENABLED' if auto_trade else '❌ DISABLED'}")
    print(f"Continuous Mode: {'✅ ENABLED' if continuous else '❌ DISABLED'}")
    print(f"Machine Learning: {'✅ ENABLED' if use_ml else '❌ DISABLED'}")
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

if __name__ == "__main__":
    main() 