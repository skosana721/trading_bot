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
    print("⚠️  ML libraries not available. Install with: pip install scikit-learn joblib")
    ML_AVAILABLE = False

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    print("⚠️  TA library not available. Install with: pip install ta")
    TA_AVAILABLE = False

from dotenv import load_dotenv
from trading_bot import TradingBot
from mt5_connector import MT5Connector

# Load environment variables
load_dotenv()

class MT5TradingBot:
    def __init__(self, symbol, timeframe, risk_per_trade=0.02, 
                 use_mt5_data=True, auto_trade=False, use_ml=True):
        """
        Initialize MT5 integrated trading bot
        
        Args:
            symbol (str): Symbol to trade (e.g., 'EURUSD', 'GBPUSD')
            timeframe (str): Timeframe for analysis ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            risk_per_trade (float): Risk per trade as percentage
            use_mt5_data (bool): Use MT5 data instead of yfinance
            auto_trade (bool): Enable automatic trading
            use_ml (bool): Enable machine learning features
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.use_mt5_data = use_mt5_data
        self.auto_trade = auto_trade
        self.use_ml = use_ml and ML_AVAILABLE
        self.account_size = None  # Will be set from MT5 account info
        
        # ML Components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_trained = False
        self.prediction_threshold = 0.65  # Confidence threshold for ML signals
        
        # Initialize components
        self.analysis_bot = None
        self.mt5_connector = None
        self.connected = False
        
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
            default_model_name = f"ml_model_{self.symbol}_{self.timeframe}.joblib"
            if self.load_ml_model(default_model_name):
                print(f"✅ Loaded existing ML model: {default_model_name}")
            else:
                print(f"📝 No existing ML model found. Will train new model when needed.")
    
    def connect_mt5(self):
        """Connect to MT5 and get account information"""
        if self.mt5_connector:
            self.connected = self.mt5_connector.connect()
            if self.connected:
                print("✅ MT5 connection established")
                # Get account size from MT5
                account_info = self.mt5_connector.get_account_summary()
                if account_info:
                    self.account_size = account_info.get('balance', 0)
                    print(f"💰 Account Balance: ${self.account_size:,.2f}")
                else:
                    print("⚠️  Could not get account balance, using default")
                    self.account_size = 10000  # Fallback default
            else:
                print("❌ MT5 connection failed")
                self.account_size = 10000  # Fallback default
        return self.connected
    
    def debug_data_structure(self, data):
        """
        Debug method to check data structure
        
        Args:
            data (pd.DataFrame): Data to debug
        """
        if data is None:
            print("❌ Data is None")
            return
        
        print(f"\n🔍 DATA STRUCTURE DEBUG:")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Data types:")
        for col in data.columns:
            print(f"     {col}: {data[col].dtype}")
        
        if len(data) > 0:
            print(f"   First row:")
            print(f"     {data.iloc[0].to_dict()}")
            print(f"   Last row:")
            print(f"     {data.iloc[-1].to_dict()}")
        
        # Check for required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"   ❌ Missing columns: {missing_columns}")
        else:
            print(f"   ✅ All required columns present")
    
    def get_market_data(self):
        """
        Get market data from MT5 or yfinance
        
        Returns:
            pd.DataFrame: Market data or None if failed
        """
        if self.use_mt5_data and self.connected:
            # Get data from MT5 - request more data for shorter timeframes
            if self.timeframe in ['1m', '5m']:
                data_count = 5000  # More data for short timeframes
            else:
                data_count = 1000  # Standard amount for longer timeframes
            
            data = self.mt5_connector.get_historical_data(self.symbol, self.timeframe, data_count)
            if data is not None:
                # Debug data structure
                self.debug_data_structure(data)
                
                # Ensure data has the correct column structure for the trading bot
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Check if we have the required columns
                if all(col in data.columns for col in required_columns):
                    print(f"✅ MT5 data structure is correct")
                    return data
                else:
                    print(f"⚠️  MT5 data missing required columns. Available: {list(data.columns)}")
                    print("⚠️  Falling back to yfinance")
        
        # Fallback to yfinance
        try:
            self.analysis_bot = TradingBot(self.symbol, self.timeframe, "forex", 
                                         self.account_size or 10000, self.risk_per_trade)
            if self.analysis_bot.fetch_data():
                data = self.analysis_bot.data
                print("✅ Using yfinance data")
                self.debug_data_structure(data)
                return data
        except Exception as e:
            print(f"❌ Error getting market data: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def create_technical_features(self, data):
        """
        Create advanced technical indicators as ML features
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if not TA_AVAILABLE:
            print("❌ TA library not available for technical indicators")
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
        
        return df
    
    def create_target_variable(self, data, lookforward=5):
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
    
    def prepare_ml_features(self, data):
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
    
    def train_ml_model(self, data):
        """
        Train machine learning model
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            bool: True if training successful
        """
        if not self.use_ml:
            print("❌ ML features disabled")
            return False
        
        try:
            print("🤖 Training ML model...")
            
            # Create technical features
            data_with_features = self.create_technical_features(data)
            
            # Prepare ML features
            X, y, feature_names = self.prepare_ml_features(data_with_features)
            
            if len(X) < 100:
                print("❌ Insufficient data for ML training (need at least 100 samples)")
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
            
            print(f"✅ ML Model Trained Successfully")
            print(f"   Training Accuracy: {train_score:.3f}")
            print(f"   Test Accuracy: {test_score:.3f}")
            print(f"   Cross-Validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"   Features Used: {len(feature_names)}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.ml_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            
            self.model_trained = True
            return True
            
        except Exception as e:
            print(f"❌ ML training failed: {e}")
            return False
    
    def get_ml_prediction(self, data):
        """
        Get ML prediction for current market conditions
        
        Args:
            data (pd.DataFrame): Current market data
            
        Returns:
            dict: ML prediction results
        """
        if not self.use_ml or not self.model_trained:
            return None
        
        try:
            # Create technical features
            data_with_features = self.create_technical_features(data)
            
            # Get latest data point
            latest_data = data_with_features[self.feature_columns].iloc[-1:].copy()
            
            # Check for NaN values
            if latest_data.isnull().any().any():
                print("⚠️  Missing features for ML prediction")
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
            print(f"❌ ML prediction failed: {e}")
            return None
    
    def analyze_market(self):
        """
        Perform advanced market analysis using the new trading rules
        
        Returns:
            dict: Analysis results or None if failed
        """
        try:
            # Get market data
            data = self.get_market_data()
            if data is None:
                print("❌ No market data available")
                return None
            
            # Verify data structure
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print(f"❌ Data missing required columns. Available: {list(data.columns)}")
                return None
            
            # Create analysis bot if not exists
            if self.analysis_bot is None:
                self.analysis_bot = TradingBot(self.symbol, self.timeframe, "forex", 
                                             self.account_size or 10000, self.risk_per_trade)
            
            # Assign data to the analysis bot
            self.analysis_bot.data = data.copy()
            
            # Perform advanced analysis using new trading rules
            print(f"\n🔍 PERFORMING ADVANCED MARKET ANALYSIS FOR {self.symbol}")
            print("=" * 70)
            
            analysis = self.analysis_bot.analyze_uptrend_advanced()
            self.last_analysis = analysis
            
            if analysis and analysis['uptrend_confirmed']:
                print(f"\n✅ ANALYSIS COMPLETE - UPTREND CONFIRMED")
                print(f"   Symbol: {self.symbol}")
                print(f"   Timeframe: {self.timeframe}")
                print(f"   Overall Strength: {analysis['overall_strength']:.1f}/100")
                print(f"   Entry Conditions: {', '.join(analysis['entry_conditions']) if analysis['entry_conditions'] else 'None'}")
            else:
                print(f"\n❌ ANALYSIS COMPLETE - NO STRONG UPTREND")
                if analysis:
                    print(f"   Overall Strength: {analysis['overall_strength']:.1f}/100")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing market: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_trading_signals(self, analysis):
        """
        Get advanced trading signals based on new trading rules
        
        Args:
            analysis (dict): Advanced market analysis results
            
        Returns:
            dict: Trading signals or None if no signals
        """
        if not analysis or not analysis['uptrend_confirmed']:
            print("❌ No strong uptrend confirmed - no trading signals")
            return None
        
        # Check for entry conditions
        if not analysis['entry_conditions']:
            print("ℹ️  No entry conditions met - waiting for breakout or continuation pattern")
            return None
        
        print(f"\n🎯 GENERATING TRADING SIGNALS")
        print("=" * 50)
        
        # Get current market price from MT5
        current_price = None
        if self.connected:
            tick = self.mt5_connector.get_symbol_info(self.symbol)
            if tick:
                # Use ask price for buy orders
                current_price = tick.get('ask', None)
        
        if current_price is None:
            # Fallback to last close price
            if self.analysis_bot and self.analysis_bot.data is not None:
                current_price = self.analysis_bot.data['Close'].iloc[-1]
        
        if current_price is None:
            print("❌ Cannot get current market price")
            return None
        
        # Initialize signal parameters
        entry_price = current_price
        stop_loss_price = None
        target_price = None
        signal_type = "BUY"
        signal_strength = analysis['overall_strength']
        
        # Get symbol info for proper calculations
        symbol_info = None
        if self.connected:
            symbol_info = self.mt5_connector.get_symbol_info(self.symbol)
        
        point_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001
        digits = symbol_info.get('digits', 5) if symbol_info else 5
        
        # Process breakout and retest signals
        if analysis['breakout_signals'] and analysis['breakout_signals']['entry_signal']:
            print("🎯 Using breakout and retest signals")
            breakout = analysis['breakout_signals']
            entry_price = breakout['entry_price']
            stop_loss_price = breakout['stop_loss']
            target_price = breakout['target']
            signal_type = "BREAKOUT_BUY"
        
        # Process continuation pattern signals
        elif analysis['continuation_patterns']:
            patterns = analysis['continuation_patterns']
            if patterns['bullish_flag'] or patterns['pennant'] or patterns['inverted_head_shoulders']:
                print("🎯 Using continuation pattern signals")
                
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
                    atr = self.calculate_atr(self.analysis_bot.h4_data, period=14)
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
            print("🎯 Using trendline-based signals")
            
            # Use H4 trendlines for entry signals
            if analysis['h4_trendlines'] and 'hl_trendline' in analysis['h4_trendlines']:
                hl_trendline = analysis['h4_trendlines']['hl_trendline']
                if hl_trendline['touches'] >= 3:  # Strong support
                    # Entry on support touch
                    entry_price = current_price
                    stop_loss_price = hl_trendline['end_price'] * 0.995  # Below support
                    
                    # Calculate target based on trend strength
                    if analysis['h4_trendlines'] and 'hh_trendline' in analysis['h4_trendlines']:
                        hh_trendline = analysis['h4_trendlines']['hh_trendline']
                        resistance_level = hh_trendline['end_price']
                        target_price = resistance_level * 1.005  # Above resistance
                    else:
                        # Use ATR-based target
                        atr = self.calculate_atr(self.analysis_bot.h4_data, period=14)
                        target_price = entry_price + (atr * 6)  # 1:3 ratio
                    
                    signal_type = "TRENDLINE_BUY"
        
        # Validate and adjust stop loss and target
        if stop_loss_price is None or target_price is None:
            print("❌ Could not calculate proper stop loss or target")
            return None
        
        # Ensure minimum stop level compliance
        if symbol_info and 'trade_stops_level' in symbol_info:
            min_stop_level_points = symbol_info['trade_stops_level']
            min_stop_distance = min_stop_level_points * point_value
            
            current_stop_distance = abs(entry_price - stop_loss_price)
            if current_stop_distance < min_stop_distance:
                print(f"⚠️  Adjusting stop loss to meet broker minimum ({min_stop_level_points} points)")
                if entry_price > stop_loss_price:  # Long position
                    stop_loss_price = entry_price - min_stop_distance
                else:  # Short position
                    stop_loss_price = entry_price + min_stop_distance
        
        # Calculate risk and reward
        risk_amount = abs(entry_price - stop_loss_price)
        reward_amount = abs(target_price - entry_price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Calculate position size based on risk
        risk_per_trade_amount = self.account_size * self.risk_per_trade
        position_size = risk_per_trade_amount / risk_amount if risk_amount > 0 else 0
        
        # Get ML prediction for confirmation
        ml_prediction = None
        if self.use_ml and self.analysis_bot:
            ml_prediction = self.get_ml_prediction(self.analysis_bot.h4_data)
        
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
            'entry_conditions': analysis['entry_conditions'],
            'ml_prediction': ml_prediction,
            'analysis_summary': {
                'overall_strength': analysis['overall_strength'],
                'weekly_strength': analysis['weekly_analysis']['trend_strength'],
                'daily_strength': analysis['daily_analysis']['trend_strength'],
                'h4_strength': analysis['h4_analysis']['trend_strength']
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
    
    def calculate_atr(self, data, period=14):
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
        ml_prediction = None
        if self.use_ml and self.analysis_bot and self.analysis_bot.data is not None:
            ml_prediction = self.get_ml_prediction(self.analysis_bot.data)
        
        # Calculate position size
        position_size = None
        if self.connected:
            position_size = self.mt5_connector.calculate_position_size(
                self.account_size * self.risk_per_trade,
                entry_price,
                stop_loss_price,
                self.symbol
            )
        
        # Determine signal strength and type
        signal_strength = 1.0  # Base strength from traditional analysis
        signal_type = 'BUY'
        
        if ml_prediction:
            # Combine traditional and ML signals
            ml_confidence = ml_prediction['confidence']
            ml_signal = ml_prediction['prediction']
            
            if ml_signal == 1 and ml_confidence >= self.prediction_threshold:
                # ML confirms buy signal
                signal_strength = min(1.0, signal_strength + ml_confidence * 0.5)
                print(f"🤖 ML Confirms BUY Signal (Confidence: {ml_confidence:.2f})")
            elif ml_signal == 0 and ml_confidence >= self.prediction_threshold:
                # ML disagrees with traditional analysis
                signal_strength = signal_strength * 0.3
                print(f"🤖 ML Disagrees with Signal (Confidence: {ml_confidence:.2f})")
            else:
                print(f"🤖 ML Neutral (Confidence: {ml_confidence:.2f})")
        
        # Only generate signal if strength is sufficient
        if signal_strength < 0.5:
            print(f"⚠️  Signal strength too low: {signal_strength:.2f}")
            return None
        
        return {
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'target': target_price,
            'position_size': position_size,
            'risk_amount': self.account_size * self.risk_per_trade,
            'potential_profit': (target_price - entry_price) * (position_size * 100000) if position_size is not None else 0,
            'timeframe': self.timeframe,
            'analysis': analysis,
            'ml_prediction': ml_prediction,
            'signal_strength': signal_strength
        }
    
    def execute_trade(self, signals):
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
            # Check current positions for this symbol
            positions = self.mt5_connector.get_positions()
            current_positions = []
            if positions:
                for pos in positions:
                    if pos['symbol'] == self.symbol:
                        current_positions.append(pos)
            
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
            
            # Calculate position size based on remaining risk allocation
            total_risk_used = sum([pos.get('risk_amount', 0) for pos in current_positions])
            available_risk = (self.account_size or 10000) * self.risk_per_trade - total_risk_used
            
            if available_risk <= 0:
                print(f"⚠️  No available risk allocation for {self.symbol}")
                return None
            
            # Adjust position size based on available risk
            adjusted_signals = signals.copy()
            if available_risk < signals['risk_amount']:
                risk_ratio = available_risk / signals['risk_amount']
                adjusted_signals['position_size'] = signals['position_size'] * risk_ratio
                adjusted_signals['risk_amount'] = available_risk
                adjusted_signals['potential_profit'] = signals['potential_profit'] * risk_ratio
                print(f"📊 Adjusted position size due to risk allocation: {risk_ratio:.2f}")
            
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
                result['risk_amount'] = adjusted_signals['risk_amount']
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'action': 'OPEN',
                    'result': result,
                    'signals': adjusted_signals
                })
                
                print(f"✅ Trade executed successfully")
                print(f"   Order ID: {result['order_id']}")
                print(f"   Entry: {result['price']}")
                print(f"   Stop Loss: {result['sl']}")
                print(f"   Target: {result['tp']}")
                print(f"   Position #{len(current_positions) + 1} for {self.symbol}")
                print(f"   Total positions: {len(current_positions) + 1}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return None
    
    def monitor_positions(self):
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
    
    def close_all_positions(self):
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
    
    def close_symbol_positions(self, symbol=None):
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
    
    def run_analysis_cycle(self):
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
            # Try to load existing model
            default_model_name = f"ml_model_{self.symbol}_{self.timeframe}.joblib"
            if self.load_ml_model(default_model_name):
                print(f"✅ Loaded existing ML model: {default_model_name}")
            else:
                print("🤖 Training new ML model with historical data...")
                self.train_ml_model(data)
        
        # Perform traditional analysis
        if self.analysis_bot is None:
            self.analysis_bot = TradingBot(self.symbol, self.timeframe, "forex", 
                                         self.account_size or 10000, self.risk_per_trade)
        
        self.analysis_bot.data = data.copy()
        analysis = self.analysis_bot.analyze_uptrend()
        
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
    
    def run_continuous_monitoring(self, interval_minutes=5, max_cycles=None):
        """
        Run continuous market monitoring
        
        Args:
            interval_minutes (int): Minutes between analysis cycles
            max_cycles (int): Maximum number of cycles (None for unlimited)
        """
        print("="*60)
        print("STARTING CONTINUOUS MARKET MONITORING")
        print("="*60)
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Interval: {interval_minutes} minutes")
        print(f"Auto Trading: {'✅ ENABLED' if self.auto_trade else '❌ DISABLED'}")
        print("="*60)
        
        cycle_count = 0
        
        try:
            while True:
                if max_cycles and cycle_count >= max_cycles:
                    print(f"\n🛑 Reached maximum cycles ({max_cycles})")
                    break
                
                cycle_count += 1
                print(f"\n🔄 Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run analysis cycle
                analysis = self.run_analysis_cycle()
                
                # Monitor positions
                if self.connected:
                    self.monitor_positions()
                
                # Wait for next cycle
                if max_cycles is None or cycle_count < max_cycles:
                    print(f"\n⏳ Waiting {interval_minutes} minutes for next cycle...")
                    time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring: {e}")
        finally:
            # Close all positions if auto trading was enabled
            if self.auto_trade and self.connected:
                print("\n🔒 Closing all positions...")
                self.close_all_positions()
            
            # Disconnect from MT5
            if self.mt5_connector:
                self.mt5_connector.disconnect()
            
            print("\n✅ Monitoring session ended")
    
    def get_trading_summary(self):
        """
        Get trading session summary
        
        Returns:
            dict: Trading summary
        """
        summary = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'account_size': self.account_size,
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
    
    def save_ml_model(self, filename=None):
        """Save trained ML model to file"""
        if not self.model_trained:
            print("❌ No trained model to save")
            return False
        
        if filename is None:
            filename = f"ml_model_{self.symbol}_{self.timeframe}.joblib"
        
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
            print(f"✅ ML model saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False
    
    def load_ml_model(self, filename):
        """Load trained ML model from file"""
        try:
            model_data = joblib.load(filename)
            
            self.ml_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_trained = True
            
            print(f"✅ ML model loaded from {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

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
        print(f"\n📈 Trading Summary:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Open Positions: {summary['open_positions']}")
        print(f"   Total Profit: ${summary['total_profit']:.2f}")
        print(f"   ML Enabled: {'✅' if summary['ml_enabled'] else '❌'}")
        print(f"   Model Trained: {'✅' if summary['model_trained'] else '❌'}")
    
    # Save ML model if trained
    if bot.use_ml and bot.model_trained:
        bot.save_ml_model()
    
    # Cleanup
    if bot.mt5_connector:
        bot.mt5_connector.disconnect()

if __name__ == "__main__":
    main() 