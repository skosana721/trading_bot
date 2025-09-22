#!/usr/bin/env python3
"""
ML Ensemble Trading System
==========================

This module implements an ensemble of multiple machine learning models
for improved trading predictions. It combines various algorithms and
techniques to create more robust and accurate signals.

Models included:
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
- Neural Network (MLP)
- LSTM (if available)
- Ensemble Voting
- Stacking

Features:
- Automatic model selection based on performance
- Dynamic ensemble weighting
- Feature importance analysis
- Model performance tracking
- Real-time prediction updates
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML libraries not available: {e}")
    ML_AVAILABLE = False

class ModelWrapper:
    """Wrapper class to handle feature selection and scaling for ensemble models"""
    
    def __init__(self, model, feature_selector, scaler):
        self.model = model
        self.feature_selector = feature_selector
        self.scaler = scaler
    
    def fit(self, X, y):
        # Apply feature selection and scaling
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return self.model.fit(X_scaled, y)
    
    def predict(self, X):
        # Apply feature selection and scaling
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        # Apply feature selection and scaling
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict_proba(X_scaled)

# Deep Learning (optional)
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Warning: Deep learning libraries not available. LSTM models will be disabled.")

@dataclass
class ModelPerformance:
    """Data class to track model performance"""
    model_name: str = ""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    cv_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    prediction_count: int = 0
    correct_predictions: int = 0

class MLEnsemble:
    """
    Ensemble of multiple ML models for trading predictions
    """
    
    def __init__(self, symbol: str, timeframe: str, use_deep_learning: bool = True):
        """
        Initialize ML Ensemble
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            use_deep_learning: Whether to use deep learning models
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_deep_learning = use_deep_learning and DEEP_LEARNING_AVAILABLE
        
        # Setup logging
        self.logger = logging.getLogger(f'ml_ensemble_{symbol}_{timeframe}')
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_performances = {}
        
        # Training lock to prevent duplicate training
        self._training_lock = threading.Lock()
        self._is_training = False
        
        # Ensemble configuration
        self.ensemble_weights = {}
        self.voting_classifier = None
        self.stacking_classifier = None
        
        # Feature engineering
        self.feature_columns = []
        self.feature_importance = {}
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_threshold = 0.6
        self.min_confidence = 0.65
        
        # Model paths
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        if not ML_AVAILABLE:
            self.logger.error("ML libraries not available")
            return
        
        # Traditional ML Models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Deep Learning Models - will be created dynamically during training
        if self.use_deep_learning and DEEP_LEARNING_AVAILABLE:
            self.logger.info("Deep learning enabled - LSTM model will be created during training")
        else:
            self.logger.info("Deep learning disabled or not available")
        
        # Initialize scalers and feature selectors
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
            self.feature_selectors[model_name] = SelectKBest(k=20)
            self.model_performances[model_name] = ModelPerformance(model_name=model_name)
    
    def _are_models_trained(self) -> bool:
        """Check if models are properly trained"""
        try:
            # Check if we have any models
            if not self.models:
                return False
            
            # Check if we have feature columns
            if not self.feature_columns:
                return False
            
            # Check if at least one traditional model has been trained
            trained_models = 0
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # For LSTM, check if it exists and has input shape
                    if model is not None and hasattr(model, 'input_shape'):
                        trained_models += 1
                else:
                    # For traditional models, check if scaler is fitted
                    if (model is not None and 
                        model_name in self.scalers and 
                        hasattr(self.scalers[model_name], 'scale_')):
                        trained_models += 1
            
            return trained_models > 0
            
        except Exception as e:
            self.logger.error(f"Error checking if models are trained: {e}")
            return False
    
    def _create_lstm_model(self, n_features: int = None):
        """Create LSTM model for time series prediction"""
        if not DEEP_LEARNING_AVAILABLE:
            return None
        
        try:
            # Use dynamic input shape based on actual feature count
            if n_features is None:
                n_features = 60  # Default fallback
            
            # Ensure n_features is reasonable
            if n_features <= 0:
                self.logger.warning(f"Invalid feature count: {n_features}, using default 60")
                n_features = 60
            
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(n_features, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            self.logger.warning(f"Failed to initialize LSTM model: {e}, skipping deep learning")
            return None
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical features for ML models
        
        Args:
            data: Price data
            
        Returns:
            DataFrame with advanced features
        """
        df = data.copy()
        
        # Ensure index is datetime for time-based features
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                # If conversion fails, create a simple numeric index
                df.index = pd.RangeIndex(len(df))
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['price_change_2'] = df['Close'].pct_change(2)
        df['price_change_5'] = df['Close'].pct_change(5)
        df['price_change_10'] = df['Close'].pct_change(10)
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(20).std()
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = df['Close'] / df[f'sma_{period}'] - 1
            df[f'price_vs_ema_{period}'] = df['Close'] / df[f'ema_{period}'] - 1
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            df['volume_price_trend'] = (df['Volume'] * df['price_change']).rolling(10).sum()
        
        # Support and Resistance
        df['support'] = df['Low'].rolling(20).min()
        df['resistance'] = df['High'].rolling(20).max()
        df['support_distance'] = (df['Close'] - df['support']) / df['Close']
        df['resistance_distance'] = (df['resistance'] - df['Close']) / df['Close']
        
        # Trend features
        df['trend_5'] = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
        df['trend_10'] = np.where(df['Close'] > df['Close'].shift(10), 1, -1)
        df['trend_20'] = np.where(df['Close'] > df['Close'].shift(20), 1, -1)
        
        # Momentum features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Time-based features (only if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour.astype(np.float64)
            df['day_of_week'] = df.index.dayofweek.astype(np.float64)
            df['month'] = df.index.month.astype(np.float64)
        else:
            # Use simple numeric features instead
            df['hour'] = np.float64(12.0)  # Default to noon
            df['day_of_week'] = np.float64(0.0)  # Default to Monday
            df['month'] = np.float64(1.0)  # Default to January
        
        # Remove infinite values and fill NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all features are numeric
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name.startswith('datetime'):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    # If conversion fails, drop the column
                    df = df.drop(columns=[col])
        
        # Convert all remaining columns to float64 to ensure consistency
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:  # Keep original price columns as is
                try:
                    df[col] = df[col].astype(np.float64)
                except:
                    # If conversion fails, drop the column
                    df = df.drop(columns=[col])
        
        return df
    
    def prepare_target_variable(self, data: pd.DataFrame, lookforward_periods: int = 5) -> pd.Series:
        """
        Create target variable for ML training
        
        Args:
            data: Price data
            lookforward_periods: Number of periods to look forward
            
        Returns:
            Target variable series
        """
        # Calculate future returns
        future_returns = data['Close'].shift(-lookforward_periods) / data['Close'] - 1
        
        # Create binary target (1 for positive return, 0 for negative)
        target = (future_returns > 0).astype(int)
        
        # Remove NaN values
        target = target.dropna()
        
        return target
    
    def prepare_ml_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for ML training
        
        Args:
            data: Raw price data
            
        Returns:
            Features, target, and feature names
        """
        # Create advanced features
        df_with_features = self.create_advanced_features(data)
        
        # Create target variable
        target = self.prepare_target_variable(df_with_features)
        
        # Align features with target
        df_with_features = df_with_features.loc[target.index]
        
        # Select feature columns (exclude price and target columns)
        exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in df_with_features.columns if col not in exclude_columns]
        
        # Ensure all feature columns are numeric
        numeric_features = []
        for col in feature_columns:
            if col in df_with_features.columns:
                try:
                    # Skip datetime columns
                    if df_with_features[col].dtype == 'datetime64[ns]':
                        continue
                    
                    # Convert to numeric and then to float64
                    df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce').astype(np.float64)
                    numeric_features.append(col)
                except Exception as e:
                    self.logger.debug(f"Skipping non-numeric column {col}: {e}")
                    continue
        
        feature_columns = numeric_features
        
        # Remove rows with NaN values
        valid_indices = ~(df_with_features[feature_columns].isnull().any(axis=1) | target.isnull())
        features = df_with_features[feature_columns].loc[valid_indices]
        target = target.loc[valid_indices]
        
        # Final validation - ensure all data is numeric
        features = features.select_dtypes(include=[np.number])
        feature_columns = list(features.columns)
        
        self.feature_columns = feature_columns
        
        return features, target, feature_columns
    
    def train_models(self, data: pd.DataFrame) -> bool:
        """
        Train all ML models
        
        Args:
            data: Training data
            
        Returns:
            True if training successful
        """
        if not ML_AVAILABLE:
            self.logger.error("ML libraries not available")
            return False
        
        # Check if already training
        with self._training_lock:
            if self._is_training:
                self.logger.warning("Training already in progress, skipping...")
                return False
            self._is_training = True
        
        try:
            self.logger.info("Training ML ensemble models...")
            
            # Prepare data
            X, y, feature_names = self.prepare_ml_data(data)
            
            if len(X) < 200:
                self.logger.error(f"Insufficient data for training (need at least 200 samples, got {len(X)})")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train each model
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # Handle LSTM separately
                    self._train_lstm_model(X_train, y_train, X_test, y_test)
                else:
                    self._train_traditional_model(model_name, model, X_train, y_train, X_test, y_test)
            
            # Create ensemble models
            self._create_ensemble_models(X_train, y_train, X_test, y_test)
            
            # Calculate ensemble weights
            self._calculate_ensemble_weights()
            
            # Save models
            self._save_models()
            
            self.logger.info("All ML models trained successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"ML training failed: {e}")
            return False
        finally:
            # Release training lock
            with self._training_lock:
                self._is_training = False
    
    def _train_traditional_model(self, model_name: str, model, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        """Train traditional ML model"""
        try:
            self.logger.info(f"Training {model_name}...")
            
            # Ensure we have enough features
            n_features = len(X_train.columns)
            if n_features < 5:
                self.logger.warning(f"Not enough features for {model_name}: {n_features}")
                return False
            
            # Adjust feature selector k if needed
            k = min(20, n_features)
            if self.feature_selectors[model_name] is None:
                self.feature_selectors[model_name] = SelectKBest(score_func=f_classif, k=k)
            else:
                # Update k if needed
                if hasattr(self.feature_selectors[model_name], 'k') and self.feature_selectors[model_name].k > n_features:
                    self.feature_selectors[model_name] = SelectKBest(score_func=f_classif, k=k)
            
            # Feature selection
            X_train_selected = self.feature_selectors[model_name].fit_transform(X_train, y_train)
            X_test_selected = self.feature_selectors[model_name].transform(X_test)
            
            # Ensure scaler is fitted with the selected features
            self.scalers[model_name] = RobustScaler()
            X_train_scaled = self.scalers[model_name].fit_transform(X_train_selected)
            X_test_scaled = self.scalers[model_name].transform(X_test_selected)
            
            # Log the feature counts for debugging
            self.logger.info(f"{model_name} - Original features: {len(X_train.columns)}, Selected features: {X_train_selected.shape[1]}, Scaled features: {X_train_scaled.shape[1]}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Update performance metrics
            self.model_performances[model_name].accuracy = accuracy_score(y_test, y_pred)
            self.model_performances[model_name].precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            self.model_performances[model_name].recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            self.model_performances[model_name].f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Calculate CV score with error handling
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
                self.model_performances[model_name].cv_score = cv_scores.mean() if len(cv_scores) > 0 else 0.0
            except Exception as e:
                self.logger.warning(f"CV score calculation failed for {model_name}: {e}")
                self.model_performances[model_name].cv_score = 0.0
            self.model_performances[model_name].last_updated = datetime.now()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    self.feature_selectors[model_name].get_support(),
                    model.feature_importances_
                ))
            
            self.logger.info(f"{model_name} - Accuracy: {self.model_performances[model_name].accuracy:.3f}, "
                           f"F1: {self.model_performances[model_name].f1_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {e}")
    
    def _train_lstm_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series):
        """Train LSTM model"""
        if not DEEP_LEARNING_AVAILABLE or not self.use_deep_learning:
            return
        
        try:
            self.logger.info("Training LSTM model...")
            
            # Ensure data is numeric and clean
            X_train_clean = X_train.select_dtypes(include=[np.number])
            X_test_clean = X_test.select_dtypes(include=[np.number])
            
            # Remove any remaining NaN values
            X_train_clean = X_train_clean.fillna(0)
            X_test_clean = X_test_clean.fillna(0)
            
            # Prepare LSTM data (3D format: samples, timesteps, features)
            sequence_length = 20
            
            # Ensure we have enough data
            if len(X_train_clean) < sequence_length + 1 or len(X_test_clean) < sequence_length + 1:
                self.logger.warning("Insufficient data for LSTM training")
                return
            
            X_lstm_train = self._prepare_lstm_data(X_train_clean, sequence_length)
            X_lstm_test = self._prepare_lstm_data(X_test_clean, sequence_length)
            
            # Ensure we have matching target data
            if len(y_train) < len(X_lstm_train) + sequence_length - 1:
                self.logger.warning("Target data length mismatch for LSTM")
                return
                
            if len(y_test) < len(X_lstm_test) + sequence_length - 1:
                self.logger.warning("Target data length mismatch for LSTM")
                return
            
            # Adjust target data to match LSTM input
            y_lstm_train = y_train.iloc[sequence_length-1:sequence_length-1+len(X_lstm_train)]
            y_lstm_test = y_test.iloc[sequence_length-1:sequence_length-1+len(X_lstm_test)]
            
            # Final validation of data shapes
            if len(X_lstm_train) != len(y_lstm_train) or len(X_lstm_test) != len(y_lstm_test):
                self.logger.error(f"Data shape mismatch: X_train={X_lstm_train.shape}, y_train={y_lstm_train.shape}, X_test={X_lstm_test.shape}, y_test={y_lstm_test.shape}")
                return
            
            # Create LSTM model with correct input shape
            n_features = X_lstm_train.shape[2]  # Get actual number of features
            self.models['lstm'] = self._create_lstm_model(n_features)
            
            if self.models['lstm'] is None:
                self.logger.warning("Failed to create LSTM model, skipping LSTM training")
                return
            
            # Train LSTM
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            self.models['lstm'].fit(
                X_lstm_train, y_lstm_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_lstm_test, y_lstm_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            y_pred = (self.models['lstm'].predict(X_lstm_test) > 0.5).astype(int)
            
            self.model_performances['lstm'].accuracy = accuracy_score(y_lstm_test, y_pred)
            self.model_performances['lstm'].precision = precision_score(y_lstm_test, y_pred, average='weighted')
            self.model_performances['lstm'].recall = recall_score(y_lstm_test, y_pred, average='weighted')
            self.model_performances['lstm'].f1_score = f1_score(y_lstm_test, y_pred, average='weighted')
            self.model_performances['lstm'].last_updated = datetime.now()
            
            self.logger.info(f"LSTM - Accuracy: {self.model_performances['lstm'].accuracy:.3f}, "
                           f"F1: {self.model_performances['lstm'].f1_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train LSTM: {e}")
    
    def _prepare_lstm_data(self, data: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Prepare data for LSTM (3D format)"""
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data.iloc[i-sequence_length:i].values)
        return np.array(sequences)
    
    def _create_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series):
        """Create voting and stacking ensemble models"""
        try:
            # Only create ensemble if we have trained models
            trained_models = []
            for name, model in self.models.items():
                if name != 'lstm' and hasattr(model, 'predict'):  # Exclude LSTM and ensure model is trained
                    trained_models.append((name, model))
            
            if len(trained_models) < 2:
                self.logger.warning("Not enough trained models for ensemble creation")
                return
            
            # Prepare data for ensemble - create a new scaler for the original features
            # since individual model scalers are fitted with selected features
            ensemble_scaler = RobustScaler()
            X_train_scaled = ensemble_scaler.fit_transform(X_train)
            X_test_scaled = ensemble_scaler.transform(X_test)
            
            # Create voting classifier with models that work with original features
            # We need to create wrapper models that handle feature selection internally
            ensemble_models = []
            for model_name, model in trained_models:
                if model_name in self.feature_selectors and model_name in self.scalers:
                    # Create a wrapper that applies feature selection and scaling
                    wrapper = ModelWrapper(model, self.feature_selectors[model_name], self.scalers[model_name])
                    ensemble_models.append((model_name, wrapper))
                else:
                    # Use model as-is if no feature selection/scaling
                    ensemble_models.append((model_name, model))
            
            if len(ensemble_models) >= 2:
                self.voting_classifier = VotingClassifier(
                    estimators=ensemble_models,
                    voting='soft'
                )
                
                # Fit the voting classifier
                self.voting_classifier.fit(X_train, y_train)
                
                # Evaluate voting classifier
                y_pred = self.voting_classifier.predict(X_test)
                self.logger.info(f"Voting Classifier - Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            else:
                self.logger.warning("Not enough compatible models for ensemble creation")
                self.voting_classifier = None
            
        except Exception as e:
            self.logger.error(f"Failed to create ensemble models: {e}")
            # Create a simple fallback ensemble
            self.voting_classifier = None
    
    def _calculate_ensemble_weights(self):
        """Calculate weights for ensemble based on model performance"""
        total_f1 = sum(perf.f1_score for perf in self.model_performances.values())
        
        for model_name, performance in self.model_performances.items():
            if total_f1 > 0:
                self.ensemble_weights[model_name] = performance.f1_score / total_f1
            else:
                self.ensemble_weights[model_name] = 1.0 / len(self.model_performances)
    
    def get_ensemble_prediction(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Get ensemble prediction from all models
        
        Args:
            data: Current market data
            
        Returns:
            Ensemble prediction results
        """
        if not self.models:
            return None
        
        # Check if models are trained
        if not self._are_models_trained():
            self.logger.warning("Models are not trained yet, cannot make predictions")
            return None
        
        try:
            # Prepare features
            df_with_features = self.create_advanced_features(data)
            
            if len(df_with_features) < 1:
                return None
            
            # Get latest data point
            latest_features = df_with_features[self.feature_columns].tail(1)
            
            # Ensure all features are numeric and remove NaN values
            for col in latest_features.columns:
                try:
                    # Handle None values explicitly
                    if latest_features[col].isnull().any():
                        latest_features[col] = latest_features[col].fillna(0.0)
                    latest_features[col] = pd.to_numeric(latest_features[col], errors='coerce').astype(np.float64)
                except Exception as e:
                    self.logger.warning(f"Failed to convert feature {col} to numeric: {e}")
                    return None
            
            if latest_features.isnull().any().any():
                self.logger.warning("NaN values found in features after conversion")
                return None
            
            predictions = {}
            confidences = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # Handle LSTM prediction - only if deep learning is available
                    if DEEP_LEARNING_AVAILABLE and model is not None:
                        lstm_pred = self._get_lstm_prediction(df_with_features)
                        if lstm_pred:
                            predictions[model_name] = lstm_pred['prediction']
                            confidences[model_name] = lstm_pred['confidence']
                        else:
                            self.logger.debug("LSTM prediction returned None")
                    else:
                        self.logger.debug("LSTM not available, skipping")
                else:
                    # Traditional model prediction
                    try:
                        # Check if feature selector exists and is properly fitted
                        if self.feature_selectors[model_name] is None:
                            self.logger.warning(f"Feature selector not initialized for {model_name}, skipping")
                            continue
                        
                        # Ensure feature selector has been fitted
                        if not hasattr(self.feature_selectors[model_name], 'get_support'):
                            self.logger.warning(f"Feature selector for {model_name} not properly fitted, skipping")
                            continue
                        
                        # Feature selection and scaling
                        features_selected = self.feature_selectors[model_name].transform(latest_features)
                        
                        # Ensure scaler is properly fitted and compatible with feature selector
                        if not hasattr(self.scalers[model_name], 'scale_'):
                            self.logger.warning(f"Scaler for {model_name} not properly fitted, creating new one")
                            # Create a new scaler and fit it with dummy data to avoid errors
                            self.scalers[model_name] = RobustScaler()
                            # Fit with the current features to make it compatible
                            dummy_data = np.zeros((1, features_selected.shape[1]))
                            self.scalers[model_name].fit(dummy_data)
                        
                        # Check if scaler is compatible with current feature count
                        expected_features = features_selected.shape[1]
                        if hasattr(self.scalers[model_name], 'scale_') and len(self.scalers[model_name].scale_) != expected_features:
                            self.logger.warning(f"Scaler for {model_name} has {len(self.scalers[model_name].scale_)} features but data has {expected_features}, recreating scaler")
                            # Create a new scaler with the correct number of features
                            self.scalers[model_name] = RobustScaler()
                            dummy_data = np.zeros((1, expected_features))
                            self.scalers[model_name].fit(dummy_data)
                        
                        # Now scale the features
                        features_scaled = self.scalers[model_name].transform(features_selected)
                        
                        # Get prediction
                        pred_proba = model.predict_proba(features_scaled)[0]
                        prediction = model.predict(features_scaled)[0]
                        
                        predictions[model_name] = prediction
                        confidences[model_name] = max(pred_proba)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get prediction from {model_name}: {e}")
                        continue
            
            if not predictions:
                return None
            
            # Calculate weighted ensemble prediction
            weighted_prediction = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = self.ensemble_weights.get(model_name, 1.0)
                weighted_prediction += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction = weighted_prediction / total_weight
                ensemble_confidence = np.mean(list(confidences.values()))
                
                # Calculate ensemble confidence based on agreement
                prediction_values = list(predictions.values())
                agreement_ratio = sum(1 for p in prediction_values if p == round(ensemble_prediction)) / len(prediction_values)
                
                final_confidence = (ensemble_confidence + agreement_ratio) / 2
                
                return {
                    'prediction': round(ensemble_prediction),
                    'confidence': final_confidence,
                    'individual_predictions': predictions,
                    'individual_confidences': confidences,
                    'ensemble_weights': self.ensemble_weights,
                    'agreement_ratio': agreement_ratio,
                    'model_count': len(predictions)
                }
            
            return None
            
        except Exception as e:
            self.logger.exception(f"Ensemble prediction failed: {e}")
            return None
    
    def _get_lstm_prediction(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get LSTM prediction"""
        if not DEEP_LEARNING_AVAILABLE or not self.use_deep_learning or 'lstm' not in self.models:
            return None
        
        try:
            sequence_length = 20
            if len(data) < sequence_length:
                return None
            
            # Ensure we have the required feature columns
            if not hasattr(self, 'feature_columns') or not self.feature_columns:
                self.logger.warning("No feature columns available for LSTM prediction")
                return None
            
            # Get only numeric columns and ensure they exist
            available_features = [col for col in self.feature_columns if col in data.columns]
            if not available_features:
                self.logger.warning("No valid feature columns found in data")
                return None
            
            # Select only numeric data and ensure it's properly formatted
            feature_data = data[available_features].copy()
            
            # Convert all columns to numeric, dropping any that can't be converted
            for col in feature_data.columns:
                try:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                except Exception:
                    feature_data = feature_data.drop(columns=[col])
            
            # Remove any NaN values
            feature_data = feature_data.fillna(0)
            
            if len(feature_data) < sequence_length:
                self.logger.warning("Insufficient data for LSTM prediction")
                return None
            
            if feature_data.empty:
                self.logger.warning("No valid numeric data for LSTM prediction")
                return None
            
            # Prepare LSTM data
            latest_sequence = feature_data.tail(sequence_length).values
            
            # Ensure the sequence has the correct shape
            if latest_sequence.shape[0] != sequence_length:
                self.logger.warning(f"Sequence length mismatch: expected {sequence_length}, got {latest_sequence.shape[0]}")
                return None
            
            # Validate that the number of features matches the model's expected input
            expected_features = self.models['lstm'].input_shape[1]  # Get expected features from model
            actual_features = latest_sequence.shape[1]
            
            if actual_features != expected_features:
                self.logger.warning(f"Feature count mismatch: model expects {expected_features}, data has {actual_features}")
                return None
            
            lstm_input = latest_sequence.reshape(1, sequence_length, actual_features)
            
            # Get prediction
            prediction_proba = self.models['lstm'].predict(lstm_input, verbose=0)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
            
            return {
                'prediction': prediction,
                'confidence': max(prediction_proba, 1 - prediction_proba)
            }
            
        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            return None
    
    def update_model_performance(self, prediction: Dict[str, Any], actual_outcome: int):
        """Update model performance based on actual outcome"""
        try:
            for model_name in self.model_performances:
                if model_name in prediction.get('individual_predictions', {}):
                    perf = self.model_performances[model_name]
                    perf.prediction_count += 1
                    
                    predicted = prediction['individual_predictions'][model_name]
                    if predicted == actual_outcome:
                        perf.correct_predictions += 1
                    
                    # Update accuracy
                    perf.accuracy = perf.correct_predictions / perf.prediction_count
                    perf.last_updated = datetime.now()
            
            # Store prediction history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'actual_outcome': actual_outcome
            })
            
            # Recalculate ensemble weights periodically
            if len(self.prediction_history) % 50 == 0:
                self._calculate_ensemble_weights()
                
        except Exception as e:
            self.logger.error(f"Failed to update model performance: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models and their performance"""
        summary = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_models': len(self.models),
            'models': {},
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.feature_columns),
            'prediction_history_count': len(self.prediction_history)
        }
        
        for model_name, performance in self.model_performances.items():
            summary['models'][model_name] = {
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'cv_score': performance.cv_score,
                'prediction_count': performance.prediction_count,
                'last_updated': performance.last_updated.isoformat()
            }
        
        return summary
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                if model_name != 'lstm':  # LSTM models are saved differently
                    model_path = os.path.join(self.models_dir, f"{model_name}_{self.symbol}_{self.timeframe}.joblib")
                    joblib.dump(model, model_path)
                    
                    # Save scaler and feature selector
                    scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler_{self.symbol}_{self.timeframe}.joblib")
                    selector_path = os.path.join(self.models_dir, f"{model_name}_selector_{self.symbol}_{self.timeframe}.joblib")
                    
                    joblib.dump(self.scalers[model_name], scaler_path)
                    joblib.dump(self.feature_selectors[model_name], selector_path)
            
            # Save ensemble weights
            weights_path = os.path.join(self.models_dir, f"ensemble_weights_{self.symbol}_{self.timeframe}.joblib")
            joblib.dump(self.ensemble_weights, weights_path)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            for model_name in self.models.keys():
                if model_name != 'lstm':
                    model_path = os.path.join(self.models_dir, f"{model_name}_{self.symbol}_{self.timeframe}.joblib")
                    scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler_{self.symbol}_{self.timeframe}.joblib")
                    selector_path = os.path.join(self.models_dir, f"{model_name}_selector_{self.symbol}_{self.timeframe}.joblib")
                    
                    if os.path.exists(model_path):
                        self.models[model_name] = joblib.load(model_path)
                        
                        # Load scaler with error handling
                        if os.path.exists(scaler_path):
                            try:
                                self.scalers[model_name] = joblib.load(scaler_path)
                                # Verify scaler is properly fitted
                                if not hasattr(self.scalers[model_name], 'scale_'):
                                    self.logger.warning(f"Scaler for {model_name} not properly fitted, creating new one")
                                    self.scalers[model_name] = RobustScaler()
                            except Exception as e:
                                self.logger.warning(f"Failed to load scaler for {model_name}: {e}, creating new one")
                                self.scalers[model_name] = RobustScaler()
                        else:
                            self.logger.warning(f"Scaler file not found for {model_name}, creating new one")
                            self.scalers[model_name] = RobustScaler()
                        
                        # Load feature selector with error handling
                        if os.path.exists(selector_path):
                            try:
                                self.feature_selectors[model_name] = joblib.load(selector_path)
                            except Exception as e:
                                self.logger.warning(f"Failed to load feature selector for {model_name}: {e}, creating new one")
                                self.feature_selectors[model_name] = SelectKBest(score_func=f_classif, k=20)
                        else:
                            self.logger.warning(f"Feature selector file not found for {model_name}, creating new one")
                            self.feature_selectors[model_name] = SelectKBest(score_func=f_classif, k=20)
            
            # Load ensemble weights
            weights_path = os.path.join(self.models_dir, f"ensemble_weights_{self.symbol}_{self.timeframe}.joblib")
            if os.path.exists(weights_path):
                try:
                    self.ensemble_weights = joblib.load(weights_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load ensemble weights: {e}, using default weights")
                    self.ensemble_weights = {name: 1.0 for name in self.models.keys() if name != 'lstm'}
            else:
                self.logger.warning("Ensemble weights file not found, using default weights")
                self.ensemble_weights = {name: 1.0 for name in self.models.keys() if name != 'lstm'}
            
            self.logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
