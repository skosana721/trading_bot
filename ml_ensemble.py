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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Deep learning libraries not available. LSTM models will be disabled.")

@dataclass
class ModelPerformance:
    """Data class to track model performance"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_score: float
    last_updated: datetime
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
            eval_metric='logloss',
            use_label_encoder=False
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
        
        # Deep Learning Models
        if self.use_deep_learning:
            self.models['lstm'] = self._create_lstm_model()
        
        # Initialize scalers and feature selectors
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
            self.feature_selectors[model_name] = SelectKBest(score_func=f_classif, k=20)
            self.model_performances[model_name] = ModelPerformance(
                model_name=model_name,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                cv_score=0.0,
                last_updated=datetime.now()
            )
    
    def _create_lstm_model(self):
        """Create LSTM model for time series prediction"""
        if not DEEP_LEARNING_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(None, 1)),
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
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical features for ML models
        
        Args:
            data: Price data
            
        Returns:
            DataFrame with advanced features
        """
        df = data.copy()
        
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
        
        # Time-based features
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        
        # Remove infinite values and fill NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
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
        
        # Remove rows with NaN values
        valid_indices = ~(df_with_features[feature_columns].isnull().any(axis=1) | target.isnull())
        features = df_with_features[feature_columns].loc[valid_indices]
        target = target.loc[valid_indices]
        
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
    
    def _train_traditional_model(self, model_name: str, model, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        """Train traditional ML model"""
        try:
            self.logger.info(f"Training {model_name}...")
            
            # Feature selection
            X_train_selected = self.feature_selectors[model_name].fit_transform(X_train, y_train)
            X_test_selected = self.feature_selectors[model_name].transform(X_test)
            
            # Scaling
            X_train_scaled = self.scalers[model_name].fit_transform(X_train_selected)
            X_test_scaled = self.scalers[model_name].transform(X_test_selected)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Update performance metrics
            self.model_performances[model_name].accuracy = accuracy_score(y_test, y_pred)
            self.model_performances[model_name].precision = precision_score(y_test, y_pred, average='weighted')
            self.model_performances[model_name].recall = recall_score(y_test, y_pred, average='weighted')
            self.model_performances[model_name].f1_score = f1_score(y_test, y_pred, average='weighted')
            self.model_performances[model_name].cv_score = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='f1_weighted'
            ).mean()
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
        if not DEEP_LEARNING_AVAILABLE or 'lstm' not in self.models:
            return
        
        try:
            self.logger.info("Training LSTM model...")
            
            # Prepare LSTM data (3D format: samples, timesteps, features)
            sequence_length = 20
            X_lstm_train = self._prepare_lstm_data(X_train, sequence_length)
            X_lstm_test = self._prepare_lstm_data(X_test, sequence_length)
            
            # Adjust target data
            y_lstm_train = y_train.iloc[sequence_length-1:]
            y_lstm_test = y_test.iloc[sequence_length-1:]
            
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
            # Prepare data for ensemble
            X_train_scaled = self.scalers['random_forest'].fit_transform(X_train)
            X_test_scaled = self.scalers['random_forest'].transform(X_test)
            
            # Create voting classifier
            estimators = []
            for name, model in self.models.items():
                if name != 'lstm':  # Exclude LSTM from voting classifier
                    estimators.append((name, model))
            
            self.voting_classifier = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            
            self.voting_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate voting classifier
            y_pred = self.voting_classifier.predict(X_test_scaled)
            self.logger.info(f"Voting Classifier - Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to create ensemble models: {e}")
    
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
        
        try:
            # Prepare features
            df_with_features = self.create_advanced_features(data)
            
            if len(df_with_features) < 1:
                return None
            
            # Get latest data point
            latest_features = df_with_features[self.feature_columns].tail(1)
            
            # Remove NaN values
            if latest_features.isnull().any().any():
                return None
            
            predictions = {}
            confidences = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # Handle LSTM prediction
                    lstm_pred = self._get_lstm_prediction(df_with_features)
                    if lstm_pred:
                        predictions[model_name] = lstm_pred['prediction']
                        confidences[model_name] = lstm_pred['confidence']
                else:
                    # Traditional model prediction
                    try:
                        # Feature selection and scaling
                        features_selected = self.feature_selectors[model_name].transform(latest_features)
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
        if not DEEP_LEARNING_AVAILABLE or 'lstm' not in self.models:
            return None
        
        try:
            sequence_length = 20
            if len(data) < sequence_length:
                return None
            
            # Prepare LSTM data
            latest_sequence = data[self.feature_columns].tail(sequence_length).values
            lstm_input = latest_sequence.reshape(1, sequence_length, len(self.feature_columns))
            
            # Get prediction
            prediction_proba = self.models['lstm'].predict(lstm_input)[0][0]
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
                        self.scalers[model_name] = joblib.load(scaler_path)
                        self.feature_selectors[model_name] = joblib.load(selector_path)
            
            # Load ensemble weights
            weights_path = os.path.join(self.models_dir, f"ensemble_weights_{self.symbol}_{self.timeframe}.joblib")
            if os.path.exists(weights_path):
                self.ensemble_weights = joblib.load(weights_path)
            
            self.logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
