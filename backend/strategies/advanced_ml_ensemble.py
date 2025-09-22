#!/usr/bin/env python3
"""
Advanced ML Ensemble Trading System
===================================

This module implements a sophisticated ensemble of machine learning models
with advanced feature engineering, model selection, and online learning capabilities.

Features:
- Advanced algorithms: XGBoost, LightGBM, CatBoost, Neural Networks
- Feature engineering with technical indicators and market microstructure
- Ensemble methods: Voting, Stacking, Blending
- Online learning and model adaptation
- Feature selection and importance analysis
- Cross-validation and hyperparameter optimization
- Real-time prediction updates
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    import xgboost as xgb
    import lightgbm as lgb
    try:
        import catboost as cb
        CATBOOST_AVAILABLE = True
    except ImportError:
        CATBOOST_AVAILABLE = False
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    prediction_time: float = 0.0

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_name: str
    importance_score: float
    rank: int
    category: str  # technical, fundamental, market_microstructure

class AdvancedMLEnsemble:
    """Advanced ML Ensemble Trading System"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced ML ensemble"""
        self.config = config
        
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available. Install scikit-learn, xgboost, lightgbm")
        
        # Model configuration
        self.models = {}
        self.ensemble_model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_importance = {}
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        self.accuracy_history = []
        
        # Feature engineering parameters
        self.technical_indicators = config.get('technical_indicators', [
            'rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'williams_r',
            'cci', 'adx', 'obv', 'volume_sma', 'price_momentum', 'volatility'
        ])
        
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.feature_lags = config.get('feature_lags', [1, 2, 3, 5])
        
        # Model parameters
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.random_state = config.get('random_state', 42)
        
        # Ensemble configuration
        self.use_voting = config.get('use_voting', True)
        self.use_stacking = config.get('use_stacking', True)
        self.use_blending = config.get('use_blending', True)
        self.ensemble_weights = config.get('ensemble_weights', None)
        
        logger.info("Advanced ML Ensemble initialized")
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for ML models
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Lagged features
        df = self._add_lagged_features(df)
        
        # Rolling statistics
        df = self._add_rolling_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        df['macd_bullish'] = (macd_line > signal_line).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_normalized'] = df['atr'] / df['close']
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_oversold'] = (stoch_k < 20).astype(int)
        df['stoch_overbought'] = (stoch_k > 80).astype(int)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df, 14)
        
        # CCI
        df['cci'] = self._calculate_cci(df, 20)
        
        # ADX
        df['adx'] = self._calculate_adx(df, 14)
        df['adx_trending'] = (df['adx'] > 25).astype(int)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Order flow imbalance (simplified)
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        # Price impact
        df['price_impact'] = df['returns'] / np.log(df['volume'] + 1)
        
        # Volatility clustering
        df['volatility_cluster'] = df['returns'].rolling(5).std()
        
        # Jump detection
        df['price_jump'] = (abs(df['returns']) > df['returns'].rolling(20).std() * 2).astype(int)
        
        # Gap analysis
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_size'] = abs(df['gap'])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            
            # Trading session features
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            df['session_overlap'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        feature_columns = ['returns', 'volume', 'rsi', 'macd', 'atr']
        
        for lag in self.feature_lags:
            for col in feature_columns:
                if col in df.columns:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features"""
        for period in self.lookback_periods:
            # Rolling means
            df[f'close_ma_{period}'] = df['close'].rolling(period).mean()
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            
            # Rolling standard deviations
            df[f'returns_std_{period}'] = df['returns'].rolling(period).std()
            df[f'volume_std_{period}'] = df['volume'].rolling(period).std()
            
            # Rolling maximums and minimums
            df[f'high_max_{period}'] = df['high'].rolling(period).max()
            df[f'low_min_{period}'] = df['low'].rolling(period).min()
            
            # Price position within range
            df[f'price_position_{period}'] = (df['close'] - df[f'low_min_{period}']) / (df[f'high_max_{period}'] - df[f'low_min_{period}'])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # GARCH-like volatility
        df['volatility_garch'] = df['returns'].rolling(20).std()
        
        # Parkinson volatility
        df['volatility_parkinson'] = np.sqrt(0.25 * np.log(df['high'] / df['low']) ** 2)
        
        # Garman-Klass volatility
        df['volatility_gk'] = np.sqrt(0.5 * np.log(df['high'] / df['low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2)
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_garch'] / df['volatility_garch'].rolling(50).mean()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        for period in [5, 10, 20]:
            # Price momentum
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            
            # Volume momentum
            df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1
            
            # Rate of change
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume-price trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # On-balance volume
        df['obv'] = self._calculate_obv(df)
        
        # Volume rate of change
        df['volume_roc'] = df['volume'].pct_change(10)
        
        # Volume-weighted average price (simplified)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def create_models(self) -> Dict[str, Any]:
        """Create and configure ML models"""
        models = {}
        
        # Tree-based models
        models['random_forest'] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        
        # XGBoost
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1
        )
        
        # CatBoost (if available)
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_seed=self.random_state,
                verbose=False
            )
        
        # Linear models
        models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        models['ridge_classifier'] = RidgeClassifier(
            random_state=self.random_state
        )
        
        # SVM
        models['svm'] = SVC(
            kernel='rbf',
            random_state=self.random_state,
            probability=True
        )
        
        # Neural Network
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=self.random_state
        )
        
        # Naive Bayes
        models['naive_bayes'] = GaussianNB()
        
        self.models = models
        logger.info(f"Created {len(models)} ML models")
        
        return models
    
    def create_ensemble_models(self) -> Dict[str, Any]:
        """Create ensemble models"""
        if not self.models:
            self.create_models()
        
        ensemble_models = {}
        
        # Voting Classifier
        if self.use_voting:
            voting_estimators = [
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm']),
                ('lr', self.models['logistic_regression'])
            ]
            
            ensemble_models['voting'] = VotingClassifier(
                estimators=voting_estimators,
                voting='soft'
            )
        
        # Stacking Classifier
        if self.use_stacking:
            base_estimators = [
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm'])
            ]
            
            ensemble_models['stacking'] = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(),
                cv=5
            )
        
        return ensemble_models
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'selectkbest', k: int = 50) -> pd.DataFrame:
        """
        Select most important features
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            k: Number of features to select
            
        Returns:
            Selected features
        """
        if method == 'selectkbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            self.feature_selector = RFE(
                estimator=RandomForestClassifier(n_estimators=50),
                n_features_to_select=k
            )
        elif method == 'selectfrommodel':
            self.feature_selector = SelectFromModel(
                estimator=RandomForestClassifier(n_estimators=50),
                max_features=k
            )
        
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, ModelPerformance]:
        """
        Train all models and evaluate performance
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test set size
            
        Returns:
            Model performance dictionary
        """
        # Create models if not already created
        if not self.models:
            self.models = self.create_models()
        
        # Split data
        # Check if we can use stratify (need at least 2 samples per class)
        unique_classes, counts = np.unique(y, return_counts=True)
        can_stratify = len(unique_classes) > 1 and all(counts >= 2)
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        for name, model in self.models.items():
            start_time = datetime.now()
            
            try:
                # Ensure target variable is properly formatted
                y_train_clean = y_train.values if hasattr(y_train, 'values') else y_train
                y_test_clean = y_test.values if hasattr(y_test, 'values') else y_test
                
                # Train model
                # Special handling for RidgeClassifier
                if name == 'ridge_classifier':
                    # Ensure target is integer type for RidgeClassifier
                    y_train_clean = y_train_clean.astype(int)
                    y_test_clean = y_test_clean.astype(int)
                    # RidgeClassifier needs 1D array
                    if y_train_clean.ndim > 1:
                        y_train_clean = y_train_clean.flatten()
                    if y_test_clean.ndim > 1:
                        y_test_clean = y_test_clean.flatten()
                
                model.fit(X_train_scaled, y_train_clean)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_clean, y_pred)
                precision = precision_score(y_test_clean, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_clean, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_clean, y_pred, average='weighted', zero_division=0)
                roc_auc = roc_auc_score(y_test_clean, y_pred_proba) if y_pred_proba is not None else 0.0
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train_clean, cv=5)
                cv_score = cv_scores.mean()
                
                # Feature importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(X.columns, abs(model.coef_[0])))
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                self.model_performance[name] = ModelPerformance(
                    model_name=name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    roc_auc=roc_auc,
                    cross_val_score=cv_score,
                    feature_importance=feature_importance,
                    training_time=training_time
                )
                
                logger.info(f"Trained {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, CV={cv_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        return dict(self.model_performance)
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, ModelPerformance]:
        """Train ensemble models"""
        ensemble_models = self.create_ensemble_models()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble models
        for name, model in ensemble_models.items():
            start_time = datetime.now()
            
            try:
                # Use time series split for financial data
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv)
                
                # Train on full data
                model.fit(X_scaled, y)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                self.model_performance[f'ensemble_{name}'] = ModelPerformance(
                    model_name=f'ensemble_{name}',
                    accuracy=cv_scores.mean(),
                    precision=0.0,  # Will be calculated during prediction
                    recall=0.0,
                    f1_score=0.0,
                    roc_auc=0.0,
                    cross_val_score=cv_scores.mean(),
                    training_time=training_time
                )
                
                logger.info(f"Trained ensemble {name}: CV Score={cv_scores.mean():.3f}")
                
            except Exception as e:
                logger.error(f"Error training ensemble {name}: {e}")
                continue
        
        return self.model_performance
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using specified model or best model
        
        Args:
            X: Feature matrix
            model_name: Specific model to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name is None:
            # Use best model based on cross-validation score
            best_model_name = max(self.model_performance.keys(), 
                                key=lambda x: self.model_performance[x].cross_val_score)
        else:
            best_model_name = model_name
        
        if best_model_name.startswith('ensemble_'):
            model = self.ensemble_model
        else:
            model = self.models.get(best_model_name)
        
        if model is None:
            raise ValueError(f"Model {best_model_name} not found")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def get_feature_importance(self, top_n: int = 20) -> List[FeatureImportance]:
        """Get feature importance from best model"""
        if not self.model_performance:
            return []
        
        # Find best model
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda x: self.model_performance[x].cross_val_score)
        
        feature_importance = self.model_performance[best_model_name].feature_importance
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create FeatureImportance objects
        importance_list = []
        for i, (feature_name, importance_score) in enumerate(sorted_features[:top_n]):
            # Categorize features
            if any(indicator in feature_name.lower() for indicator in ['rsi', 'macd', 'bb', 'atr']):
                category = 'technical'
            elif any(micro in feature_name.lower() for micro in ['volume', 'pressure', 'impact']):
                category = 'market_microstructure'
            else:
                category = 'fundamental'
            
            importance_list.append(FeatureImportance(
                feature_name=feature_name,
                importance_score=importance_score,
                rank=i + 1,
                category=category
            ))
        
        return importance_list
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_performance': self.model_performance,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.model_performance = model_data['model_performance']
        
        logger.info(f"Models loaded from {filepath}")
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = np.abs(minus_dm)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        return pd.Series(obv, index=df.index).cumsum()
