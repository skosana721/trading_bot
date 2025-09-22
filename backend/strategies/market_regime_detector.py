#!/usr/bin/env python3
"""
Market Regime Detection System
==============================

This module implements sophisticated market regime detection including:
- Trend vs. Range detection
- Volatility regime classification
- Market cycle identification
- Multi-timeframe regime analysis
- Regime transition probabilities
- Adaptive strategy selection based on regime
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.cluster import KMeans, GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import scipy.stats as stats
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime enumeration"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class VolatilityRegime(Enum):
    """Volatility regime enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RegimeMetrics:
    """Market regime metrics"""
    regime: MarketRegime
    confidence: float
    duration: int  # periods in current regime
    transition_probability: float
    volatility_regime: VolatilityRegime
    trend_strength: float
    range_boundary_upper: Optional[float] = None
    range_boundary_lower: Optional[float] = None
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

@dataclass
class RegimeTransition:
    """Regime transition data"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    probability: float
    trigger_conditions: List[str]
    timestamp: datetime

class MarketRegimeDetector:
    """Market Regime Detection System"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the market regime detector"""
        self.config = config
        
        # Detection parameters
        self.lookback_period = config.get('lookback_period', 100)
        self.volatility_window = config.get('volatility_window', 20)
        self.trend_window = config.get('trend_window', 50)
        self.regime_threshold = config.get('regime_threshold', 0.7)
        
        # Volatility thresholds
        self.vol_low_threshold = config.get('vol_low_threshold', 0.5)
        self.vol_high_threshold = config.get('vol_high_threshold', 2.0)
        self.vol_extreme_threshold = config.get('vol_extreme_threshold', 3.0)
        
        # Trend detection parameters
        self.trend_strength_threshold = config.get('trend_strength_threshold', 0.6)
        self.range_threshold = config.get('range_threshold', 0.02)  # 2% range
        
        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        self.transition_matrix = {}
        self.regime_durations = {}
        
        # ML models
        self.regime_classifier = None
        self.volatility_classifier = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        logger.info("Market Regime Detector initialized")
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeMetrics:
        """
        Detect current market regime
        
        Args:
            data: OHLCV price data
            
        Returns:
            RegimeMetrics object
        """
        if len(data) < self.lookback_period:
            return self._create_default_regime()
        
        # Normalize column names to lowercase
        data = data.copy()
        data.columns = data.columns.str.lower()
        
        # Calculate regime features
        features = self._calculate_regime_features(data)
        
        # Detect trend regime
        trend_regime = self._detect_trend_regime(data, features)
        
        # Detect volatility regime
        volatility_regime = self._detect_volatility_regime(features)
        
        # Detect range regime
        range_regime = self._detect_range_regime(data, features)
        
        # Combine regime signals
        final_regime = self._combine_regime_signals(trend_regime, volatility_regime, range_regime)
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(final_regime, features)
        
        # Calculate regime duration
        duration = self._calculate_regime_duration(final_regime)
        
        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(final_regime)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Identify support/resistance levels
        support_levels, resistance_levels = self._identify_support_resistance(data)
        
        # Get range boundaries if in ranging regime
        range_upper, range_lower = None, None
        if final_regime == MarketRegime.RANGING:
            range_upper, range_lower = self._calculate_range_boundaries(data)
        
        regime_metrics = RegimeMetrics(
            regime=final_regime,
            confidence=confidence,
            duration=duration,
            transition_probability=transition_prob,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            range_boundary_upper=range_upper,
            range_boundary_lower=range_lower,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )
        
        # Update regime history
        self._update_regime_history(regime_metrics)
        
        return regime_metrics
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for regime detection"""
        features = {}
        
        # Price-based features
        returns = data['close'].pct_change().dropna()
        features['volatility'] = returns.std() * np.sqrt(252)
        features['skewness'] = returns.skew()
        features['kurtosis'] = returns.kurtosis()
        
        # Trend features
        sma_short = data['close'].rolling(20).mean()
        sma_long = data['close'].rolling(50).mean()
        features['trend_direction'] = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else -1
        features['trend_strength'] = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        # Range features
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        features['range_size'] = (high_20.iloc[-1] - low_20.iloc[-1]) / data['close'].iloc[-1]
        features['price_position'] = (data['close'].iloc[-1] - low_20.iloc[-1]) / (high_20.iloc[-1] - low_20.iloc[-1])
        
        # Volume features
        if 'volume' in data.columns:
            volume_sma = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'].iloc[-1] / volume_sma.iloc[-1]
        else:
            features['volume_ratio'] = 1.0
        
        # Momentum features
        features['momentum_5'] = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) if len(data) > 5 else 0
        features['momentum_20'] = (data['close'].iloc[-1] / data['close'].iloc[-21] - 1) if len(data) > 20 else 0
        
        # Volatility clustering
        vol_5 = returns.rolling(5).std()
        vol_20 = returns.rolling(20).std()
        features['volatility_ratio'] = vol_5.iloc[-1] / vol_20.iloc[-1] if vol_20.iloc[-1] > 0 else 1.0
        
        # Mean reversion features
        features['mean_reversion_signal'] = self._calculate_mean_reversion_signal(data)
        
        # Breakout features
        features['breakout_signal'] = self._calculate_breakout_signal(data)
        
        return features
    
    def _detect_trend_regime(self, data: pd.DataFrame, features: Dict[str, float]) -> MarketRegime:
        """Detect trend-based regime"""
        trend_strength = features['trend_strength']
        trend_direction = features['trend_direction']
        
        if trend_strength > self.trend_strength_threshold:
            if trend_direction > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def _detect_volatility_regime(self, features: Dict[str, float]) -> VolatilityRegime:
        """Detect volatility regime"""
        volatility = features['volatility']
        
        if volatility < self.vol_low_threshold:
            return VolatilityRegime.LOW
        elif volatility < self.vol_high_threshold:
            return VolatilityRegime.MEDIUM
        elif volatility < self.vol_extreme_threshold:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _detect_range_regime(self, data: pd.DataFrame, features: Dict[str, float]) -> MarketRegime:
        """Detect range-bound regime"""
        range_size = features['range_size']
        price_position = features['price_position']
        
        if range_size < self.range_threshold:
            # Check for range breakout
            if price_position > 0.8 or price_position < 0.2:
                return MarketRegime.BREAKOUT
            else:
                return MarketRegime.RANGING
        else:
            return MarketRegime.UNKNOWN
    
    def _combine_regime_signals(self, trend_regime: MarketRegime, 
                               volatility_regime: VolatilityRegime, 
                               range_regime: MarketRegime) -> MarketRegime:
        """Combine different regime signals"""
        # Priority order: Breakout > Trend > Range > Volatility
        
        if range_regime == MarketRegime.BREAKOUT:
            return MarketRegime.BREAKOUT
        
        if trend_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            if volatility_regime == VolatilityRegime.EXTREME:
                return MarketRegime.HIGH_VOLATILITY
            else:
                return trend_regime
        
        if range_regime == MarketRegime.RANGING:
            if volatility_regime == VolatilityRegime.LOW:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.RANGING
        
        return MarketRegime.UNKNOWN
    
    def _calculate_regime_confidence(self, regime: MarketRegime, features: Dict[str, float]) -> float:
        """Calculate confidence in regime detection"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on feature strength
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence += features['trend_strength'] * 0.3
            confidence += (1 - features['range_size']) * 0.2
        
        elif regime == MarketRegime.RANGING:
            confidence += (1 - features['trend_strength']) * 0.3
            confidence += (1 - features['range_size']) * 0.2
        
        elif regime == MarketRegime.BREAKOUT:
            confidence += features['breakout_signal'] * 0.4
            confidence += features['volume_ratio'] * 0.1
        
        # Volatility adjustments
        vol_regime = self._detect_volatility_regime(features)
        if vol_regime in [VolatilityRegime.LOW, VolatilityRegime.EXTREME]:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_regime_duration(self, regime: MarketRegime) -> int:
        """Calculate duration of current regime"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].regime == regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_transition_probability(self, regime: MarketRegime) -> float:
        """Calculate probability of regime transition"""
        if not self.regime_history:
            return 0.1  # Default transition probability
        
        # Calculate historical transition probabilities
        transitions = self._calculate_transition_matrix()
        
        if regime in transitions:
            # Average transition probability for this regime
            return np.mean(list(transitions[regime].values()))
        else:
            return 0.1
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        if len(data) < 20:
            return 0.0
        
        # Use ADX-like calculation
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        # Smoothed values
        tr_smooth = tr.rolling(14).mean()
        dm_plus_smooth = dm_plus.rolling(14).mean()
        dm_minus_smooth = dm_minus.rolling(14).mean()
        
        # Directional Indicators
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean()
        
        return adx.iloc[-1] / 100.0 if not pd.isna(adx.iloc[-1]) else 0.0
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        if len(data) < 50:
            return [], []
        
        # Find local highs and lows
        highs = data['high'].rolling(10, center=True).max()
        lows = data['low'].rolling(10, center=True).min()
        
        # Identify pivot points
        resistance_levels = []
        support_levels = []
        
        for i in range(10, len(data) - 10):
            if data['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['high'].iloc[i])
            if data['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['low'].iloc[i])
        
        # Cluster similar levels
        resistance_levels = self._cluster_levels(resistance_levels)
        support_levels = self._cluster_levels(support_levels)
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.001) -> List[float]:
        """Cluster similar price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    def _calculate_range_boundaries(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate range boundaries for ranging regime"""
        if len(data) < 20:
            return None, None
        
        high_20 = data['high'].rolling(20).max().iloc[-1]
        low_20 = data['low'].rolling(20).min().iloc[-1]
        
        return high_20, low_20
    
    def _calculate_mean_reversion_signal(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion signal"""
        if len(data) < 20:
            return 0.0
        
        # Bollinger Bands
        sma = data['close'].rolling(20).mean()
        std = data['close'].rolling(20).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        current_price = data['close'].iloc[-1]
        
        if current_price > upper_band.iloc[-1]:
            return -1.0  # Overbought
        elif current_price < lower_band.iloc[-1]:
            return 1.0   # Oversold
        else:
            return 0.0   # Neutral
    
    def _calculate_breakout_signal(self, data: pd.DataFrame) -> float:
        """Calculate breakout signal"""
        if len(data) < 20:
            return 0.0
        
        # Check for volume breakout
        if 'volume' in data.columns:
            volume_sma = data['volume'].rolling(20).mean()
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma.iloc[-1]
        else:
            volume_ratio = 1.0
        
        # Check for price breakout
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        current_price = data['close'].iloc[-1]
        
        if current_price > high_20.iloc[-2] and volume_ratio > 1.5:
            return 1.0  # Bullish breakout
        elif current_price < low_20.iloc[-2] and volume_ratio > 1.5:
            return -1.0  # Bearish breakout
        else:
            return 0.0
    
    def _update_regime_history(self, regime_metrics: RegimeMetrics):
        """Update regime history"""
        self.regime_history.append(regime_metrics)
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        # Update current regime
        self.current_regime = regime_metrics.regime
    
    def _calculate_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Calculate regime transition matrix"""
        if len(self.regime_history) < 2:
            return {}
        
        transitions = {}
        
        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i].regime
            to_regime = self.regime_history[i + 1].regime
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
            
            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0
            
            transitions[from_regime][to_regime] += 1
        
        # Convert counts to probabilities
        for from_regime in transitions:
            total = sum(transitions[from_regime].values())
            for to_regime in transitions[from_regime]:
                transitions[from_regime][to_regime] /= total
        
        return transitions
    
    def _create_default_regime(self) -> RegimeMetrics:
        """Create default regime when insufficient data"""
        return RegimeMetrics(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            duration=1,
            transition_probability=0.1,
            volatility_regime=VolatilityRegime.MEDIUM,
            trend_strength=0.0
        )
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime summary"""
        if not self.regime_history:
            return {'current_regime': 'unknown', 'confidence': 0.0}
        
        current = self.regime_history[-1]
        transitions = self._calculate_transition_matrix()
        
        return {
            'current_regime': current.regime.value,
            'confidence': current.confidence,
            'duration': current.duration,
            'transition_probability': current.transition_probability,
            'volatility_regime': current.volatility_regime.value,
            'trend_strength': current.trend_strength,
            'support_levels': current.support_levels,
            'resistance_levels': current.resistance_levels,
            'range_boundaries': {
                'upper': current.range_boundary_upper,
                'lower': current.range_boundary_lower
            },
            'transition_matrix': transitions,
            'regime_history_length': len(self.regime_history)
        }
    
    def predict_next_regime(self, data: pd.DataFrame) -> List[Tuple[MarketRegime, float]]:
        """Predict next regime with probabilities"""
        if not self.regime_history:
            return [(MarketRegime.UNKNOWN, 1.0)]
        
        current_regime = self.current_regime
        transitions = self._calculate_transition_matrix()
        
        if current_regime in transitions:
            # Return possible next regimes with probabilities
            next_regimes = []
            for regime, probability in transitions[current_regime].items():
                next_regimes.append((regime, probability))
            
            # Sort by probability
            next_regimes.sort(key=lambda x: x[1], reverse=True)
            return next_regimes
        else:
            return [(MarketRegime.UNKNOWN, 1.0)]
