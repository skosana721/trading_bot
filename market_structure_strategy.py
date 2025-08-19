#!/usr/bin/env python3
"""
Market Structure & Strategy Trading System
==========================================

Implements a sophisticated trading strategy based on:
- Market structure (Higher Highs/Lower Lows)
- Trend analysis (EMA + candlestick patterns)
- Support/Resistance zones
- Engulfing candlestick patterns
- Multi-timeframe analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import talib
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction enumeration"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    CONSOLIDATION = "consolidation"

class EntryType(Enum):
    """Entry type enumeration"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class MarketStructure:
    """Market structure analysis results"""
    trend: TrendDirection
    higher_highs: List[float]
    higher_lows: List[float]
    lower_highs: List[float]
    lower_lows: List[float]
    swing_points: List[Dict[str, Any]]
    structure_strength: float

@dataclass
class SupportResistanceZone:
    """Support/Resistance zone definition"""
    zone_type: str
    high: float
    low: float
    strength: float
    touches: int
    last_touch: datetime
    zone_id: str

@dataclass
class CandlestickPattern:
    """Candlestick pattern detection results"""
    pattern_type: str
    confidence: float
    candle_index: int
    pattern_data: Dict[str, Any]

@dataclass
class TradeSignal:
    """Complete trade signal"""
    entry_type: EntryType
    symbol: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    confidence: float
    entry_reason: str
    trend_direction: TrendDirection
    zone_levels: Dict[str, float]
    pattern_info: CandlestickPattern
    timestamp: datetime

class MarketStructureStrategy:
    """Market Structure & Strategy Trading System"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the market structure strategy"""
        self.config = config
        
        # Trading parameters
        self.use_pairs = config.get('UsePairs', ['EURUSD', 'GBPUSD', 'USDJPY'])
        self.lot_size_initial = config.get('LotSizeInitial', 0.01)
        self.lot_size_reentry = config.get('LotSizeReEntry', 0.01)
        self.risk_per_trade = config.get('RiskPerTrade', 2.0) / 100.0
        self.risk_reward_ratio = config.get('RiskRewardRatio', 2.0)
        self.sl_buffer_pips = config.get('SL_Buffer_Pips', 10)
        self.tp_multiplier = config.get('TP_Multiplier', 2.0)
        
        # Trailing stop parameters
        self.enable_trailing_stop = config.get('EnableTrailingStop', False)
        self.trail_start_profit_pips = config.get('TrailStartProfitPips', 50)
        self.trail_step_pips = config.get('TrailStepPips', 10)
        
        # Analysis parameters
        self.trend_ema_period = 50
        self.min_structure_points = 2
        self.zone_touch_threshold = 3
        self.zone_strength_threshold = 0.6
        
        # State tracking
        self.open_trades = {}
        self.market_structures = {}
        self.zones = {}
        self.last_analysis = {}
        
        logger.info("Market Structure Strategy initialized")
    
    def analyze_trend(self, data: pd.DataFrame, timeframe: str) -> TrendDirection:
        """Analyze trend using EMA and candlestick patterns"""
        if len(data) < 50:
            return TrendDirection.CONSOLIDATION
        
        # Calculate EMA
        data['ema_50'] = talib.EMA(data['Close'], timeperiod=self.trend_ema_period)
        
        # Get current price and EMA
        current_price = data['Close'].iloc[-1]
        current_ema = data['ema_50'].iloc[-1]
        
        # Analyze last candle
        last_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]
        
        # Determine candle direction
        candle_bullish = last_candle['Close'] > last_candle['Open']
        candle_bearish = last_candle['Close'] < last_candle['Open']
        
        # Trend conditions
        price_above_ema = current_price > current_ema
        price_below_ema = current_price < current_ema
        
        # Check market structure
        structure = self.detect_market_structure(data)
        
        # Determine trend
        if (price_above_ema or candle_bullish) and len(structure.higher_highs) >= self.min_structure_points and len(structure.higher_lows) >= self.min_structure_points:
            return TrendDirection.UPTREND
        elif (price_below_ema or candle_bearish) and len(structure.lower_highs) >= self.min_structure_points and len(structure.lower_lows) >= self.min_structure_points:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.CONSOLIDATION
    
    def detect_market_structure(self, data: pd.DataFrame) -> MarketStructure:
        """Detect market structure (Higher Highs, Higher Lows, Lower Highs, Lower Lows)"""
        if len(data) < 20:
            return MarketStructure(
                trend=TrendDirection.CONSOLIDATION,
                higher_highs=[], higher_lows=[], lower_highs=[], lower_lows=[],
                swing_points=[], structure_strength=0.0
            )
        
        # Find swing points (peaks and troughs)
        swing_points = self._find_swing_points(data)
        
        # Analyze structure
        higher_highs = []
        higher_lows = []
        lower_highs = []
        lower_lows = []
        
        for i in range(1, len(swing_points)):
            current = swing_points[i]
            previous = swing_points[i-1]
            
            if current['type'] == 'high' and previous['type'] == 'high':
                if current['price'] > previous['price']:
                    higher_highs.append(current['price'])
                else:
                    lower_highs.append(current['price'])
            elif current['type'] == 'low' and previous['type'] == 'low':
                if current['price'] > previous['price']:
                    higher_lows.append(current['price'])
                else:
                    lower_lows.append(current['price'])
        
        # Calculate structure strength
        total_points = len(higher_highs) + len(higher_lows) + len(lower_highs) + len(lower_lows)
        structure_strength = min(total_points / 10.0, 1.0)  # Normalize to 0-1
        
        return MarketStructure(
            trend=TrendDirection.CONSOLIDATION,  # Will be set by trend analysis
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            swing_points=swing_points,
            structure_strength=structure_strength
        )
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Dict[str, Any]]:
        """Find swing highs and lows in the data"""
        swing_points = []
        
        for i in range(window, len(data) - window):
            # Check for swing high
            if all(data['High'].iloc[i] > data['High'].iloc[j] for j in range(i-window, i)) and \
               all(data['High'].iloc[i] > data['High'].iloc[j] for j in range(i+1, i+window+1)):
                swing_points.append({
                    'type': 'high',
                    'price': data['High'].iloc[i],
                    'index': i,
                    'timestamp': data.index[i]
                })
            
            # Check for swing low
            if all(data['Low'].iloc[i] < data['Low'].iloc[j] for j in range(i-window, i)) and \
               all(data['Low'].iloc[i] < data['Low'].iloc[j] for j in range(i+1, i+window+1)):
                swing_points.append({
                    'type': 'low',
                    'price': data['Low'].iloc[i],
                    'index': i,
                    'timestamp': data.index[i]
                })
        
        return swing_points
    
    def identify_support_resistance_zones(self, data: pd.DataFrame) -> List[SupportResistanceZone]:
        """Identify support and resistance zones"""
        zones = []
        
        # Find swing points
        swing_points = self._find_swing_points(data)
        
        # Group nearby swing points into zones
        tolerance = data['Close'].std() * 0.1  # 10% of standard deviation
        
        for point in swing_points:
            # Check if this point is near an existing zone
            zone_found = False
            for zone in zones:
                if abs(point['price'] - zone.high) <= tolerance or abs(point['price'] - zone.low) <= tolerance:
                    # Update zone
                    zone.high = max(zone.high, point['price'])
                    zone.low = min(zone.low, point['price'])
                    zone.touches += 1
                    zone.last_touch = point['timestamp']
                    zone_found = True
                    break
            
            if not zone_found:
                # Create new zone
                zone_type = "resistance" if point['type'] == 'high' else "support"
                new_zone = SupportResistanceZone(
                    zone_type=zone_type,
                    high=point['price'] + tolerance,
                    low=point['price'] - tolerance,
                    strength=0.5,  # Initial strength
                    touches=1,
                    last_touch=point['timestamp'],
                    zone_id=f"{zone_type}_{len(zones)}"
                )
                zones.append(new_zone)
        
        # Calculate zone strength based on touches and recency
        for zone in zones:
            # Strength based on number of touches
            touch_strength = min(zone.touches / self.zone_touch_threshold, 1.0)
            
            # Recency factor (more recent touches = stronger zone)
            days_since_touch = (datetime.now() - zone.last_touch).days
            recency_factor = max(0.1, 1.0 - (days_since_touch / 30.0))
            
            zone.strength = touch_strength * recency_factor
        
        # Filter zones by strength
        strong_zones = [zone for zone in zones if zone.strength >= self.zone_strength_threshold]
        
        return strong_zones
    
    def detect_engulfing_patterns(self, data: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect engulfing candlestick patterns"""
        patterns = []
        
        if len(data) < 2:
            return patterns
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Calculate body sizes
            current_body = abs(current['Close'] - current['Open'])
            previous_body = abs(previous['Close'] - previous['Open'])
            
            # Bullish Engulfing
            if (current_body > previous_body and
                current['Close'] > current['Open'] and  # Current is bullish
                previous['Close'] < previous['Open'] and  # Previous is bearish
                current['Close'] > previous['High'] and  # Current close above previous high
                current['Open'] < previous['Low']):  # Current open below previous low
                
                pattern = CandlestickPattern(
                    pattern_type="bullish_engulfing",
                    confidence=min(current_body / previous_body, 2.0) / 2.0,  # Normalize to 0-1
                    candle_index=i,
                    pattern_data={
                        'current_body': current_body,
                        'previous_body': previous_body,
                        'body_ratio': current_body / previous_body
                    }
                )
                patterns.append(pattern)
            
            # Bearish Engulfing
            elif (current_body > previous_body and
                  current['Close'] < current['Open'] and  # Current is bearish
                  previous['Close'] > previous['Open'] and  # Previous is bullish
                  current['Close'] < previous['Low'] and  # Current close below previous low
                  current['Open'] > previous['High']):  # Current open above previous high
                
                pattern = CandlestickPattern(
                    pattern_type="bearish_engulfing",
                    confidence=min(current_body / previous_body, 2.0) / 2.0,  # Normalize to 0-1
                    candle_index=i,
                    pattern_data={
                        'current_body': current_body,
                        'previous_body': previous_body,
                        'body_ratio': current_body / previous_body
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def check_entry_conditions(self, symbol: str, trend: TrendDirection, 
                              zones: List[SupportResistanceZone], 
                              patterns: List[CandlestickPattern],
                              current_price: float) -> Optional[TradeSignal]:
        """Check entry conditions and generate trade signals"""
        if not patterns:
            return None
        
        # Get latest pattern
        latest_pattern = patterns[-1]
        
        # Check if price is near a relevant zone
        relevant_zone = None
        for zone in zones:
            if zone.high >= current_price >= zone.low:
                relevant_zone = zone
                break
        
        if not relevant_zone:
            return None
        
        # Check trend and pattern alignment
        if (trend == TrendDirection.UPTREND and 
            latest_pattern.pattern_type == "bullish_engulfing" and
            relevant_zone.zone_type == "support"):
            
            # Buy signal
            entry_price = current_price
            stop_loss = relevant_zone.low - (self.sl_buffer_pips * 0.0001)  # Convert pips to price
            take_profit = entry_price + ((entry_price - stop_loss) * self.tp_multiplier)
            
            signal = TradeSignal(
                entry_type=EntryType.BUY,
                symbol=symbol,
                timeframe="H4",  # Entry timeframe
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=self.lot_size_initial,
                confidence=latest_pattern.confidence * relevant_zone.strength,
                entry_reason=f"Bullish engulfing at support zone in uptrend",
                trend_direction=trend,
                zone_levels={'support': relevant_zone.low, 'resistance': relevant_zone.high},
                pattern_info=latest_pattern,
                timestamp=datetime.now()
            )
            
            return signal
        
        elif (trend == TrendDirection.DOWNTREND and 
              latest_pattern.pattern_type == "bearish_engulfing" and
              relevant_zone.zone_type == "resistance"):
            
            # Sell signal
            entry_price = current_price
            stop_loss = relevant_zone.high + (self.sl_buffer_pips * 0.0001)  # Convert pips to price
            take_profit = entry_price - ((stop_loss - entry_price) * self.tp_multiplier)
            
            signal = TradeSignal(
                entry_type=EntryType.SELL,
                symbol=symbol,
                timeframe="H4",  # Entry timeframe
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=self.lot_size_initial,
                confidence=latest_pattern.confidence * relevant_zone.strength,
                entry_reason=f"Bearish engulfing at resistance zone in downtrend",
                trend_direction=trend,
                zone_levels={'support': relevant_zone.low, 'resistance': relevant_zone.high},
                pattern_info=latest_pattern,
                timestamp=datetime.now()
            )
            
            return signal
        
        return None
    
    def check_reentry_conditions(self, symbol: str, open_trade: Dict[str, Any],
                                trend: TrendDirection, zones: List[SupportResistanceZone],
                                patterns: List[CandlestickPattern], current_price: float) -> Optional[TradeSignal]:
        """Check re-entry conditions for existing trades"""
        # Check if trade has reached 25% of TP
        if 'take_profit' not in open_trade or 'entry_price' not in open_trade:
            return None
        
        tp_distance = abs(open_trade['take_profit'] - open_trade['entry_price'])
        current_distance = abs(current_price - open_trade['entry_price'])
        
        if current_distance < (tp_distance * 0.25):
            return None  # Not enough profit for re-entry
        
        # Check if trend still supports the trade direction
        trade_direction = open_trade.get('entry_type')
        if (trade_direction == 'buy' and trend != TrendDirection.UPTREND) or \
           (trade_direction == 'sell' and trend != TrendDirection.DOWNTREND):
            return None
        
        # Generate new signal with re-entry lot size
        base_signal = self.check_entry_conditions(symbol, trend, zones, patterns, current_price)
        if base_signal:
            base_signal.lot_size = self.lot_size_reentry
            base_signal.entry_reason += " (Re-entry)"
            return base_signal
        
        return None
    
    def update_trailing_stop(self, trade: Dict[str, Any], current_price: float) -> Optional[float]:
        """Update trailing stop for open trade"""
        if not self.enable_trailing_stop:
            return None
        
        entry_price = trade.get('entry_price')
        current_sl = trade.get('stop_loss')
        take_profit = trade.get('take_profit')
        entry_type = trade.get('entry_type')
        
        if not all([entry_price, current_sl, take_profit, entry_type]):
            return None
        
        # Calculate profit in pips
        if entry_type == 'buy':
            profit_pips = (current_price - entry_price) / 0.0001
            if profit_pips >= self.trail_start_profit_pips:
                new_sl = current_price - (self.trail_step_pips * 0.0001)
                if new_sl > current_sl:
                    return new_sl
        else:  # sell
            profit_pips = (entry_price - current_price) / 0.0001
            if profit_pips >= self.trail_start_profit_pips:
                new_sl = current_price + (self.trail_step_pips * 0.0001)
                if new_sl < current_sl:
                    return new_sl
        
        return None
    
    def analyze_symbol(self, symbol: str, data_d1: pd.DataFrame, data_h4: pd.DataFrame, 
                      data_h1: pd.DataFrame) -> Dict[str, Any]:
        """Complete analysis for a symbol across multiple timeframes"""
        # Analyze trend on daily timeframe
        trend = self.analyze_trend(data_d1, "D1")
        
        # Detect market structure on H4
        structure = self.detect_market_structure(data_h4)
        structure.trend = trend  # Update with daily trend
        
        # Identify zones on H4
        zones = self.identify_support_resistance_zones(data_h4)
        
        # Detect patterns on H1 (entry timeframe)
        patterns = self.detect_engulfing_patterns(data_h1)
        
        # Get current price
        current_price = data_h1['Close'].iloc[-1]
        
        # Check entry conditions
        signal = self.check_entry_conditions(symbol, trend, zones, patterns, current_price)
        
        # Check re-entry conditions if there's an open trade
        reentry_signal = None
        if symbol in self.open_trades:
            reentry_signal = self.check_reentry_conditions(
                symbol, self.open_trades[symbol], trend, zones, patterns, current_price
            )
        
        # Update trailing stop if applicable
        trailing_update = None
        if symbol in self.open_trades:
            trailing_update = self.update_trailing_stop(self.open_trades[symbol], current_price)
        
        # Store analysis results
        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'trend': trend,
            'market_structure': structure,
            'zones': zones,
            'patterns': patterns,
            'current_price': current_price,
            'signal': signal,
            'reentry_signal': reentry_signal,
            'trailing_update': trailing_update,
            'analysis_quality': self._calculate_analysis_quality(structure, zones, patterns)
        }
        
        self.last_analysis[symbol] = analysis_result
        return analysis_result
    
    def _calculate_analysis_quality(self, structure: MarketStructure, 
                                  zones: List[SupportResistanceZone],
                                  patterns: List[CandlestickPattern]) -> float:
        """Calculate the quality/confidence of the analysis"""
        # Structure quality
        structure_quality = structure.structure_strength
        
        # Zone quality (average strength of strong zones)
        zone_quality = 0.0
        if zones:
            strong_zones = [z for z in zones if z.strength >= self.zone_strength_threshold]
            if strong_zones:
                zone_quality = sum(z.strength for z in strong_zones) / len(strong_zones)
        
        # Pattern quality (confidence of latest pattern)
        pattern_quality = 0.0
        if patterns:
            pattern_quality = patterns[-1].confidence
        
        # Overall quality (weighted average)
        overall_quality = (structure_quality * 0.3 + 
                          zone_quality * 0.4 + 
                          pattern_quality * 0.3)
        
        return min(overall_quality, 1.0)
    
    def log_trade(self, trade_signal: TradeSignal, action: str, 
                  additional_info: Dict[str, Any] = None):
        """Log trade information"""
        log_entry = {
            'timestamp': datetime.now(),
            'symbol': trade_signal.symbol,
            'action': action,
            'entry_type': trade_signal.entry_type.value,
            'entry_price': trade_signal.entry_price,
            'stop_loss': trade_signal.stop_loss,
            'take_profit': trade_signal.take_profit,
            'lot_size': trade_signal.lot_size,
            'confidence': trade_signal.confidence,
            'entry_reason': trade_signal.entry_reason,
            'trend_direction': trade_signal.trend_direction.value,
            'zone_levels': trade_signal.zone_levels,
            'pattern_type': trade_signal.pattern_info.pattern_type,
            'pattern_confidence': trade_signal.pattern_info.confidence
        }
        
        if additional_info:
            log_entry.update(additional_info)
        
        logger.info(f"Trade Log: {log_entry}")
        
        # TODO: Send alerts (terminal/email/push notification)
        self._send_alert(log_entry)
    
    def _send_alert(self, log_entry: Dict[str, Any]):
        """Send trade alert (placeholder for alert system)"""
        # TODO: Implement alert system
        # - Terminal alert
        # - Email notification
        # - Push notification
        pass
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy performance and status summary"""
        return {
            'strategy_name': 'Market Structure & Strategy',
            'active_symbols': list(self.open_trades.keys()),
            'open_trades_count': len(self.open_trades),
            'last_analysis': {symbol: analysis['timestamp'] for symbol, analysis in self.last_analysis.items()},
            'configuration': {
                'use_pairs': self.use_pairs,
                'risk_per_trade': self.risk_per_trade * 100,
                'risk_reward_ratio': self.risk_reward_ratio,
                'enable_trailing_stop': self.enable_trailing_stop
            },
            'performance_metrics': {
                'total_signals': len(self.last_analysis),
                'average_confidence': np.mean([a['signal'].confidence for a in self.last_analysis.values() if a['signal']]) if any(a['signal'] for a in self.last_analysis.values()) else 0.0,
                'average_analysis_quality': np.mean([a['analysis_quality'] for a in self.last_analysis.values()])
            }
        }
