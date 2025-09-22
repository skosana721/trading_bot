#!/usr/bin/env python3
"""
Reinforcement Learning Trading System
Implements Q-learning for adaptive trading based on market conditions and outcomes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pickle
import os
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class TrendCondition(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"

class StructureType(Enum):
    HH_HL = "hh_hl"  # Higher Highs / Higher Lows
    LH_LL = "lh_ll"  # Lower Highs / Lower Lows
    MIXED = "mixed"
    NONE = "none"

class ZoneProximity(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"

class CandlestickPattern(Enum):
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    NONE = "none"

class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class TradeOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    NONE = "none"

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class TradingSession(Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "ny"
    OVERLAP = "overlap"
    CLOSED = "closed"

@dataclass
class State:
    """Enhanced trading state representation v2.0"""
    # Core market conditions
    trend_condition: TrendCondition
    structure_type: StructureType
    zone_proximity: ZoneProximity
    candlestick_pattern: CandlestickPattern
    
    # Volatility measures
    atr_normalized: float  # Normalized ATR
    bb_width: float  # Bollinger Band width
    
    # Technical indicators (normalized)
    rsi_normalized: float  # RSI normalized to [0,1]
    macd_slope: float  # MACD slope
    ema_angle: float  # EMA angle in degrees
    
    # Volume and market depth
    volume_ratio: float  # Current volume vs average
    market_depth: float  # Simplified market depth indicator
    
    # Time features
    trading_session: str  # Asia, London, NY
    day_of_week: int  # 0=Monday, 6=Sunday
    
    # Recent trade history
    recent_outcome: TradeOutcome
    win_loss_ratio: float  # Last N trades win/loss ratio
    consecutive_losses: int  # Number of consecutive losses
    
    def to_tuple(self) -> Tuple:
        """Convert state to tuple for Q-table key"""
        return (
            self.trend_condition.value,
            self.structure_type.value,
            self.zone_proximity.value,
            self.candlestick_pattern.value,
            round(self.atr_normalized, 2),
            round(self.bb_width, 2),
            round(self.rsi_normalized, 2),
            round(self.macd_slope, 2),
            round(self.ema_angle, 1),
            round(self.volume_ratio, 2),
            round(self.market_depth, 2),
            self.trading_session,
            self.day_of_week,
            self.recent_outcome.value,
            round(self.win_loss_ratio, 2),
            self.consecutive_losses
        )
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

@dataclass
class Trade:
    """Trade record for RL learning"""
    timestamp: datetime
    action: Action
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    outcome: Optional[TradeOutcome] = None
    profit_loss: Optional[float] = None
    state: Optional[State] = None
    next_state: Optional[State] = None
    risk_amount: Optional[float] = None
    market_regime: Optional[MarketRegime] = None

@dataclass
class Experience:
    """Experience replay buffer entry"""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    timestamp: datetime

class ExperienceReplayBuffer:
    """Experience replay buffer for stable learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Experience] = []
        self.position = 0
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class MarketRegimeDetector:
    """Detect market regime (trending vs ranging)"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if len(data) < self.lookback_period:
            return MarketRegime.UNKNOWN
        
        # Calculate trend strength
        prices = data['Close'].tail(self.lookback_period)
        linear_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        trend_strength = abs(linear_trend) / prices.mean()
        
        # Calculate volatility
        returns = data['Close'].pct_change().tail(self.lookback_period)
        volatility = returns.std()
        
        # Calculate range vs trend
        price_range = (prices.max() - prices.min()) / prices.mean()
        
        # Determine regime
        if trend_strength > 0.001 and price_range > 0.02:
            return MarketRegime.TRENDING
        elif volatility > 0.02:
            return MarketRegime.VOLATILE
        elif price_range < 0.01:
            return MarketRegime.RANGING
        else:
            return MarketRegime.UNKNOWN

class ReinforcementLearningTrader:
    """Reinforcement Learning Trading System using Q-learning"""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 model_path: str = "models/rl_trader.pkl",
                 use_experience_replay: bool = True,
                 replay_buffer_size: int = 10000,
                 batch_size: int = 32,
                 use_market_regime: bool = True):
        """
        Initialize RL Trader
        
        Args:
            learning_rate: Q-learning learning rate (alpha)
            discount_factor: Future reward discount factor (gamma)
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            model_path: Path to save/load Q-table
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model_path = model_path
        
        # Q-table: {state: {action: q_value}}
        self.q_table = {}
        
        # Experience replay buffer
        self.use_experience_replay = use_experience_replay
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size) if use_experience_replay else None
        self.batch_size = batch_size
        
        # Market regime detection
        self.use_market_regime = use_market_regime
        self.regime_detector = MarketRegimeDetector() if use_market_regime else None
        self.current_regime = MarketRegime.UNKNOWN
        
        # Trading history
        self.trade_history: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Risk management
        self.max_drawdown_limit = 0.1  # 10% max drawdown
        self.cooldown_period = 0  # Cooldown after consecutive losses
        
        # Load existing model if available
        self.load_model()
        
        logger.info(f"RL Trader initialized with epsilon={self.epsilon:.3f}")
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(data) < period:
            return 0.0
        
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        tr1 = high - low
        tr2 = abs(high - np.roll(close, 1))
        tr3 = abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        
        return atr
    
    def _calculate_bb_width(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Bollinger Band width"""
        if len(data) < period:
            return 0.0
        
        close = data['Close'].values
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        
        bb_upper = sma + (2 * std)
        bb_lower = sma - (2 * std)
        bb_width = (bb_upper - bb_lower) / sma
        
        return bb_width
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        if len(data) < period + 1:
            return 50.0
        
        close = data['Close'].values
        delta = np.diff(close)
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_slope(self, data: pd.DataFrame) -> float:
        """Calculate MACD slope"""
        if len(data) < 26:
            return 0.0
        
        close = data['Close'].values
        ema12 = self._calculate_ema(close, 12)
        ema26 = self._calculate_ema(close, 26)
        
        macd = ema12 - ema26
        macd_slope = macd[-1] - macd[-2] if len(macd) > 1 else 0
        
        return macd_slope
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_ema_angle(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate EMA angle in degrees"""
        if len(data) < period:
            return 0.0
        
        close = data['Close'].values
        ema = self._calculate_ema(close, period)
        
        # Calculate angle using last 5 points
        if len(ema) >= 5:
            x = np.arange(5)
            y = ema[-5:]
            slope = np.polyfit(x, y, 1)[0]
            angle = np.degrees(np.arctan(slope))
            return angle
        
        return 0.0
    
    def _calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate volume ratio vs average"""
        if len(data) < period:
            return 1.0
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].tail(period).mean()
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _estimate_market_depth(self, data: pd.DataFrame) -> float:
        """Estimate market depth (simplified)"""
        # Simplified market depth based on price volatility and volume
        if len(data) < 10:
            return 0.5
        
        price_volatility = data['Close'].tail(10).std() / data['Close'].tail(10).mean()
        volume_stability = data['Volume'].tail(10).std() / data['Volume'].tail(10).mean()
        
        # Higher depth = lower volatility and higher volume stability
        depth = 1.0 - (price_volatility + volume_stability) / 2
        return max(0.0, min(1.0, depth))
    
    def _get_trading_session(self) -> str:
        """Get current trading session"""
        now = datetime.now()
        hour = now.hour
        
        # Simplified session detection
        if 0 <= hour < 8:
            return TradingSession.ASIA.value
        elif 8 <= hour < 16:
            return TradingSession.LONDON.value
        elif 16 <= hour < 24:
            return TradingSession.NEW_YORK.value
        else:
            return TradingSession.CLOSED.value
    
    def get_state(self, market_data: pd.DataFrame, analysis: Dict[str, Any]) -> State:
        """
        Extract enhanced state from market data and analysis v2.0
        
        Args:
            market_data: OHLCV market data
            analysis: Market analysis results
            
        Returns:
            State: Current trading state
        """
        # Core market conditions (existing logic)
        trend_analysis = analysis.get('trend_analysis', {})
        if trend_analysis.get('uptrend_confirmed', False):
            trend_condition = TrendCondition.UPTREND
        elif trend_analysis.get('downtrend_confirmed', False):
            trend_condition = TrendCondition.DOWNTREND
        else:
            trend_condition = TrendCondition.RANGE
        
        # Structure type
        structure_analysis = analysis.get('market_structure', {})
        hh_count = len(structure_analysis.get('higher_highs', []))
        hl_count = len(structure_analysis.get('higher_lows', []))
        lh_count = len(structure_analysis.get('lower_highs', []))
        ll_count = len(structure_analysis.get('lower_lows', []))
        
        if hh_count > 0 and hl_count > 0 and lh_count == 0 and ll_count == 0:
            structure_type = StructureType.HH_HL
        elif lh_count > 0 and ll_count > 0 and hh_count == 0 and hl_count == 0:
            structure_type = StructureType.LH_LL
        elif hh_count > 0 or hl_count > 0 or lh_count > 0 or ll_count > 0:
            structure_type = StructureType.MIXED
        else:
            structure_type = StructureType.NONE
        
        # Zone proximity
        zones = analysis.get('zones', [])
        current_price = market_data['Close'].iloc[-1]
        zone_proximity = ZoneProximity.NEUTRAL
        
        for zone in zones:
            if zone.low <= current_price <= zone.high:
                if zone.zone_type == "support":
                    zone_proximity = ZoneProximity.SUPPORT
                elif zone.zone_type == "resistance":
                    zone_proximity = ZoneProximity.RESISTANCE
                break
        
        # Candlestick pattern
        patterns = analysis.get('patterns', [])
        candlestick_pattern = CandlestickPattern.NONE
        
        if patterns:
            latest_pattern = patterns[-1]
            if latest_pattern.pattern_type == "bullish_engulfing":
                candlestick_pattern = CandlestickPattern.BULLISH_ENGULFING
            elif latest_pattern.pattern_type == "bearish_engulfing":
                candlestick_pattern = CandlestickPattern.BEARISH_ENGULFING
        
        # Enhanced features v2.0
        # Volatility measures
        atr = self._calculate_atr(market_data)
        atr_normalized = min(atr / current_price, 0.1)  # Normalize to [0, 0.1]
        
        bb_width = self._calculate_bb_width(market_data)
        
        # Technical indicators (normalized)
        rsi = self._calculate_rsi(market_data)
        rsi_normalized = rsi / 100.0  # Normalize to [0, 1]
        
        macd_slope = self._calculate_macd_slope(market_data)
        ema_angle = self._calculate_ema_angle(market_data)
        
        # Volume and market depth
        volume_ratio = self._calculate_volume_ratio(market_data)
        market_depth = self._estimate_market_depth(market_data)
        
        # Time features
        trading_session = self._get_trading_session()
        day_of_week = datetime.now().weekday()
        
        # Recent trade history
        recent_outcome = TradeOutcome.NONE
        win_loss_ratio = 0.5  # Default neutral
        consecutive_losses = self.consecutive_losses
        
        if self.trade_history:
            last_trade = self.trade_history[-1]
            if last_trade.outcome:
                recent_outcome = last_trade.outcome
            
            # Calculate win/loss ratio for last 10 trades
            recent_trades = self.trade_history[-10:]
            if recent_trades:
                wins = sum(1 for t in recent_trades if t.outcome == TradeOutcome.WIN)
                win_loss_ratio = wins / len(recent_trades)
        
        return State(
            trend_condition=trend_condition,
            structure_type=structure_type,
            zone_proximity=zone_proximity,
            candlestick_pattern=candlestick_pattern,
            atr_normalized=atr_normalized,
            bb_width=bb_width,
            rsi_normalized=rsi_normalized,
            macd_slope=macd_slope,
            ema_angle=ema_angle,
            volume_ratio=volume_ratio,
            market_depth=market_depth,
            trading_session=trading_session,
            day_of_week=day_of_week,
            recent_outcome=recent_outcome,
            win_loss_ratio=win_loss_ratio,
            consecutive_losses=consecutive_losses
        )
    
    def choose_action(self, state: State) -> Action:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current trading state
            
        Returns:
            Action: Chosen action (BUY/SELL/HOLD)
        """
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(list(Action))
            logger.debug(f"Exploration: Random action {action.value}")
        else:
            # Exploitation: best action based on Q-values
            q_values = self.get_q_values(state)
            action = max(q_values, key=q_values.get)
            logger.debug(f"Exploitation: Best action {action.value} (Q={q_values[action]:.4f})")
        
        return action
    
    def get_q_values(self, state: State) -> Dict[Action, float]:
        """
        Get Q-values for all actions in given state
        
        Args:
            state: Trading state
            
        Returns:
            Dict[Action, float]: Q-values for each action
        """
        state_key = state.to_tuple()
        
        if state_key not in self.q_table:
            # Initialize state with zero Q-values
            self.q_table[state_key] = {
                Action.BUY: 0.0,
                Action.SELL: 0.0,
                Action.HOLD: 0.0
            }
        
        return self.q_table[state_key]
    
    def update_q_value(self, state: State, action: Action, reward: float, next_state: State):
        """
        Update Q-value using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
        """
        state_key = state.to_tuple()
        next_state_key = next_state.to_tuple()
        
        # Get current Q-value
        current_q = self.get_q_values(state)[action]
        
        # Get max Q-value for next state
        next_q_values = self.get_q_values(next_state)
        max_next_q = max(next_q_values.values())
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state_key][action] = new_q
        
        logger.debug(f"Q-update: Q({state_key}, {action.value}) = {current_q:.4f} -> {new_q:.4f}")
    
    def calculate_reward(self, trade: Trade) -> float:
        """
        Calculate enhanced reward based on trade outcome v2.0
        
        Args:
            trade: Completed trade
            
        Returns:
            float: Reward value
        """
        if trade.outcome == TradeOutcome.WIN:
            # Base reward = profit
            base_reward = trade.profit_loss if trade.profit_loss else 1.0
            
            # Risk-adjusted reward
            if trade.risk_amount and trade.risk_amount > 0:
                risk_adjusted_reward = base_reward / trade.risk_amount
            else:
                risk_adjusted_reward = base_reward
            
            # Bonus for respecting SL/TP
            sl_tp_bonus = 0.1 if trade.stop_loss and trade.take_profit else 0.0
            
            # Market regime bonus
            regime_bonus = 0.05 if trade.market_regime == MarketRegime.TRENDING else 0.0
            
            reward = risk_adjusted_reward + sl_tp_bonus + regime_bonus
            
        elif trade.outcome == TradeOutcome.LOSS:
            # Base penalty = loss
            base_penalty = trade.profit_loss if trade.profit_loss else -1.0
            
            # Risk-adjusted penalty
            if trade.risk_amount and trade.risk_amount > 0:
                risk_adjusted_penalty = base_penalty / trade.risk_amount
            else:
                risk_adjusted_penalty = base_penalty
            
            # Penalty for violating SL/TP
            sl_tp_penalty = -0.2 if trade.stop_loss and trade.take_profit else 0.0
            
            # Consecutive loss penalty
            consecutive_penalty = -0.1 * self.consecutive_losses
            
            # Drawdown penalty
            drawdown_penalty = -0.3 if self.current_drawdown > self.max_drawdown_limit else 0.0
            
            reward = risk_adjusted_penalty + sl_tp_penalty + consecutive_penalty + drawdown_penalty
            
        else:
            # No trade or pending trade
            reward = 0.0
        
        return reward
    
    def execute_trade(self, action: Action, state: State, current_price: float, 
                     stop_loss: float, take_profit: float) -> Trade:
        """
        Execute a trade action
        
        Args:
            action: Action to execute
            state: Current state
            current_price: Current market price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Trade: Trade record
        """
        if action == Action.HOLD:
            return None
        
        trade = Trade(
            timestamp=datetime.now(),
            action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            state=state
        )
        
        self.current_trade = trade
        logger.info(f"Executed {action.value} trade at {current_price:.5f}")
        
        return trade
    
    def close_trade(self, exit_price: float, outcome: TradeOutcome):
        """
        Close current trade and calculate outcome
        
        Args:
            exit_price: Exit price
            outcome: Trade outcome
        """
        if not self.current_trade:
            return
        
        self.current_trade.exit_price = exit_price
        self.current_trade.outcome = outcome
        
        # Calculate profit/loss
        if self.current_trade.action == Action.BUY:
            self.current_trade.profit_loss = exit_price - self.current_trade.entry_price
        else:  # SELL
            self.current_trade.profit_loss = self.current_trade.entry_price - exit_price
        
        # Update performance metrics
        self.total_trades += 1
        if outcome == TradeOutcome.WIN:
            self.winning_trades += 1
            self.consecutive_losses = 0  # Reset consecutive losses
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses > self.max_consecutive_losses:
                self.max_consecutive_losses = self.consecutive_losses
        
        self.total_profit += self.current_trade.profit_loss
        
        # Update drawdown
        if self.total_profit > self.peak_balance:
            self.peak_balance = self.total_profit
        else:
            self.current_drawdown = self.peak_balance - self.total_profit
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Add to history
        self.trade_history.append(self.current_trade)
        
        logger.info(f"Closed trade: {outcome.value}, P/L: {self.current_trade.profit_loss:.5f}")
        
        self.current_trade = None
    
    def learn_from_trade(self, trade: Trade, next_state: State):
        """
        Learn from completed trade with enhanced features v2.0
        
        Args:
            trade: Completed trade
            next_state: State after trade
        """
        if not trade.state:
            return
        
        # Calculate reward
        reward = self.calculate_reward(trade)
        
        # Update Q-value
        self.update_q_value(trade.state, trade.action, reward, next_state)
        
        # Add to experience replay buffer
        if self.use_experience_replay and self.replay_buffer:
            experience = Experience(
                state=trade.state,
                action=trade.action,
                reward=reward,
                next_state=next_state,
                done=True,  # Trade is complete
                timestamp=datetime.now()
            )
            self.replay_buffer.add(experience)
            
            # Train on mini-batch from replay buffer
            if len(self.replay_buffer) >= self.batch_size:
                self._train_on_replay_buffer()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update market regime if available
        if self.use_market_regime and self.regime_detector:
            # This would be updated when new market data is available
            pass
        
        logger.debug(f"Learned from trade: reward={reward:.4f}, new epsilon={self.epsilon:.3f}")
    
    def _train_on_replay_buffer(self):
        """Train on mini-batch from experience replay buffer"""
        if not self.use_experience_replay or not self.replay_buffer:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        for experience in batch:
            # Update Q-value for each experience in batch
            self.update_q_value(
                experience.state,
                experience.action,
                experience.reward,
                experience.next_state
            )
    
    def get_trading_signal(self, market_data: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trading signal based on enhanced RL policy v2.0
        
        Args:
            market_data: Market data
            analysis: Market analysis
            
        Returns:
            Dict: Trading signal with action and confidence
        """
        # Detect market regime
        if self.use_market_regime and self.regime_detector:
            self.current_regime = self.regime_detector.detect_regime(market_data)
        
        # Check risk management conditions
        if self.cooldown_period > 0:
            self.cooldown_period -= 1
            return {
                'signal_type': 'HOLD',
                'action': 'hold',
                'confidence': 0.0,
                'reason': 'cooldown_period',
                'exploration_rate': self.epsilon,
                'market_regime': self.current_regime.value if self.current_regime else 'unknown'
            }
        
        if self.current_drawdown > self.max_drawdown_limit:
            return {
                'signal_type': 'HOLD',
                'action': 'hold',
                'confidence': 0.0,
                'reason': 'max_drawdown_exceeded',
                'exploration_rate': self.epsilon,
                'market_regime': self.current_regime.value if self.current_regime else 'unknown'
            }
        
        # Get current state
        current_state = self.get_state(market_data, analysis)
        
        # Choose action
        action = self.choose_action(current_state)
        
        # Get Q-values for confidence calculation
        q_values = self.get_q_values(current_state)
        max_q = max(q_values.values())
        min_q = min(q_values.values())
        
        # Calculate confidence based on Q-value difference
        if max_q == min_q:
            confidence = 0.5  # Equal Q-values
        else:
            confidence = (q_values[action] - min_q) / (max_q - min_q)
        
        # Adjust confidence based on market regime
        if self.current_regime == MarketRegime.RANGING and action != Action.HOLD:
            confidence *= 0.7  # Reduce confidence in ranging markets
        
        # Determine signal type
        if action == Action.BUY:
            signal_type = "BUY"
        elif action == Action.SELL:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
        
        return {
            'signal_type': signal_type,
            'action': action.value,
            'confidence': confidence,
            'q_values': {a.value: q for a, q in q_values.items()},
            'state': {
                'trend': current_state.trend_condition.value,
                'structure': current_state.structure_type.value,
                'zone': current_state.zone_proximity.value,
                'pattern': current_state.candlestick_pattern.value,
                'atr_normalized': current_state.atr_normalized,
                'rsi_normalized': current_state.rsi_normalized,
                'volume_ratio': current_state.volume_ratio,
                'trading_session': current_state.trading_session,
                'win_loss_ratio': current_state.win_loss_ratio,
                'consecutive_losses': current_state.consecutive_losses,
                'recent_outcome': current_state.recent_outcome.value
            },
            'exploration_rate': self.epsilon,
            'market_regime': self.current_regime.value if self.current_regime else 'unknown',
            'risk_metrics': {
                'current_drawdown': self.current_drawdown,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_losses': self.max_consecutive_losses
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get enhanced performance metrics v2.0
        
        Returns:
            Dict: Performance metrics
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'exploration_rate': self.epsilon,
            'q_table_size': len(self.q_table),
            'market_regime': self.current_regime.value if self.current_regime else 'unknown',
            'experience_replay_size': len(self.replay_buffer) if self.replay_buffer else 0,
            'risk_management': {
                'max_drawdown_limit': self.max_drawdown_limit,
                'cooldown_period': self.cooldown_period,
                'drawdown_exceeded': self.current_drawdown > self.max_drawdown_limit
            }
        }
    
    def save_model(self):
        """Save enhanced RL model to file"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'current_regime': self.current_regime.value if self.current_regime else 'unknown',
                    'performance': {
                        'total_trades': self.total_trades,
                        'winning_trades': self.winning_trades,
                        'total_profit': self.total_profit,
                        'max_drawdown': self.max_drawdown,
                        'consecutive_losses': self.consecutive_losses,
                        'max_consecutive_losses': self.max_consecutive_losses
                    },
                    'risk_management': {
                        'max_drawdown_limit': self.max_drawdown_limit,
                        'cooldown_period': self.cooldown_period
                    }
                }, f)
            logger.info(f"Enhanced RL model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")
    
    def load_model(self):
        """Load enhanced RL model from file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon)
                
                # Load market regime
                regime_value = data.get('current_regime', 'unknown')
                if regime_value != 'unknown':
                    self.current_regime = MarketRegime(regime_value)
                
                # Load performance metrics
                performance = data.get('performance', {})
                self.total_trades = performance.get('total_trades', 0)
                self.winning_trades = performance.get('winning_trades', 0)
                self.total_profit = performance.get('total_profit', 0.0)
                self.max_drawdown = performance.get('max_drawdown', 0.0)
                self.consecutive_losses = performance.get('consecutive_losses', 0)
                self.max_consecutive_losses = performance.get('max_consecutive_losses', 0)
                
                # Load risk management settings
                risk_mgmt = data.get('risk_management', {})
                self.max_drawdown_limit = risk_mgmt.get('max_drawdown_limit', 0.1)
                self.cooldown_period = risk_mgmt.get('cooldown_period', 0)
                
                logger.info(f"Enhanced RL model loaded from {self.model_path}")
                logger.info(f"Q-table size: {len(self.q_table)} states")
                logger.info(f"Market regime: {self.current_regime.value if self.current_regime else 'unknown'}")
            else:
                logger.info("No existing RL model found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
    
    def reset(self):
        """Reset enhanced RL trader (for testing)"""
        self.q_table = {}
        self.trade_history = []
        self.current_trade = None
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.current_regime = MarketRegime.UNKNOWN
        self.cooldown_period = 0
        self.epsilon = 0.3
        
        # Reset experience replay buffer
        if self.replay_buffer:
            self.replay_buffer.buffer.clear()
            self.replay_buffer.position = 0
        
        logger.info("Enhanced RL trader reset")
