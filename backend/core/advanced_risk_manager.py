#!/usr/bin/env python3
"""
Advanced Risk Management System
===============================

This module implements sophisticated risk management techniques including:
- Kelly Criterion position sizing
- Portfolio-level risk management
- Dynamic volatility-adjusted position sizing
- Correlation analysis between positions
- Real-time risk monitoring and circuit breakers
- Multi-timeframe risk assessment
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

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionType(Enum):
    """Position type enumeration"""
    LONG = "long"
    SHORT = "short"

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float  # Expected Shortfall (CVaR)
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    correlation_risk: float
    kelly_fraction: float
    optimal_position_size: float

@dataclass
class PositionRisk:
    """Risk assessment for individual position"""
    symbol: str
    position_type: PositionType
    current_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    volatility: float
    kelly_fraction: float
    optimal_size: float
    correlation_risk: float
    timestamp: datetime

class AdvancedRiskManager:
    """Advanced Risk Management System"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced risk manager"""
        self.config = config
        
        # Risk parameters
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2% max portfolio risk
        self.max_position_risk = config.get('max_position_risk', 0.005)  # 0.5% max position risk
        self.max_correlation = config.get('max_correlation', 0.7)  # Max correlation between positions
        self.var_confidence = config.get('var_confidence', 0.95)  # VaR confidence level
        self.lookback_period = config.get('lookback_period', 252)  # Trading days for calculations
        self.min_kelly_fraction = config.get('min_kelly_fraction', 0.01)  # Minimum Kelly fraction
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)  # Maximum Kelly fraction
        
        # Circuit breakers
        self.max_daily_loss = config.get('max_daily_loss', 0.01)  # 1% max daily loss
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.05)  # 5% max drawdown
        
        # State tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.daily_pnl: List[float] = []
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        logger.info("Advanced Risk Manager initialized")
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            
        Returns:
            Kelly fraction for position sizing
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self.min_kelly_fraction
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply bounds
        kelly_fraction = max(self.min_kelly_fraction, min(kelly_fraction, self.max_kelly_fraction))
        
        return kelly_fraction
    
    def calculate_volatility_adjusted_size(self, symbol: str, base_size: float, 
                                         volatility: float, target_vol: float = 0.02) -> float:
        """
        Calculate volatility-adjusted position size
        
        Args:
            symbol: Trading symbol
            base_size: Base position size
            volatility: Current volatility
            target_vol: Target volatility level
            
        Returns:
            Adjusted position size
        """
        if volatility <= 0:
            return base_size
        
        # Adjust size inversely to volatility
        volatility_ratio = target_vol / volatility
        adjusted_size = base_size * volatility_ratio
        
        # Apply maximum size limits
        max_size = self.config.get('max_position_size', 1.0)
        adjusted_size = min(adjusted_size, max_size)
        
        return max(0.01, adjusted_size)  # Minimum size
    
    def calculate_portfolio_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk for portfolio
        
        Args:
            returns: Portfolio returns series
            confidence: Confidence level for VaR
            
        Returns:
            Value at Risk
        """
        if len(returns) < 30:
            return 0.01  # Default 1% VaR if insufficient data
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Portfolio returns series
            confidence: Confidence level
            
        Returns:
            Expected Shortfall
        """
        if len(returns) < 30:
            return 0.015  # Default 1.5% ES if insufficient data
        
        var_threshold = self.calculate_portfolio_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return tail_returns.mean()
    
    def calculate_correlation_risk(self, symbol: str, new_position: Dict[str, Any]) -> float:
        """
        Calculate correlation risk for new position
        
        Args:
            symbol: Trading symbol
            new_position: New position details
            
        Returns:
            Correlation risk score (0-1)
        """
        if len(self.positions) == 0:
            return 0.0
        
        # Calculate correlation with existing positions
        max_correlation = 0.0
        
        for existing_symbol, existing_pos in self.positions.items():
            if existing_symbol == symbol:
                continue
            
            # Simplified correlation calculation
            # In practice, you'd use historical price data
            correlation = self._estimate_correlation(symbol, existing_symbol)
            max_correlation = max(max_correlation, abs(correlation))
        
        return max_correlation
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Estimate correlation between two symbols
        This is a simplified version - in practice, use historical data
        """
        # Currency pair correlations (simplified)
        correlations = {
            ('EURUSD', 'GBPUSD'): 0.7,
            ('EURUSD', 'USDJPY'): -0.5,
            ('GBPUSD', 'USDJPY'): -0.6,
            ('XAUUSD', 'USDJPY'): -0.3,
            ('XAUUSD', 'EURUSD'): 0.4,
        }
        
        # Check both directions
        key1 = (symbol1, symbol2)
        key2 = (symbol2, symbol1)
        
        if key1 in correlations:
            return correlations[key1]
        elif key2 in correlations:
            return correlations[key2]
        else:
            return 0.0  # Default no correlation
    
    def assess_position_risk(self, symbol: str, position_type: PositionType,
                           current_price: float, stop_loss: float, take_profit: float,
                           historical_data: pd.DataFrame) -> PositionRisk:
        """
        Assess risk for a new position
        
        Args:
            symbol: Trading symbol
            position_type: Long or short position
            current_price: Current market price
            stop_loss: Stop loss price
            take_profit: Take profit price
            historical_data: Historical price data
            
        Returns:
            PositionRisk assessment
        """
        # Calculate basic risk metrics
        if position_type == PositionType.LONG:
            risk_amount = current_price - stop_loss
            reward_amount = take_profit - current_price
        else:
            risk_amount = stop_loss - current_price
            reward_amount = current_price - take_profit
        
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Calculate volatility
        if len(historical_data) > 20:
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            volatility = 0.02  # Default 2% volatility
        
        # Calculate Kelly fraction (simplified)
        win_rate = 0.6  # Default win rate - should be calculated from historical data
        avg_win = reward_amount / current_price
        avg_loss = risk_amount / current_price
        
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Calculate optimal position size
        account_balance = self.config.get('account_balance', 10000)
        risk_per_trade = self.config.get('risk_per_trade', 0.01)
        max_risk_amount = account_balance * risk_per_trade
        
        optimal_size = min(
            max_risk_amount / risk_amount,
            account_balance * kelly_fraction / current_price
        )
        
        # Calculate correlation risk
        correlation_risk = self.calculate_correlation_risk(symbol, {
            'position_type': position_type,
            'current_price': current_price
        })
        
        return PositionRisk(
            symbol=symbol,
            position_type=position_type,
            current_price=current_price,
            position_size=optimal_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            volatility=volatility,
            kelly_fraction=kelly_fraction,
            optimal_size=optimal_size,
            correlation_risk=correlation_risk,
            timestamp=datetime.now()
        )
    
    def check_risk_limits(self, position_risk: PositionRisk) -> Tuple[bool, str]:
        """
        Check if position meets risk limits
        
        Args:
            position_risk: Position risk assessment
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check individual position risk
        if position_risk.risk_amount > self.max_position_risk:
            return False, f"Position risk {position_risk.risk_amount:.4f} exceeds limit {self.max_position_risk:.4f}"
        
        # Check correlation risk
        if position_risk.correlation_risk > self.max_correlation:
            return False, f"Correlation risk {position_risk.correlation_risk:.2f} exceeds limit {self.max_correlation:.2f}"
        
        # Check Kelly fraction
        if position_risk.kelly_fraction > self.max_kelly_fraction:
            return False, f"Kelly fraction {position_risk.kelly_fraction:.3f} exceeds limit {self.max_kelly_fraction:.3f}"
        
        # Check circuit breakers
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        if self.current_drawdown > self.max_drawdown_limit:
            return False, f"Drawdown {self.current_drawdown:.3f} exceeds limit {self.max_drawdown_limit:.3f}"
        
        return True, "Risk limits satisfied"
    
    def update_portfolio_metrics(self, daily_pnl: float):
        """
        Update portfolio-level risk metrics
        
        Args:
            daily_pnl: Daily profit/loss
        """
        self.daily_pnl.append(daily_pnl)
        
        # Update consecutive losses
        if daily_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update drawdown
        if daily_pnl > 0:
            self.peak_equity = max(self.peak_equity, sum(self.daily_pnl))
        
        current_equity = sum(self.daily_pnl)
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Keep only recent data
        if len(self.daily_pnl) > self.lookback_period:
            self.daily_pnl = self.daily_pnl[-self.lookback_period:]
    
    def get_portfolio_risk_metrics(self) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Returns:
            RiskMetrics object
        """
        if len(self.daily_pnl) < 30:
            return RiskMetrics(
                var_95=0.01,
                var_99=0.015,
                expected_shortfall=0.02,
                sharpe_ratio=0.0,
                max_drawdown=self.current_drawdown,
                volatility=0.02,
                correlation_risk=0.0,
                kelly_fraction=0.01,
                optimal_position_size=0.01
            )
        
        returns = pd.Series(self.daily_pnl)
        
        # Calculate VaR
        var_95 = self.calculate_portfolio_var(returns, 0.95)
        var_99 = self.calculate_portfolio_var(returns, 0.99)
        
        # Calculate Expected Shortfall
        expected_shortfall = self.calculate_expected_shortfall(returns, 0.95)
        
        # Calculate Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate correlation risk
        correlation_risk = 0.0
        if len(self.positions) > 1:
            correlations = []
            position_list = list(self.positions.values())
            for i in range(len(position_list)):
                for j in range(i + 1, len(position_list)):
                    corr = self._estimate_correlation(
                        position_list[i].symbol, 
                        position_list[j].symbol
                    )
                    correlations.append(abs(corr))
            correlation_risk = max(correlations) if correlations else 0.0
        
        # Calculate optimal Kelly fraction
        win_rate = len(returns[returns > 0]) / len(returns)
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
        
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.current_drawdown,
            volatility=volatility,
            correlation_risk=correlation_risk,
            kelly_fraction=kelly_fraction,
            optimal_position_size=kelly_fraction
        )
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be stopped due to risk limits
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check daily loss limit
        if len(self.daily_pnl) > 0 and self.daily_pnl[-1] < -self.max_daily_loss:
            return True, f"Daily loss {self.daily_pnl[-1]:.4f} exceeds limit {self.max_daily_loss:.4f}"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_limit:
            return True, f"Drawdown {self.current_drawdown:.3f} exceeds limit {self.max_drawdown_limit:.3f}"
        
        return False, "Risk limits satisfied"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary
        
        Returns:
            Risk summary dictionary
        """
        portfolio_metrics = self.get_portfolio_risk_metrics()
        
        return {
            'portfolio_metrics': {
                'var_95': portfolio_metrics.var_95,
                'var_99': portfolio_metrics.var_99,
                'expected_shortfall': portfolio_metrics.expected_shortfall,
                'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                'max_drawdown': portfolio_metrics.max_drawdown,
                'volatility': portfolio_metrics.volatility,
                'correlation_risk': portfolio_metrics.correlation_risk,
                'kelly_fraction': portfolio_metrics.kelly_fraction
            },
            'circuit_breakers': {
                'consecutive_losses': self.consecutive_losses,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl[-1] if self.daily_pnl else 0,
                'should_stop_trading': self.should_stop_trading()[0]
            },
            'positions': {
                symbol: {
                    'risk_amount': pos.risk_amount,
                    'correlation_risk': pos.correlation_risk,
                    'kelly_fraction': pos.kelly_fraction,
                    'optimal_size': pos.optimal_size
                }
                for symbol, pos in self.positions.items()
            },
            'timestamp': datetime.now().isoformat()
        }
