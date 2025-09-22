#!/usr/bin/env python3
"""
Advanced Backtesting Framework
==============================

This module provides comprehensive backtesting capabilities including:
- Walk-forward analysis
- Monte Carlo simulation
- Performance attribution
- Risk-adjusted metrics
- Strategy comparison
- Out-of-sample testing
- Portfolio-level backtesting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = 'unknown'
    strategy: str = 'default'

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    consecutive_wins: int
    consecutive_losses: int
    var_95: float
    var_99: float
    expected_shortfall: float
    skewness: float
    kurtosis: float
    start_date: datetime
    end_date: datetime
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None

class AdvancedBacktester:
    """Advanced Backtesting Framework"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the backtester"""
        self.config = config
        
        # Backtesting parameters
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05%
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        
        # Portfolio parameters
        self.max_positions = config.get('max_positions', 5)
        self.position_sizing = config.get('position_sizing', 'fixed')  # 'fixed', 'kelly', 'volatility'
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2%
        
        # State tracking
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        logger.info("Advanced Backtester initialized")
    
    def run_backtest(self, data: pd.DataFrame, strategy_func: callable, 
                    benchmark_data: Optional[pd.DataFrame] = None) -> BacktestResults:
        """
        Run comprehensive backtest
        
        Args:
            data: Historical price data
            strategy_func: Strategy function that returns signals
            benchmark_data: Benchmark data for comparison
            
        Returns:
            BacktestResults object
        """
        logger.info("Starting backtest...")
        
        # Reset state
        self._reset_state()
        
        # Generate signals
        signals = strategy_func(data)
        
        # Process each time period
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # Update existing positions
            self._update_positions(timestamp, current_price, row)
            
            # Check for new signals
            if i < len(signals):
                signal = signals.iloc[i]
                self._process_signal(timestamp, signal, current_price, row)
            
            # Update equity curve
            self._update_equity_curve(timestamp)
        
        # Close all remaining positions
        self._close_all_positions(data.index[-1], data.iloc[-1]['close'])
        
        # Calculate results
        results = self._calculate_results(data, benchmark_data)
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        
        return results
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, strategy_func: callable,
                                 train_period: int = 252, test_period: int = 63,
                                 step_size: int = 21) -> List[BacktestResults]:
        """
        Run walk-forward analysis
        
        Args:
            data: Historical price data
            strategy_func: Strategy function
            train_period: Training period in days
            test_period: Testing period in days
            step_size: Step size for rolling window
            
        Returns:
            List of BacktestResults for each test period
        """
        logger.info("Starting walk-forward analysis...")
        
        results = []
        total_periods = len(data)
        
        for start_idx in range(train_period, total_periods - test_period, step_size):
            # Define train and test periods
            train_start = start_idx - train_period
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + test_period, total_periods)
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            logger.info(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Train strategy on training data
            # Note: This is simplified - in practice, you'd retrain models here
            strategy_func(train_data)
            
            # Test on out-of-sample data
            result = self.run_backtest(test_data, strategy_func)
            result.start_date = test_data.index[0]
            result.end_date = test_data.index[-1]
            results.append(result)
        
        logger.info(f"Walk-forward analysis completed. {len(results)} periods tested.")
        
        return results
    
    def run_monte_carlo_simulation(self, backtest_results: BacktestResults, 
                                  n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on backtest results
        
        Args:
            backtest_results: Results from backtest
            n_simulations: Number of simulations to run
            
        Returns:
            Monte Carlo simulation results
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations...")
        
        # Extract trade returns
        trade_returns = [trade.pnl_pct for trade in backtest_results.trades]
        
        if len(trade_returns) < 10:
            logger.warning("Insufficient trades for Monte Carlo simulation")
            return {}
        
        # Run simulations
        simulation_results = []
        
        for _ in range(n_simulations):
            # Bootstrap sample of trades
            simulated_trades = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + np.array(simulated_trades))
            
            # Calculate metrics
            total_return = cumulative_returns[-1] - 1
            volatility = np.std(simulated_trades) * np.sqrt(252)
            sharpe_ratio = (np.mean(simulated_trades) * 252 - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            simulation_results.append({
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': cumulative_returns[-1] * self.initial_capital
            })
        
        # Calculate statistics
        results_df = pd.DataFrame(simulation_results)
        
        monte_carlo_results = {
            'simulations': simulation_results,
            'statistics': {
                'mean_return': results_df['total_return'].mean(),
                'std_return': results_df['total_return'].std(),
                'percentile_5': results_df['total_return'].quantile(0.05),
                'percentile_25': results_df['total_return'].quantile(0.25),
                'percentile_50': results_df['total_return'].quantile(0.50),
                'percentile_75': results_df['total_return'].quantile(0.75),
                'percentile_95': results_df['total_return'].quantile(0.95),
                'mean_sharpe': results_df['sharpe_ratio'].mean(),
                'mean_max_drawdown': results_df['max_drawdown'].mean(),
                'worst_case_return': results_df['total_return'].min(),
                'best_case_return': results_df['total_return'].max()
            },
            'probability_of_loss': (results_df['total_return'] < 0).mean(),
            'probability_of_beating_benchmark': (results_df['total_return'] > 0.1).mean()  # Assuming 10% benchmark
        }
        
        logger.info("Monte Carlo simulation completed")
        
        return monte_carlo_results
    
    def _reset_state(self):
        """Reset backtester state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def _update_positions(self, timestamp: datetime, current_price: float, row: pd.Series):
        """Update existing positions"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            # Check stop loss
            if position['stop_loss'] is not None:
                if position['side'] == 'long' and current_price <= position['stop_loss']:
                    positions_to_close.append((symbol, 'stop_loss'))
                elif position['side'] == 'short' and current_price >= position['stop_loss']:
                    positions_to_close.append((symbol, 'stop_loss'))
            
            # Check take profit
            if position['take_profit'] is not None:
                if position['side'] == 'long' and current_price >= position['take_profit']:
                    positions_to_close.append((symbol, 'take_profit'))
                elif position['side'] == 'short' and current_price <= position['take_profit']:
                    positions_to_close.append((symbol, 'take_profit'))
        
        # Close positions
        for symbol, exit_reason in positions_to_close:
            self._close_position(timestamp, symbol, current_price, exit_reason)
    
    def _process_signal(self, timestamp: datetime, signal: pd.Series, current_price: float, row: pd.Series):
        """Process trading signal"""
        if signal.get('action') in ['buy', 'sell'] and len(self.positions) < self.max_positions:
            symbol = signal.get('symbol', 'default')
            
            # Calculate position size
            position_size = self._calculate_position_size(current_price, signal)
            
            if position_size > 0:
                # Open new position
                self._open_position(timestamp, symbol, signal['action'], current_price, position_size, signal)
    
    def _calculate_position_size(self, price: float, signal: pd.Series) -> float:
        """Calculate position size based on sizing method"""
        if self.position_sizing == 'fixed':
            return self.current_capital * self.risk_per_trade / price
        elif self.position_sizing == 'kelly':
            # Simplified Kelly criterion
            win_rate = signal.get('win_rate', 0.6)
            avg_win = signal.get('avg_win', 0.02)
            avg_loss = signal.get('avg_loss', 0.01)
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.01, min(kelly_fraction, 0.25))  # Bound Kelly fraction
            return self.current_capital * kelly_fraction / price
        elif self.position_sizing == 'volatility':
            # Volatility-adjusted position sizing
            volatility = signal.get('volatility', 0.02)
            target_vol = 0.02
            vol_adjustment = target_vol / volatility if volatility > 0 else 1.0
            return self.current_capital * self.risk_per_trade * vol_adjustment / price
        else:
            return self.current_capital * self.risk_per_trade / price
    
    def _open_position(self, timestamp: datetime, symbol: str, action: str, 
                      price: float, quantity: float, signal: pd.Series):
        """Open new position"""
        side = 'long' if action == 'buy' else 'short'
        
        # Apply slippage
        if side == 'long':
            entry_price = price * (1 + self.slippage_rate)
        else:
            entry_price = price * (1 - self.slippage_rate)
        
        # Calculate commission
        commission = abs(quantity * entry_price * self.commission_rate)
        
        # Update capital
        if side == 'long':
            self.current_capital -= (quantity * entry_price + commission)
        else:
            self.current_capital += (quantity * entry_price - commission)
        
        # Store position
        self.positions[symbol] = {
            'entry_time': timestamp,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'commission': commission,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'strategy': signal.get('strategy', 'default')
        }
    
    def _close_position(self, timestamp: datetime, symbol: str, price: float, exit_reason: str):
        """Close existing position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage
        if position['side'] == 'long':
            exit_price = price * (1 - self.slippage_rate)
        else:
            exit_price = price * (1 + self.slippage_rate)
        
        # Calculate P&L
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        pnl_pct = pnl / (position['entry_price'] * position['quantity'])
        
        # Calculate commission
        commission = abs(position['quantity'] * exit_price * self.commission_rate)
        total_commission = position['commission'] + commission
        
        # Update capital
        if position['side'] == 'long':
            self.current_capital += (position['quantity'] * exit_price - commission)
        else:
            self.current_capital -= (position['quantity'] * exit_price + commission)
        
        # Create trade record
        trade = Trade(
            entry_time=position['entry_time'],
            exit_time=timestamp,
            symbol=symbol,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            slippage=self.slippage_rate,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            exit_reason=exit_reason,
            strategy=position['strategy']
        )
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
    
    def _close_all_positions(self, timestamp: datetime, price: float):
        """Close all remaining positions"""
        for symbol in list(self.positions.keys()):
            self._close_position(timestamp, symbol, price, 'end_of_data')
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve"""
        # Calculate current portfolio value
        portfolio_value = self.current_capital
        
        # Add unrealized P&L from open positions
        for position in self.positions.values():
            # This is simplified - in practice, you'd use current market prices
            unrealized_pnl = 0  # Would calculate based on current price
            portfolio_value += unrealized_pnl
        
        self.equity_curve.append((timestamp, portfolio_value))
    
    def _calculate_results(self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            logger.warning("No trades executed during backtest")
            return self._create_empty_results()
        
        # Create equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        equity_curve = equity_df['equity']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_periods)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades])) if losing_trades else float('inf')
        
        largest_win = max([t.pnl_pct for t in self.trades]) if self.trades else 0
        largest_loss = min([t.pnl_pct for t in self.trades]) if self.trades else 0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0
        
        # Expected shortfall (CVaR)
        tail_returns = returns[returns <= var_95]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else 0
        
        # Higher moments
        skewness = returns.skew() if len(returns) > 0 else 0
        kurtosis = returns.kurtosis() if len(returns) > 0 else 0
        
        # Benchmark comparison
        benchmark_return = None
        alpha = None
        beta = None
        information_ratio = None
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            benchmark_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0] - 1)
            
            # Align returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 1:
                # Calculate beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate alpha
                alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
                
                # Information ratio
                excess_returns = aligned_returns - aligned_benchmark
                information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_curve,
            returns=returns,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            skewness=skewness,
            kurtosis=kurtosis,
            start_date=data.index[0],
            end_date=data.index[-1],
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio
        )
    
    def _create_empty_results(self) -> BacktestResults:
        """Create empty results for cases with no trades"""
        return BacktestResults(
            trades=[],
            equity_curve=pd.Series([self.initial_capital]),
            returns=pd.Series([0]),
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_wins=0,
            consecutive_losses=0,
            var_95=0,
            var_99=0,
            expected_shortfall=0,
            skewness=0,
            kurtosis=0,
            start_date=datetime.now(),
            end_date=datetime.now()
        )
    
    def _calculate_max_drawdown_duration(self, drawdown_periods: pd.Series) -> int:
        """Calculate maximum drawdown duration"""
        max_duration = 0
        current_duration = 0
        
        for is_drawdown in drawdown_periods:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not self.trades:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def plot_results(self, results: BacktestResults, save_path: Optional[str] = None):
        """Plot backtest results"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value')
        
        # Drawdown
        peak = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - peak) / peak
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        
        # Returns distribution
        axes[1, 0].hist(results.returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        
        # Trade P&L
        trade_pnl = [trade.pnl_pct for trade in results.trades]
        axes[1, 1].bar(range(len(trade_pnl)), trade_pnl, 
                      color=['green' if pnl > 0 else 'red' for pnl in trade_pnl])
        axes[1, 1].set_title('Trade P&L')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('P&L %')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtest report"""
        report = f"""
BACKTEST REPORT
===============

Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${results.equity_curve.iloc[-1]:,.2f}

PERFORMANCE METRICS
-------------------
Total Return: {results.total_return:.2%}
Annualized Return: {results.annualized_return:.2%}
Volatility: {results.volatility:.2%}
Sharpe Ratio: {results.sharpe_ratio:.2f}
Sortino Ratio: {results.sortino_ratio:.2f}
Calmar Ratio: {results.calmar_ratio:.2f}
Maximum Drawdown: {results.max_drawdown:.2%}
Max Drawdown Duration: {results.max_drawdown_duration} days

TRADE STATISTICS
----------------
Total Trades: {results.total_trades}
Winning Trades: {results.winning_trades}
Losing Trades: {results.losing_trades}
Win Rate: {results.win_rate:.2%}
Profit Factor: {results.profit_factor:.2f}
Average Win: {results.avg_win:.2%}
Average Loss: {results.avg_loss:.2%}
Largest Win: {results.largest_win:.2%}
Largest Loss: {results.largest_loss:.2%}
Max Consecutive Wins: {results.consecutive_wins}
Max Consecutive Losses: {results.consecutive_losses}

RISK METRICS
------------
VaR (95%): {results.var_95:.2%}
VaR (99%): {results.var_99:.2%}
Expected Shortfall: {results.expected_shortfall:.2%}
Skewness: {results.skewness:.2f}
Kurtosis: {results.kurtosis:.2f}
"""
        
        if results.benchmark_return is not None:
            report += f"""
BENCHMARK COMPARISON
-------------------
Benchmark Return: {results.benchmark_return:.2%}
Alpha: {results.alpha:.2%}
Beta: {results.beta:.2f}
Information Ratio: {results.information_ratio:.2f}
"""
        
        return report
