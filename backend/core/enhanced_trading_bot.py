#!/usr/bin/env python3
"""
Enhanced Trading Bot with Advanced Features
==========================================

This module integrates all the advanced improvements including:
- Advanced risk management with Kelly Criterion
- Enhanced ML ensemble with sophisticated algorithms
- Market regime detection
- Real-time data processing
- Comprehensive backtesting
- Multi-asset correlation analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import advanced components
try:
    from backend.core.advanced_risk_manager import AdvancedRiskManager, RiskMetrics
    from backend.strategies.advanced_ml_ensemble import AdvancedMLEnsemble
    from backend.strategies.market_regime_detector import MarketRegimeDetector, MarketRegime
    from backend.core.real_time_data_pipeline import RealTimeDataPipeline
    from backend.analysis.advanced_backtesting import AdvancedBacktester, BacktestResults
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Import existing components
try:
    from backend.core.mt5_trading_bot import MT5TradingBot
    from backend.connectors.mt5_connector import MT5Connector
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    """Enhanced Trading Bot with Advanced Features"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced trading bot"""
        self.config = config
        
        if not ADVANCED_FEATURES_AVAILABLE:
            raise ImportError("Advanced features not available. Please install required dependencies.")
        
        # Initialize components
        self.risk_manager = AdvancedRiskManager(config.get('risk_management', {}))
        self.ml_ensemble = AdvancedMLEnsemble(config.get('ml_ensemble', {}))
        self.regime_detector = MarketRegimeDetector(config.get('regime_detection', {}))
        self.data_pipeline = RealTimeDataPipeline(config.get('data_pipeline', {}))
        self.backtester = AdvancedBacktester(config.get('backtesting', {}))
        
        # Trading parameters
        self.symbols = config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        self.timeframes = config.get('timeframes', ['5m', '15m', '1h'])
        self.auto_trade = config.get('auto_trade', False)
        
        # State tracking
        self.is_running = False
        self.current_positions = {}
        self.performance_metrics = {}
        self.regime_history = []
        
        # MT5 integration (if available)
        self.mt5_connector = None
        if MT5_AVAILABLE:
            try:
                self.mt5_connector = MT5Connector()
            except Exception as e:
                logger.warning(f"MT5 connector not available: {e}")
        
        logger.info("Enhanced Trading Bot initialized")
    
    def start(self):
        """Start the enhanced trading bot"""
        if self.is_running:
            logger.warning("Bot is already running")
            return
        
        self.is_running = True
        
        # Start data pipeline
        self.data_pipeline.start()
        
        # Add event handlers
        self.data_pipeline.add_event_handler('price', self._handle_price_event)
        self.data_pipeline.add_event_handler('news', self._handle_news_event)
        self.data_pipeline.add_event_handler('economic', self._handle_economic_event)
        
        logger.info("Enhanced Trading Bot started")
    
    def stop(self):
        """Stop the enhanced trading bot"""
        self.is_running = False
        
        # Stop data pipeline
        self.data_pipeline.stop()
        
        # Close all positions
        self._close_all_positions()
        
        logger.info("Enhanced Trading Bot stopped")
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, BacktestResults]:
        """
        Run comprehensive backtest on historical data
        
        Args:
            data: Dictionary of symbol -> DataFrame
            benchmark_data: Optional benchmark data
            
        Returns:
            Dictionary of backtest results by symbol
        """
        logger.info("Starting comprehensive backtest...")
        
        results = {}
        
        for symbol, symbol_data in data.items():
            logger.info(f"Backtesting {symbol}...")
            
            # Create strategy function
            strategy_func = self._create_strategy_function(symbol)
            
            # Run backtest
            result = self.backtester.run_backtest(symbol_data, strategy_func, 
                                                benchmark_data.get(symbol) if benchmark_data else None)
            
            results[symbol] = result
            
            logger.info(f"Backtest completed for {symbol}: "
                       f"Return={result.total_return:.2%}, "
                       f"Sharpe={result.sharpe_ratio:.2f}, "
                       f"Max DD={result.max_drawdown:.2%}")
        
        return results
    
    def run_walk_forward_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[BacktestResults]]:
        """Run walk-forward analysis for all symbols"""
        logger.info("Starting walk-forward analysis...")
        
        results = {}
        
        for symbol, symbol_data in data.items():
            logger.info(f"Walk-forward analysis for {symbol}...")
            
            strategy_func = self._create_strategy_function(symbol)
            
            symbol_results = self.backtester.run_walk_forward_analysis(
                symbol_data, strategy_func,
                train_period=252,  # 1 year
                test_period=63,    # 3 months
                step_size=21       # 1 month
            )
            
            results[symbol] = symbol_results
            
            # Calculate average performance
            avg_return = np.mean([r.total_return for r in symbol_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in symbol_results])
            
            logger.info(f"Walk-forward analysis completed for {symbol}: "
                       f"Avg Return={avg_return:.2%}, "
                       f"Avg Sharpe={avg_sharpe:.2f}")
        
        return results
    
    def run_monte_carlo_analysis(self, backtest_results: Dict[str, BacktestResults], 
                                n_simulations: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Run Monte Carlo analysis on backtest results"""
        logger.info(f"Starting Monte Carlo analysis with {n_simulations} simulations...")
        
        results = {}
        
        for symbol, backtest_result in backtest_results.items():
            logger.info(f"Monte Carlo analysis for {symbol}...")
            
            mc_result = self.backtester.run_monte_carlo_simulation(
                backtest_result, n_simulations
            )
            
            results[symbol] = mc_result
            
            if mc_result:
                prob_loss = mc_result.get('probability_of_loss', 0)
                worst_case = mc_result.get('statistics', {}).get('worst_case_return', 0)
                
                logger.info(f"Monte Carlo analysis completed for {symbol}: "
                           f"Prob of Loss={prob_loss:.2%}, "
                           f"Worst Case={worst_case:.2%}")
        
        return results
    
    def _create_strategy_function(self, symbol: str):
        """Create strategy function for backtesting"""
        def strategy(data: pd.DataFrame) -> pd.DataFrame:
            signals = []
            
            for i in range(len(data)):
                if i < 50:  # Need enough data for analysis
                    signals.append({'action': 'hold', 'symbol': symbol})
                    continue
                
                # Get current data slice
                current_data = data.iloc[:i+1]
                
                # Detect market regime
                regime_metrics = self.regime_detector.detect_regime(current_data)
                
                # Get ML prediction
                ml_prediction = self._get_ml_prediction(current_data, symbol)
                
                # Get risk assessment
                risk_assessment = self._assess_trade_risk(current_data, symbol, regime_metrics)
                
                # Generate signal
                signal = self._generate_trading_signal(
                    current_data, symbol, regime_metrics, ml_prediction, risk_assessment
                )
                
                signals.append(signal)
            
            return pd.DataFrame(signals, index=data.index)
        
        return strategy
    
    def _get_ml_prediction(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get ML prediction for current market state"""
        try:
            # Create features
            features_df = self.ml_ensemble.create_advanced_features(data)
            
            # Select features
            if hasattr(self.ml_ensemble, 'feature_selector') and self.ml_ensemble.feature_selector:
                X = features_df.select_dtypes(include=[np.number]).fillna(0)
                y = (data['close'].shift(-1) > data['close']).astype(int).fillna(0)
                
                if len(X) > 0 and len(y) > 0:
                    X_selected = self.ml_ensemble.select_features(X, y, method='selectkbest', k=20)
                    
                    # Get prediction
                    if len(X_selected) > 0:
                        prediction, probabilities = self.ml_ensemble.predict(X_selected.iloc[-1:].values.reshape(1, -1))
                        
                        return {
                            'prediction': prediction[0],
                            'probability': probabilities[0][1] if probabilities is not None else 0.5,
                            'confidence': abs(probabilities[0][1] - 0.5) * 2 if probabilities is not None else 0.0
                        }
            
            return {'prediction': 0, 'probability': 0.5, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return {'prediction': 0, 'probability': 0.5, 'confidence': 0.0}
    
    def _assess_trade_risk(self, data: pd.DataFrame, symbol: str, 
                          regime_metrics) -> Dict[str, Any]:
        """Assess risk for potential trade"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Calculate stop loss and take profit
            atr = data['close'].rolling(20).std().iloc[-1]
            stop_loss = current_price - 2 * atr  # Simplified
            take_profit = current_price + 3 * atr  # Simplified
            
            # Assess position risk
            position_risk = self.risk_manager.assess_position_risk(
                symbol=symbol,
                position_type='long',  # Simplified
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                historical_data=data
            )
            
            # Check risk limits
            is_allowed, reason = self.risk_manager.check_risk_limits(position_risk)
            
            return {
                'position_risk': position_risk,
                'is_allowed': is_allowed,
                'reason': reason,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            logger.error(f"Error assessing trade risk: {e}")
            return {
                'position_risk': None,
                'is_allowed': False,
                'reason': f"Error: {e}",
                'stop_loss': None,
                'take_profit': None
            }
    
    def _generate_trading_signal(self, data: pd.DataFrame, symbol: str, 
                                regime_metrics, ml_prediction: Dict[str, Any], 
                                risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on all factors"""
        signal = {
            'action': 'hold',
            'symbol': symbol,
            'confidence': 0.0,
            'reason': 'No signal generated'
        }
        
        # Check if trade is allowed
        if not risk_assessment['is_allowed']:
            signal['reason'] = f"Risk limits: {risk_assessment['reason']}"
            return signal
        
        # Regime-based signal generation
        regime = regime_metrics.regime
        regime_confidence = regime_metrics.confidence
        
        # ML prediction
        ml_confidence = ml_prediction['confidence']
        ml_probability = ml_prediction['probability']
        
        # Combine signals
        if regime == MarketRegime.TRENDING_UP and regime_confidence > 0.7:
            if ml_probability > 0.6 and ml_confidence > 0.5:
                signal.update({
                    'action': 'buy',
                    'confidence': (regime_confidence + ml_confidence) / 2,
                    'reason': f'Trending up regime (conf: {regime_confidence:.2f}) + ML bullish (conf: {ml_confidence:.2f})',
                    'stop_loss': risk_assessment['stop_loss'],
                    'take_profit': risk_assessment['take_profit']
                })
        
        elif regime == MarketRegime.TRENDING_DOWN and regime_confidence > 0.7:
            if ml_probability < 0.4 and ml_confidence > 0.5:
                signal.update({
                    'action': 'sell',
                    'confidence': (regime_confidence + ml_confidence) / 2,
                    'reason': f'Trending down regime (conf: {regime_confidence:.2f}) + ML bearish (conf: {ml_confidence:.2f})',
                    'stop_loss': risk_assessment['stop_loss'],
                    'take_profit': risk_assessment['take_profit']
                })
        
        elif regime == MarketRegime.RANGING and regime_confidence > 0.6:
            # Mean reversion strategy in ranging markets
            if ml_probability > 0.7:  # Oversold
                signal.update({
                    'action': 'buy',
                    'confidence': regime_confidence * 0.8,
                    'reason': f'Ranging regime + Mean reversion buy signal',
                    'stop_loss': risk_assessment['stop_loss'],
                    'take_profit': risk_assessment['take_profit']
                })
            elif ml_probability < 0.3:  # Overbought
                signal.update({
                    'action': 'sell',
                    'confidence': regime_confidence * 0.8,
                    'reason': f'Ranging regime + Mean reversion sell signal',
                    'stop_loss': risk_assessment['stop_loss'],
                    'take_profit': risk_assessment['take_profit']
                })
        
        return signal
    
    def _handle_price_event(self, event):
        """Handle real-time price events"""
        try:
            symbol = event.symbol
            price_data = event.data
            
            # Update risk manager with new price
            # This would trigger position monitoring and risk checks
            
            # Check for trading opportunities
            if self.auto_trade:
                self._process_trading_opportunity(symbol, price_data)
                
        except Exception as e:
            logger.error(f"Error handling price event: {e}")
    
    def _handle_news_event(self, event):
        """Handle news events"""
        try:
            # Update sentiment analysis
            # Adjust risk parameters based on news impact
            # Trigger regime re-evaluation if high impact news
            
            if event.impact_level == 'high':
                logger.info(f"High impact news: {event.title}")
                # Could trigger position adjustments or trading halt
                
        except Exception as e:
            logger.error(f"Error handling news event: {e}")
    
    def _handle_economic_event(self, event):
        """Handle economic calendar events"""
        try:
            # Adjust trading parameters based on economic events
            # Could reduce position sizes or halt trading before high impact events
            
            if event.impact == 'high':
                logger.info(f"High impact economic event: {event.event}")
                
        except Exception as e:
            logger.error(f"Error handling economic event: {e}")
    
    def _process_trading_opportunity(self, symbol: str, price_data: Dict[str, Any]):
        """Process trading opportunity in real-time"""
        try:
            # This would implement real-time trading logic
            # For now, just log the opportunity
            
            logger.debug(f"Processing trading opportunity for {symbol}: {price_data}")
            
        except Exception as e:
            logger.error(f"Error processing trading opportunity: {e}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            if self.mt5_connector:
                # Close MT5 positions
                self.mt5_connector.close_all_positions()
            
            # Clear internal position tracking
            self.current_positions.clear()
            
            logger.info("All positions closed")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'risk_metrics': self.risk_manager.get_risk_summary(),
            'regime_summary': self.regime_detector.get_regime_summary(),
            'pipeline_metrics': self.data_pipeline.get_pipeline_metrics(),
            'current_positions': len(self.current_positions),
            'is_running': self.is_running,
            'timestamp': datetime.now().isoformat()
        }
    
    def train_ml_models(self, training_data: Dict[str, pd.DataFrame]):
        """Train ML models on historical data"""
        logger.info("Training ML models...")
        
        for symbol, data in training_data.items():
            logger.info(f"Training models for {symbol}...")
            
            # Create features
            features_df = self.ml_ensemble.create_advanced_features(data)
            
            # Prepare target variable
            X = features_df.select_dtypes(include=[np.number]).fillna(0)
            y = (data['close'].shift(-1) > data['close']).astype(int).fillna(0)
            
            if len(X) > 100 and len(y) > 100:  # Need sufficient data
                # Select features
                X_selected = self.ml_ensemble.select_features(X, y, method='selectkbest', k=30)
                
                # Train models
                self.ml_ensemble.train_models(X_selected, y)
                
                # Train ensemble
                self.ml_ensemble.train_ensemble(X_selected, y)
                
                logger.info(f"Models trained for {symbol}")
            else:
                logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
        
        logger.info("ML model training completed")
    
    def save_models(self, filepath: str):
        """Save trained models"""
        try:
            self.ml_ensemble.save_models(filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            self.ml_ensemble.load_models(filepath)
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def generate_trading_report(self, backtest_results: Dict[str, BacktestResults]) -> str:
        """Generate comprehensive trading report"""
        report = "ENHANCED TRADING BOT REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall performance
        total_return = np.mean([r.total_return for r in backtest_results.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in backtest_results.values()])
        avg_max_dd = np.mean([r.max_drawdown for r in backtest_results.values()])
        
        report += f"OVERALL PERFORMANCE\n"
        report += f"Average Return: {total_return:.2%}\n"
        report += f"Average Sharpe Ratio: {avg_sharpe:.2f}\n"
        report += f"Average Max Drawdown: {avg_max_dd:.2%}\n\n"
        
        # Per-symbol performance
        report += "PER-SYMBOL PERFORMANCE\n"
        report += "-" * 30 + "\n"
        
        for symbol, result in backtest_results.items():
            report += f"\n{symbol}:\n"
            report += f"  Return: {result.total_return:.2%}\n"
            report += f"  Sharpe: {result.sharpe_ratio:.2f}\n"
            report += f"  Max DD: {result.max_drawdown:.2%}\n"
            report += f"  Win Rate: {result.win_rate:.2%}\n"
            report += f"  Total Trades: {result.total_trades}\n"
        
        # Risk metrics
        risk_summary = self.risk_manager.get_risk_summary()
        report += f"\nRISK METRICS\n"
        report += "-" * 15 + "\n"
        report += f"Portfolio VaR (95%): {risk_summary['portfolio_metrics']['var_95']:.2%}\n"
        report += f"Expected Shortfall: {risk_summary['portfolio_metrics']['expected_shortfall']:.2%}\n"
        report += f"Current Drawdown: {risk_summary['circuit_breakers']['current_drawdown']:.2%}\n"
        
        # Regime analysis
        regime_summary = self.regime_detector.get_regime_summary()
        report += f"\nMARKET REGIME\n"
        report += "-" * 15 + "\n"
        report += f"Current Regime: {regime_summary['current_regime']}\n"
        report += f"Confidence: {regime_summary['confidence']:.2f}\n"
        report += f"Duration: {regime_summary['duration']} periods\n"
        
        return report
