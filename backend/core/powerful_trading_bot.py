#!/usr/bin/env python3
"""
Powerful Trading Bot - Full Integration
======================================

This is the ultimate trading bot that combines all advanced features:
- Advanced risk management with Kelly Criterion
- Enhanced ML ensemble with sophisticated algorithms
- Market regime detection and adaptive strategies
- Real-time data processing with news sentiment
- Comprehensive backtesting and analysis
- Multi-asset correlation analysis
- Smart Money Concept integration
- Reinforcement learning
- Full MT5 integration
"""

import pandas as pd
import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Core imports
from dotenv import load_dotenv
load_dotenv()

# Advanced ML and Data Science
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
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
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Advanced Components
try:
    from backend.core.advanced_risk_manager import AdvancedRiskManager, RiskMetrics
    from backend.strategies.advanced_ml_ensemble import AdvancedMLEnsemble
    from backend.strategies.market_regime_detector import MarketRegimeDetector, MarketRegime
    from backend.core.real_time_data_pipeline import RealTimeDataPipeline
    from backend.analysis.advanced_backtesting import AdvancedBacktester, BacktestResults
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

# Existing Components
try:
    from backend.strategies.market_structure_strategy import MarketStructureStrategy
    from backend.strategies.smart_money_concept import SmartMoneyConcept
    from backend.strategies.reinforcement_learning_trader import ReinforcementLearningTrader
    from backend.connectors.mt5_connector import MT5Connector
    EXISTING_COMPONENTS_AVAILABLE = True
except ImportError:
    EXISTING_COMPONENTS_AVAILABLE = False

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)

class PowerfulTradingBot:
    """Ultimate Trading Bot with All Advanced Features"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the powerful trading bot"""
        self.config = config
        
        # Basic configuration
        self.symbol = config.get('symbol', 'EURUSD')
        self.timeframe = config.get('timeframe', '5m')
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.auto_trade = config.get('auto_trade', False)
        self.account_balance = config.get('account_balance', 10000)
        
        # Feature flags
        self.use_advanced_risk = config.get('use_advanced_risk', True)
        self.use_regime_detection = config.get('use_regime_detection', True)
        self.use_advanced_ml = config.get('use_advanced_ml', True)
        self.use_real_time_data = config.get('use_real_time_data', True)
        self.use_smart_money = config.get('use_smart_money', True)
        self.use_reinforcement_learning = config.get('use_reinforcement_learning', True)
        self.use_market_structure = config.get('use_market_structure', True)
        
        # Initialize logger
        self.logger = logging.getLogger('powerful_trading_bot')
        
        # State tracking
        self.is_running = False
        self.connected = False
        self.current_positions = {}
        self.performance_metrics = {}
        self.trade_history = []
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Powerful Trading Bot initialized with all advanced features")
    
    def _initialize_components(self):
        """Initialize all trading bot components"""
        
        # 1. Advanced Risk Management
        if self.use_advanced_risk and ADVANCED_FEATURES_AVAILABLE:
            risk_config = {
                'max_portfolio_risk': 0.02,
                'max_position_risk': 0.005,
                'max_correlation': 0.7,
                'max_drawdown_limit': 0.05,
                'max_consecutive_losses': 5,
                'account_balance': self.account_balance,
                'risk_per_trade': self.risk_per_trade
            }
            self.risk_manager = AdvancedRiskManager(risk_config)
            self.logger.info("Advanced Risk Management initialized")
        else:
            self.risk_manager = None
            self.logger.warning("Advanced Risk Management not available")
        
        # 2. Advanced ML Ensemble
        if self.use_advanced_ml and ADVANCED_FEATURES_AVAILABLE:
            ml_config = {
                'technical_indicators': [
                    'rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'williams_r',
                    'cci', 'adx', 'obv', 'volume_sma', 'price_momentum', 'volatility'
                ],
                'lookback_periods': [5, 10, 20, 50],
                'feature_lags': [1, 2, 3, 5],
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'use_voting': True,
                'use_stacking': True,
                'use_blending': True
            }
            self.ml_ensemble = AdvancedMLEnsemble(ml_config)
            self.logger.info("Advanced ML Ensemble initialized")
        else:
            self.ml_ensemble = None
            self.logger.warning("Advanced ML Ensemble not available")
        
        # 3. Market Regime Detection
        if self.use_regime_detection and ADVANCED_FEATURES_AVAILABLE:
            regime_config = {
                'lookback_period': 100,
                'volatility_window': 20,
                'trend_window': 50,
                'regime_threshold': 0.7,
                'trend_strength_threshold': 0.6,
                'range_threshold': 0.02
            }
            self.regime_detector = MarketRegimeDetector(regime_config)
            self.logger.info("Market Regime Detection initialized")
        else:
            self.regime_detector = None
            self.logger.warning("Market Regime Detection not available")
        
        # 4. Real-time Data Pipeline
        if self.use_real_time_data and ADVANCED_FEATURES_AVAILABLE:
            pipeline_config = {
                'update_frequency': 1.0,
                'buffer_size': 1000,
                'max_latency': 0.1
            }
            self.data_pipeline = RealTimeDataPipeline(pipeline_config)
            self.logger.info("Real-time Data Pipeline initialized")
        else:
            self.data_pipeline = None
            self.logger.warning("Real-time Data Pipeline not available")
        
        # 5. Smart Money Concept
        if self.use_smart_money and EXISTING_COMPONENTS_AVAILABLE:
            self.smc_analyzer = SmartMoneyConcept
            self.logger.info("Smart Money Concept initialized")
        else:
            self.smc_analyzer = None
            self.logger.warning("Smart Money Concept not available")
        
        # 6. Reinforcement Learning
        if self.use_reinforcement_learning and EXISTING_COMPONENTS_AVAILABLE:
            self.rl_trader = ReinforcementLearningTrader(
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=0.3,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                model_path=f"models/rl_trader_{self.symbol}_{self.timeframe}.pkl"
            )
            self.logger.info("Reinforcement Learning initialized")
        else:
            self.rl_trader = None
            self.logger.warning("Reinforcement Learning not available")
        
        # 7. Market Structure Strategy
        if self.use_market_structure and EXISTING_COMPONENTS_AVAILABLE:
            strategy_config = {
                'UsePairs': [self.symbol],
                'LotSizeInitial': 0.01,
                'LotSizeReEntry': 0.01,
                'RiskPerTrade': 2.0,
                'RiskRewardRatio': 2.0,
                'SL_Buffer_Pips': 10,
                'TP_Multiplier': 2.0
            }
            self.market_structure_strategy = MarketStructureStrategy(strategy_config)
            self.logger.info("Market Structure Strategy initialized")
        else:
            self.market_structure_strategy = None
            self.logger.warning("Market Structure Strategy not available")
        
        # 8. MT5 Connector
        try:
            self.mt5_connector = MT5Connector()
            self.logger.info("MT5 Connector initialized")
        except Exception as e:
            self.mt5_connector = None
            self.logger.warning(f"MT5 Connector not available: {e}")
        
        # 9. Advanced Backtester
        if ADVANCED_FEATURES_AVAILABLE:
            backtest_config = {
                'initial_capital': self.account_balance,
                'commission_rate': 0.001,
                'slippage_rate': 0.0005,
                'risk_free_rate': 0.02,
                'max_positions': 5,
                'position_sizing': 'kelly',
                'risk_per_trade': self.risk_per_trade
            }
            self.backtester = AdvancedBacktester(backtest_config)
            self.logger.info("Advanced Backtester initialized")
        else:
            self.backtester = None
            self.logger.warning("Advanced Backtester not available")
    
    def start(self):
        """Start the powerful trading bot"""
        if self.is_running:
            self.logger.warning("Bot is already running")
            return
        
        self.is_running = True
        
        # Connect to MT5
        if self.mt5_connector:
            self.connected = self.mt5_connector.connect()
            if self.connected:
                self.logger.info("Connected to MT5")
            else:
                self.logger.warning("Could not connect to MT5")
        
        # Start real-time data pipeline
        if self.data_pipeline:
            self.data_pipeline.start()
            self.logger.info("Real-time data pipeline started")
        
        # Start main trading loop
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("Powerful Trading Bot started")
    
    def stop(self):
        """Stop the powerful trading bot"""
        self.is_running = False
        
        # Stop data pipeline
        if self.data_pipeline:
            self.data_pipeline.stop()
        
        # Disconnect from MT5
        if self.mt5_connector and self.connected:
            self.mt5_connector.disconnect()
            self.connected = False
        
        self.logger.info("Powerful Trading Bot stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Get market data
                market_data = self._get_market_data()
                if market_data is None:
                    time.sleep(5)
                    continue
                
                # Analyze market conditions
                analysis = self._analyze_market(market_data)
                
                # Generate trading signals
                signals = self._generate_signals(market_data, analysis)
                
                # Execute trades
                if signals and self.auto_trade:
                    self._execute_trades(signals)
                
                # Monitor positions
                self._monitor_positions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                time.sleep(1)  # 1 second loop
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get market data from various sources"""
        try:
            # Try MT5 first
            if self.mt5_connector and self.connected:
                data = self.mt5_connector.get_historical_data(self.symbol, self.timeframe, 100)
                if data is not None and not data.empty:
                    # Ensure MT5 data has the right column names
                    if 'Close' in data.columns and 'close' not in data.columns:
                        data = data.rename(columns={'Close': 'close'})
                    if 'Open' in data.columns and 'open' not in data.columns:
                        data = data.rename(columns={'Open': 'open'})
                    if 'High' in data.columns and 'high' not in data.columns:
                        data = data.rename(columns={'High': 'high'})
                    if 'Low' in data.columns and 'low' not in data.columns:
                        data = data.rename(columns={'Low': 'low'})
                    if 'Volume' in data.columns and 'volume' not in data.columns:
                        data = data.rename(columns={'Volume': 'volume'})
                    return data
            
            # Fallback to yfinance
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1mo", interval="5m")
            if not data.empty:
                data = data.reset_index()
                # Handle different column names from yfinance
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'datetime'})
                elif 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'datetime'})
                
                # Ensure we have the right columns
                required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    # Try to map common column names
                    column_mapping = {
                        'Open': 'open', 'High': 'high', 'Low': 'low', 
                        'Close': 'close', 'Volume': 'volume'
                    }
                    for old_col, new_col in column_mapping.items():
                        if old_col in data.columns:
                            data = data.rename(columns={old_col: new_col})
                
                return data.tail(100)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
    
    def _analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive market analysis"""
        analysis = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'data_points': len(data)
        }
        
        # 1. Market Regime Analysis
        if self.regime_detector:
            try:
                # Ensure data has the required columns
                if 'close' not in data.columns and 'Close' in data.columns:
                    data = data.rename(columns={'Close': 'close'})
                if 'close' not in data.columns:
                    self.logger.warning("No 'close' column found in data for regime detection")
                    analysis['regime'] = {'type': 'unknown', 'confidence': 0.0}
                else:
                    regime_metrics = self.regime_detector.detect_regime(data)
                    analysis['regime'] = {
                        'type': regime_metrics.regime.value if regime_metrics else 'unknown',
                        'confidence': regime_metrics.confidence if regime_metrics else 0.0,
                        'trend_strength': regime_metrics.trend_strength if regime_metrics else 0.0,
                        'volatility_regime': regime_metrics.volatility_regime.value if regime_metrics else 'unknown'
                    }
            except Exception as e:
                self.logger.error(f"Error in regime detection: {e}")
                analysis['regime'] = {'type': 'unknown', 'confidence': 0.0}
        
        # 2. ML Analysis
        if self.ml_ensemble:
            try:
                # Ensure data has the required columns
                if 'close' not in data.columns and 'Close' in data.columns:
                    data = data.rename(columns={'Close': 'close'})
                if 'close' not in data.columns:
                    self.logger.warning("No 'close' column found in data for ML analysis")
                    analysis['ml_prediction'] = {'prediction': 0, 'probability': 0.5, 'confidence': 0.0}
                else:
                    # Create features
                    features_df = self.ml_ensemble.create_advanced_features(data)
                    X = features_df.select_dtypes(include=[np.number]).fillna(0)
                    
                    if len(X) > 0:
                        # Get prediction
                        prediction, probabilities = self.ml_ensemble.predict(X.iloc[-1:].values.reshape(1, -1))
                        analysis['ml_prediction'] = {
                            'prediction': prediction[0] if len(prediction) > 0 else 0,
                            'probability': probabilities[0][1] if probabilities is not None else 0.5,
                            'confidence': abs(probabilities[0][1] - 0.5) * 2 if probabilities is not None else 0.0
                        }
            except Exception as e:
                self.logger.error(f"Error in ML analysis: {e}")
                analysis['ml_prediction'] = {'prediction': 0, 'probability': 0.5, 'confidence': 0.0}
        
        # 3. Smart Money Concept Analysis
        if self.smc_analyzer:
            try:
                # Ensure data has the required columns
                if 'close' not in data.columns and 'Close' in data.columns:
                    data = data.rename(columns={'Close': 'close'})
                if 'close' not in data.columns:
                    self.logger.warning("No 'close' column found in data for SMC analysis")
                    analysis['smc'] = {'signals': {}}
                else:
                    smc = self.smc_analyzer(data, self.timeframe)
                    current_price = data['close'].iloc[-1] if len(data) > 0 else 0
                    smc_signals = smc.get_smc_signals(current_price)
                    analysis['smc'] = {
                        'order_blocks': len(smc.order_blocks) if hasattr(smc, 'order_blocks') else 0,
                        'fair_value_gaps': len(smc.fair_value_gaps) if hasattr(smc, 'fair_value_gaps') else 0,
                        'liquidity_zones': len(smc.liquidity_zones) if hasattr(smc, 'liquidity_zones') else 0,
                        'signals': smc_signals
                    }
            except Exception as e:
                self.logger.error(f"Error in SMC analysis: {e}")
                analysis['smc'] = {'signals': {}}
        
        # 4. Market Structure Analysis
        if self.market_structure_strategy:
            try:
                # Ensure data has the required columns
                if 'close' not in data.columns and 'Close' in data.columns:
                    data = data.rename(columns={'Close': 'close'})
                if 'close' not in data.columns:
                    self.logger.warning("No 'close' column found in data for market structure analysis")
                    analysis['market_structure'] = {'trend': 'unknown', 'should_trade': False}
                else:
                    # Use the correct method name
                    trend = self.market_structure_strategy.analyze_trend(data, self.timeframe)
                    analysis['market_structure'] = {
                        'trend': trend.value if hasattr(trend, 'value') else str(trend),
                        'structure_strength': 0.7,  # Default strength
                        'should_trade': True  # Default to allow trading
                    }
            except Exception as e:
                self.logger.error(f"Error in market structure analysis: {e}")
                analysis['market_structure'] = {'trend': 'unknown', 'should_trade': False}
        
        # 5. Reinforcement Learning Analysis
        if self.rl_trader:
            try:
                # Ensure data has the required columns
                if 'close' not in data.columns and 'Close' in data.columns:
                    data = data.rename(columns={'Close': 'close'})
                if 'close' not in data.columns or len(data) == 0:
                    self.logger.warning("No 'close' column found in data for RL analysis")
                    analysis['rl_action'] = 'HOLD'
                else:
                    # Create a simple state from the data
                    current_price = data['close'].iloc[-1]
                    # Create a basic state representation
                    state = {
                        'price': current_price,
                        'volume': data['volume'].iloc[-1] if 'volume' in data.columns else 0,
                        'trend': 1 if len(data) > 1 and data['close'].iloc[-1] > data['close'].iloc[-2] else -1
                    }
                    rl_action = self.rl_trader.choose_action(state)
                    analysis['rl_action'] = rl_action
            except Exception as e:
                self.logger.error(f"Error in RL analysis: {e}")
                analysis['rl_action'] = 'HOLD'
        
        # 6. Real-time Data Analysis
        if self.data_pipeline:
            try:
                real_time_features = self.data_pipeline.get_real_time_features(self.symbol)
                analysis['real_time'] = real_time_features
            except Exception as e:
                self.logger.error(f"Error in real-time analysis: {e}")
                analysis['real_time'] = {}
        
        return analysis
    
    def _generate_signals(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate comprehensive trading signals"""
        try:
            # Combine all analysis results
            signals = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'confidence': 0.0,
                'action': 'HOLD',
                'reason': 'No clear signal',
                'risk_assessment': {},
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }
            
            # 1. Risk Management Check
            if self.risk_manager:
                should_stop, reason = self.risk_manager.should_stop_trading()
                if should_stop:
                    signals['action'] = 'HOLD'
                    signals['reason'] = f"Risk management: {reason}"
                    return signals
            
            # 2. Regime-based Signal Generation
            regime = analysis.get('regime', {})
            regime_type = regime.get('type', 'unknown')
            regime_confidence = regime.get('confidence', 0.0)
            
            # 3. ML-based Signal Generation
            ml_prediction = analysis.get('ml_prediction', {})
            ml_confidence = ml_prediction.get('confidence', 0.0)
            ml_probability = ml_prediction.get('probability', 0.5)
            
            # 4. Smart Money Concept Signals
            smc = analysis.get('smc', {})
            smc_signals = smc.get('signals', {})
            
            # 5. Market Structure Signals
            market_structure = analysis.get('market_structure', {})
            structure_should_trade = market_structure.get('should_trade', False)
            
            # 6. Reinforcement Learning Signals
            rl_action = analysis.get('rl_action', 'HOLD')
            
            # Combine signals with weighted approach
            signal_weights = {
                'regime': 0.3,
                'ml': 0.25,
                'smc': 0.2,
                'structure': 0.15,
                'rl': 0.1
            }
            
            # Calculate combined signal
            buy_score = 0.0
            sell_score = 0.0
            
            # Regime contribution
            if regime_type == 'trending_up' and regime_confidence > 0.7:
                buy_score += signal_weights['regime']
            elif regime_type == 'trending_down' and regime_confidence > 0.7:
                sell_score += signal_weights['regime']
            
            # ML contribution
            if ml_confidence > 0.6:
                if ml_probability > 0.6:
                    buy_score += signal_weights['ml'] * ml_confidence
                elif ml_probability < 0.4:
                    sell_score += signal_weights['ml'] * ml_confidence
            
            # SMC contribution
            if 'buy' in smc_signals and smc_signals['buy']:
                buy_score += signal_weights['smc']
            elif 'sell' in smc_signals and smc_signals['sell']:
                sell_score += signal_weights['smc']
            
            # Structure contribution
            if structure_should_trade:
                if market_structure.get('trend') == 'uptrend':
                    buy_score += signal_weights['structure']
                elif market_structure.get('trend') == 'downtrend':
                    sell_score += signal_weights['structure']
            
            # RL contribution
            if rl_action == 'BUY':
                buy_score += signal_weights['rl']
            elif rl_action == 'SELL':
                sell_score += signal_weights['rl']
            
            # Determine final action
            min_confidence = 0.6  # Minimum confidence threshold
            
            if buy_score > sell_score and buy_score > min_confidence:
                signals['action'] = 'BUY'
                signals['confidence'] = buy_score
                signals['reason'] = f"Buy signal (score: {buy_score:.2f})"
            elif sell_score > buy_score and sell_score > min_confidence:
                signals['action'] = 'SELL'
                signals['confidence'] = sell_score
                signals['reason'] = f"Sell signal (score: {sell_score:.2f})"
            else:
                signals['action'] = 'HOLD'
                signals['confidence'] = max(buy_score, sell_score)
                signals['reason'] = f"Insufficient confidence (max: {max(buy_score, sell_score):.2f})"
            
            # Calculate position size and risk parameters
            if signals['action'] != 'HOLD':
                signals.update(self._calculate_position_parameters(data, signals, analysis))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return None
    
    def _calculate_position_parameters(self, data: pd.DataFrame, signals: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position size and risk parameters"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Calculate ATR for stop loss
            atr = data['close'].rolling(20).std().iloc[-1]
            if pd.isna(atr) or atr == 0:
                atr = current_price * 0.01  # 1% fallback
            
            # Calculate stop loss and take profit
            if signals['action'] == 'BUY':
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
            else:  # SELL
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
            
            # Calculate position size using Kelly Criterion if available
            position_size = 0.01  # Default position size
            
            if self.risk_manager:
                try:
                    # Assess position risk
                    position_risk = self.risk_manager.assess_position_risk(
                        symbol=self.symbol,
                        position_type='long' if signals['action'] == 'BUY' else 'short',
                        current_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        historical_data=data
                    )
                    
                    if position_risk:
                        position_size = position_risk.optimal_size
                        
                        # Check risk limits
                        is_allowed, reason = self.risk_manager.check_risk_limits(position_risk)
                        if not is_allowed:
                            signals['action'] = 'HOLD'
                            signals['reason'] = f"Risk limits: {reason}"
                            return signals
                
                except Exception as e:
                    self.logger.error(f"Error in risk assessment: {e}")
            
            # Adjust position size based on regime
            regime = analysis.get('regime', {})
            if regime.get('type') == 'high_volatility' and regime.get('confidence', 0) > 0.7:
                position_size *= 0.5  # Reduce position size in high volatility
            
            return {
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_price': current_price,
                'risk_amount': abs(current_price - stop_loss) * position_size,
                'potential_profit': abs(take_profit - current_price) * position_size
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position parameters: {e}")
            return {
                'position_size': 0.01,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'entry_price': 0.0,
                'risk_amount': 0.0,
                'potential_profit': 0.0
            }
    
    def _execute_trades(self, signals: Dict[str, Any]):
        """Execute trades based on signals"""
        try:
            if not self.connected or signals['action'] == 'HOLD':
                return
            
            # Execute trade via MT5
            if self.mt5_connector:
                trade_result = self.mt5_connector.place_order(
                    symbol=self.symbol,
                    order_type=signals['action'],
                    volume=signals['position_size'],
                    price=signals['entry_price'],
                    sl=signals['stop_loss'],
                    tp=signals['take_profit']
                )
                
                if trade_result:
                    self.logger.info(f"✅ Trade executed: {signals['action']} {self.symbol} at {signals['entry_price']}")
                    
                    # Update RL trader
                    if self.rl_trader:
                        self.rl_trader.update(signals, trade_result)
                    
                    # Record trade
                    self.trade_history.append({
                        'timestamp': signals['timestamp'],
                        'symbol': self.symbol,
                        'action': signals['action'],
                        'entry_price': signals['entry_price'],
                        'position_size': signals['position_size'],
                        'stop_loss': signals['stop_loss'],
                        'take_profit': signals['take_profit'],
                        'confidence': signals['confidence'],
                        'reason': signals['reason']
                    })
                else:
                    self.logger.warning(f"❌ Trade execution failed: {signals['action']} {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def _monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            if not self.connected:
                return
            
            # Get current positions
            positions = self.mt5_connector.get_positions() if self.mt5_connector else []
            
            for position in positions:
                if position['symbol'] == self.symbol:
                    # Update risk manager
                    if self.risk_manager:
                        # Update portfolio metrics with position P&L
                        pnl = position.get('profit', 0)
                        self.risk_manager.update_portfolio_metrics(pnl)
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            self.performance_metrics = {
                'timestamp': datetime.now(),
                'total_trades': len(self.trade_history),
                'winning_trades': len([t for t in self.trade_history if t.get('profit', 0) > 0]),
                'losing_trades': len([t for t in self.trade_history if t.get('profit', 0) < 0]),
                'current_positions': len(self.current_positions),
                'account_balance': self.account_balance
            }
            
            # Add risk metrics
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                self.performance_metrics['risk_metrics'] = risk_summary
            
            # Add regime metrics
            if self.regime_detector:
                regime_summary = self.regime_detector.get_regime_summary()
                self.performance_metrics['regime_metrics'] = regime_summary
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        return {
            'bot_status': {
                'is_running': self.is_running,
                'connected': self.connected,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'auto_trade': self.auto_trade
            },
            'features': {
                'advanced_risk_management': self.use_advanced_risk and self.risk_manager is not None,
                'market_regime_detection': self.use_regime_detection and self.regime_detector is not None,
                'advanced_ml': self.use_advanced_ml and self.ml_ensemble is not None,
                'real_time_data': self.use_real_time_data and self.data_pipeline is not None,
                'smart_money_concept': self.use_smart_money and self.smc_analyzer is not None,
                'reinforcement_learning': self.use_reinforcement_learning and self.rl_trader is not None,
                'market_structure': self.use_market_structure and self.market_structure_strategy is not None
            },
            'performance': self.performance_metrics,
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def run_backtest(self, data: pd.DataFrame) -> Optional[BacktestResults]:
        """Run comprehensive backtest"""
        if not self.backtester:
            self.logger.warning("Backtester not available")
            return None
        
        try:
            # Create strategy function
            def strategy_func(df):
                signals = []
                for i in range(len(df)):
                    if i < 100:  # Need enough data
                        signals.append({'action': 'hold', 'symbol': self.symbol})
                        continue
                    
                    current_data = df.iloc[:i+1]
                    analysis = self._analyze_market(current_data)
                    signal = self._generate_signals(current_data, analysis)
                    
                    if signal:
                        signals.append({
                            'action': signal['action'].lower(),
                            'symbol': self.symbol,
                            'stop_loss': signal.get('stop_loss'),
                            'take_profit': signal.get('take_profit')
                        })
                    else:
                        signals.append({'action': 'hold', 'symbol': self.symbol})
                
                return pd.DataFrame(signals, index=df.index)
            
            # Run backtest
            result = self.backtester.run_backtest(data, strategy_func)
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return None
