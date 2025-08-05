#!/usr/bin/env python3
"""
Flask Backend API for Dynamic Trading Bot
========================================

This Flask application provides a REST API for the trading bot with:
- XM account configuration
- Dynamic symbol and timeframe selection
- Auto trading controls
- Real-time market analysis
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import threading
import time
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Import trading bot components
from mt5_connector import MT5Connector
from mt5_trading_bot import MT5TradingBot

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

# Global variables for bot state
bot_instance = None
bot_thread = None
bot_running = False
bot_config = {
    'account_number': os.getenv('XM_ACCOUNT_NUMBER'),
    'password': os.getenv('XM_PASSWORD'),
    'server': os.getenv('XM_SERVER', 'XMGlobal-Demo'),
    'symbol': 'EURUSD',
    'timeframe': '5m',
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
            'account_size': None,  # No longer used - balance fetched from MT5
    'auto_trade': os.getenv('AUTO_TRADE', 'false').lower() == 'true',
    'use_ml': os.getenv('USE_ML', 'true').lower() == 'true'
}

# Available symbols and timeframes
AVAILABLE_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURGBP', 'EURJPY', 'GBPJPY', 'CHFJPY', 'AUDCAD', 'AUDCHF', 'AUDJPY',
    'AUDNZD', 'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY'
]

AVAILABLE_TIMEFRAMES = [
    {'value': '1m', 'label': '1 Minute'},
    {'value': '5m', 'label': '5 Minutes'},
    {'value': '15m', 'label': '15 Minutes'},
    {'value': '30m', 'label': '30 Minutes'},
    {'value': '1h', 'label': '1 Hour'},
    {'value': '4h', 'label': '4 Hours'},
    {'value': '1d', 'label': '1 Day'}
]

@app.route('/')
def index():
    """Serve the Vue frontend"""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current bot configuration"""
    return jsonify({
        'config': bot_config,
        'available_symbols': AVAILABLE_SYMBOLS,
        'available_timeframes': AVAILABLE_TIMEFRAMES
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update bot configuration"""
    global bot_config
    
    try:
        data = request.get_json()
        
        # Update configuration
        if 'account_number' in data:
            bot_config['account_number'] = data['account_number']
        if 'password' in data:
            bot_config['password'] = data['password']
        if 'server' in data:
            bot_config['server'] = data['server']
        if 'symbol' in data:
            bot_config['symbol'] = data['symbol']
        if 'timeframe' in data:
            bot_config['timeframe'] = data['timeframe']
        if 'risk_per_trade' in data:
            bot_config['risk_per_trade'] = float(data['risk_per_trade'])
        if 'auto_trade' in data:
            bot_config['auto_trade'] = bool(data['auto_trade'])
        if 'use_ml' in data:
            bot_config['use_ml'] = bool(data['use_ml'])
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/env-config', methods=['GET'])
def get_env_config():
    """Get configuration from environment variables"""
    env_config = {
        'account_number': os.getenv('XM_ACCOUNT_NUMBER', ''),
        'password': os.getenv('XM_PASSWORD', ''),
        'server': os.getenv('XM_SERVER', 'XMGlobal-Demo'),
        'symbol': 'EURUSD',
        'timeframe': '5m',
        'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
        'account_size': None,  # No longer used - balance fetched from MT5
        'auto_trade': os.getenv('AUTO_TRADE', 'false').lower() == 'true',
        'use_ml': os.getenv('USE_ML', 'true').lower() == 'true'
    }
    return jsonify({
        'config': env_config,
        'available_symbols': AVAILABLE_SYMBOLS,
        'available_timeframes': AVAILABLE_TIMEFRAMES
    })

@app.route('/api/connect', methods=['POST'])
def connect_mt5():
    """Connect to MT5 with XM credentials"""
    global bot_instance
    
    try:
        data = request.get_json()
        account_number = data.get('account_number')
        password = data.get('password')
        server = data.get('server', '')
        
        print(f"\n🔗 CONNECTING TO MT5")
        print(f"   Account: {account_number}")
        print(f"   Server: {server}")
        
        if not account_number or not password or not server:
            return jsonify({'success': False, 'error': 'Account number, password, and server are required'}), 400
        
        # Validate account number format
        try:
            int(account_number)
        except ValueError:
            return jsonify({'success': False, 'error': 'Account number must be numeric'}), 400
        
        # Create MT5 connector
        print("🔄 Creating MT5 connector...")
        connector = MT5Connector(account_number, password, server)
        
        # Try to connect
        print("🔄 Attempting connection...")
        if connector.connect():
            print("✅ MT5 connection successful")
            
            # Test basic functionality
            print("🧪 Testing MT5 functionality...")
            if not connector.test_mt5_functionality():
                connector.disconnect()
                return jsonify({
                    'success': False, 
                    'error': 'MT5 connection established but functionality test failed. Please check your MT5 terminal settings.'
                }), 400
            
            # Get account info
            account_info = connector.get_account_summary()
            if not account_info:
                connector.disconnect()
                return jsonify({'success': False, 'error': 'Failed to retrieve account information'}), 400
            
            print(f"💰 Account Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"💰 Account Equity: ${account_info.get('equity', 0):,.2f}")
            
            # Update global config with credentials
            bot_config.update({
                'account_number': account_number,
                'password': password,
                'server': server
            })
            
            # Create or update bot instance
            print("🤖 Creating/updating bot instance...")
            bot_instance = MT5TradingBot(
                symbol=bot_config.get('symbol', 'EURUSD'),
                timeframe=bot_config.get('timeframe', '5m'),
                risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                use_mt5_data=True,
                auto_trade=False,
                use_ml=bot_config.get('use_ml', True)
            )
            
            # Assign the working connector to the bot
            bot_instance.mt5_connector = connector
            bot_instance.connected = True
            # Account balance is now fetched dynamically from MT5
            
            # Test market data retrieval
            print("📊 Testing market data retrieval...")
            test_data = bot_instance.get_market_data()
            if test_data is None or len(test_data) < 10:
                print("⚠️  Limited market data available, but connection is working")
            else:
                print(f"✅ Market data test successful ({len(test_data)} data points)")
            
            return jsonify({
                'success': True,
                'message': 'Successfully connected to MT5',
                'account_info': {
                    'login': account_info.get('login'),
                    'server': account_info.get('server'),
                    'balance': account_info.get('balance', 0),
                    'equity': account_info.get('equity', 0),
                    'margin': account_info.get('margin', 0),
                    'margin_free': account_info.get('margin_free', 0),
                    'currency': account_info.get('currency', 'USD')
                },
                'connection_details': {
                    'account_number': account_number,
                    'server': server,
                    'connected': True,
                    'data_available': test_data is not None and len(test_data) > 0
                }
            })
        else:
            print("❌ MT5 connection failed")
            return jsonify({
                'success': False, 
                'error': 'Failed to connect to MT5. Please check your credentials, server name, and ensure MT5 terminal is running.'
            }), 400
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Connection error: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_market():
    """Perform market analysis for a symbol and timeframe"""
    global bot_instance
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not symbol or not timeframe:
            return jsonify({'success': False, 'error': 'Symbol and timeframe are required'}), 400
        
        # Check if we have an existing bot instance with MT5 connection
        if bot_instance and bot_instance.connected:
            # Update the existing bot with new symbol and timeframe
            bot_instance.symbol = symbol
            bot_instance.timeframe = timeframe
            bot_instance.risk_per_trade = bot_config.get('risk_per_trade', 0.02)
            bot_instance.use_ml = bot_config.get('use_ml', True)
        else:
            # Create new trading bot instance
            bot_instance = MT5TradingBot(
                symbol=symbol,
                timeframe=timeframe,
                risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                use_mt5_data=True,
                auto_trade=False,  # Don't auto trade for analysis
                use_ml=bot_config.get('use_ml', True)
            )
        
        # Update config
        bot_config['symbol'] = symbol
        bot_config['timeframe'] = timeframe
        
        # Run analysis
        analysis = bot_instance.run_analysis_cycle()
        print(f"Analysis result: {type(analysis)}")
        if analysis:
            print(f"Analysis keys: {list(analysis.keys())}")
            try:
                # Get ML prediction if available
                ml_prediction = None
                if bot_instance.use_ml and bot_instance.model_trained and bot_instance.analysis_bot and bot_instance.analysis_bot.data is not None:
                    ml_prediction = bot_instance.get_ml_prediction(bot_instance.analysis_bot.data)
                
                # Get trading signals
                signals = bot_instance.get_trading_signals(analysis)
                
                # Ensure analysis data is JSON serializable and has expected keys
                serializable_analysis = {}
                for key, value in analysis.items():
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        serializable_analysis[key] = value
                    else:
                        serializable_analysis[key] = str(value)
                
                # Ensure all expected keys are present for frontend compatibility
                expected_keys = {
                    'entry_conditions': [],
                    'uptrend_confirmed': False,
                    'downtrend_confirmed': False,
                    'trend_direction': 'SIDEWAYS',
                    'trend_strength': 0,
                    'overall_strength': 0,
                    'primary_timeframe': timeframe,
                    'primary_analysis': {
                        'hh_count': 0,
                        'hl_count': 0,
                        'lh_count': 0,
                        'll_count': 0,
                        'higher_highs': [],
                        'higher_lows': [],
                        'lower_highs': [],
                        'lower_lows': [],
                        'uptrend_confirmed': False,
                        'downtrend_confirmed': False
                    },
                    'uptrend_analysis': {
                        'hh_count': 0,
                        'hl_count': 0,
                        'uptrend_confirmed': False
                    },
                    'downtrend_analysis': {
                        'lh_count': 0,
                        'll_count': 0,
                        'downtrend_confirmed': False
                    },
                    'h4_analysis': {
                        'hh_count': 0,
                        'hl_count': 0,
                        'uptrend_confirmed': False
                    },
                    'strong_trendlines': [],
                    'breakout_signals': None,
                    'continuation_patterns': None,
                    'timeframe': timeframe,
                    'trading_rules_followed': {
                        'multi_timeframe_confirmed': False,
                        'min_hh_hl_met': False,
                        'strong_trendlines': False,
                        'breakout_retest_ready': False,
                        'continuation_patterns_ready': False
                    }
                }
                
                def merge_defaults(target, defaults):
                    """Recursively merge default values into target dict"""
                    for key, default_value in defaults.items():
                        if key not in target:
                            target[key] = default_value
                        elif isinstance(default_value, dict) and isinstance(target.get(key), dict):
                            merge_defaults(target[key], default_value)
                
                merge_defaults(serializable_analysis, expected_keys)
                
                return jsonify({
                    'success': True,
                    'analysis': serializable_analysis,
                    'ml_prediction': ml_prediction,
                    'signals': signals,
                    'symbol': symbol,
                    'timeframe': timeframe
                })
            except Exception as e:
                print(f"Error serializing analysis response: {e}")
                print(f"Analysis keys: {list(analysis.keys()) if analysis else 'None'}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': f'Error serializing analysis: {str(e)}'}), 500
        else:
            return jsonify({'success': False, 'error': 'Analysis failed'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Start automated trading"""
    global bot_instance, bot_thread, bot_running
    
    try:
        data = request.get_json()
        auto_trade = data.get('auto_trade', True)
        symbol = data.get('symbol', bot_config.get('symbol', 'EURUSD'))
        timeframe = data.get('timeframe', bot_config.get('timeframe', '5m'))
        
        print(f"\n🚀 STARTING TRADING SESSION")
        print(f"   Symbol: {symbol}")
        print(f"   Timeframe: {timeframe}")
        print(f"   Auto Trade: {auto_trade}")
        
        # Check if trading is already running
        if bot_running:
            return jsonify({'success': False, 'error': 'Trading is already running'}), 400
        
        # Create or recreate bot instance with proper MT5 connection
        try:
            print("🔄 Creating/updating bot instance...")
            
            # Create new bot instance with MT5 connection
            new_bot = MT5TradingBot(
                symbol=symbol,
                timeframe=timeframe,
                risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                use_mt5_data=True,
                auto_trade=auto_trade,
                use_ml=bot_config.get('use_ml', True)
            )
            
            # Try to establish MT5 connection using stored credentials
            if bot_config.get('account_number') and bot_config.get('password') and bot_config.get('server'):
                print("🔗 Attempting MT5 connection with stored credentials...")
                
                # Create MT5 connector with credentials
                from mt5_connector import MT5Connector
                connector = MT5Connector(
                    account_number=bot_config['account_number'],
                    password=bot_config['password'],
                    server=bot_config['server']
                )
                
                # Try to connect
                if connector.connect():
                    print("✅ MT5 connection established")
                    new_bot.mt5_connector = connector
                    new_bot.connected = True
                    
                    # Get account info
                    account_info = connector.get_account_summary()
                    if account_info:
                        # Account balance is now fetched dynamically from MT5
                        print(f"💰 Account Balance: ${account_info.get('balance', 0):,.2f}")
                    
                    # Update global bot instance
                    bot_instance = new_bot
                    
                else:
                    return jsonify({
                        'success': False, 
                        'error': 'Failed to connect to MT5. Please check your credentials and try connecting first.'
                    }), 400
            else:
                return jsonify({
                    'success': False, 
                    'error': 'MT5 credentials not configured. Please connect to MT5 first.'
                }), 400
            
        except Exception as e:
            print(f"❌ Error creating bot instance: {e}")
            return jsonify({'success': False, 'error': f'Failed to create bot instance: {str(e)}'}), 500
        
        # Verify connection is working
        if not bot_instance or not bot_instance.connected:
            return jsonify({
                'success': False, 
                'error': 'Not connected to MT5. Please connect first.'
            }), 400
        
        # Test MT5 functionality before starting trading
        print("🧪 Testing MT5 functionality...")
        if not bot_instance.mt5_connector.test_mt5_functionality():
            return jsonify({
                'success': False, 
                'error': 'MT5 functionality test failed. Please check your MT5 terminal.'
            }), 400
        
        # Run initial analysis to ensure signals can be generated
        print("📊 Running initial market analysis...")
        initial_analysis = bot_instance.run_analysis_cycle()
        if not initial_analysis:
            return jsonify({
                'success': False, 
                'error': 'Failed to perform initial market analysis. Insufficient data or connection issues.'
            }), 400
        
        # Update bot configuration
        bot_instance.auto_trade = auto_trade
        bot_config['auto_trade'] = auto_trade
        bot_config['symbol'] = symbol
        bot_config['timeframe'] = timeframe
        
        # Start trading in a separate thread
        print("🎯 Starting trading loop...")
        bot_running = True
        bot_thread = threading.Thread(target=run_trading_loop, name="TradingThread")
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Trading started successfully for {symbol} on {timeframe} timeframe',
            'details': {
                'symbol': symbol,
                'timeframe': timeframe,
                'auto_trade': auto_trade,
                'account_balance': None,  # Balance fetched dynamically from MT5
                'risk_per_trade': bot_config.get('risk_per_trade', 0.02) * 100,
                'connected': True
            }
        })
    
    except Exception as e:
        print(f"❌ Error starting trading: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop-trading', methods=['POST'])
def stop_trading():
    """Stop automated trading"""
    global bot_running, bot_instance
    
    try:
        bot_running = False
        
        if bot_instance:
            # Close all positions
            bot_instance.close_all_positions()
            # Disconnect from MT5
            if bot_instance.mt5_connector:
                bot_instance.mt5_connector.disconnect()
        
        return jsonify({
            'success': True,
            'message': 'Trading stopped successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current bot status"""
    global bot_instance, bot_running
    
    status = {
        'running': bot_running,
        'config': bot_config,
        'connected': False,
        'positions': [],
        'account_info': None,
        'connection_details': {
            'has_bot_instance': bot_instance is not None,
            'has_mt5_connector': False,
            'mt5_initialized': False,
            'credentials_configured': False
        }
    }
    
    # Check if credentials are configured
    if (bot_config.get('account_number') and 
        bot_config.get('password') and 
        bot_config.get('server')):
        status['connection_details']['credentials_configured'] = True
    
    # Check bot instance and connection
    if bot_instance:
        status['connection_details']['has_bot_instance'] = True
        
        if hasattr(bot_instance, 'mt5_connector') and bot_instance.mt5_connector:
            status['connection_details']['has_mt5_connector'] = True
            
            if hasattr(bot_instance, 'connected') and bot_instance.connected:
                status['connected'] = True
                status['connection_details']['mt5_initialized'] = True
                
                try:
                    # Get positions with error handling
                    positions = bot_instance.monitor_positions()
                    status['positions'] = positions if positions is not None else []
                    
                    # Get account info with error handling
                    account_info = bot_instance.mt5_connector.get_account_summary()
                    if account_info:
                        status['account_info'] = {
                            'login': account_info.get('login'),
                            'server': account_info.get('server'),
                            'balance': account_info.get('balance', 0),
                            'equity': account_info.get('equity', 0),
                            'margin': account_info.get('margin', 0),
                            'margin_free': account_info.get('margin_free', 0),
                            'currency': account_info.get('currency', 'USD')
                        }
                    
                    # Test connection health
                    try:
                        test_price = bot_instance.mt5_connector.get_current_price('EURUSD')
                        status['connection_details']['connection_healthy'] = test_price is not None
                    except:
                        status['connection_details']['connection_healthy'] = False
                        
                except Exception as e:
                    print(f"Error getting detailed status: {e}")
                    status['positions'] = []
                    status['account_info'] = None
                    status['connection_details']['connection_healthy'] = False
    
    return jsonify(status)

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({'positions': []})
    
    positions = bot_instance.monitor_positions()
    return jsonify({'positions': positions or []})

@app.route('/api/close-positions', methods=['POST'])
def close_positions():
    """Close all positions"""
    global bot_instance
    
    try:
        if not bot_instance or not bot_instance.connected:
            return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400
        
        success = bot_instance.close_all_positions()
        
        return jsonify({
            'success': success,
            'message': 'Positions closed' if success else 'Failed to close positions'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close-position/<int:ticket>', methods=['POST'])
def close_position(ticket):
    """Close a specific position by ticket ID"""
    global bot_instance
    
    try:
        if not bot_instance or not bot_instance.connected:
            return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400
        
        success = bot_instance.mt5_connector.close_position(ticket)
        
        return jsonify({
            'success': success,
            'message': 'Position closed' if success else 'Failed to close position'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close-symbol-positions/<symbol>', methods=['POST'])
def close_symbol_positions(symbol):
    """Close all positions for a specific symbol"""
    global bot_instance
    
    try:
        if not bot_instance or not bot_instance.connected:
            return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400
        
        success = bot_instance.close_symbol_positions(symbol)
        
        return jsonify({
            'success': success,
            'message': f'Positions closed for {symbol}' if success else f'Failed to close positions for {symbol}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/smc_analysis/<symbol>/<timeframe>')
def get_smc_analysis(symbol, timeframe):
    """Get Smart Money Concept analysis for a symbol and timeframe"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get market data
        data = bot.get_market_data()
        if data is None or len(data) < 100:
            # If MT5 data is not available, create sample data for demo
            print(f"⚠️ MT5 data not available for {symbol}, using sample data for demo")
            import pandas as pd
            import numpy as np
            
            # Create sample data
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
            np.random.seed(42)
            
            base_price = 1.1000
            price_changes = np.random.normal(0, 0.0005, len(dates))
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] + change
                prices.append(max(0.9, min(1.3, new_price)))
            
            data = pd.DataFrame({
                'Open': prices,
                'High': [p + abs(np.random.normal(0, 0.0002)) for p in prices],
                'Low': [p - abs(np.random.normal(0, 0.0002)) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Perform SMC analysis
        smc_results = bot.analyze_smc(data)
        if not smc_results:
            return jsonify({'error': 'SMC analysis failed'}), 500
        
        # Format response
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'current_price': smc_results['current_price'],
            'market_structure': smc_results['summary']['market_structure'],
            'order_blocks': {
                'total': smc_results['summary']['order_blocks']['total_count'],
                'bullish': smc_results['summary']['order_blocks']['bullish_count'],
                'bearish': smc_results['summary']['order_blocks']['bearish_count'],
                'mitigated': smc_results['summary']['order_blocks']['mitigated_count'],
                'average_strength': round(smc_results['summary']['order_blocks']['average_strength'], 2)
            },
            'fair_value_gaps': {
                'total': smc_results['summary']['fair_value_gaps']['total_count'],
                'bullish': smc_results['summary']['fair_value_gaps']['bullish_count'],
                'bearish': smc_results['summary']['fair_value_gaps']['bearish_count'],
                'mitigated': smc_results['summary']['fair_value_gaps']['mitigated_count'],
                'average_gap_size': round(smc_results['summary']['fair_value_gaps']['average_gap_size'], 5)
            },
            'liquidity_zones': {
                'total': smc_results['summary']['liquidity_zones']['total_count'],
                'equal_highs': smc_results['summary']['liquidity_zones']['equal_highs_count'],
                'equal_lows': smc_results['summary']['liquidity_zones']['equal_lows_count'],
                'average_strength': round(smc_results['summary']['liquidity_zones']['average_strength'], 2)
            },
            'institutional_order_blocks': {
                'total': smc_results['summary']['institutional_order_blocks']['total_count'],
                'bullish': smc_results['summary']['institutional_order_blocks']['bullish_count'],
                'bearish': smc_results['summary']['institutional_order_blocks']['bearish_count'],
                'average_volume_ratio': round(smc_results['summary']['institutional_order_blocks']['average_volume_ratio'], 2)
            },
            'signals': {
                'order_block_signals': len(smc_results['signals'].get('order_block_signals', [])),
                'fvg_signals': len(smc_results['signals'].get('fvg_signals', [])),
                'liquidity_signals': len(smc_results['signals'].get('liquidity_signals', [])),
                'market_structure_signals': len(smc_results['signals'].get('market_structure_signals', [])),
                'premium_discount_signals': len(smc_results['signals'].get('premium_discount_signals', [])),
                'institutional_signals': len(smc_results['signals'].get('institutional_signals', []))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'SMC analysis error: {str(e)}'}), 500

@app.route('/api/smc_signals/<symbol>/<timeframe>')
def get_smc_trading_signals(symbol, timeframe):
    """Get Smart Money Concept trading signals for a symbol and timeframe"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get market data
        data = bot.get_market_data()
        if data is None or len(data) < 100:
            # If MT5 data is not available, create sample data for demo
            print(f"⚠️ MT5 data not available for {symbol}, using sample data for demo")
            import pandas as pd
            import numpy as np
            
            # Create sample data
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
            np.random.seed(42)
            
            base_price = 1.1000
            price_changes = np.random.normal(0, 0.0005, len(dates))
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] + change
                prices.append(max(0.9, min(1.3, new_price)))
            
            data = pd.DataFrame({
                'Open': prices,
                'High': [p + abs(np.random.normal(0, 0.0002)) for p in prices],
                'Low': [p - abs(np.random.normal(0, 0.0002)) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Perform SMC analysis
        smc_results = bot.analyze_smc(data)
        if not smc_results:
            return jsonify({'error': 'SMC analysis failed'}), 500
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Get SMC trading signals
        smc_signals = bot.get_smc_trading_signals(current_price)
        
        # Format signals for response
        formatted_signals = []
        if smc_signals:
            for signal in smc_signals:
                formatted_signals.append({
                    'type': signal['type'],
                    'entry_price': round(signal['entry_price'], 5),
                    'stop_loss': round(signal['stop_loss'], 5),
                    'target': round(signal['target'], 5),
                    'source': signal['source'],
                    'strength': round(signal['strength'], 3),
                    'timestamp': signal['timestamp'].isoformat() if hasattr(signal['timestamp'], 'isoformat') else str(signal['timestamp'])
                })
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': round(current_price, 5),
            'timestamp': datetime.now().isoformat(),
            'signals_count': len(formatted_signals),
            'signals': formatted_signals
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'SMC signals error: {str(e)}'}), 500

@app.route('/api/combined_analysis/<symbol>/<timeframe>')
def get_combined_analysis(symbol, timeframe):
    """Get combined analysis including traditional TA, SMC, and ML"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Perform comprehensive analysis
        combined_signal = bot.analyze_market()
        
        if not combined_signal:
            return jsonify({'error': 'No trading signals generated'}), 404
        
        # Format response
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'signal_type': combined_signal['signal_type'],
            'signal_strength': round(combined_signal['signal_strength'], 3),
            'signal_sources': combined_signal['signal_sources'],
            'entry_price': round(combined_signal['entry_price'], 5) if combined_signal['entry_price'] else None,
            'stop_loss': round(combined_signal['stop_loss'], 5) if combined_signal['stop_loss'] else None,
            'target': round(combined_signal['target'], 5) if combined_signal['target'] else None,
            'position_size': round(combined_signal['position_size'], 2) if combined_signal['position_size'] else None,
            'potential_profit': round(combined_signal['potential_profit'], 2) if combined_signal['potential_profit'] else None,
            'risk_amount': round(combined_signal['risk_amount'], 2) if combined_signal['risk_amount'] else None,
            'analysis': {
                'traditional_ta': combined_signal.get('analysis', {}),
                'ml_prediction': combined_signal.get('ml_prediction', {}),
                'smc_signals': combined_signal.get('smc_signals', {})
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Combined analysis error: {str(e)}'}), 500

def run_trading_loop():
    """Run the trading loop in a separate thread"""
    global bot_instance, bot_running
    
    print("🎯 Trading loop started")
    cycle_count = 0
    
    try:
        while bot_running and bot_instance:
            cycle_count += 1
            print(f"\n🔄 Trading Cycle #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                # Check MT5 connection status
                if not bot_instance.connected or not bot_instance.mt5_connector:
                    print("❌ MT5 connection lost. Attempting to reconnect...")
                    
                    # Try to reconnect
                    if bot_instance.mt5_connector:
                        if bot_instance.mt5_connector.connect():
                            bot_instance.connected = True
                            print("✅ MT5 reconnected successfully")
                        else:
                            print("❌ Failed to reconnect to MT5. Stopping trading.")
                            break
                    else:
                        print("❌ No MT5 connector available. Stopping trading.")
                        break
                
                # Run analysis cycle
                print("📊 Running market analysis...")
                analysis = bot_instance.run_analysis_cycle()
                
                if analysis:
                    print("✅ Analysis completed successfully")
                    
                    # Generate trading signals if analysis is good
                    signals = bot_instance.get_trading_signals(analysis)
                    if signals and bot_instance.auto_trade:
                        print(f"🎯 Trading signal detected: {signals['signal_type']}")
                        
                        # Execute trade if conditions are met
                        result = bot_instance.execute_trade(signals)
                        if result:
                            print(f"✅ Trade executed: Order #{result['order_id']}")
                        else:
                            print("⚠️  Trade execution failed or skipped")
                    else:
                        if not signals:
                            print("ℹ️  No trading signals generated")
                        elif not bot_instance.auto_trade:
                            print("ℹ️  Auto trading disabled - signals not executed")
                else:
                    print("⚠️  Analysis failed this cycle")
                
                # Monitor existing positions
                if bot_instance.connected:
                    print("📊 Monitoring positions...")
                    positions = bot_instance.monitor_positions()
                    if positions:
                        print(f"📈 Currently managing {len(positions)} position(s)")
                    else:
                        print("📈 No open positions")
                else:
                    print("❌ Cannot monitor positions - MT5 not connected")
                
            except Exception as cycle_error:
                print(f"❌ Error in trading cycle #{cycle_count}: {cycle_error}")
                import traceback
                traceback.print_exc()
                
                # Continue to next cycle unless it's a critical error
                if "connection" in str(cycle_error).lower():
                    print("🔄 Connection issue detected, will attempt reconnection next cycle")
                else:
                    print("⚠️  Non-critical error, continuing to next cycle")
            
            # Wait for next cycle (5 minutes) with status updates
            if bot_running:
                print(f"⏳ Waiting 5 minutes for next cycle...")
                for i in range(300):  # 5 minutes = 300 seconds
                    if not bot_running:
                        print("🛑 Trading loop stopped by user")
                        break
                    
                    # Show progress every minute
                    if i > 0 and i % 60 == 0:
                        minutes_left = (300 - i) // 60
                        print(f"⏳ {minutes_left} minute(s) remaining until next cycle...")
                    
                    time.sleep(1)
    
    except Exception as e:
        print(f"❌ Critical error in trading loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 Trading loop ending...")
        bot_running = False
        
        # Close all positions if they exist
        if bot_instance and bot_instance.connected and bot_instance.auto_trade:
            print("🔒 Auto-closing all positions before stopping...")
            try:
                bot_instance.close_all_positions()
            except Exception as close_error:
                print(f"⚠️  Error closing positions: {close_error}")
        
        print("✅ Trading loop stopped")

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Trading Bot API Server...")
    print("📊 Available symbols:", len(AVAILABLE_SYMBOLS))
    print("⏰ Available timeframes:", len(AVAILABLE_TIMEFRAMES))
    print("🌐 Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 