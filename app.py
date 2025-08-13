#!/usr/bin/env python3
"""
Flask Backend API for Dynamic Trading Bot
========================================

This Flask application provides a REST API for the trading bot with:
- XM account configuration
- Dynamic symbol and timeframe selection
- Auto trading controls
- Real-time market analysis
- Enhanced logging and error handling
"""

import logging
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import secrets
import json
import threading
import time
from datetime import datetime
import traceback
from dotenv import load_dotenv
from functools import wraps

# Import trading bot components
from mt5_connector import MT5Connector
from mt5_trading_bot import MT5TradingBot
from trading_bot import TradingBot  # Import the enhanced trading bot

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create specific loggers
    app_logger = logging.getLogger('trading_bot.app')
    trading_logger = logging.getLogger('trading_bot.trading')
    mt5_logger = logging.getLogger('trading_bot.mt5')
    
    return app_logger, trading_logger, mt5_logger

# Setup loggers
app_logger, trading_logger, mt5_logger = setup_logging()

app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

# Simple API key auth decorator
API_KEY = os.getenv('API_KEY')
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            return f(*args, **kwargs)
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != API_KEY:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

# Global variables for bot state
bot_instance = None
enhanced_bot_instance = None  # Enhanced trading bot instance
bot_thread = None
bot_running = False
# Multi-symbol trading support
trading_bots = {}
bot_config = {
    'account_number': '',
    'password': '',
    'server': 'XMGlobal-Demo',
    'symbol': 'EURUSD',
    'timeframe': '5m',
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
    'account_size': None,  # No longer used - balance fetched from MT5
    'auto_trade': os.getenv('AUTO_TRADE', 'false').lower() == 'true',
    'use_ml': os.getenv('USE_ML', 'true').lower() == 'true',
    'enable_automation': os.getenv('ENABLE_AUTOMATION', 'true').lower() == 'true',
    # Symbols must be selected from the frontend; start empty
    'symbols_to_trade': []
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

# Market types for enhanced trading bot
MARKET_TYPES = [
    {'value': 'stock', 'label': 'Stock Market'},
    {'value': 'forex', 'label': 'Forex Market'},
    {'value': 'crypto', 'label': 'Cryptocurrency'},
    {'value': 'commodities', 'label': 'Commodities'}
]

# Error handling decorator
def handle_errors(f):
    """Decorator for consistent error handling across endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app_logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                'success': False, 
                'error': f'Internal server error: {str(e)}'
            }), 500
    return decorated_function

# Validation decorator
def validate_required_fields(required_fields):
    """Decorator to validate required fields in request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json() or {}
            missing_fields = [field for field in required_fields if field not in data or not data[field]]
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def initialize_enhanced_bot(symbol, timeframe, market_type='forex', enable_automation=True):
    """Initialize the enhanced trading bot"""
    global enhanced_bot_instance
    
    try:
        # Create enhanced trading bot instance
        enhanced_bot_instance = TradingBot(
            symbol=symbol,
            period=timeframe,
            market_type=market_type,
            account_size=10000,  # Will be updated from MT5 balance
            risk_per_trade=bot_config['risk_per_trade'],
            enable_automation=enable_automation,
            mt5_config=bot_config
        )
        
        app_logger.info(f"Enhanced trading bot initialized for {symbol} ({market_type})")
        return True
        
    except Exception as e:
        app_logger.error(f"Failed to initialize enhanced trading bot: {e}")
        return False

@app.route('/')
def index():
    """Serve the Vue frontend"""
    app_logger.info("Frontend requested")
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
@handle_errors
def get_config():
    """Get current bot configuration"""
    app_logger.info("Configuration requested")
    return jsonify({
        'config': bot_config,
        'available_symbols': AVAILABLE_SYMBOLS,
        'available_timeframes': AVAILABLE_TIMEFRAMES,
        'market_types': MARKET_TYPES,
        'enhanced_bot_available': True
    })

@app.route('/api/config', methods=['POST'])
@handle_errors
@require_api_key
@validate_required_fields([])  # No required fields for config update
def update_config():
    """Update bot configuration"""
    global bot_config
    
    data = request.get_json()
    app_logger.info(f"Updating configuration: {list(data.keys())}")
    
    # Update configuration with validation
    config_updates = {}
    if 'account_number' in data:
        config_updates['account_number'] = data['account_number']
    if 'password' in data:
        config_updates['password'] = data['password']
    if 'server' in data:
        config_updates['server'] = data['server']
    if 'symbol' in data:
        if data['symbol'] in AVAILABLE_SYMBOLS:
            config_updates['symbol'] = data['symbol']
        else:
            return jsonify({'success': False, 'error': f'Invalid symbol: {data["symbol"]}'}), 400
    if 'symbols_to_trade' in data:
        symbols = data.get('symbols_to_trade') or []
        if not isinstance(symbols, list):
            return jsonify({'success': False, 'error': 'symbols_to_trade must be a list'}), 400
        symbols = [s for s in symbols if s in AVAILABLE_SYMBOLS]
        symbols = list(dict.fromkeys(symbols))  # dedupe, preserve order
        if len(symbols) == 0 or len(symbols) > 5:
            return jsonify({'success': False, 'error': 'Select between 1 and 5 symbols'}), 400
        config_updates['symbols_to_trade'] = symbols
    if 'timeframe' in data:
        valid_timeframes = [tf['value'] for tf in AVAILABLE_TIMEFRAMES]
        if data['timeframe'] in valid_timeframes:
            config_updates['timeframe'] = data['timeframe']
        else:
            return jsonify({'success': False, 'error': f'Invalid timeframe: {data["timeframe"]}'}), 400
    if 'risk_per_trade' in data:
        try:
            risk = float(data['risk_per_trade'])
            if 0.001 <= risk <= 0.1:  # 0.1% to 10%
                config_updates['risk_per_trade'] = risk
            else:
                return jsonify({'success': False, 'error': 'Risk per trade must be between 0.1% and 10%'}), 400
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid risk per trade value'}), 400
    if 'auto_trade' in data:
        config_updates['auto_trade'] = bool(data['auto_trade'])
    if 'use_ml' in data:
        config_updates['use_ml'] = bool(data['use_ml'])
    if 'enable_automation' in data:
        config_updates['enable_automation'] = bool(data['enable_automation'])
    
    # Apply updates
    bot_config.update(config_updates)
    
    app_logger.info(f"Configuration updated successfully: {list(config_updates.keys())}")
    return jsonify({'success': True, 'message': 'Configuration updated'})

@app.route('/api/env-config', methods=['GET'])
def get_env_config():
    """Provide default configuration without pulling credentials from environment"""
    env_config = {
        'account_number': '',
        'password': '',
        'server': 'XMGlobal-Demo',
        'symbol': 'EURUSD',
        'timeframe': '5m',
        'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
        'account_size': None,
        'auto_trade': False,
        'use_ml': os.getenv('USE_ML', 'true').lower() == 'true',
        'enable_automation': os.getenv('ENABLE_AUTOMATION', 'true').lower() == 'true'
    }
    return jsonify({
        'config': env_config,
        'available_symbols': AVAILABLE_SYMBOLS,
        'available_timeframes': AVAILABLE_TIMEFRAMES
    })

@app.route('/api/connect', methods=['POST'])
@handle_errors
def connect_mt5():
    """Connect to MT5 with XM credentials"""
    global bot_instance
    
    data = request.get_json()
    account_number = data.get('account_number')
    password = data.get('password')
    server = data.get('server', '')
    
    app_logger.info(f"Attempting MT5 connection for account: {account_number}")
    
    if not account_number or not password or not server:
        return jsonify({'success': False, 'error': 'Account number, password, and server are required'}), 400
    
    # Validate account number format
    try:
        int(account_number)
    except ValueError:
        return jsonify({'success': False, 'error': 'Account number must be numeric'}), 400
    
    # Create MT5 connector
    app_logger.info("Creating MT5 connector...")
    connector = MT5Connector(account_number, password, server)
    
    # Try to connect
    app_logger.info("Attempting MT5 connection...")
    if connector.connect():
        app_logger.info("MT5 connection successful")
        
        # Test basic functionality
        app_logger.info("Testing MT5 functionality...")
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
        
        app_logger.info(f"Account Balance: ${account_info.get('balance', 0):,.2f}")
        app_logger.info(f"Account Equity: ${account_info.get('equity', 0):,.2f}")
        
        # Update global config with credentials
        bot_config.update({
            'account_number': account_number,
            'password': password,
            'server': server
        })
        
        # Create or update per-symbol bot instances for selected symbols
        app_logger.info("Creating/updating per-symbol bot instances...")
        global trading_bots, bot_instance
        trading_bots = {}
        selected = bot_config.get('symbols_to_trade') or []
        if len(selected) == 0:
            # If none selected, default to primary symbol only to allow minimal operation
            selected = [bot_config.get('symbol', 'EURUSD')]
        for sym in selected:
            b = MT5TradingBot(
                symbol=sym,
                timeframe=bot_config.get('timeframe', '5m'),
                risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                use_mt5_data=True,
                auto_trade=True,
                use_ml=bot_config.get('use_ml', True)
            )
            b.mt5_connector = connector
            b.connected = True
            b.auto_trade = True
            b.last_analysis = None
            trading_bots[sym] = b
        # Set primary instance to the configured symbol
        primary_symbol = bot_config.get('symbol', 'EURUSD')
        bot_instance = trading_bots.get(primary_symbol) or next(iter(trading_bots.values()))
        # Account balance is now fetched dynamically from MT5
        
        # Test market data retrieval on primary symbol
        app_logger.info("Testing market data retrieval (primary symbol)...")
        test_data = bot_instance.get_market_data()
        if test_data is None or len(test_data) < 10:
            app_logger.warning("Limited market data available, but connection is working")
        else:
            app_logger.info(f"Market data test successful ({len(test_data)} data points)")
        
        # Auto-generate an API key upon successful connection if not set, or rotate
        global API_KEY
        API_KEY = secrets.token_urlsafe(32)
        os.environ['API_KEY'] = API_KEY
        app_logger.info("API key generated for session")

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
            },
            'api_key': API_KEY
        })
    else:
        app_logger.error("MT5 connection failed")
        try:
            last_err = connector.get_last_error()
        except Exception:
            last_err = None
        return jsonify({
            'success': False, 
            'error': 'Failed to connect to MT5. Please check your credentials, server name, and ensure MT5 terminal is running.',
            'details': str(last_err) if last_err else None
        }), 400

@app.route('/api/analyze', methods=['POST'])
@handle_errors
def analyze_market():
    """Perform market analysis for one or multiple symbols"""
    global bot_instance, trading_bots

    data = request.get_json() or {}
    symbols = data.get('symbols')
    symbol = data.get('symbol')
    timeframe = data.get('timeframe') or bot_config.get('timeframe', '5m')

    if not symbols and not symbol:
        return jsonify({'success': False, 'error': 'Provide symbol or symbols[]'}), 400

    symbols_to_run = []
    if symbols:
        if not isinstance(symbols, list):
            return jsonify({'success': False, 'error': 'symbols must be a list'}), 400
        symbols = [s for s in symbols if s in AVAILABLE_SYMBOLS]
        symbols = list(dict.fromkeys(symbols))
        if len(symbols) == 0 or len(symbols) > 5:
            return jsonify({'success': False, 'error': 'Select between 1 and 5 symbols'}), 400
        symbols_to_run = symbols
    else:
        if symbol not in AVAILABLE_SYMBOLS:
            return jsonify({'success': False, 'error': f'Invalid symbol: {symbol}'}), 400
        symbols_to_run = [symbol]

    app_logger.info(f"Running analysis for symbols: {symbols_to_run} ({timeframe})")

    results = {}
    for sym in symbols_to_run:
        # Reuse existing per-symbol bot when available
        b = trading_bots.get(sym)
        if not b:
            b = MT5TradingBot(
                symbol=sym,
                timeframe=timeframe,
                risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                use_mt5_data=True,
                auto_trade=False,
                use_ml=bot_config.get('use_ml', True)
            )
            if bot_instance and bot_instance.mt5_connector:
                b.mt5_connector = bot_instance.mt5_connector
                b.connected = True
        combined_signal = b.run_analysis_cycle()
        b.last_analysis = {'timestamp': datetime.now()} if combined_signal else b.last_analysis

        if combined_signal:
            try:
                results[sym] = {
                    'success': True,
                    'signal_type': combined_signal.get('signal_type'),
                    'signal_strength': combined_signal.get('signal_strength'),
                    'timeframe': combined_signal.get('timeframe'),
                    'entry_price': round(combined_signal['entry_price'], 5) if combined_signal['entry_price'] else None,
                    'stop_loss': round(combined_signal['stop_loss'], 5) if combined_signal['stop_loss'] else None,
                    'target': round(combined_signal['target'], 5) if combined_signal['target'] else None,
                    'position_size': round(combined_signal.get('position_size') or 0, 2),
                    'potential_profit': round(combined_signal.get('potential_profit') or 0, 2),
                    'risk_amount': round(combined_signal.get('risk_amount') or 0, 2),
                }
            except Exception as e:
                app_logger.error(f"Error serializing {sym} analysis: {e}")
                results[sym] = {'success': False, 'error': f'Error serializing analysis: {str(e)}'}
        else:
            results[sym] = {'success': False, 'error': 'Analysis failed'}

    return jsonify({'success': True, 'results': results, 'timeframe': timeframe})

@app.route('/api/start-trading', methods=['POST'])
@handle_errors
@require_api_key
def start_trading():
    """Start automated trading"""
    global bot_instance, bot_thread, bot_running
    
    # Accept empty/non-JSON bodies from alias endpoints or simple POSTs
    data = request.get_json(silent=True) or {}
    auto_trade = data.get('auto_trade', True)
    symbol = data.get('symbol', bot_config.get('symbol', 'EURUSD'))
    timeframe = data.get('timeframe', bot_config.get('timeframe', '5m'))

    # Optionally accept credentials in this request to streamline starting
    account_number = data.get('account_number')
    password = data.get('password')
    server = data.get('server')
    if account_number and password and server:
        bot_config.update({
            'account_number': str(account_number),
            'password': password,
            'server': server
        })
    
    app_logger.info(f"\nStarting trading session for {symbol} on {timeframe} timeframe")
    app_logger.info(f"   Auto Trade: {auto_trade}")
    
    # Check if trading is already running
    if bot_running:
        return jsonify({'success': False, 'error': 'Trading is already running'}), 400
    
    # Create or recreate bot instance with proper MT5 connection
    try:
        app_logger.info("Creating/updating bot instance...")
        
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
            app_logger.info("Attempting MT5 connection with stored credentials...")
            
            # Create MT5 connector with credentials
            from mt5_connector import MT5Connector
            connector = MT5Connector(
                account_number=bot_config['account_number'],
                password=bot_config['password'],
                server=bot_config['server']
            )
            
            # Try to connect
            if connector.connect():
                app_logger.info("MT5 connection established")
                new_bot.mt5_connector = connector
                new_bot.connected = True
                new_bot.auto_trade = auto_trade
                
                # Get account info
                account_info = connector.get_account_summary()
                if account_info:
                    # Account balance is now fetched dynamically from MT5
                    app_logger.info(f"Account Balance: ${account_info.get('balance', 0):,.2f}")
                
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
        app_logger.error(f"Error creating bot instance: {e}")
        return jsonify({'success': False, 'error': f'Failed to create bot instance: {str(e)}'}), 500
    
    # Verify connection is working
    if not bot_instance or not bot_instance.connected:
        return jsonify({
            'success': False, 
            'error': 'Not connected to MT5. Please connect first.'
        }), 400
    
    # Test MT5 functionality before starting trading
    app_logger.info("Testing MT5 functionality...")
    if not bot_instance.mt5_connector.test_mt5_functionality():
        return jsonify({
            'success': False, 
            'error': 'MT5 functionality test failed. Please check your MT5 terminal.'
        }), 400
    
    # Run initial analysis per selected symbols to ensure signals can be generated
    app_logger.info("Running initial market analysis for selected symbols...")
    selected = bot_config.get('symbols_to_trade') or [bot_config.get('symbol', 'EURUSD')]
    analyzed = {}
    for sym in selected:
        b = trading_bots.get(sym) or bot_instance
        if b.symbol != sym:
            b.symbol = sym
        # Ensure auto trading preference is propagated
        b.auto_trade = auto_trade
        res = b.run_analysis_cycle()
        analyzed[sym] = bool(res)
    if not any(analyzed.values()):
        return jsonify({
            'success': False, 
            'error': 'Failed to perform initial market analysis for all symbols.'
        }), 400
    
    # Update bot configuration
    bot_instance.auto_trade = auto_trade
    bot_config['auto_trade'] = auto_trade
    bot_config['symbol'] = symbol
    bot_config['timeframe'] = timeframe
    
    # Start trading in a separate thread
    app_logger.info("Starting trading loop...")
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

@app.route('/api/stop-trading', methods=['POST'])
@handle_errors
@require_api_key
def stop_trading():
    """Stop automated trading"""
    global bot_running, bot_instance
    
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

# Alias endpoints for unified frontend compatibility
@app.route('/api/start-automated-trading', methods=['POST'])
@handle_errors
def start_automated_trading_alias():
    """Alias to start automated trading (compatibility)"""
    return start_trading()

@app.route('/api/stop-automated-trading', methods=['POST'])
@handle_errors
def stop_automated_trading_alias():
    """Alias to stop automated trading (compatibility)"""
    return stop_trading()

@app.route('/api/status', methods=['GET'])
@handle_errors
def get_status():
    """Get current bot status"""
    global bot_instance, bot_running
    
    status = {
        'running': bot_running,
        'config': bot_config,
        'connected': False,
        'positions': [],
        'account_info': None,
        'trading_bots': {},
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
                    app_logger.error(f"Error getting detailed status: {e}")
                    status['positions'] = []
                    status['account_info'] = None
                    status['connection_details']['connection_healthy'] = False
    
    # Populate trading_bots summary
    if trading_bots:
        for sym, b in trading_bots.items():
            status['trading_bots'][sym] = {
                'connected': getattr(b, 'connected', False),
                'last_analysis': b.last_analysis['timestamp'].isoformat() if getattr(b, 'last_analysis', None) else None
            }

    return jsonify(status)

@app.route('/api/positions', methods=['GET'])
@handle_errors
def get_positions():
    """Get current positions"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({'positions': []})
    
    positions = bot_instance.monitor_positions()
    return jsonify({'positions': positions or []})

@app.route('/api/close-positions', methods=['POST'])
@handle_errors
@require_api_key
def close_positions():
    """Close all positions"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400
    
    success = bot_instance.close_all_positions()
    
    return jsonify({
        'success': success,
        'message': 'Positions closed' if success else 'Failed to close positions'
    })

@app.route('/api/open-position', methods=['POST'])
@handle_errors
@require_api_key
def open_position():
    """Open a manual position for a given symbol with type and optional sl/tp."""
    global trading_bots, bot_instance
    data = request.get_json() or {}
    symbol = data.get('symbol')
    order_type = data.get('type')  # BUY/SELL
    volume = float(data.get('volume') or 0)
    sl = data.get('sl')
    tp = data.get('tp')

    if symbol not in AVAILABLE_SYMBOLS:
        return jsonify({'success': False, 'error': 'Invalid symbol'}), 400
    if order_type not in ['BUY', 'SELL']:
        return jsonify({'success': False, 'error': 'Invalid order type'}), 400
    if volume <= 0:
        return jsonify({'success': False, 'error': 'Volume must be greater than 0'}), 400
    
    b = trading_bots.get(symbol) or bot_instance
    if not b or not b.connected or not b.mt5_connector:
        return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400

    # Final volume validation
    sym_info = b.mt5_connector.get_symbol_info(symbol)
    if sym_info:
        min_vol = sym_info.get('volume_min', 0.01)
        max_vol = sym_info.get('volume_max', 100.0)
        step = sym_info.get('volume_step', 0.01)
        # Round to step
        volume = round(volume / step) * step
        if volume < min_vol or volume > max_vol:
            return jsonify({'success': False, 'error': f'Volume out of range [{min_vol}, {max_vol}]'}), 400

    # Place order
    price = None  # market
    result = b.mt5_connector.place_order(symbol, order_type, volume, price, sl, tp, comment='Manual Open from UI')
    if result:
        return jsonify({'success': True, 'order': result})
    return jsonify({'success': False, 'error': 'Order placement failed'}), 400

@app.route('/api/close-position/<int:ticket>', methods=['POST'])
@handle_errors
def close_position(ticket):
    """Close a specific position by ticket ID"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400
    
    success = bot_instance.mt5_connector.close_position(ticket)
    
    return jsonify({
        'success': success,
        'message': 'Position closed' if success else 'Failed to close position'
    })

@app.route('/api/close-symbol-positions/<symbol>', methods=['POST'])
@handle_errors
def close_symbol_positions(symbol):
    """Close all positions for a specific symbol"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({'success': False, 'error': 'Not connected to MT5'}), 400
    
    success = bot_instance.close_symbol_positions(symbol)
    
    return jsonify({
        'success': success,
        'message': f'Positions closed for {symbol}' if success else f'Failed to close positions for {symbol}'
    })

@app.route('/api/smc_analysis/<symbol>/<timeframe>')
@handle_errors
def get_smc_analysis(symbol, timeframe):
    """Get Smart Money Concept analysis for a symbol and timeframe"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get market data
        data = bot.get_market_data()
        if data is None or len(data) < 100:
            # If MT5 data is not available, create sample data for demo
            app_logger.warning(f"MT5 data not available for {symbol}, using sample data for demo")
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
        app_logger.error(f"SMC analysis error: {str(e)}")
        return jsonify({'error': f'SMC analysis error: {str(e)}'}), 500

@app.route('/api/smc_signals/<symbol>/<timeframe>')
@handle_errors
def get_smc_trading_signals(symbol, timeframe):
    """Get Smart Money Concept trading signals for a symbol and timeframe"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get market data
        data = bot.get_market_data()
        if data is None or len(data) < 100:
            # If MT5 data is not available, create sample data for demo
            app_logger.warning(f"MT5 data not available for {symbol}, using sample data for demo")
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
        app_logger.error(f"SMC signals error: {str(e)}")
        return jsonify({'error': f'SMC signals error: {str(e)}'}), 500

@app.route('/api/combined_analysis/<symbol>/<timeframe>')
@handle_errors
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
        app_logger.error(f"Combined analysis error: {str(e)}")
        return jsonify({'error': f'Combined analysis error: {str(e)}'}), 500

def run_trading_loop():
    """Run the trading loop in a separate thread"""
    global bot_instance, bot_running
    
    app_logger.info("Trading loop started")
    cycle_count = 0
    
    try:
        while bot_running and bot_instance:
            cycle_count += 1
            app_logger.info(f"\nTrading Cycle #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                # Check MT5 connection status
                if not bot_instance.connected or not bot_instance.mt5_connector:
                    app_logger.warning("MT5 connection lost. Attempting to reconnect...")
                    
                    # Try to reconnect
                    if bot_instance.mt5_connector:
                        if bot_instance.mt5_connector.connect():
                            bot_instance.connected = True
                            app_logger.info("MT5 reconnected successfully")
                        else:
                            app_logger.error("Failed to reconnect to MT5. Stopping trading.")
                            break
                    else:
                        app_logger.error("No MT5 connector available. Stopping trading.")
                        break
                
                # Run analysis cycle for selected symbols automatically
                selected = bot_config.get('symbols_to_trade') or [bot_config.get('symbol', 'EURUSD')]
                app_logger.info(f"Running market analysis for: {selected}")
                for sym in selected:
                    b = trading_bots.get(sym) or bot_instance
                    if b.symbol != sym:
                        b.symbol = sym
                    analysis = b.run_analysis_cycle()
                
                    if analysis:
                        app_logger.info(f"Analysis completed for {sym}")
                        # Generate trading signals if analysis is good
                        signals = b.get_trading_signals(analysis)
                        if signals and b.auto_trade:
                            app_logger.info(f"Trading signal detected for {sym}: {signals['signal_type']}")
                            result = b.execute_trade(signals)
                            if result:
                                app_logger.info(f"Trade executed for {sym}: Order #{result['order_id']}")
                            else:
                                app_logger.warning(f"Trade execution failed or skipped for {sym}")
                        else:
                            if not signals:
                                app_logger.info(f"No trading signals generated for {sym}")
                            elif not b.auto_trade:
                                app_logger.info(f"Auto trading disabled - signals not executed for {sym}")
                    else:
                        app_logger.warning(f"Analysis failed this cycle for {sym}")
                
                # Monitor existing positions
                if bot_instance.connected:
                    app_logger.info("Monitoring positions...")
                    positions = bot_instance.monitor_positions()
                    if positions:
                        app_logger.info(f"Currently managing {len(positions)} position(s)")
                    else:
                        app_logger.info("No open positions")
                else:
                    app_logger.error("Cannot monitor positions - MT5 not connected")
                
            except Exception as cycle_error:
                app_logger.error(f"Error in trading cycle #{cycle_count}: {cycle_error}")
                import traceback
                traceback.print_exc()
                
                # Continue to next cycle unless it's a critical error
                if "connection" in str(cycle_error).lower():
                    app_logger.warning("Connection issue detected, will attempt reconnection next cycle")
                else:
                    app_logger.warning("Non-critical error, continuing to next cycle")
            
            # Wait for next cycle (5 minutes) with status updates
            if bot_running:
                app_logger.info(f"Waiting 5 minutes for next cycle...")
                for i in range(300):  # 5 minutes = 300 seconds
                    if not bot_running:
                        app_logger.info("Trading loop stopped by user")
                        break
                    
                    # Show progress every minute
                    if i > 0 and i % 60 == 0:
                        minutes_left = (300 - i) // 60
                        app_logger.info(f"Waiting {minutes_left} minute(s) remaining until next cycle...")
                    
                    time.sleep(1)
    
    except Exception as e:
        app_logger.error(f"Critical error in trading loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app_logger.info("Trading loop ending...")
        bot_running = False
        
        # Close all positions if they exist
        if bot_instance and bot_instance.connected and bot_instance.auto_trade:
            app_logger.info("Auto-closing all positions before stopping...")
            try:
                bot_instance.close_all_positions()
            except Exception as close_error:
                app_logger.warning(f"Error closing positions: {close_error}")
        
        app_logger.info("Trading loop stopped")

# ===== ENHANCED TRADING BOT ENDPOINTS =====

@app.route('/api/enhanced/initialize', methods=['POST'])
@handle_errors
@validate_required_fields(['symbol', 'timeframe'])
def initialize_enhanced_bot_endpoint():
    """Initialize the enhanced trading bot"""
    data = request.get_json()
    symbol = data['symbol']
    timeframe = data['timeframe']
    market_type = data.get('market_type', 'forex')
    enable_automation = data.get('enable_automation', True)
    
    app_logger.info(f"Initializing enhanced trading bot for {symbol} ({timeframe})")
    
    success = initialize_enhanced_bot(symbol, timeframe, market_type, enable_automation)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Enhanced trading bot initialized for {symbol}',
            'bot_status': enhanced_bot_instance.get_enhanced_status() if enhanced_bot_instance else None
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to initialize enhanced trading bot'
        }), 500

@app.route('/api/enhanced/connect', methods=['POST'])
@handle_errors
def connect_enhanced_bot():
    """Connect enhanced trading bot to MT5"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    data = request.get_json() or {}
    account_number = data.get('account_number')
    password = data.get('password')
    server = data.get('server')
    
    app_logger.info("Connecting enhanced trading bot to MT5")
    
    success = enhanced_bot_instance.connect_mt5(account_number, password, server)
    
    return jsonify({
        'success': success,
        'message': 'Connected to MT5' if success else 'Failed to connect to MT5',
        'bot_status': enhanced_bot_instance.get_enhanced_status()
    })

@app.route('/api/enhanced/disconnect', methods=['POST'])
@handle_errors
def disconnect_enhanced_bot():
    """Disconnect enhanced trading bot from MT5"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    app_logger.info("Disconnecting enhanced trading bot from MT5")
    enhanced_bot_instance.disconnect_mt5()
    
    return jsonify({
        'success': True,
        'message': 'Disconnected from MT5',
        'bot_status': enhanced_bot_instance.get_enhanced_status()
    })

@app.route('/api/enhanced/analyze', methods=['POST'])
@handle_errors
def analyze_with_enhanced_bot():
    """Run analysis with enhanced trading bot"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    app_logger.info("Running analysis with enhanced trading bot")
    
    try:
        # Fetch data and run analysis
        enhanced_bot_instance.fetch_data()
        trend_analysis = enhanced_bot_instance.identify_higher_highs_lows()
        
        if not trend_analysis:
            return jsonify({
                'success': False,
                'error': 'No trend analysis available'
            }), 400
        
        # Get trading signals
        signals = enhanced_bot_instance.get_day_trading_signals(trend_analysis)
        
        result = {
            'success': True,
            'symbol': enhanced_bot_instance.symbol,
            'timeframe': enhanced_bot_instance.period,
            'market_type': enhanced_bot_instance.market_type,
            'trend_analysis': trend_analysis,
            'trading_signals': signals,
            'bot_status': enhanced_bot_instance.get_enhanced_status()
        }
        
        return jsonify(result)
        
    except Exception as e:
        app_logger.error(f"Error in enhanced analysis: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/enhanced/automated-analysis', methods=['POST'])
@handle_errors
def run_automated_analysis():
    """Run automated analysis cycle with enhanced trading bot"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    app_logger.info("Running automated analysis cycle")
    
    try:
        analysis_result = enhanced_bot_instance.run_automated_analysis_cycle()
        
        if not analysis_result:
            return jsonify({
                'success': False,
                'error': 'Automated analysis failed'
            }), 500
        
        return jsonify({
            'success': True,
            'analysis_result': analysis_result
        })
        
    except Exception as e:
        app_logger.error(f"Error in automated analysis: {e}")
        return jsonify({
            'success': False,
            'error': f'Automated analysis failed: {str(e)}'
        }), 500

@app.route('/api/enhanced/execute-trade', methods=['POST'])
@handle_errors
def execute_enhanced_trade():
    """Execute trade with enhanced trading bot"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    if not enhanced_bot_instance.connected:
        return jsonify({
            'success': False,
            'error': 'Not connected to MT5'
        }), 400
    
    data = request.get_json() or {}
    signal = data.get('signal')
    
    if not signal:
        return jsonify({
            'success': False,
            'error': 'No trading signal provided'
        }), 400
    
    app_logger.info("Executing trade with enhanced trading bot")
    
    try:
        result = enhanced_bot_instance.execute_automated_trade(signal)
        
        return jsonify({
            'success': True,
            'trade_result': result,
            'bot_status': enhanced_bot_instance.get_enhanced_status()
        })
        
    except Exception as e:
        app_logger.error(f"Error executing trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Trade execution failed: {str(e)}'
        }), 500

@app.route('/api/enhanced/start-automation', methods=['POST'])
@handle_errors
def start_enhanced_automation():
    """Start automated trading with enhanced trading bot"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    if not enhanced_bot_instance.connected:
        return jsonify({
            'success': False,
            'error': 'Not connected to MT5'
        }), 400
    
    data = request.get_json() or {}
    interval_minutes = data.get('interval_minutes', 5)
    max_cycles = data.get('max_cycles')
    
    app_logger.info(f"Starting enhanced automation (interval: {interval_minutes} minutes)")
    
    try:
        success = enhanced_bot_instance.start_automated_trading()
        
        if success:
            # Start automation in a separate thread
            def run_automation():
                enhanced_bot_instance.run_continuous_automation(interval_minutes, max_cycles)
            
            automation_thread = threading.Thread(target=run_automation, daemon=True)
            automation_thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Automated trading started',
                'bot_status': enhanced_bot_instance.get_enhanced_status()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start automated trading'
            }), 500
            
    except Exception as e:
        app_logger.error(f"Error starting automation: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to start automation: {str(e)}'
        }), 500

@app.route('/api/enhanced/stop-automation', methods=['POST'])
@handle_errors
def stop_enhanced_automation():
    """Stop automated trading with enhanced trading bot"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    app_logger.info("Stopping enhanced automation")
    enhanced_bot_instance.stop_automated_trading()
    
    return jsonify({
        'success': True,
        'message': 'Automated trading stopped',
        'bot_status': enhanced_bot_instance.get_enhanced_status()
    })

@app.route('/api/enhanced/status', methods=['GET'])
@handle_errors
def get_enhanced_status():
    """Get enhanced trading bot status"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    return jsonify({
        'success': True,
        'bot_status': enhanced_bot_instance.get_enhanced_status(),
        'automation_summary': enhanced_bot_instance.get_automation_summary()
    })

@app.route('/api/enhanced/positions', methods=['GET'])
@handle_errors
def get_enhanced_positions():
    """Get MT5 positions for enhanced trading bot"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    try:
        positions = enhanced_bot_instance.get_mt5_positions()
        
        return jsonify({
            'success': True,
            'positions': positions,
            'count': len(positions)
        })
        
    except Exception as e:
        app_logger.error(f"Error getting positions: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get positions: {str(e)}'
        }), 500

@app.route('/api/enhanced/close-position', methods=['POST'])
@handle_errors
@validate_required_fields(['ticket'])
def close_enhanced_position():
    """Close a specific MT5 position"""
    if not enhanced_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Enhanced trading bot not initialized'
        }), 400
    
    if not enhanced_bot_instance.connected:
        return jsonify({
            'success': False,
            'error': 'Not connected to MT5'
        }), 400
    
    data = request.get_json()
    ticket = data['ticket']
    
    app_logger.info(f"Closing position {ticket}")
    
    try:
        success = enhanced_bot_instance.close_mt5_position(ticket)
        
        return jsonify({
            'success': success,
            'message': f'Position {ticket} closed' if success else f'Failed to close position {ticket}',
            'bot_status': enhanced_bot_instance.get_enhanced_status()
        })
        
    except Exception as e:
        app_logger.error(f"Error closing position: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to close position: {str(e)}'
        }), 500

@app.route('/api/enhanced/capabilities', methods=['GET'])
@handle_errors
def get_enhanced_capabilities():
    """Get enhanced trading bot capabilities"""
    return jsonify({
        'success': True,
        'capabilities': {
            'manual_analysis': True,
            'automated_trading': True,
            'mt5_integration': True,
            'multi_timeframe': True,
            'risk_management': True,
            'continuous_automation': True,
            'position_management': True,
            'market_types': ['stock', 'forex', 'crypto', 'commodities'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        },
        'available_symbols': AVAILABLE_SYMBOLS,
        'available_timeframes': AVAILABLE_TIMEFRAMES,
        'market_types': MARKET_TYPES
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app_logger.info("Starting Trading Bot API Server...")
    app_logger.info(f"Available symbols: {len(AVAILABLE_SYMBOLS)}")
    app_logger.info(f"Available timeframes: {len(AVAILABLE_TIMEFRAMES)}")
    app_logger.info(f"Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 