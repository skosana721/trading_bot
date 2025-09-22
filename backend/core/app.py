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

# Configure TensorFlow warnings before any other imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import backend.config.tensorflow_config

import logging
import sys
from flask import Flask, request, jsonify, render_template, session as flask_session
from flask_cors import CORS
import os
import secrets
import json
import threading
import time
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps

# Admin portal is now separate and runs on port 5001
ADMIN_AVAILABLE = False

# Import deployment configuration
from backend.config.deployment_config import DeploymentConfig, DEPLOYMENT_MODE, PLATFORM_LIMITATIONS

# Import trading bot components
try:
    from backend.connectors.mt5_connector import MT5Connector
    from backend.core.mt5_trading_bot import MT5TradingBot
    from backend.core.powerful_trading_bot import PowerfulTradingBot
    from backend.config.powerful_trading_config import POWERFUL_TRADING_CONFIG
    MT5_AVAILABLE = True
    POWERFUL_BOT_AVAILABLE = True
except ImportError:
    print("Warning: MetaTrader5 components not available - running in simulation mode")
    MT5_AVAILABLE = False
    POWERFUL_BOT_AVAILABLE = False

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

# Set up template and static folders
# We'll use the frontend folder for templates
main_template_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend')
main_static_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static')

app = Flask(__name__, template_folder=main_template_folder, static_folder=main_static_folder)
CORS(app)  # Enable CORS for Vue frontend

# Session secret key (for signed client-side sessions)
app.secret_key = os.getenv('FLASK_SECRET') or secrets.token_hex(32)

# Admin portal runs separately on port 5001

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
powerful_bot_instance = None  # Powerful trading bot instance
bot_thread = None
bot_running = False
# Multi-symbol trading support
trading_bots = {}
bot_config = {
    'account_number': '',
    'password': '',
    'server': '',
    'symbol': '',
    'timeframe': '1h',
    # Risk will be provided by the frontend via /api/config
    'risk_per_trade': None,
    'account_size': None,  # No longer used - balance fetched from MT5
    'auto_trade': os.getenv('AUTO_TRADE', 'false').lower() == 'true',
    'use_ml': os.getenv('USE_ML', 'true').lower() == 'true',
    'enable_automation': os.getenv('ENABLE_AUTOMATION', 'true').lower() == 'true',
    # Symbols must be selected from the frontend; start empty
    'symbols_to_trade': []
}

# Available symbols and timeframes
AVAILABLE_SYMBOLS = [
    # Forex Major Pairs
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    # Forex Cross Pairs
    'EURGBP', 'EURJPY', 'GBPJPY', 'CHFJPY', 'AUDCAD', 'AUDCHF', 'AUDJPY',
    'AUDNZD', 'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
    # Commodities
    'XAUUSD',  # Gold
    # Indices
    'US30',    # Dow Jones
    'NAS100',  # NASDAQ 100
    # Cryptocurrency
    'BTCUSD'   # Bitcoin
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
        enhanced_bot_instance = MT5TradingBot(
            symbol=symbol,
            timeframe=timeframe,
            risk_per_trade=(bot_config['risk_per_trade'] if bot_config['risk_per_trade'] is not None else 0.02),
            use_mt5_data=True,
            auto_trade=enable_automation
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
        'server': '',
        'symbol': '',
        'timeframe': '5m',
        # Risk should be provided by frontend; do not source from env here
        'risk_per_trade': None,
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

        # Persist session flags (avoid storing password)
        try:
            flask_session['mt5_connected'] = True
            flask_session['account_number'] = str(account_number)
            flask_session['server'] = server
            flask_session['connected_at'] = datetime.now().isoformat()
        except Exception:
            pass
        
        # Store the MT5 connector globally for later use
        global trading_bots, bot_instance
        trading_bots = {}
        
        # Create a minimal bot instance for connection testing (without requiring symbols)
        # This allows users to connect first, then configure symbols later
        test_bot = MT5TradingBot(
            symbol='EURUSD',  # Use a default symbol for testing
            timeframe=bot_config.get('timeframe', '5m'),
            risk_per_trade=(bot_config.get('risk_per_trade') if bot_config.get('risk_per_trade') is not None else 0.02),
            use_mt5_data=True,
            auto_trade=False,  # Don't start auto trading yet
            use_ml=bot_config.get('use_ml', True)
        )
        test_bot.mt5_connector = connector
        test_bot.connected = True
        test_bot.auto_trade = False
        test_bot.last_analysis = None
        
        # Set as primary instance for now
        bot_instance = test_bot
        
        # Test market data retrieval with default symbol
        app_logger.info("Testing market data retrieval with default symbol...")
        test_data = bot_instance.get_market_data()
        if test_data is None or len(test_data) < 10:
            app_logger.warning("Limited market data available, but connection is working")
        else:
            app_logger.info(f"Market data test successful ({len(test_data)} data points)")
        
        # Note: Trading bot instances for specific symbols will be created when symbols are configured
        # via /api/config or when starting automated trading
        
        # Auto-generate an API key upon successful connection if not set, or rotate
        global API_KEY
        API_KEY = secrets.token_urlsafe(32)
        os.environ['API_KEY'] = API_KEY
        app_logger.info("API key generated for session")

        return jsonify({
            'success': True,
            'message': 'Successfully connected to MT5. You can now configure symbols and start trading.',
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
                'data_available': test_data is not None and len(test_data) > 0,
                'symbols_configured': len(bot_config.get('symbols_to_trade', [])) > 0 or bot_config.get('symbol') is not None
            },
            'api_key': API_KEY,
            'next_steps': 'Configure symbols via /api/config to start trading'
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
                risk_per_trade=(bot_config.get('risk_per_trade') if bot_config.get('risk_per_trade') is not None else 0.02),
                use_mt5_data=True,
                auto_trade=False,
                use_ml=bot_config.get('use_ml', True)
            )
            if bot_instance and bot_instance.mt5_connector:
                b.mt5_connector = bot_instance.mt5_connector
                b.connected = True
        
        # Ensure we have a proper MT5TradingBot instance
        if not hasattr(b, 'get_recent_data'):
            app_logger.warning(f"Bot instance for {sym} missing get_recent_data method, recreating...")
            # Recreate the bot instance
            b = MT5TradingBot(
                symbol=sym,
                timeframe=timeframe,
                risk_per_trade=(bot_config.get('risk_per_trade') if bot_config.get('risk_per_trade') is not None else 0.02),
                use_mt5_data=True,
                auto_trade=False,
                use_ml=bot_config.get('use_ml', True)
            )
            if bot_instance and hasattr(bot_instance, 'mt5_connector'):
                b.mt5_connector = bot_instance.mt5_connector
                b.connected = getattr(bot_instance, 'connected', False)
            trading_bots[sym] = b
        
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
    symbol = data.get('symbol', bot_config.get('symbol'))
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
            from connectors.mt5_connector import MT5Connector
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
    
    # Check if symbols are configured for trading
    selected = bot_config.get('symbols_to_trade') or []
    if len(selected) == 0:
        # If no multi-symbol selection, use single configured symbol when provided
        fallback_symbol = bot_config.get('symbol')
        if fallback_symbol:
            selected = [fallback_symbol]
        else:
            return jsonify({
                'success': False, 
                'error': 'No symbols configured for trading. Please configure symbols via /api/config before starting trading.',
                'help': 'You can configure symbols by sending a POST request to /api/config with {"symbols_to_trade": ["EURUSD", "GBPUSD"]} or {"symbol": "EURUSD"}'
            }), 400
    
    # Create trading bot instances for the selected symbols
    app_logger.info(f"Creating trading bot instances for symbols: {selected}")
    for sym in selected:
        b = MT5TradingBot(
            symbol=sym,
            timeframe=timeframe,
            risk_per_trade=bot_config.get('risk_per_trade', 0.02),
            use_mt5_data=True,
            auto_trade=auto_trade,
            use_ml=bot_config.get('use_ml', True)
        )
        b.mt5_connector = bot_instance.mt5_connector
        b.connected = True
        b.auto_trade = auto_trade
        b.last_analysis = None
        trading_bots[sym] = b
    
    # Set primary instance to the first symbol
    primary_symbol = selected[0]
    bot_instance = trading_bots.get(primary_symbol)
    
    # Warm-up analysis in background to speed up start (model load, caches)
    def _warm_up(symbols):
        try:
            app_logger.info("Warming up symbols in background...")
            for sym in symbols:
                b = trading_bots.get(sym) or bot_instance
                if b.symbol != sym:
                    b.symbol = sym
                b.auto_trade = auto_trade
                
                # Ensure we have a proper MT5TradingBot instance
                if not hasattr(b, 'get_recent_data'):
                    app_logger.warning(f"Bot instance for {sym} missing get_recent_data method, recreating...")
                    # Recreate the bot instance
                    b = MT5TradingBot(
                        symbol=sym,
                        timeframe=bot_config.get('timeframe', '5m'),
                        risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                        use_mt5_data=True,
                        auto_trade=auto_trade,
                        use_ml=bot_config.get('use_ml', True)
                    )
                    if bot_instance and hasattr(bot_instance, 'mt5_connector'):
                        b.mt5_connector = bot_instance.mt5_connector
                        b.connected = getattr(bot_instance, 'connected', False)
                    trading_bots[sym] = b
                
                try:
                    b.run_analysis_cycle()
                    app_logger.info(f"Warm-up analysis completed for {sym}")
                except Exception as e:
                    app_logger.warning(f"Warm-up analysis failed for {sym}: {e}")
        except Exception as e:
            app_logger.warning(f"Warm-up task error: {e}")

    threading.Thread(target=_warm_up, args=(selected,), name="WarmUpThread", daemon=True).start()
    
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
            'risk_per_trade': ((bot_config.get('risk_per_trade') or 0.02) * 100),
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
    
    # Clear session and sensitive config inputs when session ends
    try:
        flask_session.clear()
    except Exception:
        pass
    
    # Clear sensitive bot_config fields
    bot_config.update({
        'account_number': '',
        'password': '',
        'server': ''
    })
    # Optionally clear symbol selection to force re-input on next session
    # Keep timeframe and other preferences
    # bot_config['symbol'] = ''
    # bot_config['symbols_to_trade'] = []
    
    return jsonify({
        'success': True,
        'message': 'Trading stopped successfully and session cleared'
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
        'deployment_info': {
            'mode': DEPLOYMENT_MODE,
            'limitations': PLATFORM_LIMITATIONS,
            'mt5_available': MT5_AVAILABLE
        },
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

    # Final volume validation using symbol-specific constraints
    sym_info = b.mt5_connector.get_symbol_info(symbol)
    if sym_info:
        min_vol = sym_info.get('volume_min', 0.01)
        max_vol = sym_info.get('volume_max', 100.0)
        step = sym_info.get('volume_step', 0.01)
        print(f"📊 Symbol {symbol} volume constraints: min={min_vol}, max={max_vol}, step={step}")
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
            },
            'ml_ensemble_details': combined_signal.get('ml_prediction', {}).get('ensemble_details', {}) if combined_signal.get('ml_prediction') else {}
        }
        
        return jsonify(response)
        
    except Exception as e:
        app_logger.error(f"Combined analysis error: {str(e)}")
        return jsonify({'error': f'Combined analysis error: {str(e)}'}), 500

@app.route('/api/ml_ensemble_summary/<symbol>/<timeframe>', methods=['GET'])
def get_ml_ensemble_summary(symbol, timeframe):
    """Get ML ensemble summary and performance metrics"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get ML ensemble summary
        ensemble_summary = bot.get_ml_ensemble_summary()
        
        if not ensemble_summary:
            return jsonify({'error': 'ML ensemble not available or not trained'}), 404
        
        return jsonify(ensemble_summary)
        
    except Exception as e:
        app_logger.error(f"ML ensemble summary error: {str(e)}")
        return jsonify({'error': f'ML ensemble summary error: {str(e)}'}), 500

@app.route('/api/market_structure_analysis/<symbol>/<timeframe>', methods=['GET'])
@handle_errors
def get_market_structure_analysis(symbol, timeframe):
    """Get market structure analysis results"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get market structure analysis
        analysis = bot.run_market_structure_analysis()
        
        if not analysis:
            return jsonify({'error': 'Market structure analysis not available'}), 404
        
        return jsonify(analysis)
        
    except Exception as e:
        app_logger.error(f"Market structure analysis error: {str(e)}")
        return jsonify({'error': f'Market structure analysis error: {str(e)}'}), 500

@app.route('/api/market_structure_summary/<symbol>/<timeframe>', methods=['GET'])
@handle_errors
def get_market_structure_summary(symbol, timeframe):
    """Get market structure strategy summary"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get market structure summary
        summary = bot.get_market_structure_summary()
        
        if not summary:
            return jsonify({'error': 'Market structure strategy not available'}), 404
        
        return jsonify(summary)
        
    except Exception as e:
        app_logger.error(f"Market structure summary error: {str(e)}")
        return jsonify({'error': f'Market structure summary error: {str(e)}'}), 500

@app.route('/api/rl_analysis/<symbol>/<timeframe>', methods=['GET'])
@handle_errors
def get_rl_analysis(symbol, timeframe):
    """Get reinforcement learning analysis results"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get RL analysis
        rl_analysis = bot.run_rl_analysis()
        
        if not rl_analysis:
            return jsonify({'error': 'RL analysis not available'}), 404
        
        return jsonify(rl_analysis)
        
    except Exception as e:
        app_logger.error(f"RL analysis error: {str(e)}")
        return jsonify({'error': f'RL analysis error: {str(e)}'}), 500

@app.route('/api/rl_summary/<symbol>/<timeframe>', methods=['GET'])
@handle_errors
def get_rl_summary(symbol, timeframe):
    """Get reinforcement learning summary and performance metrics"""
    try:
        # Create trading bot instance
        bot = MT5TradingBot(symbol, timeframe, use_smc=True, use_ml=True)
        
        # Get RL summary
        summary = bot.get_rl_summary()
        
        if not summary:
            return jsonify({'error': 'RL summary not available'}), 404
        
        return jsonify(summary)
        
    except Exception as e:
        app_logger.error(f"RL summary error: {str(e)}")
        return jsonify({'error': f'RL summary error: {str(e)}'}), 500

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
                selected = bot_config.get('symbols_to_trade') or []
                if len(selected) == 0:
                    # If no multi-symbol selection, use single configured symbol when provided
                    fallback_symbol = bot_config.get('symbol')
                    if fallback_symbol:
                        selected = [fallback_symbol]
                    else:
                        app_logger.warning("No symbols configured for trading. Skipping analysis cycle.")
                        continue
                
                app_logger.info(f"Running market analysis for: {selected}")
                for sym in selected:
                    b = trading_bots.get(sym) or bot_instance
                    if b.symbol != sym:
                        b.symbol = sym
                    
                    # Ensure we have a proper MT5TradingBot instance
                    if not hasattr(b, 'get_recent_data'):
                        app_logger.warning(f"Bot instance for {sym} missing get_recent_data method, recreating...")
                        # Recreate the bot instance
                        b = MT5TradingBot(
                            symbol=sym,
                            timeframe=bot_config.get('timeframe', '5m'),
                            risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                            use_mt5_data=True,
                            auto_trade=bot_config.get('auto_trade', False),
                            use_ml=bot_config.get('use_ml', True)
                        )
                        if bot_instance and hasattr(bot_instance, 'mt5_connector'):
                            b.mt5_connector = bot_instance.mt5_connector
                            b.connected = getattr(bot_instance, 'connected', False)
                        trading_bots[sym] = b
                    
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

# ============================================================================
# TRADE JOURNAL API ENDPOINTS
# ============================================================================

# Global trade journal storage (in production, use a database)
trade_journal = []
trade_stats = {
    'total_trades': 0,
    'total_wins': 0,
    'total_losses': 0,
    'win_rate': 0.0,
    'total_profit': 0.0,
    'total_loss': 0.0,
    'net_pnl': 0.0
}

@app.route('/api/trade-journal', methods=['POST'])
@handle_errors
@require_api_key
def log_trade():
    """Log a new trade to the journal"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['symbol', 'entry_price', 'take_profit', 'stop_loss', 'trade_type']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create trade entry
        trade_entry = {
            'id': len(trade_journal) + 1,
            'symbol': data['symbol'],
            'trade_type': data['trade_type'].upper(),  # BUY or SELL
            'entry_price': float(data['entry_price']),
            'take_profit': float(data['take_profit']),
            'stop_loss': float(data['stop_loss']),
            'entry_date': data.get('entry_date', datetime.now().isoformat()),
            'notes': data.get('notes', ''),
            'status': 'OPEN',  # OPEN, WIN, LOSS
            'exit_price': None,
            'exit_date': None,
            'pnl': 0.0,
            'pnl_percentage': 0.0
        }
        
        # Add to journal
        trade_journal.append(trade_entry)
        
        # Update statistics
        update_trade_statistics()
        
        app_logger.info(f"Trade logged: {trade_entry['symbol']} {trade_entry['trade_type']} at {trade_entry['entry_price']}")
        
        return jsonify({
            'success': True,
            'message': 'Trade logged successfully',
            'trade_id': trade_entry['id'],
            'trade': trade_entry
        })
        
    except Exception as e:
        app_logger.error(f"Error logging trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to log trade: {str(e)}'
        }), 500

@app.route('/api/trade-journal', methods=['GET'])
@handle_errors
def get_trade_journal():
    """Get trades from the journal with pagination support"""
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status_filter = request.args.get('status', None)
        symbol_filter = request.args.get('symbol', None)
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 10
        
        # Sort by entry date (newest first)
        sorted_trades = sorted(trade_journal, key=lambda x: x['entry_date'], reverse=True)
        
        # Apply filters
        filtered_trades = sorted_trades
        if status_filter:
            filtered_trades = [t for t in filtered_trades if t['status'].upper() == status_filter.upper()]
        if symbol_filter:
            filtered_trades = [t for t in filtered_trades if t['symbol'].upper() == symbol_filter.upper()]
        
        # Calculate pagination
        total_trades = len(filtered_trades)
        total_pages = (total_trades + per_page - 1) // per_page
        
        # Get trades for current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_trades = filtered_trades[start_idx:end_idx]
        
        # Calculate pagination info
        pagination_info = {
            'current_page': page,
            'per_page': per_page,
            'total_trades': total_trades,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_page': page + 1 if page < total_pages else None,
            'prev_page': page - 1 if page > 1 else None
        }
        
        return jsonify({
            'success': True,
            'trades': page_trades,
            'statistics': trade_stats,
            'pagination': pagination_info,
            'filters': {
                'status': status_filter,
                'symbol': symbol_filter
            }
        })
        
    except Exception as e:
        app_logger.error(f"Error retrieving trade journal: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve trade journal: {str(e)}'
        }), 500

@app.route('/api/trade-journal/<int:trade_id>', methods=['PUT'])
@handle_errors
@require_api_key
def update_trade(trade_id):
    """Update a trade entry (e.g., mark as closed)"""
    try:
        data = request.get_json()
        
        # Find the trade
        trade = next((t for t in trade_journal if t['id'] == trade_id), None)
        if not trade:
            return jsonify({
                'success': False,
                'error': 'Trade not found'
            }), 404
        
        # Update trade fields
        if 'exit_price' in data:
            trade['exit_price'] = float(data['exit_price'])
            trade['exit_date'] = data.get('exit_date', datetime.now().isoformat())
            
            # Calculate PnL
            if trade['trade_type'] == 'BUY':
                pnl = trade['exit_price'] - trade['entry_price']
            else:  # SELL
                pnl = trade['entry_price'] - trade['exit_price']
            
            trade['pnl'] = pnl
            trade['pnl_percentage'] = (pnl / trade['entry_price']) * 100
            
            # Determine if WIN or LOSS
            if trade['trade_type'] == 'BUY':
                if trade['exit_price'] >= trade['take_profit']:
                    trade['status'] = 'WIN'
                elif trade['exit_price'] <= trade['stop_loss']:
                    trade['status'] = 'LOSS'
            else:  # SELL
                if trade['exit_price'] <= trade['take_profit']:
                    trade['status'] = 'WIN'
                elif trade['exit_price'] >= trade['stop_loss']:
                    trade['status'] = 'LOSS'
        
        # Update other fields
        for field in ['notes', 'status']:
            if field in data:
                trade[field] = data[field]
        
        # Update statistics
        update_trade_statistics()
        
        app_logger.info(f"Trade {trade_id} updated: {trade['status']}")
        
        return jsonify({
            'success': True,
            'message': 'Trade updated successfully',
            'trade': trade
        })
        
    except Exception as e:
        app_logger.error(f"Error updating trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to update trade: {str(e)}'
        }), 500

@app.route('/api/trade-journal/<int:trade_id>', methods=['DELETE'])
@handle_errors
@require_api_key
def delete_trade(trade_id):
    """Delete a trade from the journal"""
    try:
        # Find and remove the trade
        global trade_journal
        trade_journal = [t for t in trade_journal if t['id'] != trade_id]
        
        # Update statistics
        update_trade_statistics()
        
        app_logger.info(f"Trade {trade_id} deleted")
        
        return jsonify({
            'success': True,
            'message': 'Trade deleted successfully'
        })
        
    except Exception as e:
        app_logger.error(f"Error deleting trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to delete trade: {str(e)}'
        }), 500

@app.route('/api/trade-journal/statistics', methods=['GET'])
@handle_errors
def get_trade_statistics():
    """Get comprehensive trade statistics"""
    try:
        # Calculate additional statistics
        closed_trades = [t for t in trade_journal if t['status'] in ['WIN', 'LOSS']]
        open_trades = [t for t in trade_journal if t['status'] == 'OPEN']
        
        # Calculate win streak and loss streak
        win_streak = 0
        loss_streak = 0
        current_streak = 0
        current_streak_type = None
        
        for trade in sorted(closed_trades, key=lambda x: x['exit_date'] or x['entry_date']):
            if trade['status'] == 'WIN':
                if current_streak_type == 'WIN':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = 'WIN'
                win_streak = max(win_streak, current_streak)
            else:  # LOSS
                if current_streak_type == 'LOSS':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = 'LOSS'
                loss_streak = max(loss_streak, current_streak)
        
        # Calculate average win/loss
        wins = [t for t in closed_trades if t['status'] == 'WIN']
        losses = [t for t in closed_trades if t['status'] == 'LOSS']
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        # Calculate profit factor
        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        extended_stats = {
            **trade_stats,
            'open_trades': len(open_trades),
            'closed_trades': len(closed_trades),
            'win_streak': win_streak,
            'loss_streak': loss_streak,
            'current_streak': current_streak,
            'current_streak_type': current_streak_type,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': max((t['pnl'] for t in wins), default=0),
            'largest_loss': min((t['pnl'] for t in losses), default=0)
        }
        
        return jsonify({
            'success': True,
            'statistics': extended_stats
        })
        
    except Exception as e:
        app_logger.error(f"Error calculating trade statistics: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to calculate statistics: {str(e)}'
        }), 500

@app.route('/api/trade-journal/evaluate', methods=['POST'])
@handle_errors
@require_api_key
def evaluate_trade_outcome():
    """Evaluate trade outcome based on current market price"""
    try:
        data = request.get_json()
        trade_id = data.get('trade_id')
        current_price = data.get('current_price')
        
        if not trade_id or not current_price:
            return jsonify({
                'success': False,
                'error': 'Missing trade_id or current_price'
            }), 400
        
        # Find the trade
        trade = next((t for t in trade_journal if t['id'] == trade_id), None)
        if not trade:
            return jsonify({
                'success': False,
                'error': 'Trade not found'
            }), 404
        
        if trade['status'] != 'OPEN':
            return jsonify({
                'success': False,
                'error': 'Trade is already closed'
            }), 400
        
        # Evaluate outcome
        outcome = None
        if trade['trade_type'] == 'BUY':
            if current_price >= trade['take_profit']:
                outcome = 'WIN'
            elif current_price <= trade['stop_loss']:
                outcome = 'LOSS'
        else:  # SELL
            if current_price <= trade['take_profit']:
                outcome = 'WIN'
            elif current_price >= trade['stop_loss']:
                outcome = 'LOSS'
        
        if outcome:
            # Update trade
            trade['exit_price'] = float(current_price)
            trade['exit_date'] = datetime.now().isoformat()
            trade['status'] = outcome
            
            # Calculate PnL
            if trade['trade_type'] == 'BUY':
                pnl = trade['exit_price'] - trade['entry_price']
            else:  # SELL
                pnl = trade['entry_price'] - trade['exit_price']
            
            trade['pnl'] = pnl
            trade['pnl_percentage'] = (pnl / trade['entry_price']) * 100
            
            # Update statistics
            update_trade_statistics()
            
            return jsonify({
                'success': True,
                'message': f'Trade evaluated as {outcome}',
                'trade': trade,
                'outcome': outcome
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Trade is still open - no TP or SL hit',
                'trade': trade,
                'outcome': 'OPEN'
            })
        
    except Exception as e:
        app_logger.error(f"Error evaluating trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to evaluate trade: {str(e)}'
        }), 500

def update_trade_statistics():
    """Update global trade statistics"""
    global trade_stats
    
    closed_trades = [t for t in trade_journal if t['status'] in ['WIN', 'LOSS']]
    
    trade_stats['total_trades'] = len(trade_journal)
    trade_stats['total_wins'] = len([t for t in closed_trades if t['status'] == 'WIN'])
    trade_stats['total_losses'] = len([t for t in closed_trades if t['status'] == 'LOSS'])
    
    if trade_stats['total_trades'] > 0:
        trade_stats['win_rate'] = (trade_stats['total_wins'] / trade_stats['total_trades']) * 100
    else:
        trade_stats['win_rate'] = 0.0
    
    trade_stats['total_profit'] = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)
    trade_stats['total_loss'] = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
    trade_stats['net_pnl'] = sum(t['pnl'] for t in closed_trades)

# ============================================================================
# MT5 TRADING HISTORY API ENDPOINTS
# ============================================================================

@app.route('/api/mt5-trading-history', methods=['GET'])
@handle_errors
def get_mt5_trading_history():
    """Get trading history from MT5 account"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({
            'success': False,
            'error': 'Not connected to MT5'
        }), 400
    
    try:
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        symbol = request.args.get('symbol', None)
        
        # Validate days parameter
        if days < 1 or days > 365:
            return jsonify({
                'success': False,
                'error': 'Days parameter must be between 1 and 365'
            }), 400
        
        # Get trading history from MT5
        history = bot_instance.mt5_connector.get_trading_history(days, symbol)
        
        if history is None:
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve trading history from MT5'
            }), 500
        
        # Convert datetime objects to ISO format for JSON serialization
        for record in history:
            if isinstance(record['time'], datetime):
                record['time'] = record['time'].isoformat()
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history),
            'period_days': days,
            'symbol_filter': symbol
        })
        
    except Exception as e:
        app_logger.error(f"Error getting MT5 trading history: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get trading history: {str(e)}'
        }), 500

@app.route('/api/mt5-trading-history/summary', methods=['GET'])
@handle_errors
def get_mt5_trading_history_summary():
    """Get trading history summary from MT5 account"""
    global bot_instance
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({
            'success': False,
            'error': 'Not connected to MT5'
        }), 400
    
    try:
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        
        # Validate days parameter
        if days < 1 or days > 365:
            return jsonify({
                'success': False,
                'error': 'Days parameter must be between 1 and 365'
            }), 400
        
        # Get trading history summary from MT5
        summary = bot_instance.mt5_connector.get_trading_history_summary(days)
        
        if summary is None:
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve trading history summary from MT5'
            }), 500
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        app_logger.error(f"Error getting MT5 trading history summary: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get trading history summary: {str(e)}'
        }), 500

@app.route('/api/mt5-trading-history/import', methods=['POST'])
@handle_errors
@require_api_key
def import_mt5_trading_history():
    """Import trading history from MT5 into the trade journal"""
    global bot_instance, trade_journal
    
    if not bot_instance or not bot_instance.connected:
        return jsonify({
            'success': False,
            'error': 'Not connected to MT5'
        }), 400
    
    try:
        data = request.get_json() or {}
        days = data.get('days', 30)
        symbol = data.get('symbol', None)
        overwrite = data.get('overwrite', False)
        
        # Validate days parameter
        if days < 1 or days > 365:
            return jsonify({
                'success': False,
                'error': 'Days parameter must be between 1 and 365'
            }), 400
        
        # Get trading history from MT5
        history = bot_instance.mt5_connector.get_trading_history(days, symbol)
        
        if history is None:
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve trading history from MT5'
            }), 500
        
        app_logger.info(f"Retrieved {len(history)} trading history records from MT5")
        
        # Log sample record for debugging
        if history:
            sample_record = history[0]
            app_logger.info(f"Sample record types: {[(k, type(v).__name__) for k, v in sample_record.items()]}")
        
        # Group deals by order ticket to create complete trades
        trades_by_order = {}
        for i, record in enumerate(history):
            try:
                order_ticket = record.get('order_ticket')
                if order_ticket is None:
                    app_logger.warning(f"Record {i} missing order_ticket: {record}")
                    continue
                
                # Ensure order_ticket is consistently treated as an integer
                if isinstance(order_ticket, str):
                    try:
                        order_ticket = int(order_ticket)
                    except ValueError:
                        app_logger.warning(f"Invalid order ticket format: {order_ticket}")
                        continue
                elif not isinstance(order_ticket, int):
                    try:
                        order_ticket = int(order_ticket)
                    except (ValueError, TypeError):
                        app_logger.warning(f"Cannot convert order ticket to int: {order_ticket} (type: {type(order_ticket)})")
                        continue
                
                if order_ticket not in trades_by_order:
                    trades_by_order[order_ticket] = {'entry': None, 'exit': None}
                
                if record.get('entry') == 'IN':
                    trades_by_order[order_ticket]['entry'] = record
                else:
                    trades_by_order[order_ticket]['exit'] = record
                    
            except Exception as e:
                app_logger.error(f"Error processing record {i}: {e}")
                app_logger.error(f"Record data: {record}")
                continue
        
        # Convert to trade journal format
        imported_trades = []
        imported_count = 0
        
        for order_ticket, trade_data in trades_by_order.items():
            entry = trade_data['entry']
            exit = trade_data['exit']
            
            if not entry:
                continue  # Skip incomplete trades
            
            # Check if trade already exists in journal (ensure type consistency)
            existing_trade = None
            for t in trade_journal:
                existing_ticket = t.get('mt5_order_ticket')
                if existing_ticket is not None:
                    # Convert both to integers for comparison
                    try:
                        existing_ticket_int = int(existing_ticket) if isinstance(existing_ticket, str) else existing_ticket
                        if existing_ticket_int == order_ticket:
                            existing_trade = t
                            break
                    except (ValueError, TypeError):
                        continue
            if existing_trade and not overwrite:
                continue  # Skip existing trades unless overwrite is True
            
            # Create trade entry with proper type validation
            try:
                trade_entry = {
                    'id': len(trade_journal) + 1 + imported_count,
                    'symbol': str(entry.get('symbol', '')),
                    'trade_type': str(entry.get('trade_type', '')),
                    'entry_price': float(entry.get('price', 0.0)),
                    'take_profit': None,  # Not available in MT5 history
                    'stop_loss': None,    # Not available in MT5 history
                    'entry_date': entry['time'].isoformat() if isinstance(entry['time'], datetime) else str(entry.get('time', '')),
                    'notes': f"Imported from MT5 - {entry.get('comment', '')}",
                    'status': 'OPEN',
                    'exit_price': None,
                    'exit_date': None,
                    'pnl': 0.0,
                    'pnl_percentage': 0.0,
                    'volume': float(entry.get('volume', 0.0)),
                    'commission': float(entry.get('commission', 0.0)),
                    'swap': float(entry.get('swap', 0.0)),
                    'mt5_ticket': int(entry.get('ticket', 0)),
                    'mt5_order_ticket': int(order_ticket),
                    'magic': int(entry.get('magic', 0))
                }
            except (ValueError, TypeError) as e:
                app_logger.warning(f"Error creating trade entry for order {order_ticket}: {e}")
                continue
            
            # If there's an exit deal, update the trade
            if exit:
                try:
                    trade_entry['exit_price'] = float(exit.get('price', 0.0))
                    trade_entry['exit_date'] = exit['time'].isoformat() if isinstance(exit['time'], datetime) else str(exit.get('time', ''))
                    trade_entry['pnl'] = float(exit.get('profit', 0.0))
                    trade_entry['commission'] += float(exit.get('commission', 0.0))
                    trade_entry['swap'] += float(exit.get('swap', 0.0))
                    
                    # Determine status based on profit
                    exit_profit = float(exit.get('profit', 0.0))
                    if exit_profit > 0:
                        trade_entry['status'] = 'WIN'
                    elif exit_profit < 0:
                        trade_entry['status'] = 'LOSS'
                    else:
                        trade_entry['status'] = 'BREAKEVEN'
                    
                    # Calculate PnL percentage
                    if trade_entry['entry_price'] > 0:
                        trade_entry['pnl_percentage'] = (trade_entry['pnl'] / trade_entry['entry_price']) * 100
                except (ValueError, TypeError) as e:
                    app_logger.warning(f"Error processing exit deal for order {order_ticket}: {e}")
                    # Continue with the trade entry even if exit processing fails
            
            # Remove existing trade if overwrite is True
            if existing_trade and overwrite:
                trade_journal.remove(existing_trade)
            
            # Add to journal
            trade_journal.append(trade_entry)
            imported_trades.append(trade_entry)
            imported_count += 1
        
        # Update statistics
        update_trade_statistics()
        
        app_logger.info(f"Imported {imported_count} trades from MT5 history")
        
        return jsonify({
            'success': True,
            'message': f'Successfully imported {imported_count} trades from MT5',
            'imported_count': imported_count,
            'total_history_records': len(history),
            'imported_trades': imported_trades[:10]  # Return first 10 for preview
        })
        
    except Exception as e:
        app_logger.error(f"Error importing MT5 trading history: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to import trading history: {str(e)}'
        }), 500

# Powerful Trading Bot API Endpoints
@app.route('/api/powerful/start', methods=['POST'])
@handle_errors
def start_powerful_bot():
    """Start the powerful trading bot"""
    global powerful_bot_instance
    
    try:
        if not POWERFUL_BOT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Powerful trading bot not available'
            }), 400
        
        # Get configuration from request
        config = request.json if request.json else POWERFUL_TRADING_CONFIG.copy()
        
        # Initialize powerful bot
        powerful_bot_instance = PowerfulTradingBot(config)
        
        # Start the bot
        powerful_bot_instance.start()
        
        return jsonify({
            'success': True,
            'message': 'Powerful trading bot started successfully',
            'config': config
        })
        
    except Exception as e:
        app_logger.error(f"Error starting powerful bot: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to start powerful bot: {str(e)}'
        }), 500

@app.route('/api/powerful/stop', methods=['POST'])
@handle_errors
def stop_powerful_bot():
    """Stop the powerful trading bot"""
    global powerful_bot_instance
    
    try:
        if not powerful_bot_instance:
            return jsonify({
                'success': False,
                'error': 'Powerful trading bot not initialized'
            }), 400
        
        # Stop the bot
        powerful_bot_instance.stop()
        powerful_bot_instance = None
        
        return jsonify({
            'success': True,
            'message': 'Powerful trading bot stopped successfully'
        })
        
    except Exception as e:
        app_logger.error(f"Error stopping powerful bot: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to stop powerful bot: {str(e)}'
        }), 500

@app.route('/api/powerful/status', methods=['GET'])
@handle_errors
def get_powerful_bot_status():
    """Get powerful trading bot status"""
    if not powerful_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Powerful trading bot not initialized'
        }), 400
    
    try:
        status = powerful_bot_instance.get_comprehensive_status()
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        app_logger.error(f"Error getting powerful bot status: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get status: {str(e)}'
        }), 500

@app.route('/api/powerful/backtest', methods=['POST'])
@handle_errors
def run_powerful_backtest():
    """Run backtest with powerful trading bot"""
    if not powerful_bot_instance:
        return jsonify({
            'success': False,
            'error': 'Powerful trading bot not initialized'
        }), 400
    
    try:
        # Get backtest parameters
        params = request.json if request.json else {}
        symbol = params.get('symbol', 'EURUSD')
        days = params.get('days', 30)
        
        # Generate sample data for backtest
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d", interval="5m")
        
        if data.empty:
            return jsonify({
                'success': False,
                'error': f'No data available for {symbol}'
            }), 400
        
        # Convert to required format
        data = data.reset_index()
        data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        # Run backtest
        result = powerful_bot_instance.run_backtest(data)
        
        if result:
            return jsonify({
                'success': True,
                'backtest_result': {
                    'total_return': result.total_return,
                    'annualized_return': result.annualized_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Backtest failed'
            }), 500
            
    except Exception as e:
        app_logger.error(f"Error running backtest: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to run backtest: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app_logger.info("Starting Trading Bot API Server...")
    app_logger.info(f"Available symbols: {len(AVAILABLE_SYMBOLS)}")
    app_logger.info(f"Available timeframes: {len(AVAILABLE_TIMEFRAMES)}")
    app_logger.info(f"Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 