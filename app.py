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
    'account_number': None,
    'password': None,
    'server': None,
    'symbol': None,
    'timeframe': None,
    'risk_per_trade': 0.02,
    'auto_trade': False,
    'use_ml': True
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

@app.route('/api/connect', methods=['POST'])
def connect_mt5():
    """Connect to MT5 with XM credentials"""
    global bot_instance
    
    try:
        data = request.get_json()
        account_number = data.get('account_number')
        password = data.get('password')
        server = data.get('server', '')
        
        if not account_number or not password or not server:
            return jsonify({'success': False, 'error': 'Account number, password, and server are required'}), 400
        
        # Create MT5 connector
        connector = MT5Connector(account_number, password, server)
        
        # Try to connect
        if connector.connect():
            # Get account info
            account_info = connector.get_account_summary()
            
            # Update global config
            bot_config.update({
                'account_number': account_number,
                'password': password,
                'server': server
            })
            
            # Store connector globally for reuse
            bot_instance = MT5TradingBot(
                symbol=bot_config.get('symbol', 'EURUSD'),
                timeframe=bot_config.get('timeframe', '5m'),
                risk_per_trade=bot_config.get('risk_per_trade', 0.02),
                use_mt5_data=True,
                auto_trade=False,
                use_ml=bot_config.get('use_ml', True)
            )
            bot_instance.mt5_connector = connector
            bot_instance.connected = True
            bot_instance.account_size = account_info.get('balance', 10000) if account_info else 10000
            
            return jsonify({
                'success': True,
                'message': 'Successfully connected to MT5',
                'account_info': account_info
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to connect to MT5'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
        
        if analysis:
            # Get ML prediction if available
            ml_prediction = None
            if bot_instance.use_ml and bot_instance.model_trained and bot_instance.analysis_bot and bot_instance.analysis_bot.data is not None:
                ml_prediction = bot_instance.get_ml_prediction(bot_instance.analysis_bot.data)
            
            # Get trading signals
            signals = bot_instance.get_trading_signals(analysis)
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'ml_prediction': ml_prediction,
                'signals': signals,
                'symbol': symbol,
                'timeframe': timeframe
            })
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
        
        if not bot_instance:
            return jsonify({'success': False, 'error': 'No bot instance. Run analysis first.'}), 400
        
        if not bot_instance.connected:
            return jsonify({'success': False, 'error': 'Not connected to MT5. Please connect first.'}), 400
        
        if bot_running:
            return jsonify({'success': False, 'error': 'Trading is already running'}), 400
        
        # Update bot configuration
        bot_instance.auto_trade = auto_trade
        bot_config['auto_trade'] = auto_trade
        
        # Start trading in a separate thread
        bot_running = True
        bot_thread = threading.Thread(target=run_trading_loop)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Trading started successfully'
        })
    
    except Exception as e:
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
        'account_info': None
    }
    
    if bot_instance and hasattr(bot_instance, 'connected') and bot_instance.connected:
        status['connected'] = True
        try:
            status['positions'] = bot_instance.monitor_positions() or []
            status['account_info'] = bot_instance.mt5_connector.get_account_summary()
        except Exception as e:
            print(f"Error getting status: {e}")
            status['positions'] = []
            status['account_info'] = None
    
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

def run_trading_loop():
    """Run the trading loop in a separate thread"""
    global bot_instance, bot_running
    
    try:
        while bot_running and bot_instance:
            # Run analysis cycle
            analysis = bot_instance.run_analysis_cycle()
            
            # Monitor positions
            if bot_instance.connected:
                bot_instance.monitor_positions()
            
            # Wait for next cycle (5 minutes)
            for _ in range(300):  # 5 minutes = 300 seconds
                if not bot_running:
                    break
                time.sleep(1)
    
    except Exception as e:
        print(f"Error in trading loop: {e}")
        traceback.print_exc()
    finally:
        bot_running = False

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Trading Bot API Server...")
    print("📊 Available symbols:", len(AVAILABLE_SYMBOLS))
    print("⏰ Available timeframes:", len(AVAILABLE_TIMEFRAMES))
    print("🌐 Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 