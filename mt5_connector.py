#!/usr/bin/env python3
"""
MetaTrader 5 Connector for XM Trading Account
============================================

This module provides integration between the trading bot and MetaTrader 5
for executing trades on XM trading account.
"""

import MetaTrader5 as mt5
import logging
import time as _time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from utils.backoff import backoff

# Load environment variables
load_dotenv()

from typing import Any, Dict, List, Optional, Tuple


class MT5Connector:
    def __init__(self, account_number: Optional[str] = None, password: Optional[str] = None, server: Optional[str] = None) -> None:
        """
        Initialize MT5 connector for XM trading account
        
        Args:
            account_number (str): XM account number
            password (str): XM account password
            server (str): XM server name (e.g., 'XMGlobal-Demo' for demo, 'XMGlobal-Live' for live)
        """
        self.account_number = account_number
        self.password = password
        self.server = server or 'XMGlobal-Demo'
        self.connected = False
        self.account_info = None
        self.last_error = None
        # Caching for performance (symbol info and ticks)
        self._cache = {}
        self.cache_ttl_seconds = 1.0
        # Logger
        self.logger = logging.getLogger('mt5_connector')
        
        # XM specific settings
        self.xm_settings = {
            'demo': {
                'server': 'XMGlobal-Demo',
                'description': 'XM Demo Account'
            },
            'live': {
                'server': 'XMGlobal-Live',
                'description': 'XM Live Account'
            }
        }
        
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 with XM account credentials
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.last_error = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {self.last_error}")
                return False
            
            # Login to XM account
            if not mt5.login(
                login=int(self.account_number),
                password=self.password,
                server=self.server
            ):
                self.last_error = mt5.last_error()
                self.logger.error(f"MT5 login failed: {self.last_error} | account={self.account_number} server={self.server}")
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.logger.error("Failed to get account information")
                return False
            
            self.connected = True
            
            self.logger.info("Successfully connected to XM trading account")
            self.logger.info(f"Login={self.account_info.login} Server={self.account_info.server} Balance=${self.account_info.balance:.2f} Equity=${self.account_info.equity:.2f} Margin=${self.account_info.margin:.2f} FreeMargin=${self.account_info.margin_free:.2f}")
            
            # Check AutoTrading status
            self.check_autotrading_enabled()
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.exception(f"Connection error: {e}")
            return False

    def get_last_error(self) -> Optional[Any]:
        """Return the last MT5 error encountered during connect/login."""
        return self.last_error
    
    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
    
    @backoff(retries=3, base_delay_seconds=0.2, max_delay_seconds=1.5)
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information for trading
        
        Args:
            symbol (str): Symbol to get info for (e.g., 'EURUSD', 'GBPUSD')
            
        Returns:
            dict: Symbol information or None if failed
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return None
        
        try:
            # Cache lookup
            cache_key = f"symbol_info:{symbol}"
            cached = self._cache.get(cache_key)
            now = _time.time()
            if cached and (now - cached['ts'] < self.cache_ttl_seconds):
                return cached['value']
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return None
            
            # Enable symbol for trading
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select symbol {symbol}")
                    return None
            
            # Create info dict with safe attribute access
            info = {
                'symbol': symbol,
                'digits': getattr(symbol_info, 'digits', 5),
                'spread': getattr(symbol_info, 'spread', 0),
                'trade_mode': getattr(symbol_info, 'trade_mode', 0),
                'volume_min': getattr(symbol_info, 'volume_min', 0.01),
                'volume_max': getattr(symbol_info, 'volume_max', 100.0),
                'volume_step': getattr(symbol_info, 'volume_step', 0.01),
                'point': getattr(symbol_info, 'point', 0.00001),
                'tick_size': getattr(symbol_info, 'tick_size', 0.00001),
                'trade_stops_level': getattr(symbol_info, 'trade_stops_level', 10),
                'trade_contract_size': getattr(symbol_info, 'trade_contract_size', 100000)
            }
            
            # Try to get tick_value safely
            try:
                info['tick_value'] = symbol_info.tick_value
            except AttributeError:
                # Calculate tick_value based on point and contract size
                contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
                info['tick_value'] = info['point'] * contract_size
            
            # Store in cache
            self._cache[cache_key] = {'value': info, 'ts': now}
            return info
            
        except Exception as e:
            self.logger.exception(f"Error getting symbol info: {e}")
            return None

    def get_pip_value_per_lot(self, symbol: str) -> float:
        """
        Compute pip value per 1.00 lot for the given symbol using MT5 tick data.
        Falls back to $10 per pip per lot for most FX majors if unavailable.
        """
        try:
            info = self.get_symbol_info(symbol)
            if not info:
                return 10.0
            # Determine pip size
            pip_size = 0.01 if 'JPY' in symbol.upper() else 0.0001
            tick_size = info.get('tick_size', 0.00001)
            # Obtain tick_value using MT5 symbol_info as it reflects value per tick for 1 lot
            si = mt5.symbol_info(symbol)
            tick_value = getattr(si, 'tick_value', None)
            if tick_value is None or tick_value <= 0 or tick_size <= 0:
                return 10.0
            # Scale pip value from tick value
            pip_value_per_lot = tick_value * (pip_size / tick_size)
            # Sanity bounds
            if pip_value_per_lot <= 0:
                return 10.0
            return pip_value_per_lot
        except Exception:
            return 10.0

    def calculate_position_size_robust(self, symbol: str, order_type: str, risk_percent: float, entry_price: float, stop_loss_price: float, margin_buffer: float = 0.95) -> float:
        """
        Calculate a safe position size constrained by risk amount and free margin.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD')
            order_type (str): 'BUY' or 'SELL'
            risk_percent (float): Fraction of balance to risk (e.g., 0.02)
            entry_price (float): Proposed entry price
            stop_loss_price (float): Proposed stop loss price
            margin_buffer (float): Fraction of free margin allowed to be used (e.g., 0.95)

        Returns:
            float: volume in lots (rounded to step), or 0.0 if cannot open safely
        """
        try:
            info = self.get_symbol_info(symbol)
            if not info or entry_price is None or stop_loss_price is None:
                return 0.0

            # Get account summary
            acct = self.get_account_summary() if hasattr(self, 'get_account_summary') else None
            if not acct:
                return 0.0
            balance = float(acct.get('balance', 0) or 0)
            free_margin = float(acct.get('margin_free', 0) or 0)

            # Pip computations
            pip_size = 0.01 if 'JPY' in symbol.upper() else 0.0001
            pip_value_per_lot = self.get_pip_value_per_lot(symbol)
            stop_loss_pips = abs(entry_price - stop_loss_price) / pip_size
            if stop_loss_pips <= 0:
                # Fallback to min volume
                return max(info.get('volume_min', 0.01), 0.01)

            # Risk-based volume
            risk_amount = balance * float(risk_percent)
            if risk_amount <= 0:
                return 0.0
            vol_by_risk = risk_amount / (stop_loss_pips * pip_value_per_lot)

            # Apply broker constraints
            min_vol = info.get('volume_min', 0.01)
            max_vol = info.get('volume_max', 100.0)
            step = info.get('volume_step', 0.01)

            def round_to_step(v):
                return max(min_vol, min(max_vol, round(v / step) * step))

            vol_by_risk = round_to_step(vol_by_risk)

            # Margin constraint using order_calc_margin
            order_const = mt5.ORDER_TYPE_BUY if str(order_type).upper() == 'BUY' else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            ref_price = getattr(tick, 'ask', None) if order_const == mt5.ORDER_TYPE_BUY else getattr(tick, 'bid', None)
            if ref_price is None:
                ref_price = entry_price

            # Margin per lot
            margin_one = mt5.order_calc_margin(order_const, symbol, 1.0, ref_price)
            max_by_margin = None
            if margin_one is not None and margin_one > 0:
                max_by_margin = math.floor(((free_margin * margin_buffer) / margin_one) / step) * step
                max_by_margin = round_to_step(max_by_margin)

            if max_by_margin is not None:
                volume = min(vol_by_risk, max_by_margin)
            else:
                volume = vol_by_risk

            # Ensure not below min lot due to constraints
            if volume < min_vol:
                return 0.0

            return round_to_step(volume)
        except Exception as e:
            self.logger.exception(f"Error in robust position sizing: {e}")
            return 0.0
    
    @backoff(retries=2, base_delay_seconds=0.5, max_delay_seconds=2.0)
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get historical data from MT5
        
        Args:
            symbol (str): Symbol to get data for
            timeframe (str): Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            count (int): Number of candles to get
            
        Returns:
            pd.DataFrame: Historical data or None if failed
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return None
        
        try:
            # Convert timeframe string to MT5 timeframe
            mt5_timeframe = self.convert_timeframe(timeframe)
            if mt5_timeframe is None:
                return None
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None:
                self.logger.error(f"Failed to get historical data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match yfinance format
            column_mapping = {
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }
            
            # Rename columns that exist
            for mt5_col, yf_col in column_mapping.items():
                if mt5_col in df.columns:
                    df.rename(columns={mt5_col: yf_col}, inplace=True)
            
            # Ensure we have all required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Volume':
                        # If volume is missing, add a default volume
                        df['Volume'] = 1000
                    else:
                        self.logger.error(f"Missing required column: {col}")
                        return None
            
            # Sort by date to ensure chronological order
            df = df.sort_values('Date').reset_index(drop=True)
            
            self.logger.debug(f"Got {len(df)} data points for {symbol} on {timeframe} timeframe | Columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.exception(f"Error getting historical data: {e}")
            return None
    
    def convert_timeframe(self, timeframe: str) -> Optional[int]:
        """
        Convert timeframe string to MT5 timeframe constant
        
        Args:
            timeframe (str): Timeframe string
            
        Returns:
            int: MT5 timeframe constant or None if invalid
        """
        timeframe_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
        return timeframe_map.get(timeframe)
    
    @backoff(retries=2, base_delay_seconds=0.2, max_delay_seconds=1.0)
    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market price for a symbol
        
        Args:
            symbol (str): Symbol to get price for
            
        Returns:
            dict: Current bid/ask prices or None if failed
        """
        if not self.connected:
            return None
        
        try:
            # Cache lookup
            cache_key = f"tick:{symbol}"
            cached = self._cache.get(cache_key)
            now = _time.time()
            if cached and (now - cached['ts'] < self.cache_ttl_seconds):
                return cached['value']
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            value = {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': tick.time
            }
            # Store in cache
            self._cache[cache_key] = {'value': value, 'ts': now}
            return value
            
        except Exception as e:
            self.logger.exception(f"Error getting current price: {e}")
            return None
    
    def validate_stop_levels(self, symbol: str, entry_price: float, sl: Optional[float] = None, tp: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Validate and adjust stop loss and take profit levels
        
        Args:
            symbol (str): Symbol to trade
            entry_price (float): Entry price
            sl (float): Stop loss price
            tp (float): Take profit price
            
        Returns:
            tuple: (adjusted_sl, adjusted_tp) or (None, None) if invalid
        """
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                self.logger.error("Could not get symbol info")
                return None, None
            
            # Get current market price
            current_prices = self.get_current_price(symbol)
            if current_prices is None:
                self.logger.error("Could not get current prices")
                return None, None
            
            # Use appropriate price based on order type
            if entry_price and entry_price > 0:
                current_price = current_prices['ask'] if entry_price > current_prices['bid'] else current_prices['bid']
            else:
                current_price = current_prices['ask']  # Default to ask price
            
            # Get minimum stop level directly from symbol info
            min_stop_level = 0  # Initialize variable
            try:
                symbol_info_raw = mt5.symbol_info(symbol)
                if symbol_info_raw is None:
                    self.logger.error("Could not get raw symbol info")
                    return None, None
                
                stop_level_points = symbol_info_raw.trade_stops_level
                min_stop_level = stop_level_points  # Set the variable
                point = symbol_info_raw.point
                min_distance = stop_level_points * point  # This is in price units
                
                self.logger.debug(f"Broker stop level: {stop_level_points} points | point={point} | min_distance={min_distance:.5f}")
                
                # If min_distance is 0, use a reasonable default
                if min_distance == 0:
                    min_distance = 0.00050  # 50 pips for major pairs (more conservative)
                    self.logger.warning(f"Using default minimum distance: {min_distance:.5f}")
                
            except Exception as e:
                self.logger.warning(f"Error getting broker stop level: {e}")
                # Fallback to symbol_info dict
                min_stop_level = symbol_info.get('trade_stops_level', 10)
                point = symbol_info['point']
                min_distance = min_stop_level * point
                
                if min_distance == 0:
                    min_distance = 0.00050  # 50 pips for major pairs (more conservative)
                    self.logger.warning(f"Using fallback minimum distance: {min_distance:.5f}")
            
            # Add extra buffer for safety (200% more than minimum for extra safety)
            safe_distance = min_distance * 3.0
            
            self.logger.debug(f"STOP LEVEL VALIDATION | symbol={symbol} current={current_price:.5f} min_stop_level={min_stop_level} point={point} min_dist={min_distance:.5f} safe={safe_distance:.5f}")
            
            adjusted_sl = sl
            adjusted_tp = tp
            
            # Validate stop loss
            if sl is not None:
                sl_distance = abs(current_price - sl)
                self.logger.debug(f"Original SL: {sl:.5f} (distance: {sl_distance:.5f})")
                
                if sl_distance < safe_distance:
                    # Adjust stop loss to safe distance
                    if sl < current_price:  # Stop loss below current price (for BUY)
                        adjusted_sl = current_price - safe_distance
                    else:  # Stop loss above current price (for SELL)
                        adjusted_sl = current_price + safe_distance
                        self.logger.warning(f"Stop loss adjusted to safe distance: {adjusted_sl:.5f}")
                else:
                    self.logger.debug("Stop loss is valid")
            
            # Validate take profit
            if tp is not None:
                tp_distance = abs(current_price - tp)
                self.logger.debug(f"Original TP: {tp:.5f} (distance: {tp_distance:.5f})")
                
                if tp_distance < safe_distance:
                    # Adjust take profit to safe distance
                    if tp > current_price:  # Take profit above current price (for BUY)
                        adjusted_tp = current_price + safe_distance
                    else:  # Take profit below current price (for SELL)
                        adjusted_tp = current_price - safe_distance
                    self.logger.warning(f"Take profit adjusted to safe distance: {adjusted_tp:.5f}")
                else:
                    self.logger.debug("Take profit is valid")
            
            return adjusted_sl, adjusted_tp
            
        except Exception as e:
            self.logger.exception(f"Error validating stop levels: {e}")
            return None, None
    
    def get_symbol_stop_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed symbol stop level information
        
        Args:
            symbol (str): Symbol to get info for
            
        Returns:
            dict: Stop level information or None if failed
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            current_prices = self.get_current_price(symbol)
            if current_prices is None:
                return None
            
            # Get minimum stop level directly from symbol info
            min_stop_level = 0  # Initialize variable
            try:
                symbol_info_raw = mt5.symbol_info(symbol)
                if symbol_info_raw is None:
                    return None
                
                stop_level_points = symbol_info_raw.trade_stops_level
                min_stop_level = stop_level_points  # Set the variable
                point = symbol_info_raw.point
                min_distance = stop_level_points * point  # This is in price units
                
                # If min_distance is 0, use a reasonable default
                if min_distance == 0:
                    min_distance = 0.00050  # 50 pips for major pairs (more conservative)
                
            except Exception as e:
                # Fallback to symbol_info dict
                min_stop_level = symbol_info.get('trade_stops_level', 10)
                point = symbol_info['point']
                min_distance = min_stop_level * point
                
                if min_distance == 0:
                    min_distance = 0.00050  # 50 pips for major pairs (more conservative)
            
            # Add extra buffer for safety (200% more than minimum for extra safety)
            safe_distance = min_distance * 3.0
            
            return {
                'symbol': symbol,
                'current_bid': current_prices['bid'],
                'current_ask': current_prices['ask'],
                'min_stop_level': min_stop_level,
                'point': point,
                'min_distance': min_distance,
                'safe_distance': safe_distance,
                'digits': symbol_info.get('digits', 5)
            }
            
        except Exception as e:
            self.logger.exception(f"Error getting symbol stop info: {e}")
            return None
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """
        Get MT5 terminal information
        
        Returns:
            dict: Terminal information or None if failed
        """
        if not self.connected:
            return None
        
        try:
            terminal_info = mt5.terminal_info()
            info = {}
            
            # Get attributes that exist with safe access
            info['connected'] = getattr(terminal_info, 'connected', False)
            info['trade_allowed'] = getattr(terminal_info, 'trade_allowed', False)
            info['expert_allowed'] = getattr(terminal_info, 'expert_allowed', False)
            info['dlls_allowed'] = getattr(terminal_info, 'dlls_allowed', False)
            info['trade_timeout'] = getattr(terminal_info, 'trade_timeout', 0)
            info['path'] = getattr(terminal_info, 'path', '')
            info['data_path'] = getattr(terminal_info, 'data_path', '')
            info['common_path'] = getattr(terminal_info, 'common_path', '')
            
            return info
            
        except Exception as e:
            print(f"❌ Error getting terminal info: {e}")
            return None
    
    def check_autotrading_enabled(self) -> bool:
        """
        Check if AutoTrading is enabled in MT5
        
        Returns:
            bool: True if AutoTrading is enabled, False otherwise
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return False
        
        try:
            # Get terminal info
            terminal_info = self.get_terminal_info()
            if terminal_info is None:
                return False
            
            # Check if AutoTrading is enabled
            if not terminal_info['trade_allowed']:
                self.logger.warning("AutoTrading is disabled in MetaTrader 5")
                self.logger.info(f"Terminal Status | trade_allowed={terminal_info['trade_allowed']} expert_allowed={terminal_info['expert_allowed']} dlls_allowed={terminal_info['dlls_allowed']} trade_timeout={terminal_info['trade_timeout']}")
                return False

            self.logger.info("AutoTrading is enabled")
            self.logger.info(f"Expert Advisors: {'Allowed' if terminal_info['expert_allowed'] else 'Disabled'} | DLLs: {'Allowed' if terminal_info['dlls_allowed'] else 'Disabled'}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error checking AutoTrading status: {e}")
            return False
    
    def test_stop_levels(self, symbol: str, entry_price: Optional[float], sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        """
        Test stop levels without placing an order
        
        Args:
            symbol (str): Symbol to test
            entry_price (float): Entry price
            sl (float): Stop loss price
            tp (float): Take profit price
            
        Returns:
            bool: True if stop levels are valid, False otherwise
        """
        try:
            # Get current market price if entry_price is not provided
            if entry_price is None or entry_price == 0:
                current_prices = self.get_current_price(symbol)
                if current_prices is None:
                    print("❌ Could not get current market price")
                    return False
                entry_price = current_prices['ask']  # Use ask price as default
                print(f"📊 Using current ask price: {entry_price:.5f}")
            
            print(f"\n🧪 TESTING STOP LEVELS FOR {symbol}")
            print(f"   Entry Price: {entry_price:.5f}")
            print(f"   Stop Loss: {sl:.5f}" if sl else "   Stop Loss: None")
            print(f"   Take Profit: {tp:.5f}" if tp else "   Take Profit: None")
            
            # Validate stop levels
            adjusted_sl, adjusted_tp = self.validate_stop_levels(symbol, entry_price, sl, tp)
            
            if adjusted_sl is None and adjusted_tp is None:
                print("❌ Stop level validation failed")
                return False
            
            print(f"\n📊 FINAL STOP LEVELS:")
            print(f"   Stop Loss: {adjusted_sl:.5f}" if adjusted_sl else "   Stop Loss: None")
            print(f"   Take Profit: {adjusted_tp:.5f}" if adjusted_tp else "   Take Profit: None")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing stop levels: {e}")
            return False
    
    def test_mt5_functionality(self) -> bool:
        """
        Test if MT5 is working properly
        
        Returns:
            bool: True if MT5 is working, False otherwise
        """
        if not self.connected:
            print("❌ Not connected to MT5")
            return False
        
        try:
            print("\n🧪 TESTING MT5 FUNCTIONALITY:")
            
            # Test symbol info
            symbol = "EURUSD"
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Could not get symbol info for {symbol}")
                return False
            print(f"✅ Symbol info OK for {symbol}")
            
            # Test symbol selection
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Could not select symbol {symbol}")
                return False
            print(f"✅ Symbol selection OK for {symbol}")
            
            # Test current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"❌ Could not get current price for {symbol}")
                return False
            print(f"✅ Current price OK: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
            
            # Test account info
            account = mt5.account_info()
            if account is None:
                print(f"❌ Could not get account info")
                return False
            print(f"✅ Account info OK: Balance=${account.balance:.2f}")
            
            print("✅ MT5 functionality test passed")
            return True
            
        except Exception as e:
            print(f"❌ MT5 functionality test failed: {e}")
            return False
    
    def place_order_no_stops(self, symbol: str, order_type: str, volume: float, price: Optional[float] = None, comment: str = "") -> Optional[Dict[str, Any]]:
        """
        Place a trade order without stop loss or take profit
        
        Args:
            symbol (str): Symbol to trade
            order_type (str): 'BUY' or 'SELL'
            volume (float): Position size in lots
            price (float): Entry price (None for market order)
            comment (str): Order comment
            
        Returns:
            dict: Order result or None if failed
        """
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        # Check if AutoTrading is enabled
        if not self.check_autotrading_enabled():
            return None
        
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Get current market price for market orders
            if price is None:
                current_prices = self.get_current_price(symbol)
                if current_prices is None:
                    print("❌ Could not get current market price")
                    return None
                
                # Use ask price for BUY orders, bid price for SELL orders
                if order_type.upper() == 'BUY':
                    price = current_prices['ask']
                else:
                    price = current_prices['bid']
                
                print(f"📊 Using market price: {price:.5f}")
            
            print(f"\n🚀 PLACING ORDER WITHOUT STOPS:")
            print(f"   Symbol: {symbol}")
            print(f"   Type: {order_type}")
            print(f"   Volume: {volume}")
            print(f"   Price: {price:.5f}")
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": None,
                "tp": None,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            # Check if result is None (order failed)
            if result is None:
                print(f"❌ Order failed: MT5 returned None")
                print(f"   Last error: {mt5.last_error()}")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Order failed: {result.retcode} - {result.comment}")
                return None
            
            print(f"✅ Order placed successfully")
            print(f"   Symbol: {symbol}")
            print(f"   Type: {order_type}")
            print(f"   Volume: {volume}")
            print(f"   Price: {result.price}")
            print(f"   Order ID: {result.order}")
            
            return {
                'order_id': result.order,
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'price': result.price,
                'sl': None,
                'tp': None,
                'comment': comment
            }
            
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            return None
    
    def place_order(self, symbol: str, order_type: str, volume: float, price: Optional[float] = None, sl: Optional[float] = None, tp: Optional[float] = None, comment: str = "") -> Optional[Dict[str, Any]]:
        """
        Place a trade order
        
        Args:
            symbol (str): Symbol to trade
            order_type (str): 'BUY' or 'SELL'
            volume (float): Position size in lots
            price (float): Entry price (None for market order)
            sl (float): Stop loss price
            tp (float): Take profit price
            comment (str): Order comment
            
        Returns:
            dict: Order result or None if failed
        """
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        # Check if AutoTrading is enabled
        if not self.check_autotrading_enabled():
            return None
        
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Test MT5 functionality first
            if not self.test_mt5_functionality():
                self.logger.error("MT5 functionality test failed")
                return None
            
            # Test stop levels first
            if not self.test_stop_levels(symbol, price or 0, sl, tp):
                self.logger.error("Stop level test failed")
                return None
            
            # Get current market price for market orders
            if price is None:
                current_prices = self.get_current_price(symbol)
                if current_prices is None:
                    self.logger.error("Could not get current market price")
                    return None
                
                # Use ask price for BUY orders, bid price for SELL orders
                if order_type.upper() == 'BUY':
                    price = current_prices['ask']
                else:
                    price = current_prices['bid']
                
                self.logger.debug(f"Using market price: {price:.5f}")
            
            # Validate and adjust stop levels
            adjusted_sl, adjusted_tp = self.validate_stop_levels(symbol, price, sl, tp)
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": adjusted_sl,
                "tp": adjusted_tp,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
            }
            
            # Validate request before sending
            self.logger.info(f"ORDER REQUEST | symbol={request['symbol']} type={order_type} volume={request['volume']} price={request['price']} sl={request['sl']} tp={request['tp']}")
            
            # Check if symbol is available for trading
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Symbol {symbol} is not available for trading")
                return None
            
            # Send order
            result = mt5.order_send(request)
            
            # Check if result is None (order failed)
            if result is None:
                self.logger.error("Order failed: MT5 returned None | last_error=%s", mt5.last_error())
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                
                # Provide specific guidance for common error codes
                if result.retcode == 10019:  # No money
                    self.logger.warning("ERROR 10019 - NO MONEY: insufficient margin to place the trade")
                    
                    # Get current account status for debugging
                    try:
                        account_info = self.get_account_summary()
                        if account_info:
                            self.logger.info("Account Status | balance=$%s equity=$%s margin=$%s free_margin=$%s margin_level=%s%%",
                                            f"{account_info.get('balance', 0):,.2f}",
                                            f"{account_info.get('equity', 0):,.2f}",
                                            f"{account_info.get('margin', 0):,.2f}",
                                            f"{account_info.get('margin_free', 0):,.2f}",
                                            f"{account_info.get('margin_level', 0):,.2f}")
                    except:
                        pass
                
                elif result.retcode == 10018:  # Market closed
                    self.logger.warning("ERROR 10018 - MARKET CLOSED: The market is currently closed for this symbol")
                
                elif result.retcode == 10004:  # Requote
                    self.logger.warning("ERROR 10004 - REQUOTE: Price has changed. Try placing the order again")
                
                elif result.retcode == 10006:  # Request rejected
                    self.logger.warning("ERROR 10006 - REQUEST REJECTED: Order was rejected by the broker")
                
                elif result.retcode == 10014:  # Invalid volume
                    self.logger.warning(f"ERROR 10014 - INVALID VOLUME: requested volume={volume}")
                    
                    # Get symbol info for debugging
                    try:
                        symbol_info = self.get_symbol_info(symbol)
                        if symbol_info:
                            self.logger.info(f"Symbol Volume Requirements | min={symbol_info.get('volume_min', 'N/A')} max={symbol_info.get('volume_max', 'N/A')} step={symbol_info.get('volume_step', 'N/A')} requested={volume}")
                    except:
                        pass
                
                else:
                    self.logger.info(f"For more information about error {result.retcode}, check MT5 documentation")
                
                return None
            
            self.logger.info(f"Order placed successfully | symbol={symbol} type={order_type} volume={volume} price={result.price} sl={adjusted_sl} tp={adjusted_tp} order_id={result.order}")
            
            return {
                'order_id': result.order,
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'price': result.price,
                'sl': adjusted_sl,
                'tp': adjusted_tp,
                'comment': comment
            }
            
        except Exception as e:
            self.logger.exception(f"Error placing order: {e}")
            return None
    
    def close_position(self, position_id: int) -> bool:
        """
        Close a specific position
        
        Args:
            position_id (int): Position ticket ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return False
        
        try:
            # Get position info
            position = mt5.positions_get(ticket=position_id)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False
            
            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            # Check if result is None (order failed)
            if result is None:
                self.logger.error("Close order failed: MT5 returned None | last_error=%s", mt5.last_error())
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close order failed: {result.retcode} - {result.comment}")
                return False
            
            self.logger.info(f"Position {position_id} closed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error closing position: {e}")
            return False
    
    def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get all open positions
        
        Returns:
            list: List of open positions or None if failed
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return None
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'comment': pos.comment
                }
                for pos in positions
            ]
            
        except Exception as e:
            self.logger.exception(f"Error getting positions: {e}")
            return None
    
    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get account summary information
        
        Returns:
            dict: Account summary or None if failed
        """
        if not self.connected or self.account_info is None:
            self.logger.error("Not connected to MT5")
            return None
        
        try:
            return {
                'login': self.account_info.login,
                'server': self.account_info.server,
                'balance': self.account_info.balance,
                'equity': self.account_info.equity,
                'margin': self.account_info.margin,
                'margin_free': self.account_info.margin_free,
                'margin_level': self.account_info.margin_level,
                'currency': self.account_info.currency
            }
            
        except Exception as e:
            self.logger.exception(f"Error getting account summary: {e}")
            return None
    
    def calculate_position_size(self, risk_amount: float, entry_price: float, stop_loss_price: float, symbol: str) -> Optional[float]:
        """
        Calculate position size based on risk amount
        
        Args:
            risk_amount (float): Risk amount in account currency
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            symbol (str): Symbol to trade
            
        Returns:
            float: Position size in lots or None if failed
        """
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Calculate price difference
            price_diff = abs(entry_price - stop_loss_price)
            
            # Calculate position size
            # For forex: 1 lot = 100,000 units
            # Risk = (price_diff * position_size * 100000) / account_currency_rate
            position_size = risk_amount / (price_diff * 100000)
            
            # Round to symbol's volume step
            position_size = round(position_size / symbol_info['volume_step']) * symbol_info['volume_step']
            
            # Check volume limits
            if position_size < symbol_info['volume_min']:
                position_size = symbol_info['volume_min']
            elif position_size > symbol_info['volume_max']:
                position_size = symbol_info['volume_max']
            
            return position_size
            
        except Exception as e:
            self.logger.exception(f"Error calculating position size: {e}")
            return None

# Example usage and testing
def test_mt5_connection():
    """Test MT5 connection with XM account"""
    print("="*60)
    print("TESTING MT5 CONNECTION TO XM ACCOUNT")
    print("="*60)
    
    # Create connector
    connector = MT5Connector()
    
    # Try to connect
    if connector.connect():
        print("\n✅ Connection successful!")
        
        # Get account summary
        summary = connector.get_account_summary()
        if summary:
            print(f"\n📊 Account Summary:")
            print(f"   Login: {summary['login']}")
            print(f"   Server: {summary['server']}")
            print(f"   Balance: ${summary['balance']:.2f}")
            print(f"   Equity: ${summary['equity']:.2f}")
            print(f"   Free Margin: ${summary['margin_free']:.2f}")
        
        # Get symbol info
        symbol_info = connector.get_symbol_info("EURUSD")
        if symbol_info:
            print(f"\n📈 Symbol Info (EURUSD):")
            print(f"   Digits: {symbol_info['digits']}")
            print(f"   Spread: {symbol_info['spread']}")
            print(f"   Min Volume: {symbol_info['volume_min']}")
            print(f"   Max Volume: {symbol_info['volume_max']}")
        
        # Get historical data
        data = connector.get_historical_data("EURUSD", "1h", 100)
        if data is not None:
            print(f"\n📊 Historical Data:")
            print(f"   Data points: {len(data)}")
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Disconnect
        connector.disconnect()
    else:
        print("\n❌ Connection failed!")
        print("Please check your XM account credentials in .env file")

if __name__ == "__main__":
    test_mt5_connection() 