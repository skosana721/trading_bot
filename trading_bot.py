import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingBot:
    def __init__(self, symbol, period="1d", market_type="stock", account_size=10000, risk_per_trade=0.02):
        """
        Initialize the day trading bot
        
        Args:
            symbol (str): Symbol (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')
            period (str): Data period for day trading ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            market_type (str): Type of market ('stock', 'crypto', 'forex')
            account_size (float): Total account size in USD
            risk_per_trade (float): Risk per trade as percentage of account (default 2%)
        """
        self.symbol = symbol
        self.period = period
        self.market_type = market_type
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.data = None
        self.pivot_points = None
        self.trendlines = None
        self.risk_reward_ratio = 3.0  # 1:3 risk-to-reward ratio
        
        # Multi-timeframe data for advanced analysis
        self.daily_data = None
        self.weekly_data = None
        self.h4_data = None
        
        # Advanced trend analysis components
        self.support_resistance_levels = []
        self.breakout_levels = []
        self.continuation_patterns = []
        self.trend_strength = 0  # 0-100 scale
        
        # Validate timeframe for day trading
        self.valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if self.period not in self.valid_timeframes:
            print(f"⚠️  Warning: {self.period} is not a standard day trading timeframe.")
            print(f"   Recommended timeframes: {', '.join(self.valid_timeframes)}")
        
        # Define symbol mappings for different markets
        self.symbol_mappings = {
            'crypto': {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'ADA': 'ADA-USD',
                'DOT': 'DOT-USD',
                'LINK': 'LINK-USD',
                'LTC': 'LTC-USD',
                'XRP': 'XRP-USD',
                'BCH': 'BCH-USD',
                'BNB': 'BNB-USD',
                'SOL': 'SOL-USD'
            },
            'forex': {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'USDCHF': 'USDCHF=X',
                'AUDUSD': 'AUDUSD=X',
                'USDCAD': 'USDCAD=X',
                'NZDUSD': 'NZDUSD=X',
                'EURGBP': 'EURGBP=X',
                'EURJPY': 'EURJPY=X',
                'GBPJPY': 'GBPJPY=X'
            },
            'commodities': {
                'GOLD': 'GC=F',
                'SILVER': 'SI=F',
                'OIL': 'CL=F',
                'COPPER': 'HG=F',
                'NATURAL_GAS': 'NG=F'
            }
        }
        
    def get_full_symbol(self):
        """Get the full symbol for the given market type"""
        if self.market_type == 'crypto' and self.symbol in self.symbol_mappings['crypto']:
            return self.symbol_mappings['crypto'][self.symbol]
        elif self.market_type == 'forex' and self.symbol in self.symbol_mappings['forex']:
            return self.symbol_mappings['forex'][self.symbol]
        elif self.market_type == 'commodities' and self.symbol in self.symbol_mappings['commodities']:
            return self.symbol_mappings['commodities'][self.symbol]
        else:
            return self.symbol
        
    def get_data_period_for_timeframe(self):
        """
        Get the appropriate data period based on timeframe for day trading
        
        Returns:
            str: Data period for yfinance
        """
        # For day trading, we need enough data to analyze patterns
        # but not too much to avoid noise
        timeframe_periods = {
            '1m': '7d',     # 7 days for 1-minute data
            '5m': '60d',    # 60 days for 5-minute data (increased for more pivot points)
            '15m': '30d',   # 30 days for 15-minute data
            '30m': '60d',   # 60 days for 30-minute data
            '1h': '90d',    # 90 days for 1-hour data
            '4h': '180d',   # 180 days for 4-hour data
            '1d': '1y',     # 1 year for daily data
        }
        
        return timeframe_periods.get(self.period, '60d')
    
    def fetch_multi_timeframe_data(self):
        """
        Fetch data for multiple timeframes (Weekly, Daily, H4) for advanced analysis
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_symbol = self.get_full_symbol()
            
            # Fetch weekly data (for trend confirmation)
            print(f"📊 Fetching weekly data for {self.symbol}...")
            weekly_ticker = yf.Ticker(full_symbol)
            self.weekly_data = weekly_ticker.history(period='2y', interval='1wk')
            if self.weekly_data.empty:
                print(f"❌ Failed to fetch weekly data for {self.symbol}")
                return False
            
            # Fetch daily data (for trend confirmation)
            print(f"📊 Fetching daily data for {self.symbol}...")
            daily_ticker = yf.Ticker(full_symbol)
            self.daily_data = daily_ticker.history(period='1y', interval='1d')
            if self.daily_data.empty:
                print(f"❌ Failed to fetch daily data for {self.symbol}")
                return False
            
            # Fetch H4 data (for entry signals)
            print(f"📊 Fetching H4 data for {self.symbol}...")
            h4_ticker = yf.Ticker(full_symbol)
            self.h4_data = h4_ticker.history(period='90d', interval='4h')
            if self.h4_data.empty:
                print(f"❌ Failed to fetch H4 data for {self.symbol}")
                return False
            
            # Standardize column names
            for data in [self.weekly_data, self.daily_data, self.h4_data]:
                data.reset_index(inplace=True)
                data.rename(columns={'Datetime': 'Date'}, inplace=True)
                data['Date'] = pd.to_datetime(data['Date'])
            
            print(f"✅ Multi-timeframe data fetched successfully")
            print(f"   Weekly: {len(self.weekly_data)} bars")
            print(f"   Daily: {len(self.daily_data)} bars")
            print(f"   H4: {len(self.h4_data)} bars")
            
            return True
            
        except Exception as e:
            print(f"❌ Error fetching multi-timeframe data: {e}")
            return False
    
    def find_pivot_points_advanced(self, data, window=5):
        """
        Find pivot points with advanced filtering for trend analysis
        
        Args:
            data (pd.DataFrame): Price data
            window (int): Window size for pivot detection
        """
        if data is None or len(data) < window * 2:
            return pd.DataFrame()
        
        pivot_points = []
        
        for i in range(window, len(data) - window):
            # Check for pivot high
            if all(data['High'].iloc[i] >= data['High'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['High'].iloc[i] >= data['High'].iloc[i+j] for j in range(1, window+1)):
                pivot_points.append({
                    'date': data['Date'].iloc[i],
                    'price': data['High'].iloc[i],
                    'type': 'high',
                    'strength': self.calculate_pivot_strength(data, i, 'high')
                })
            
            # Check for pivot low
            if all(data['Low'].iloc[i] <= data['Low'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['Low'].iloc[i] <= data['Low'].iloc[i+j] for j in range(1, window+1)):
                pivot_points.append({
                    'date': data['Date'].iloc[i],
                    'price': data['Low'].iloc[i],
                    'type': 'low',
                    'strength': self.calculate_pivot_strength(data, i, 'low')
                })
        
        return pd.DataFrame(pivot_points)
    
    def calculate_pivot_strength(self, data, index, pivot_type):
        """
        Calculate the strength of a pivot point based on surrounding price action
        
        Args:
            data (pd.DataFrame): Price data
            index (int): Index of the pivot point
            pivot_type (str): 'high' or 'low'
            
        Returns:
            float: Pivot strength (0-100)
        """
        if index < 5 or index >= len(data) - 5:
            return 50  # Default strength for edge cases
        
        window = 5
        pivot_price = data['High'].iloc[index] if pivot_type == 'high' else data['Low'].iloc[index]
        
        # Calculate price difference from surrounding bars
        if pivot_type == 'high':
            surrounding_prices = data['High'].iloc[index-window:index+window+1]
            max_diff = max(abs(pivot_price - p) for p in surrounding_prices)
        else:
            surrounding_prices = data['Low'].iloc[index-window:index+window+1]
            max_diff = max(abs(pivot_price - p) for p in surrounding_prices)
        
        # Calculate volume confirmation
        avg_volume = data['Volume'].iloc[index-window:index+window+1].mean()
        current_volume = data['Volume'].iloc[index]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate strength based on price difference and volume
        price_strength = min(100, (max_diff / pivot_price) * 10000)  # Scale appropriately
        volume_strength = min(100, volume_ratio * 50)
        
        return (price_strength + volume_strength) / 2
    
    def identify_higher_highs_lows_advanced(self, data, min_points=2):
        """
        Advanced HH/HL identification with trendline analysis
        
        Args:
            data (pd.DataFrame): Price data
            min_points (int): Minimum number of HH/HL points to confirm uptrend
        """
        if data is None or len(data) < 10:
            return None
        
        # Find pivot points
        pivot_points = self.find_pivot_points_advanced(data)
        if len(pivot_points) < 4:
            return None
        
        # Separate highs and lows
        highs = pivot_points[pivot_points['type'] == 'high'].copy()
        lows = pivot_points[pivot_points['type'] == 'low'].copy()
        
        # Find higher highs with strength filtering
        higher_highs = []
        for i in range(1, len(highs)):
            if highs['price'].iloc[i] > highs['price'].iloc[i-1] and \
               highs['strength'].iloc[i] > 30:  # Minimum strength threshold
                higher_highs.append({
                    'date': highs['date'].iloc[i],
                    'price': highs['price'].iloc[i],
                    'type': 'HH',
                    'strength': highs['strength'].iloc[i]
                })
        
        # Find higher lows with strength filtering
        higher_lows = []
        for i in range(1, len(lows)):
            if lows['price'].iloc[i] > lows['price'].iloc[i-1] and \
               lows['strength'].iloc[i] > 30:  # Minimum strength threshold
                higher_lows.append({
                    'date': lows['date'].iloc[i],
                    'price': lows['price'].iloc[i],
                    'type': 'HL',
                    'strength': lows['strength'].iloc[i]
                })
        
        # Check if we have enough HH and HL points to confirm uptrend
        uptrend_confirmed = len(higher_highs) >= min_points and len(higher_lows) >= min_points
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(higher_highs, higher_lows)
        
        result = {
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'uptrend_confirmed': uptrend_confirmed,
            'hh_count': len(higher_highs),
            'hl_count': len(higher_lows),
            'trend_strength': trend_strength,
            'pivot_points': pivot_points,
            'timeframe': self.period
        }
        
        return result
    
    def calculate_trend_strength(self, higher_highs, higher_lows):
        """
        Calculate the strength of the uptrend based on HH/HL quality
        
        Args:
            higher_highs (list): List of higher high points
            higher_lows (list): List of higher low points
            
        Returns:
            float: Trend strength (0-100)
        """
        if not higher_highs or not higher_lows:
            return 0
        
        # Calculate average strength of HH and HL
        hh_strength = sum(hh['strength'] for hh in higher_highs) / len(higher_highs)
        hl_strength = sum(hl['strength'] for hl in higher_lows) / len(higher_lows)
        
        # Calculate price progression consistency
        hh_progression = 0
        if len(higher_highs) >= 2:
            for i in range(1, len(higher_highs)):
                price_diff = higher_highs[i]['price'] - higher_highs[i-1]['price']
                hh_progression += price_diff
        
        hl_progression = 0
        if len(higher_lows) >= 2:
            for i in range(1, len(higher_lows)):
                price_diff = higher_lows[i]['price'] - higher_lows[i-1]['price']
                hl_progression += price_diff
        
        # Normalize progression
        avg_price = (higher_highs[0]['price'] + higher_lows[0]['price']) / 2
        hh_progression_ratio = abs(hh_progression) / avg_price * 100
        hl_progression_ratio = abs(hl_progression) / avg_price * 100
        
        # Calculate final strength
        strength = (hh_strength + hl_strength) / 2 * 0.6 + \
                   (hh_progression_ratio + hl_progression_ratio) / 2 * 0.4
        
        return min(100, strength)
    
    def create_trendlines_advanced(self, trend_analysis):
        """
        Create advanced trendlines with support/resistance validation
        
        Args:
            trend_analysis (dict): Result from identify_higher_highs_lows_advanced
        """
        if trend_analysis is None or not trend_analysis['uptrend_confirmed']:
            return None
        
        trendlines = {}
        
        # Create trendline for higher highs (resistance)
        if len(trend_analysis['higher_highs']) >= 2:
            hh_points = trend_analysis['higher_highs'][-2:]  # Use last 2 HH points
            hh_trendline = self.calculate_trendline(hh_points[0], hh_points[1])
            hh_trendline['type'] = 'HH_trendline'
            hh_trendline['touches'] = self.count_trendline_touches(trend_analysis['pivot_points'], hh_trendline)
            trendlines['hh_trendline'] = hh_trendline
        
        # Create trendline for higher lows (support)
        if len(trend_analysis['higher_lows']) >= 2:
            hl_points = trend_analysis['higher_lows'][-2:]  # Use last 2 HL points
            hl_trendline = self.calculate_trendline(hl_points[0], hl_points[1])
            hl_trendline['type'] = 'HL_trendline'
            hl_trendline['touches'] = self.count_trendline_touches(trend_analysis['pivot_points'], hl_trendline)
            trendlines['hl_trendline'] = hl_trendline
        
        return trendlines
    
    def calculate_trendline(self, point1, point2):
        """
        Calculate trendline equation between two points
        
        Args:
            point1 (dict): First point with 'date' and 'price'
            point2 (dict): Second point with 'date' and 'price'
            
        Returns:
            dict: Trendline parameters
        """
        # Convert dates to numeric values for calculation
        date1 = pd.to_datetime(point1['date']).timestamp()
        date2 = pd.to_datetime(point2['date']).timestamp()
        
        # Calculate slope and intercept
        slope = (point2['price'] - point1['price']) / (date2 - date1)
        intercept = point1['price'] - slope * date1
        
        return {
            'start_date': point1['date'],
            'end_date': point2['date'],
            'start_price': point1['price'],
            'end_price': point2['price'],
            'slope': slope,
            'intercept': intercept
        }
    
    def count_trendline_touches(self, pivot_points, trendline):
        """
        Count how many times price touches a trendline
        
        Args:
            pivot_points (pd.DataFrame): All pivot points
            trendline (dict): Trendline parameters
            
        Returns:
            int: Number of touches
        """
        touches = 0
        tolerance = 0.001  # Price tolerance for touch detection
        
        for _, point in pivot_points.iterrows():
            # Calculate expected price on trendline at this date
            date_timestamp = pd.to_datetime(point['date']).timestamp()
            expected_price = trendline['slope'] * date_timestamp + trendline['intercept']
            
            # Check if actual price is close to expected price
            if abs(point['price'] - expected_price) / expected_price < tolerance:
                touches += 1
        
        return touches
    
    def detect_breakout_retest(self, data, trend_analysis, trendlines):
        """
        Detect breakout and retest patterns following the specific trading rules:
        - Every breakout has a retest
        - Wait for retest to confirm previous resistance becomes new support
        - Strong resistance requires 3-4 touches minimum
        
        Args:
            data (pd.DataFrame): Price data
            trend_analysis (dict): Trend analysis results
            trendlines (dict): Trendline analysis
            
        Returns:
            dict: Breakout/retest signals with retest confirmation
        """
        if not trendlines or data.empty:
            return None
        
        signals = {
            'breakout_detected': False,
            'retest_detected': False,
            'retest_confirmed': False,
            'entry_signal': False,
            'entry_price': None,
            'stop_loss': None,
            'target': None,
            'breakout_level': None,
            'retest_level': None,
            'pattern_strength': 0
        }
        
        # Check for breakout of HH trendline (resistance) - must be strong (3+ touches)
        if 'hh_trendline' in trendlines:
            hh_trendline = trendlines['hh_trendline']
            if hh_trendline['touches'] >= 3:  # Strong resistance requirement
                current_price = data['Close'].iloc[-1]
                current_date = pd.to_datetime(data['Date'].iloc[-1]).timestamp()
                resistance_level = hh_trendline['slope'] * current_date + hh_trendline['intercept']
                
                # Check for breakout (price above resistance)
                if current_price > resistance_level * 1.001:  # 0.1% above resistance
                    signals['breakout_detected'] = True
                    signals['breakout_level'] = resistance_level
                    signals['pattern_strength'] += 30  # Base strength for breakout
                    
                    print(f"   🚀 Breakout detected at {resistance_level:.5f}")
                    
                    # Look for retest (price comes back to test the breakout level)
                    recent_data = data.tail(15)  # Check last 15 bars for retest
                    retest_found = False
                    
                    for i, (_, bar) in enumerate(recent_data.iterrows()):
                        bar_date = pd.to_datetime(bar['Date']).timestamp()
                        bar_resistance = hh_trendline['slope'] * bar_date + hh_trendline['intercept']
                        
                        # Check if price retested the breakout level (touched or came close)
                        retest_tolerance = 0.003  # 0.3% tolerance for retest
                        if (bar['Low'] <= bar_resistance * (1 + retest_tolerance) and 
                            bar['High'] >= bar_resistance * (1 - retest_tolerance)):
                            
                            signals['retest_detected'] = True
                            signals['retest_level'] = bar_resistance
                            signals['pattern_strength'] += 20  # Additional strength for retest
                            
                            print(f"   🔄 Retest detected at {bar_resistance:.5f}")
                            
                            # Check if retest was successful (price bounced back up)
                            # Look at subsequent bars after retest
                            if i < len(recent_data) - 1:
                                subsequent_data = recent_data.iloc[i+1:]
                                bounce_confirmed = False
                                
                                for _, subsequent_bar in subsequent_data.iterrows():
                                    if subsequent_bar['Close'] > bar_resistance * 1.002:  # 0.2% above retest level
                                        bounce_confirmed = True
                                        signals['retest_confirmed'] = True
                                        signals['entry_signal'] = True
                                        signals['entry_price'] = subsequent_bar['Close']
                                        signals['stop_loss'] = bar_resistance * 0.995  # 0.5% below retest level
                                        signals['target'] = signals['entry_price'] + (signals['entry_price'] - signals['stop_loss']) * 3  # 1:3 ratio
                                        signals['pattern_strength'] += 50  # Full strength for confirmed retest
                                        
                                        print(f"   ✅ Retest confirmed - Previous resistance now support")
                                        print(f"   🎯 Entry signal generated")
                                        break
                                
                                if bounce_confirmed:
                                    break
                            
                            retest_found = True
                            break
                    
                    if not retest_found:
                        print(f"   ⏳ Breakout detected, waiting for retest...")
        
        # Also check for breakout of horizontal resistance levels
        if 'hl_trendline' in trendlines:
            hl_trendline = trendlines['hl_trendline']
            if hl_trendline['touches'] >= 3:  # Strong support
                current_price = data['Close'].iloc[-1]
                current_date = pd.to_datetime(data['Date'].iloc[-1]).timestamp()
                support_level = hl_trendline['slope'] * current_date + hl_trendline['intercept']
                
                # Check for breakout above support (continuation)
                if current_price > support_level * 1.002:  # 0.2% above support
                    signals['breakout_detected'] = True
                    signals['breakout_level'] = support_level
                    signals['pattern_strength'] += 20
                    
                    print(f"   🚀 Support breakout detected at {support_level:.5f}")
        
        # Calculate overall pattern strength
        if signals['pattern_strength'] > 0:
            signals['pattern_strength'] = min(signals['pattern_strength'], 100)
        
        return signals
    
    def detect_continuation_patterns(self, data):
        """
        Detect continuation patterns: bullish flag, pennant, inverted head and shoulders
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            dict: Pattern detection results
        """
        patterns = {
            'bullish_flag': False,
            'pennant': False,
            'inverted_head_shoulders': False,
            'pattern_details': {}
        }
        
        if len(data) < 20:
            return patterns
        
        # Detect bullish flag pattern
        flag_pattern = self.detect_bullish_flag(data)
        if flag_pattern:
            patterns['bullish_flag'] = True
            patterns['pattern_details']['bullish_flag'] = flag_pattern
        
        # Detect pennant pattern
        pennant_pattern = self.detect_pennant(data)
        if pennant_pattern:
            patterns['pennant'] = True
            patterns['pattern_details']['pennant'] = pennant_pattern
        
        # Detect inverted head and shoulders
        ihs_pattern = self.detect_inverted_head_shoulders(data)
        if ihs_pattern:
            patterns['inverted_head_shoulders'] = True
            patterns['pattern_details']['inverted_head_shoulders'] = ihs_pattern
        
        return patterns
    
    def detect_bullish_flag(self, data):
        """
        Detect bullish flag pattern
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            dict: Flag pattern details or None
        """
        if len(data) < 15:
            return None
        
        # Look for strong upward move followed by consolidation
        recent_data = data.tail(15)
        
        # Check for strong upward move in first 5 bars
        first_5 = recent_data.head(5)
        price_change = (first_5['Close'].iloc[-1] - first_5['Open'].iloc[0]) / first_5['Open'].iloc[0]
        
        if price_change < 0.02:  # Less than 2% move
            return None
        
        # Check for consolidation in remaining bars
        last_10 = recent_data.tail(10)
        high_range = last_10['High'].max() - last_10['Low'].min()
        avg_price = last_10['Close'].mean()
        range_ratio = high_range / avg_price
        
        if range_ratio < 0.03:  # Tight consolidation
            return {
                'start_date': first_5['Date'].iloc[0],
                'end_date': last_10['Date'].iloc[-1],
                'flag_pole_height': price_change,
                'consolidation_range': range_ratio
            }
        
        return None
    
    def detect_pennant(self, data):
        """
        Detect pennant pattern (similar to flag but with converging trendlines)
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            dict: Pennant pattern details or None
        """
        if len(data) < 15:
            return None
        
        # Similar to flag but with converging highs and lows
        recent_data = data.tail(15)
        
        # Check for strong upward move
        first_5 = recent_data.head(5)
        price_change = (first_5['Close'].iloc[-1] - first_5['Open'].iloc[0]) / first_5['Open'].iloc[0]
        
        if price_change < 0.02:
            return None
        
        # Check for converging trendlines in consolidation
        last_10 = recent_data.tail(10)
        
        # Calculate trendlines for highs and lows
        highs = last_10[['Date', 'High']].copy()
        lows = last_10[['Date', 'Low']].copy()
        
        # Simple linear regression for trendlines
        high_slope = np.polyfit(range(len(highs)), highs['High'], 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows['Low'], 1)[0]
        
        # Check if trendlines are converging (high slope negative, low slope positive)
        if high_slope < -0.0001 and low_slope > 0.0001:
            return {
                'start_date': first_5['Date'].iloc[0],
                'end_date': last_10['Date'].iloc[-1],
                'high_slope': high_slope,
                'low_slope': low_slope
            }
        
        return None
    
    def detect_inverted_head_shoulders(self, data):
        """
        Detect inverted head and shoulders pattern
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            dict: IHS pattern details or None
        """
        if len(data) < 25:
            return None
        
        # Find pivot lows
        pivot_points = self.find_pivot_points_advanced(data)
        lows = pivot_points[pivot_points['type'] == 'low'].copy()
        
        if len(lows) < 5:
            return None
        
        # Look for IHS pattern in last 5-7 pivot lows
        recent_lows = lows.tail(7)
        
        if len(recent_lows) < 5:
            return None
        
        # Check for IHS pattern: left shoulder, head (lower), right shoulder
        prices = recent_lows['price'].values
        
        # Find the lowest point (head)
        head_index = np.argmin(prices)
        
        if head_index == 0 or head_index == len(prices) - 1:
            return None  # Head should be in the middle
        
        # Check if left shoulder is higher than head
        left_shoulder = prices[head_index - 1]
        head = prices[head_index]
        right_shoulder = prices[head_index + 1]
        
        if left_shoulder > head and right_shoulder > head:
            # Check if shoulders are roughly at same level
            shoulder_diff = abs(left_shoulder - right_shoulder) / head
            if shoulder_diff < 0.02:  # Shoulders within 2% of each other
                return {
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': (left_shoulder + right_shoulder) / 2,
                    'pattern_height': (left_shoulder + right_shoulder) / 2 - head
                }
        
        return None
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculate position size based on risk management rules
        
        Args:
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            
        Returns:
            dict: Position sizing information
        """
        # Calculate risk amount
        risk_amount = self.account_size * self.risk_per_trade
        
        # Calculate price risk per share/unit
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return None
        
        # Calculate position size
        position_size = risk_amount / price_risk
        
        # Calculate target price based on 1:3 risk-to-reward ratio
        price_reward = price_risk * self.risk_reward_ratio
        target_price = entry_price + price_reward if entry_price > stop_loss_price else entry_price - price_reward
        
        # Calculate potential profit
        potential_profit = position_size * price_reward
        
        # Calculate position value
        position_value = position_size * entry_price
        
        return {
            'position_size': position_size,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'target_price': target_price,
            'risk_amount': risk_amount,
            'potential_profit': potential_profit,
            'position_value': position_value,
            'risk_reward_ratio': self.risk_reward_ratio
        }
    
    def get_day_trading_signals(self, trend_analysis):
        """
        Generate day trading specific signals with money management
        
        Args:
            trend_analysis (dict): Result from identify_higher_highs_lows
            
        Returns:
            dict: Day trading signals with position sizing
        """
        if not trend_analysis or not trend_analysis['uptrend_confirmed']:
            return None
        
        # Get current price (last close price)
        current_price = self.data['Close'].iloc[-1]
        
        # Find recent higher low for stop loss
        recent_hl = None
        if trend_analysis['higher_lows']:
            recent_hl = trend_analysis['higher_lows'][-1]['price']
        
        # Find recent higher high for target
        recent_hh = None
        if trend_analysis['higher_highs']:
            recent_hh = trend_analysis['higher_highs'][-1]['price']
        
        # Calculate entry, stop loss, and target
        entry_price = current_price
        stop_loss_price = recent_hl if recent_hl else current_price * 0.98  # 2% below current if no HL
        target_price = recent_hh if recent_hh else current_price * 1.06  # 6% above current if no HH
        
        # Calculate position sizing
        position_info = self.calculate_position_size(entry_price, stop_loss_price)
        
        if not position_info:
            return None
        
        # Day trading specific recommendations based on timeframe
        timeframe_rules = {
            '1m': {
                'max_holding_time': '15-30 minutes',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            },
            '5m': {
                'max_holding_time': '1-2 hours',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            },
            '15m': {
                'max_holding_time': '2-4 hours',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            },
            '30m': {
                'max_holding_time': '4-6 hours',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            },
            '1h': {
                'max_holding_time': '6-8 hours',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            },
            '4h': {
                'max_holding_time': '1-2 days',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            },
            '1d': {
                'max_holding_time': '2-5 days',
                'trailing_stop': 'Move stop loss to breakeven after 1:1 R:R',
                'partial_profit': 'Take 50% profit at 1:2 R:R',
                'risk_management': f'Risk {self.risk_per_trade*100}% of account per trade'
            }
        }
        
        signals = {
            'signal_type': 'DAY_TRADING_LONG',
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'target_price': target_price,
            'position_sizing': position_info,
            'timeframe': self.period,
            'day_trading_rules': timeframe_rules.get(self.period, timeframe_rules['1h'])
        }
        
        return signals
    
    def debug_data_structure(self):
        """Debug method to check data structure"""
        if self.data is None:
            print("❌ Data is None")
            return
        
        print(f"\n🔍 TRADING BOT DATA STRUCTURE DEBUG:")
        print(f"   Shape: {self.data.shape}")
        print(f"   Columns: {list(self.data.columns)}")
        print(f"   Data types:")
        for col in self.data.columns:
            print(f"     {col}: {self.data[col].dtype}")
        
        if len(self.data) > 0:
            print(f"   First row:")
            print(f"     {self.data.iloc[0].to_dict()}")
            print(f"   Last row:")
            print(f"     {self.data.iloc[-1].to_dict()}")
        
        # Check for required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            print(f"   ❌ Missing columns: {missing_columns}")
        else:
            print(f"   ✅ All required columns present")
    
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance for day trading timeframes"""
        try:
            full_symbol = self.get_full_symbol()
            ticker = yf.Ticker(full_symbol)
            
            # Get appropriate data period for the timeframe
            data_period = self.get_data_period_for_timeframe()
            
            # For day trading, use the timeframe as interval
            interval = self.period
            
            # Try to use the actual minute intervals for better scalping analysis
            if self.period in ['1m', '5m', '15m', '30m']:
                # First try with the actual minute interval
                try:
                    self.data = ticker.history(period=data_period, interval=self.period)
                    if not self.data.empty and len(self.data) > 50:  # Ensure we have enough data
                        print(f"✅ Successfully fetched {self.period} interval data")
                        # Skip the second fetch since we already have data
                        if self.data.empty:
                            print(f"No data available for {full_symbol} ({self.market_type}) with period={data_period}, interval={self.period}")
                            return False
                    else:
                        # Fallback to daily data if minute data is insufficient
                        interval = "1d"
                        print(f"⚠️  {self.period} data insufficient. Using daily data for analysis.")
                        self.data = ticker.history(period=data_period, interval=interval)
                except Exception as e:
                    # Fallback to daily data if minute data fails
                    interval = "1d"
                    print(f"⚠️  {self.period} data unavailable. Using daily data for analysis.")
                    self.data = ticker.history(period=data_period, interval=interval)
            else:
                interval = self.period
                self.data = ticker.history(period=data_period, interval=interval)
            
            if self.data.empty:
                print(f"No data available for {full_symbol} ({self.market_type}) with period={data_period}, interval={interval}")
                print(f"Try using a different timeframe or check if the symbol is correct")
                return False
            
            self.data.reset_index(inplace=True)
            
            # Debug: Show original column names
            print(f"Original columns from yfinance: {list(self.data.columns)}")
            
            # Standardize column names to ensure consistency
            column_mapping = {
                'date': 'Date',
                'datetime': 'Date',
                'Datetime': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in self.data.columns:
                    self.data.rename(columns={old_col: new_col}, inplace=True)
                    print(f"Renamed column '{old_col}' to '{new_col}'")
            
            # Show columns after renaming
            print(f"Columns after renaming: {list(self.data.columns)}")
            
            # Ensure we have all required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in self.data.columns:
                    if col == 'Volume':
                        # If volume is missing, add a default volume
                        self.data['Volume'] = 1000
                    else:
                        print(f"❌ Missing required column: {col}")
                        print(f"Available columns: {list(self.data.columns)}")
                        return False
            
            print(f"Successfully fetched {len(self.data)} data points for {full_symbol} ({self.market_type}) on {self.period} timeframe")
            print(f"Columns: {list(self.data.columns)}")
            
            # Debug data structure
            self.debug_data_structure()
            
            return True
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            print(f"Try using a different symbol or timeframe")
            return False
    
    def find_pivot_points(self, window=5):
        """
        Find pivot points (highs and lows) in the price data
        
        Args:
            window (int): Window size for pivot detection
        """
        if self.data is None or len(self.data) == 0:
            print("No data available. Please fetch data first.")
            return
        
        if len(self.data) < (2 * window + 1):
            print(f"Not enough data points for pivot detection. Need at least {2 * window + 1} points, got {len(self.data)}")
            return
        
        highs = []
        lows = []
        
        for i in range(window, len(self.data) - window):
            # Check for pivot high
            if all(self.data['High'].iloc[i] >= self.data['High'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.data['High'].iloc[i] >= self.data['High'].iloc[i+j] for j in range(1, window+1)):
                highs.append({
                    'index': i,
                    'date': self.data['Date'].iloc[i],
                    'price': self.data['High'].iloc[i],
                    'type': 'high'
                })
            
            # Check for pivot low
            if all(self.data['Low'].iloc[i] <= self.data['Low'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.data['Low'].iloc[i] <= self.data['Low'].iloc[i+j] for j in range(1, window+1)):
                lows.append({
                    'index': i,
                    'date': self.data['Date'].iloc[i],
                    'price': self.data['Low'].iloc[i],
                    'type': 'low'
                })
        
        # Only create DataFrame if we have pivot points
        if highs or lows:
            self.pivot_points = pd.DataFrame(highs + lows).sort_values('date').reset_index(drop=True)
            print(f"Found {len(highs)} pivot highs and {len(lows)} pivot lows on {self.period} timeframe")
        else:
            self.pivot_points = pd.DataFrame()
            print("No pivot points found in the data")
        
        return self.pivot_points
    
    def identify_higher_highs_lows(self, min_points=2):
        """
        Identify higher highs (HH) and higher lows (HL) patterns
        
        Args:
            min_points (int): Minimum number of HH/HL points to confirm uptrend
        """
        if self.pivot_points is None or len(self.pivot_points) == 0:
            print("No pivot points available. Cannot analyze trends.")
            return None
        
        if len(self.pivot_points) < 4:
            print(f"Not enough pivot points to analyze trends. Need at least 4, got {len(self.pivot_points)}")
            return None
        
        # Separate highs and lows
        highs = self.pivot_points[self.pivot_points['type'] == 'high'].copy()
        lows = self.pivot_points[self.pivot_points['type'] == 'low'].copy()
        
        # Find higher highs
        higher_highs = []
        for i in range(1, len(highs)):
            if highs['price'].iloc[i] > highs['price'].iloc[i-1]:
                higher_highs.append({
                    'date': highs['date'].iloc[i],
                    'price': highs['price'].iloc[i],
                    'type': 'HH'
                })
        
        # Find higher lows
        higher_lows = []
        for i in range(1, len(lows)):
            if lows['price'].iloc[i] > lows['price'].iloc[i-1]:
                higher_lows.append({
                    'date': lows['date'].iloc[i],
                    'price': lows['price'].iloc[i],
                    'type': 'HL'
                })
        
        # Check if we have enough HH and HL points to confirm uptrend
        uptrend_confirmed = len(higher_highs) >= min_points and len(higher_lows) >= min_points
        
        result = {
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'uptrend_confirmed': uptrend_confirmed,
            'hh_count': len(higher_highs),
            'hl_count': len(higher_lows),
            'timeframe': self.period
        }
        
        return result
    
    def create_trendlines(self, trend_analysis):
        """
        Create trendlines connecting pivot points
        
        Args:
            trend_analysis (dict): Result from identify_higher_highs_lows
        """
        if trend_analysis is None or not trend_analysis['uptrend_confirmed']:
            print("No confirmed uptrend to create trendlines")
            return
        
        # Create trendline for higher highs
        hh_trendline = None
        if len(trend_analysis['higher_highs']) >= 2:
            hh_points = trend_analysis['higher_highs'][-2:]  # Use last 2 HH points
            hh_trendline = {
                'start_date': hh_points[0]['date'],
                'end_date': hh_points[1]['date'],
                'start_price': hh_points[0]['price'],
                'end_price': hh_points[1]['price'],
                'type': 'HH_trendline'
            }
        
        # Create trendline for higher lows
        hl_trendline = None
        if len(trend_analysis['higher_lows']) >= 2:
            hl_points = trend_analysis['higher_lows'][-2:]  # Use last 2 HL points
            hl_trendline = {
                'start_date': hl_points[0]['date'],
                'end_date': hl_points[1]['date'],
                'start_price': hl_points[0]['price'],
                'end_price': hl_points[1]['price'],
                'type': 'HL_trendline'
            }
        
        self.trendlines = {
            'hh_trendline': hh_trendline,
            'hl_trendline': hl_trendline
        }
        
        return self.trendlines
    
    def analyze_uptrend_advanced(self):
        """
        Advanced uptrend analysis following the specific trading rules:
        
        UPTREND DEFINITION:
        - Market moves in higher highs (HH) and higher lows (HL)
        - Market respects the lows and breaks the highs
        - Need at least 2 HH and 2 HL to confirm uptrend
        
        MULTI-TIMEFRAME APPROACH:
        - Weekly chart or Daily for trend identification
        - H4 for entry signals
        
        TRENDLINE ANALYSIS:
        - Join pivot points to create diagonal support/resistance
        - Takes 3-4 touches to confirm strong support/resistance
        - Every breakout has a retest
        - Wait for retest to confirm previous resistance becomes new support
        
        CONTINUATION PATTERNS:
        - Bullish flag, pennant, inverted head and shoulders
        - Used to catch the wave continuation
        """
        print(f"\n🔍 ADVANCED UPTREND ANALYSIS FOR {self.symbol}")
        print("=" * 70)
        print("📋 FOLLOWING SPECIFIC TRADING RULES:")
        print("   • Multi-timeframe: Weekly/Daily for trend, H4 for entries")
        print("   • Minimum 2 HH and 2 HL to confirm uptrend")
        print("   • Trendline support with 3-4 touches validation")
        print("   • Breakout + retest confirmation")
        print("   • Continuation patterns for wave catching")
        print("=" * 70)
        
        # Step 1: Multi-timeframe data collection
        print("\n📊 STEP 1: Multi-timeframe Data Collection")
        if not self.fetch_multi_timeframe_data():
            print("❌ Failed to fetch multi-timeframe data")
            return None
        
        # Step 2: Weekly/Daily trend confirmation (Primary trend identification)
        print("\n📈 STEP 2: Primary Trend Confirmation (Weekly/Daily)")
        
        # Try Weekly first, fallback to Daily
        primary_timeframe = "Weekly"
        primary_data = self.weekly_data
        primary_analysis = self.identify_higher_highs_lows_advanced(self.weekly_data, min_points=2)
        
        if not primary_analysis or not primary_analysis['uptrend_confirmed']:
            print("⚠️  Weekly timeframe does not confirm uptrend, checking Daily...")
            primary_timeframe = "Daily"
            primary_data = self.daily_data
            primary_analysis = self.identify_higher_highs_lows_advanced(self.daily_data, min_points=2)
            
            if not primary_analysis or not primary_analysis['uptrend_confirmed']:
                print("❌ Neither Weekly nor Daily timeframe confirms uptrend")
                print(f"   Weekly HH: {self.identify_higher_highs_lows_advanced(self.weekly_data, min_points=2)['hh_count'] if self.weekly_data is not None else 0}")
                print(f"   Weekly HL: {self.identify_higher_highs_lows_advanced(self.weekly_data, min_points=2)['hl_count'] if self.weekly_data is not None else 0}")
                print(f"   Daily HH: {primary_analysis['hh_count'] if primary_analysis else 0}")
                print(f"   Daily HL: {primary_analysis['hl_count'] if primary_analysis else 0}")
                return None
        
        print(f"✅ {primary_timeframe} uptrend confirmed")
        print(f"   Higher Highs: {primary_analysis['hh_count']}")
        print(f"   Higher Lows: {primary_analysis['hl_count']}")
        print(f"   Trend Strength: {primary_analysis['trend_strength']:.1f}/100")
        
        # Step 3: H4 entry analysis (Entry timeframe)
        print("\n📈 STEP 3: H4 Entry Analysis")
        h4_analysis = self.identify_higher_highs_lows_advanced(self.h4_data, min_points=2)
        if not h4_analysis:
            print("❌ Insufficient H4 data for analysis")
            return None
        
        print(f"   H4 Higher Highs: {h4_analysis['hh_count']}")
        print(f"   H4 Higher Lows: {h4_analysis['hl_count']}")
        print(f"   H4 Trend Strength: {h4_analysis['trend_strength']:.1f}/100")
        
        # Step 4: Advanced trendline analysis with 3-4 touch validation
        print("\n📏 STEP 4: Advanced Trendline Analysis (3-4 Touch Validation)")
        
        # Create trendlines for primary timeframe and H4
        primary_trendlines = self.create_trendlines_advanced(primary_analysis)
        h4_trendlines = self.create_trendlines_advanced(h4_analysis)
        
        # Validate trendline strength (3-4 touches minimum)
        strong_trendlines = {}
        
        if primary_trendlines:
            print(f"✅ {primary_timeframe} trendlines created:")
            for trendline_type, trendline in primary_trendlines.items():
                touches = trendline['touches']
                if touches >= 3:
                    strong_trendlines[f"{primary_timeframe}_{trendline_type}"] = trendline
                    print(f"   ✅ {trendline_type}: {touches} touches (STRONG)")
                else:
                    print(f"   ⚠️  {trendline_type}: {touches} touches (WEAK - need 3+)")
        
        if h4_trendlines:
            print(f"✅ H4 trendlines created:")
            for trendline_type, trendline in h4_trendlines.items():
                touches = trendline['touches']
                if touches >= 3:
                    strong_trendlines[f"H4_{trendline_type}"] = trendline
                    print(f"   ✅ {trendline_type}: {touches} touches (STRONG)")
                else:
                    print(f"   ⚠️  {trendline_type}: {touches} touches (WEAK - need 3+)")
        
        # Step 5: Breakout and retest detection (Every breakout has a retest)
        print("\n🎯 STEP 5: Breakout and Retest Detection")
        print("   Rule: Every breakout has a retest - wait for retest confirmation")
        
        breakout_signals = self.detect_breakout_retest(self.h4_data, h4_analysis, h4_trendlines)
        
        if breakout_signals and breakout_signals['entry_signal']:
            print("✅ Breakout and retest pattern detected!")
            print(f"   Entry Price: {breakout_signals['entry_price']:.5f}")
            print(f"   Stop Loss: {breakout_signals['stop_loss']:.5f}")
            print(f"   Target: {breakout_signals['target']:.5f}")
            print(f"   Retest Confirmed: {breakout_signals.get('retest_confirmed', False)}")
        else:
            print("ℹ️  No breakout and retest pattern detected")
            print("   Waiting for: Breakout → Retest → Confirmation")
        
        # Step 6: Continuation pattern detection (Wave catching)
        print("\n🔍 STEP 6: Continuation Pattern Detection (Wave Catching)")
        continuation_patterns = self.detect_continuation_patterns(self.h4_data)
        
        pattern_count = sum([
            continuation_patterns['bullish_flag'],
            continuation_patterns['pennant'],
            continuation_patterns['inverted_head_shoulders']
        ])
        
        if pattern_count > 0:
            print(f"✅ {pattern_count} continuation pattern(s) detected for wave catching:")
            if continuation_patterns['bullish_flag']:
                print("   🚩 Bullish Flag - Continuation pattern")
            if continuation_patterns['pennant']:
                print("   🚩 Pennant - Continuation pattern")
            if continuation_patterns['inverted_head_shoulders']:
                print("   🚩 Inverted Head and Shoulders - Reversal to continuation")
        else:
            print("ℹ️  No continuation patterns detected")
            print("   Waiting for: Bullish Flag, Pennant, or Inverted H&S")
        
        # Step 7: Final analysis summary with strict rules
        print("\n📋 STEP 7: Final Analysis Summary")
        
        # Calculate overall trend strength
        overall_strength = (primary_analysis['trend_strength'] + h4_analysis['trend_strength']) / 2
        
        # Strict uptrend confirmation rules
        uptrend_confirmed = (
            primary_analysis['uptrend_confirmed'] and  # Weekly/Daily confirms trend
            primary_analysis['hh_count'] >= 2 and      # At least 2 HH
            primary_analysis['hl_count'] >= 2 and      # At least 2 HL
            h4_analysis['uptrend_confirmed'] and       # H4 confirms for entries
            overall_strength > 60                      # Strong trend strength
        )
        
        # Entry conditions based on specific rules
        entry_conditions = []
        
        # Breakout and retest condition
        if breakout_signals and breakout_signals['entry_signal']:
            if breakout_signals.get('retest_confirmed', False):
                entry_conditions.append("Breakout + Retest Confirmed")
            else:
                entry_conditions.append("Breakout Detected (Waiting for Retest)")
        
        # Continuation pattern condition
        if pattern_count > 0:
            entry_conditions.append("Continuation Pattern Detected")
        
        # Strong trendline support condition
        if len(strong_trendlines) >= 2:  # At least 2 strong trendlines
            entry_conditions.append("Strong Trendline Support")
        
        final_analysis = {
            'uptrend_confirmed': uptrend_confirmed,
            'overall_strength': overall_strength,
            'primary_timeframe': primary_timeframe,
            'primary_analysis': primary_analysis,
            'h4_analysis': h4_analysis,
            'strong_trendlines': strong_trendlines,
            'breakout_signals': breakout_signals,
            'continuation_patterns': continuation_patterns,
            'entry_conditions': entry_conditions,
            'timeframe': self.period,
            'trading_rules_followed': {
                'multi_timeframe_confirmed': primary_analysis['uptrend_confirmed'] and h4_analysis['uptrend_confirmed'],
                'min_hh_hl_met': primary_analysis['hh_count'] >= 2 and primary_analysis['hl_count'] >= 2,
                'strong_trendlines': len(strong_trendlines) >= 2,
                'breakout_retest_ready': breakout_signals and breakout_signals.get('retest_confirmed', False),
                'continuation_patterns_ready': pattern_count > 0
            }
        }
        
        if uptrend_confirmed:
            print(f"✅ STRONG UPTREND CONFIRMED for {self.symbol}")
            print(f"   Overall Strength: {overall_strength:.1f}/100")
            print(f"   Primary Timeframe: {primary_timeframe}")
            print(f"   Entry Conditions: {', '.join(entry_conditions) if entry_conditions else 'None'}")
            
            if entry_conditions:
                print(f"   🎯 READY FOR ENTRY SIGNALS")
                print(f"   📊 Trading Rules Compliance:")
                for rule, status in final_analysis['trading_rules_followed'].items():
                    print(f"      {'✅' if status else '❌'} {rule.replace('_', ' ').title()}")
            else:
                print(f"   ⏳ Waiting for entry conditions")
        else:
            print(f"❌ Uptrend not strong enough for {self.symbol}")
            print(f"   Overall Strength: {overall_strength:.1f}/100")
            print(f"   Missing Requirements:")
            if not primary_analysis['uptrend_confirmed']:
                print(f"      ❌ {primary_timeframe} uptrend not confirmed")
            if primary_analysis['hh_count'] < 2:
                print(f"      ❌ Insufficient Higher Highs ({primary_analysis['hh_count']}/2)")
            if primary_analysis['hl_count'] < 2:
                print(f"      ❌ Insufficient Higher Lows ({primary_analysis['hl_count']}/2)")
            if overall_strength <= 60:
                print(f"      ❌ Trend strength too weak ({overall_strength:.1f}/60)")
        
        return final_analysis
    
    def analyze_uptrend(self):
        """
        Complete uptrend analysis including pivot points, HH/HL detection, and trendlines
        (Legacy method - use analyze_uptrend_advanced for new rules)
        """
        if not self.fetch_data():
            print(f"Failed to fetch data for {self.symbol}. Cannot perform analysis.")
            return None
        
        # Check if we have enough data
        if len(self.data) == 0:
            print(f"No data available for {self.symbol}. Cannot perform analysis.")
            return None
        
        if len(self.data) < 10:
            print(f"Insufficient data for {self.symbol}. Need at least 10 data points, got {len(self.data)}")
            return None
        
        self.find_pivot_points()
        trend_analysis = self.identify_higher_highs_lows()
        
        if trend_analysis:
            if trend_analysis['uptrend_confirmed']:
                self.create_trendlines(trend_analysis)
                print(f"✅ UPTREND CONFIRMED for {self.symbol} ({self.market_type}) on {self.period} timeframe")
                print(f"   Higher Highs: {trend_analysis['hh_count']}")
                print(f"   Higher Lows: {trend_analysis['hl_count']}")
            else:
                print(f"❌ No uptrend confirmed for {self.symbol} ({self.market_type}) on {self.period} timeframe")
                print(f"   Higher Highs: {trend_analysis['hh_count']}")
                print(f"   Higher Lows: {trend_analysis['hl_count']}")
        else:
            print(f"❌ Cannot analyze trends for {self.symbol} ({self.market_type}) on {self.period} timeframe - insufficient data")
        
        return trend_analysis
    
    def plot_analysis(self, trend_analysis):
        """
        Create an interactive plot showing the analysis
        
        Args:
            trend_analysis (dict): Result from identify_higher_highs_lows
        """
        if self.data is None or trend_analysis is None:
            print("No data or analysis to plot")
            return
        
        # Create subplot
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'{self.symbol} ({self.market_type}) - {self.period} Timeframe Analysis'])
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=self.data['Date'],
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add pivot points
        if self.pivot_points is not None:
            highs = self.pivot_points[self.pivot_points['type'] == 'high']
            lows = self.pivot_points[self.pivot_points['type'] == 'low']
            
            fig.add_trace(go.Scatter(
                x=highs['date'],
                y=highs['price'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-down'),
                name='Pivot Highs'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=lows['date'],
                y=lows['price'],
                mode='markers',
                marker=dict(color='green', size=8, symbol='triangle-up'),
                name='Pivot Lows'
            ), row=1, col=1)
        
        # Add higher highs and lows
        if trend_analysis['higher_highs']:
            hh_dates = [hh['date'] for hh in trend_analysis['higher_highs']]
            hh_prices = [hh['price'] for hh in trend_analysis['higher_highs']]
            
            fig.add_trace(go.Scatter(
                x=hh_dates,
                y=hh_prices,
                mode='markers+lines',
                marker=dict(color='darkred', size=10, symbol='diamond'),
                line=dict(color='darkred', width=2),
                name='Higher Highs (HH)'
            ), row=1, col=1)
        
        if trend_analysis['higher_lows']:
            hl_dates = [hl['date'] for hl in trend_analysis['higher_lows']]
            hl_prices = [hl['price'] for hl in trend_analysis['higher_lows']]
            
            fig.add_trace(go.Scatter(
                x=hl_dates,
                y=hl_prices,
                mode='markers+lines',
                marker=dict(color='darkgreen', size=10, symbol='diamond'),
                line=dict(color='darkgreen', width=2),
                name='Higher Lows (HL)'
            ), row=1, col=1)
        
        # Add trendlines
        if self.trendlines:
            if self.trendlines['hh_trendline']:
                tl = self.trendlines['hh_trendline']
                fig.add_trace(go.Scatter(
                    x=[tl['start_date'], tl['end_date']],
                    y=[tl['start_price'], tl['end_price']],
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    name='HH Trendline'
                ), row=1, col=1)
            
            if self.trendlines['hl_trendline']:
                tl = self.trendlines['hl_trendline']
                fig.add_trace(go.Scatter(
                    x=[tl['start_date'], tl['end_date']],
                    y=[tl['start_price'], tl['end_price']],
                    mode='lines',
                    line=dict(color='green', width=3, dash='dash'),
                    name='HL Trendline'
                ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} ({self.market_type}) - {self.period} Timeframe Analysis - {"✅ UPTREND CONFIRMED" if trend_analysis["uptrend_confirmed"] else "❌ No Uptrend"}',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            showlegend=True
        )
        
        fig.show()
        
    def generate_report(self, trend_analysis):
        """
        Generate a detailed analysis report
        
        Args:
            trend_analysis (dict): Result from identify_higher_highs_lows
        """
        if trend_analysis is None:
            print("No analysis to report")
            return
        
        print("\n" + "="*60)
        print(f"DAY TRADING ANALYSIS REPORT - {self.symbol} ({self.market_type.upper()})")
        print("="*60)
        
        print(f"Timeframe: {self.period}")
        print(f"Analysis Period: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Total Data Points: {len(self.data)}")
        print(f"Pivot Points Found: {len(self.pivot_points) if self.pivot_points is not None else 0}")
        
        print(f"\nUPTREND ANALYSIS:")
        print(f"  Higher Highs (HH): {trend_analysis['hh_count']}")
        print(f"  Higher Lows (HL): {trend_analysis['hl_count']}")
        print(f"  Uptrend Confirmed: {'✅ YES' if trend_analysis['uptrend_confirmed'] else '❌ NO'}")
        
        if trend_analysis['uptrend_confirmed']:
            print(f"\n✅ TRADING SIGNAL: UPTREND DETECTED on {self.period} timeframe")
            print(f"   The {self.market_type} market is showing a confirmed uptrend with:")
            print(f"   - At least 2 higher highs (HH)")
            print(f"   - At least 2 higher lows (HL)")
            print(f"   - Trendlines connecting pivot points")
            
            if self.trendlines:
                if self.trendlines['hh_trendline']:
                    tl = self.trendlines['hh_trendline']
                    print(f"\n   HH Trendline: {tl['start_date'].strftime('%Y-%m-%d')} ({tl['start_price']:.2f}) → {tl['end_date'].strftime('%Y-%m-%d')} ({tl['end_price']:.2f})")
                
                if self.trendlines['hl_trendline']:
                    tl = self.trendlines['hl_trendline']
                    print(f"   HL Trendline: {tl['start_date'].strftime('%Y-%m-%d')} ({tl['start_price']:.2f}) → {tl['end_date'].strftime('%Y-%m-%d')} ({tl['end_price']:.2f})")
        else:
            print(f"\n❌ NO UPTREND DETECTED on {self.period} timeframe")
            print(f"   Insufficient higher highs and/or higher lows to confirm uptrend")
        
        print("="*60)
    
    def generate_day_trading_report(self, trend_analysis):
        """
        Generate day trading specific report with money management
        
        Args:
            trend_analysis (dict): Result from identify_higher_highs_lows
        """
        if trend_analysis is None:
            print("No analysis to report")
            return
        
        # Get day trading signals
        signals = self.get_day_trading_signals(trend_analysis)
        
        print("\n" + "="*60)
        print(f"DAY TRADING REPORT - {self.symbol} ({self.market_type.upper()}) - {self.period} TIMEFRAME")
        print("="*60)
        
        print(f"Account Size: ${self.account_size:,.2f}")
        print(f"Risk Per Trade: {self.risk_per_trade*100}% (${self.account_size * self.risk_per_trade:,.2f})")
        print(f"Risk-to-Reward Ratio: 1:{self.risk_reward_ratio}")
        print(f"Timeframe: {self.period}")
        
        if signals:
            pos = signals['position_sizing']
            print(f"\n🎯 DAY TRADING SIGNAL:")
            print(f"   Signal Type: {signals['signal_type']}")
            print(f"   Entry Price: ${pos['entry_price']:.2f}")
            print(f"   Stop Loss: ${pos['stop_loss_price']:.2f}")
            print(f"   Target Price: ${pos['target_price']:.2f}")
            print(f"   Position Size: {pos['position_size']:.2f} units")
            print(f"   Position Value: ${pos['position_value']:,.2f}")
            print(f"   Risk Amount: ${pos['risk_amount']:,.2f}")
            print(f"   Potential Profit: ${pos['potential_profit']:,.2f}")
            
            print(f"\n📋 DAY TRADING RULES for {self.period} timeframe:")
            for rule, description in signals['day_trading_rules'].items():
                print(f"   {rule.replace('_', ' ').title()}: {description}")
        else:
            print(f"\n❌ NO DAY TRADING SIGNAL")
            print(f"   Insufficient data for position sizing")
        
        print("="*60)
    
    def get_available_symbols(self):
        """Get list of available symbols for each market type"""
        return self.symbol_mappings
    
    def get_available_timeframes(self):
        """Get list of available timeframes for day trading"""
        return self.valid_timeframes 