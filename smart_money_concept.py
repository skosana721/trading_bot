#!/usr/bin/env python3
"""
Smart Money Concept (SMC) Trading Strategy
==========================================

This module implements Smart Money Concept trading strategies including:
- Order Blocks (OB) identification and analysis
- Fair Value Gaps (FVG) detection
- Liquidity Zones and Institutional Order Flow
- Market Structure (Higher Highs/Lows, Break of Structure)
- Premium/Discount Zones
- Institutional Order Blocks (IOB)
- Mitigation and Retest Patterns

Combined with existing technical analysis for enhanced trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SmartMoneyConcept:
    def __init__(self, data, timeframe='5m'):
        """
        Initialize Smart Money Concept analyzer
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            timeframe (str): Timeframe for analysis
        """
        self.data = data.copy()
        self.timeframe = timeframe
        self.order_blocks = []
        self.fair_value_gaps = []
        self.liquidity_zones = []
        self.market_structure = {}
        self.premium_discount_zones = []
        self.institutional_order_blocks = []
        
        # SMC Parameters
        self.ob_lookback = 20  # Lookback for order block identification
        self.fvg_threshold = 0.0005  # Minimum gap size for FVG
        self.liquidity_threshold = 0.001  # Minimum swing for liquidity zones
        
        # Initialize analysis
        self.analyze_smc_components()
    
    def analyze_smc_components(self):
        """Analyze all SMC components"""
        print("🔍 Analyzing Smart Money Concept Components...")
        
        # 1. Market Structure Analysis
        self.analyze_market_structure()
        
        # 2. Order Blocks
        self.identify_order_blocks()
        
        # 3. Fair Value Gaps
        self.identify_fair_value_gaps()
        
        # 4. Liquidity Zones
        self.identify_liquidity_zones()
        
        # 5. Premium/Discount Zones
        self.identify_premium_discount_zones()
        
        # 6. Institutional Order Blocks
        self.identify_institutional_order_blocks()
        
        print("✅ SMC Analysis Complete")
    
    def analyze_market_structure(self):
        """Analyze market structure (Higher Highs, Higher Lows, Break of Structure)"""
        print("📊 Analyzing Market Structure...")
        
        # Find swing highs and lows
        swing_highs = self.find_swing_points('high', window=5)
        swing_lows = self.find_swing_points('low', window=5)
        
        # Identify Higher Highs and Higher Lows
        higher_highs = self.identify_higher_highs(swing_highs)
        higher_lows = self.identify_higher_lows(swing_lows)
        
        # Identify Lower Highs and Lower Lows
        lower_highs = self.identify_lower_highs(swing_highs)
        lower_lows = self.identify_lower_lows(swing_lows)
        
        # Detect Break of Structure (BOS)
        bos_points = self.detect_break_of_structure(higher_highs, higher_lows, lower_highs, lower_lows)
        
        # Detect Change of Character (CHoCH)
        choch_points = self.detect_change_of_character(higher_highs, higher_lows, lower_highs, lower_lows)
        
        self.market_structure = {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'lower_highs': lower_highs,
            'lower_lows': lower_lows,
            'break_of_structure': bos_points,
            'change_of_character': choch_points,
            'trend_direction': self.determine_trend_direction(higher_highs, higher_lows, lower_highs, lower_lows)
        }
    
    def find_swing_points(self, point_type, window=5):
        """Find swing highs or lows"""
        points = []
        
        for i in range(window, len(self.data) - window):
            if point_type == 'high':
                current_high = float(self.data['High'].iloc[i])
                window_highs = self.data['High'].iloc[i-window:i+window+1]
                max_high = float(window_highs.max())
                if abs(current_high - max_high) < 0.00001:  # Use small tolerance for float comparison
                    points.append({
                        'index': i,
                        'price': current_high,
                        'timestamp': self.data.index[i]
                    })
            else:  # low
                current_low = float(self.data['Low'].iloc[i])
                window_lows = self.data['Low'].iloc[i-window:i+window+1]
                min_low = float(window_lows.min())
                if abs(current_low - min_low) < 0.00001:  # Use small tolerance for float comparison
                    points.append({
                        'index': i,
                        'price': current_low,
                        'timestamp': self.data.index[i]
                    })
        
        return points
    
    def identify_higher_highs(self, swing_highs):
        """Identify Higher Highs (HH)"""
        higher_highs = []
        
        for i in range(1, len(swing_highs)):
            if swing_highs[i]['price'] > swing_highs[i-1]['price']:
                higher_highs.append({
                    'current': swing_highs[i],
                    'previous': swing_highs[i-1],
                    'type': 'HH'
                })
        
        return higher_highs
    
    def identify_higher_lows(self, swing_lows):
        """Identify Higher Lows (HL)"""
        higher_lows = []
        
        for i in range(1, len(swing_lows)):
            if swing_lows[i]['price'] > swing_lows[i-1]['price']:
                higher_lows.append({
                    'current': swing_lows[i],
                    'previous': swing_lows[i-1],
                    'type': 'HL'
                })
        
        return higher_lows
    
    def identify_lower_highs(self, swing_highs):
        """Identify Lower Highs (LH)"""
        lower_highs = []
        
        for i in range(1, len(swing_highs)):
            if swing_highs[i]['price'] < swing_highs[i-1]['price']:
                lower_highs.append({
                    'current': swing_highs[i],
                    'previous': swing_highs[i-1],
                    'type': 'LH'
                })
        
        return lower_highs
    
    def identify_lower_lows(self, swing_lows):
        """Identify Lower Lows (LL)"""
        lower_lows = []
        
        for i in range(1, len(swing_lows)):
            if swing_lows[i]['price'] < swing_lows[i-1]['price']:
                lower_lows.append({
                    'current': swing_lows[i],
                    'previous': swing_lows[i-1],
                    'type': 'LL'
                })
        
        return lower_lows
    
    def detect_break_of_structure(self, higher_highs, higher_lows, lower_highs, lower_lows):
        """Detect Break of Structure (BOS) points"""
        bos_points = []
        
        # Bullish BOS: Break above previous lower high
        for lh in lower_highs:
            # Find if price breaks above this lower high
            break_index = self.find_break_above(lh['current']['index'], lh['current']['price'])
            if break_index:
                bos_points.append({
                    'index': break_index,
                    'price': lh['current']['price'],
                    'type': 'Bullish_BOS',
                    'level': lh['current']['price']
                })
        
        # Bearish BOS: Break below previous higher low
        for hl in higher_lows:
            # Find if price breaks below this higher low
            break_index = self.find_break_below(hl['current']['index'], hl['current']['price'])
            if break_index:
                bos_points.append({
                    'index': break_index,
                    'price': hl['current']['price'],
                    'type': 'Bearish_BOS',
                    'level': hl['current']['price']
                })
        
        return bos_points
    
    def detect_change_of_character(self, higher_highs, higher_lows, lower_highs, lower_lows):
        """Detect Change of Character (CHoCH) points"""
        choch_points = []
        
        # Bullish CHoCH: First higher high after downtrend
        if len(lower_lows) > 0 and len(higher_highs) > 0:
            last_ll = lower_lows[-1]
            first_hh_after_ll = None
            
            for hh in higher_highs:
                if hh['current']['index'] > last_ll['current']['index']:
                    first_hh_after_ll = hh
                    break
            
            if first_hh_after_ll:
                choch_points.append({
                    'index': first_hh_after_ll['current']['index'],
                    'price': first_hh_after_ll['current']['price'],
                    'type': 'Bullish_CHoCH'
                })
        
        # Bearish CHoCH: First lower low after uptrend
        if len(higher_highs) > 0 and len(lower_lows) > 0:
            last_hh = higher_highs[-1]
            first_ll_after_hh = None
            
            for ll in lower_lows:
                if ll['current']['index'] > last_hh['current']['index']:
                    first_ll_after_hh = ll
                    break
            
            if first_ll_after_hh:
                choch_points.append({
                    'index': first_ll_after_hh['current']['index'],
                    'price': first_ll_after_hh['current']['price'],
                    'type': 'Bearish_CHoCH'
                })
        
        return choch_points
    
    def determine_trend_direction(self, higher_highs, higher_lows, lower_highs, lower_lows):
        """Determine current trend direction"""
        if len(higher_highs) > len(lower_highs) and len(higher_lows) > len(lower_lows):
            return 'UPTREND'
        elif len(lower_highs) > len(higher_highs) and len(lower_lows) > len(higher_lows):
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def find_break_above(self, start_index, level):
        level = float(level)
        for i in range(start_index + 1, len(self.data)):
            if float(self.data['High'].iloc[i]) > level:
                return i
        return None
    
    def find_break_below(self, start_index, level):
        level = float(level)
        for i in range(start_index + 1, len(self.data)):
            if float(self.data['Low'].iloc[i]) < level:
                return i
        return None 

    def identify_order_blocks(self):
        """Identify Order Blocks (OB) - institutional order zones"""
        print("📦 Identifying Order Blocks...")
        
        order_blocks = []
        
        # Look for strong moves followed by retracements
        for i in range(20, len(self.data) - 5):
            # Bullish Order Block: Strong up move followed by retracement
            if self.is_bullish_order_block(i):
                ob = self.create_bullish_order_block(i)
                if ob:
                    order_blocks.append(ob)
            
            # Bearish Order Block: Strong down move followed by retracement
            if self.is_bearish_order_block(i):
                ob = self.create_bearish_order_block(i)
                if ob:
                    order_blocks.append(ob)
        
        self.order_blocks = order_blocks
    
    def is_bullish_order_block(self, index):
        """Check if index contains a bullish order block"""
        # Look for strong up move
        up_move = self.data['Close'].iloc[index] - self.data['Open'].iloc[index]
        up_move_pct = up_move / self.data['Open'].iloc[index]
        
        # Check volume
        volume_avg = self.data['Volume'].iloc[index-10:index].mean()
        current_volume = self.data['Volume'].iloc[index]
        
        # Check for retracement after the move
        retracement = False
        if index + 5 < len(self.data):
            low_after = self.data['Low'].iloc[index+1:index+6].min()
            if low_after < self.data['Close'].iloc[index]:
                retracement = True
        
        return (up_move_pct > 0.002 and  # 0.2% move
                current_volume > volume_avg * 1.5 and  # High volume
                retracement)
    
    def is_bearish_order_block(self, index):
        """Check if index contains a bearish order block"""
        # Look for strong down move
        down_move = self.data['Open'].iloc[index] - self.data['Close'].iloc[index]
        down_move_pct = down_move / self.data['Open'].iloc[index]
        
        # Check volume
        volume_avg = self.data['Volume'].iloc[index-10:index].mean()
        current_volume = self.data['Volume'].iloc[index]
        
        # Check for retracement after the move
        retracement = False
        if index + 5 < len(self.data):
            high_after = self.data['High'].iloc[index+1:index+6].max()
            if high_after > self.data['Close'].iloc[index]:
                retracement = True
        
        return (down_move_pct > 0.002 and  # 0.2% move
                current_volume > volume_avg * 1.5 and  # High volume
                retracement)
    
    def create_bullish_order_block(self, index):
        """Create bullish order block structure"""
        return {
            'index': index,
            'timestamp': self.data.index[index],
            'type': 'Bullish_OB',
            'high': self.data['High'].iloc[index],
            'low': self.data['Low'].iloc[index],
            'open': self.data['Open'].iloc[index],
            'close': self.data['Close'].iloc[index],
            'volume': self.data['Volume'].iloc[index],
            'strength': self.calculate_ob_strength(index, 'bullish'),
            'mitigated': False
        }
    
    def create_bearish_order_block(self, index):
        """Create bearish order block structure"""
        return {
            'index': index,
            'timestamp': self.data.index[index],
            'type': 'Bearish_OB',
            'high': self.data['High'].iloc[index],
            'low': self.data['Low'].iloc[index],
            'open': self.data['Open'].iloc[index],
            'close': self.data['Close'].iloc[index],
            'volume': self.data['Volume'].iloc[index],
            'strength': self.calculate_ob_strength(index, 'bearish'),
            'mitigated': False
        }
    
    def calculate_ob_strength(self, index, ob_type):
        """Calculate order block strength"""
        # Base strength on volume and price movement
        volume_ratio = self.data['Volume'].iloc[index] / self.data['Volume'].iloc[index-10:index].mean()
        
        if ob_type == 'bullish':
            price_move = (self.data['Close'].iloc[index] - self.data['Open'].iloc[index]) / self.data['Open'].iloc[index]
        else:
            price_move = (self.data['Open'].iloc[index] - self.data['Close'].iloc[index]) / self.data['Open'].iloc[index]
        
        strength = (volume_ratio * 0.6 + price_move * 1000 * 0.4) * 100
        return min(strength, 100)  # Cap at 100
    
    def identify_fair_value_gaps(self):
        """Identify Fair Value Gaps (FVG) - price inefficiencies"""
        print("🕳️ Identifying Fair Value Gaps...")
        
        fair_value_gaps = []
        
        for i in range(1, len(self.data) - 1):
            # Bullish FVG: Current low > Previous high
            if self.data['Low'].iloc[i] > self.data['High'].iloc[i-1]:
                gap_size = self.data['Low'].iloc[i] - self.data['High'].iloc[i-1]
                if gap_size > self.fvg_threshold:
                    fvg = {
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'Bullish_FVG',
                        'gap_high': self.data['Low'].iloc[i],
                        'gap_low': self.data['High'].iloc[i-1],
                        'gap_size': gap_size,
                        'mitigated': False
                    }
                    fair_value_gaps.append(fvg)
            
            # Bearish FVG: Current high < Previous low
            if self.data['High'].iloc[i] < self.data['Low'].iloc[i-1]:
                gap_size = self.data['Low'].iloc[i-1] - self.data['High'].iloc[i]
                if gap_size > self.fvg_threshold:
                    fvg = {
                        'index': i,
                        'timestamp': self.data.index[i],
                        'type': 'Bearish_FVG',
                        'gap_high': self.data['Low'].iloc[i-1],
                        'gap_low': self.data['High'].iloc[i],
                        'gap_size': gap_size,
                        'mitigated': False
                    }
                    fair_value_gaps.append(fvg)
        
        self.fair_value_gaps = fair_value_gaps
    
    def identify_liquidity_zones(self):
        """Identify Liquidity Zones - areas of stop losses"""
        print("💧 Identifying Liquidity Zones...")
        
        liquidity_zones = []
        
        # Find swing highs and lows
        swing_highs = self.find_swing_points('high', window=3)
        swing_lows = self.find_swing_points('low', window=3)
        
        # Equal highs (liquidity above)
        equal_highs = self.find_equal_highs(swing_highs)
        for eh in equal_highs:
            liquidity_zones.append({
                'type': 'Equal_Highs',
                'price': eh['price'],
                'indices': eh['indices'],
                'strength': len(eh['indices']),
                'timestamp': self.data.index[eh['indices'][-1]]
            })
        
        # Equal lows (liquidity below)
        equal_lows = self.find_equal_lows(swing_lows)
        for el in equal_lows:
            liquidity_zones.append({
                'type': 'Equal_Lows',
                'price': el['price'],
                'indices': el['indices'],
                'strength': len(el['indices']),
                'timestamp': self.data.index[el['indices'][-1]]
            })
        
        self.liquidity_zones = liquidity_zones
    
    def find_equal_highs(self, swing_highs):
        """Find equal highs for liquidity zones"""
        equal_highs = []
        tolerance = 0.0002  # Price tolerance
        
        for i in range(len(swing_highs)):
            current_group = [swing_highs[i]]
            
            for j in range(i + 1, len(swing_highs)):
                if abs(swing_highs[j]['price'] - swing_highs[i]['price']) <= tolerance:
                    current_group.append(swing_highs[j])
            
            if len(current_group) >= 2:
                # Check if this group is not already included
                already_included = False
                for existing in equal_highs:
                    # Ensure existing is a dictionary with 'indices' key
                    if isinstance(existing, dict) and 'indices' in existing:
                        # Check if any index from current_group is in existing indices
                        current_indices = [gh['index'] for gh in current_group]
                        existing_indices = existing['indices']
                        if any(idx in existing_indices for idx in current_indices):
                            already_included = True
                            break
                
                if not already_included:
                    equal_highs.append({
                        'price': swing_highs[i]['price'],
                        'indices': [gh['index'] for gh in current_group]
                    })
        
        return equal_highs
    
    def find_equal_lows(self, swing_lows):
        """Find equal lows for liquidity zones"""
        equal_lows = []
        tolerance = 0.0002  # Price tolerance
        
        for i in range(len(swing_lows)):
            current_group = [swing_lows[i]]
            
            for j in range(i + 1, len(swing_lows)):
                if abs(swing_lows[j]['price'] - swing_lows[i]['price']) <= tolerance:
                    current_group.append(swing_lows[j])
            
            if len(current_group) >= 2:
                # Check if this group is not already included
                already_included = False
                for existing in equal_lows:
                    # Ensure existing is a dictionary with 'indices' key
                    if isinstance(existing, dict) and 'indices' in existing:
                        # Check if any index from current_group is in existing indices
                        current_indices = [gl['index'] for gl in current_group]
                        existing_indices = existing['indices']
                        if any(idx in existing_indices for idx in current_indices):
                            already_included = True
                            break
                
                if not already_included:
                    equal_lows.append({
                        'price': swing_lows[i]['price'],
                        'indices': [gl['index'] for gl in current_group]
                    })
        
        return equal_lows
    
    def identify_premium_discount_zones(self):
        """Identify Premium and Discount Zones"""
        print("💰 Identifying Premium/Discount Zones...")
        
        premium_zones = []
        discount_zones = []
        
        # Calculate moving averages for reference
        ma20 = self.data['Close'].rolling(window=20).mean()
        ma50 = self.data['Close'].rolling(window=50).mean()
        
        for i in range(50, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_ma20 = ma20.iloc[i]
            current_ma50 = ma50.iloc[i]
            
            # Premium zone: Price significantly above moving averages
            if current_price > current_ma20 * 1.005 and current_price > current_ma50 * 1.01:
                premium_zones.append({
                    'index': i,
                    'timestamp': self.data.index[i],
                    'price': current_price,
                    'ma20': current_ma20,
                    'ma50': current_ma50,
                    'premium_pct': ((current_price - current_ma50) / current_ma50) * 100
                })
            
            # Discount zone: Price significantly below moving averages
            elif current_price < current_ma20 * 0.995 and current_price < current_ma50 * 0.99:
                discount_zones.append({
                    'index': i,
                    'timestamp': self.data.index[i],
                    'price': current_price,
                    'ma20': current_ma20,
                    'ma50': current_ma50,
                    'discount_pct': ((current_ma50 - current_price) / current_ma50) * 100
                })
        
        self.premium_discount_zones = {
            'premium': premium_zones,
            'discount': discount_zones
        }
    
    def identify_institutional_order_blocks(self):
        """Identify Institutional Order Blocks (IOB) - large volume zones"""
        print("🏢 Identifying Institutional Order Blocks...")
        
        institutional_obs = []
        
        # Calculate volume statistics
        volume_mean = self.data['Volume'].mean()
        volume_std = self.data['Volume'].std()
        
        for i in range(20, len(self.data)):
            current_volume = self.data['Volume'].iloc[i]
            
            # IOB criteria: Very high volume (> 2 standard deviations above mean)
            if current_volume > volume_mean + (2 * volume_std):
                # Check for strong price movement
                price_change = abs(self.data['Close'].iloc[i] - self.data['Open'].iloc[i])
                price_change_pct = price_change / self.data['Open'].iloc[i]
                
                if price_change_pct > 0.003:  # 0.3% move
                    iob = {
                        'index': i,
                        'timestamp': self.data.index[i],
                        'volume': current_volume,
                        'volume_ratio': current_volume / volume_mean,
                        'price_change': price_change_pct,
                        'high': self.data['High'].iloc[i],
                        'low': self.data['Low'].iloc[i],
                        'open': self.data['Open'].iloc[i],
                        'close': self.data['Close'].iloc[i],
                        'type': 'Bullish_IOB' if self.data['Close'].iloc[i] > self.data['Open'].iloc[i] else 'Bearish_IOB'
                    }
                    institutional_obs.append(iob)
        
        self.institutional_order_blocks = institutional_obs
    
    def get_smc_signals(self, current_price):
        """Generate SMC-based trading signals"""
        signals = {
            'order_block_signals': self.get_order_block_signals(current_price),
            'fvg_signals': self.get_fvg_signals(current_price),
            'liquidity_signals': self.get_liquidity_signals(current_price),
            'market_structure_signals': self.get_market_structure_signals(),
            'premium_discount_signals': self.get_premium_discount_signals(current_price),
            'institutional_signals': self.get_institutional_signals(current_price)
        }
        
        return signals
    
    def get_order_block_signals(self, current_price):
        """Get signals from order blocks"""
        signals = []
        
        for ob in self.order_blocks:
            if ob['type'] == 'Bullish_OB' and not ob['mitigated']:
                # Check if price is near bullish order block
                if current_price >= ob['low'] and current_price <= ob['high']:
                    signals.append({
                        'type': 'Bullish_OB_Entry',
                        'price': current_price,
                        'ob_level': ob['high'],
                        'stop_loss': ob['low'],
                        'strength': ob['strength'],
                        'timestamp': ob['timestamp']
                    })
            
            elif ob['type'] == 'Bearish_OB' and not ob['mitigated']:
                # Check if price is near bearish order block
                if current_price >= ob['low'] and current_price <= ob['high']:
                    signals.append({
                        'type': 'Bearish_OB_Entry',
                        'price': current_price,
                        'ob_level': ob['low'],
                        'stop_loss': ob['high'],
                        'strength': ob['strength'],
                        'timestamp': ob['timestamp']
                    })
        
        return signals
    
    def get_fvg_signals(self, current_price):
        """Get signals from fair value gaps"""
        signals = []
        
        for fvg in self.fair_value_gaps:
            if not fvg['mitigated']:
                if fvg['type'] == 'Bullish_FVG':
                    # Check if price is filling bullish FVG
                    if current_price >= fvg['gap_low'] and current_price <= fvg['gap_high']:
                        signals.append({
                            'type': 'Bullish_FVG_Fill',
                            'price': current_price,
                            'fvg_level': fvg['gap_high'],
                            'stop_loss': fvg['gap_low'],
                            'gap_size': fvg['gap_size'],
                            'timestamp': fvg['timestamp']
                        })
                
                elif fvg['type'] == 'Bearish_FVG':
                    # Check if price is filling bearish FVG
                    if current_price >= fvg['gap_low'] and current_price <= fvg['gap_high']:
                        signals.append({
                            'type': 'Bearish_FVG_Fill',
                            'price': current_price,
                            'fvg_level': fvg['gap_low'],
                            'stop_loss': fvg['gap_high'],
                            'gap_size': fvg['gap_size'],
                            'timestamp': fvg['timestamp']
                        })
        
        return signals
    
    def get_liquidity_signals(self, current_price):
        """Get signals from liquidity zones"""
        signals = []
        
        for lz in self.liquidity_zones:
            # Check if price is approaching liquidity zone
            distance = abs(current_price - lz['price']) / lz['price']
            
            if distance < 0.001:  # Within 0.1%
                if lz['type'] == 'Equal_Highs':
                    signals.append({
                        'type': 'Liquidity_Above',
                        'price': current_price,
                        'liquidity_level': lz['price'],
                        'strength': lz['strength'],
                        'timestamp': lz['timestamp']
                    })
                elif lz['type'] == 'Equal_Lows':
                    signals.append({
                        'type': 'Liquidity_Below',
                        'price': current_price,
                        'liquidity_level': lz['price'],
                        'strength': lz['strength'],
                        'timestamp': lz['timestamp']
                    })
        
        return signals
    
    def get_market_structure_signals(self):
        """Get signals from market structure"""
        signals = []
        
        # Break of Structure signals
        for bos in self.market_structure['break_of_structure']:
            if bos['type'] == 'Bullish_BOS':
                signals.append({
                    'type': 'Bullish_BOS',
                    'level': bos['level'],
                    'timestamp': self.data.index[bos['index']]
                })
            elif bos['type'] == 'Bearish_BOS':
                signals.append({
                    'type': 'Bearish_BOS',
                    'level': bos['level'],
                    'timestamp': self.data.index[bos['index']]
                })
        
        # Change of Character signals
        for choch in self.market_structure['change_of_character']:
            signals.append({
                'type': choch['type'],
                'level': choch['price'],
                'timestamp': self.data.index[choch['index']]
            })
        
        return signals
    
    def get_premium_discount_signals(self, current_price):
        """Get signals from premium/discount zones"""
        signals = []
        
        # Check if currently in premium zone
        current_premium = None
        for pz in self.premium_discount_zones['premium']:
            if abs(current_price - pz['price']) / pz['price'] < 0.001:
                current_premium = pz
                break
        
        if current_premium:
            signals.append({
                'type': 'Premium_Zone',
                'price': current_price,
                'premium_pct': current_premium['premium_pct'],
                'timestamp': current_premium['timestamp']
            })
        
        # Check if currently in discount zone
        current_discount = None
        for dz in self.premium_discount_zones['discount']:
            if abs(current_price - dz['price']) / dz['price'] < 0.001:
                current_discount = dz
                break
        
        if current_discount:
            signals.append({
                'type': 'Discount_Zone',
                'price': current_price,
                'discount_pct': current_discount['discount_pct'],
                'timestamp': current_discount['timestamp']
            })
        
        return signals
    
    def get_institutional_signals(self, current_price):
        """Get signals from institutional order blocks"""
        signals = []
        
        for iob in self.institutional_order_blocks:
            # Check if price is near institutional order block
            if (current_price >= iob['low'] and current_price <= iob['high']):
                signals.append({
                    'type': f"{iob['type']}_Signal",
                    'price': current_price,
                    'volume_ratio': iob['volume_ratio'],
                    'price_change': iob['price_change'],
                    'timestamp': iob['timestamp']
                })
        
        return signals
    
    def get_smc_summary(self):
        """Get comprehensive SMC analysis summary"""
        summary = {
            'market_structure': {
                'trend_direction': self.market_structure['trend_direction'],
                'higher_highs_count': len(self.market_structure['higher_highs']),
                'higher_lows_count': len(self.market_structure['higher_lows']),
                'lower_highs_count': len(self.market_structure['lower_highs']),
                'lower_lows_count': len(self.market_structure['lower_lows']),
                'bos_count': len(self.market_structure['break_of_structure']),
                'choch_count': len(self.market_structure['change_of_character'])
            },
            'order_blocks': {
                'total_count': len(self.order_blocks),
                'bullish_count': len([ob for ob in self.order_blocks if ob['type'] == 'Bullish_OB']),
                'bearish_count': len([ob for ob in self.order_blocks if ob['type'] == 'Bearish_OB']),
                'mitigated_count': len([ob for ob in self.order_blocks if ob['mitigated']]),
                'average_strength': np.mean([ob['strength'] for ob in self.order_blocks]) if self.order_blocks else 0
            },
            'fair_value_gaps': {
                'total_count': len(self.fair_value_gaps),
                'bullish_count': len([fvg for fvg in self.fair_value_gaps if fvg['type'] == 'Bullish_FVG']),
                'bearish_count': len([fvg for fvg in self.fair_value_gaps if fvg['type'] == 'Bearish_FVG']),
                'mitigated_count': len([fvg for fvg in self.fair_value_gaps if fvg['mitigated']]),
                'average_gap_size': np.mean([fvg['gap_size'] for fvg in self.fair_value_gaps]) if self.fair_value_gaps else 0
            },
            'liquidity_zones': {
                'total_count': len(self.liquidity_zones),
                'equal_highs_count': len([lz for lz in self.liquidity_zones if lz['type'] == 'Equal_Highs']),
                'equal_lows_count': len([lz for lz in self.liquidity_zones if lz['type'] == 'Equal_Lows']),
                'average_strength': np.mean([lz['strength'] for lz in self.liquidity_zones]) if self.liquidity_zones else 0
            },
            'institutional_order_blocks': {
                'total_count': len(self.institutional_order_blocks),
                'bullish_count': len([iob for iob in self.institutional_order_blocks if iob['type'] == 'Bullish_IOB']),
                'bearish_count': len([iob for iob in self.institutional_order_blocks if iob['type'] == 'Bearish_IOB']),
                'average_volume_ratio': np.mean([iob['volume_ratio'] for iob in self.institutional_order_blocks]) if self.institutional_order_blocks else 0
            }
        }
        
        return summary 