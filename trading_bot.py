#!/usr/bin/env python3
"""
Simple TradingBot replacement for backward compatibility
This provides basic functionality needed by mt5_trading_bot.py
"""

import pandas as pd
import yfinance as yf
import logging
from typing import Optional, Dict, Any

class TradingBot:
    """Simple TradingBot replacement for backward compatibility"""
    
    def __init__(self, symbol: str, timeframe: str, market_type: str = "forex", 
                 account_size: float = 10000, risk_per_trade: float = 0.02):
        self.symbol = symbol
        self.timeframe = timeframe
        self.market_type = market_type
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.data = None
        self.logger = logging.getLogger(__name__)
        
        # Map timeframes to yfinance intervals
        self.timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
    
    def fetch_data(self) -> bool:
        """Fetch market data using yfinance"""
        try:
            # Convert symbol for yfinance
            yf_symbol = self.symbol
            if self.market_type == "forex" and not self.symbol.endswith('=X'):
                yf_symbol = f"{self.symbol}=X"
            
            # Get interval
            interval = self.timeframe_map.get(self.timeframe, '1h')
            
            # Fetch data
            self.data = yf.download(yf_symbol, period="1mo", interval=interval)
            
            if self.data is not None and len(self.data) > 0:
                self.logger.info(f"Fetched {len(self.data)} data points for {self.symbol}")
                return True
            else:
                self.logger.warning(f"No data fetched for {self.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return False
    
    def analyze_market_trend(self) -> Optional[Dict[str, Any]]:
        """Basic market trend analysis"""
        if self.data is None or len(self.data) < 20:
            return None
        
        try:
            # Simple trend analysis
            current_price = self.data['Close'].iloc[-1]
            sma_20 = self.data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = self.data['Close'].rolling(window=50).mean().iloc[-1]
            
            # Determine trend
            if current_price > sma_20 > sma_50:
                trend = "uptrend"
            elif current_price < sma_20 < sma_50:
                trend = "downtrend"
            else:
                trend = "sideways"
            
            return {
                'trend': trend,
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'uptrend_confirmed': trend == "uptrend",
                'downtrend_confirmed': trend == "downtrend"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market trend: {e}")
            return None
    
    def generate_report(self, analysis: Dict[str, Any]) -> None:
        """Generate basic report"""
        if analysis:
            self.logger.info(f"Market Analysis for {self.symbol}:")
            self.logger.info(f"  Trend: {analysis.get('trend', 'Unknown')}")
            self.logger.info(f"  Current Price: {analysis.get('current_price', 0):.5f}")
    
    def generate_day_trading_report(self, analysis: Dict[str, Any]) -> None:
        """Generate day trading report"""
        self.generate_report(analysis)
