#!/usr/bin/env python3
"""
Script to check available symbols in MT5 broker
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.mt5_connector import MT5Connector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_symbols():
    """Check available symbols in MT5"""
    connector = MT5Connector()
    
    # Try to connect (this will use credentials from environment or config)
    if not connector.connect():
        logger.error("Failed to connect to MT5. Please check your credentials.")
        return
    
    logger.info("Connected to MT5 successfully!")
    
    # Get all available symbols
    all_symbols = connector.get_available_symbols()
    logger.info(f"Total symbols available: {len(all_symbols)}")
    
    # Check for specific symbol patterns
    patterns = {
        'Gold': ['*GOLD*', '*XAU*'],
        'Dow Jones': ['*US30*', '*DJI*', '*DOW*'],
        'NASDAQ': ['*US100*', '*NAS*', '*NDX*', '*QQQ*'],
        'Bitcoin': ['*BTC*', '*BITCOIN*'],
        'Forex': ['*USD*', '*EUR*', '*GBP*', '*JPY*']
    }
    
    for category, pattern_list in patterns.items():
        logger.info(f"\n{category} symbols:")
        found_symbols = []
        for pattern in pattern_list:
            symbols = connector.get_available_symbols(pattern)
            found_symbols.extend(symbols)
        
        # Remove duplicates and sort
        found_symbols = sorted(list(set(found_symbols)))
        if found_symbols:
            for symbol in found_symbols[:10]:  # Show first 10
                logger.info(f"  {symbol}")
            if len(found_symbols) > 10:
                logger.info(f"  ... and {len(found_symbols) - 10} more")
        else:
            logger.info("  No symbols found")
    
    # Test specific symbols we're trying to use
    test_symbols = ['XAUUSD', 'US30', 'NAS100', 'BTCUSD']
    logger.info(f"\nTesting specific symbols:")
    for symbol in test_symbols:
        mapped = connector.map_symbol_name(symbol)
        info = connector.get_symbol_info(symbol)
        if info:
            logger.info(f"  {symbol} -> {mapped}: ✅ Available")
        else:
            logger.info(f"  {symbol} -> {mapped}: ❌ Not available")
    
    connector.disconnect()

if __name__ == "__main__":
    check_symbols()
