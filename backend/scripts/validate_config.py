#!/usr/bin/env python3
"""
Configuration Validator for Enhanced Trading Bot
==============================================

This script validates the configuration and provides recommendations
for optimal settings.
"""

import sys
import os
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_risk_config(config: Dict[str, Any]) -> List[str]:
    """Validate risk management configuration"""
    issues = []
    
    risk_config = config.get('risk_management', {})
    
    # Check portfolio risk
    max_portfolio_risk = risk_config.get('max_portfolio_risk', 0.02)
    if max_portfolio_risk > 0.05:
        issues.append("⚠️  max_portfolio_risk is quite high (>5%). Consider reducing to 2-3%")
    elif max_portfolio_risk < 0.01:
        issues.append("ℹ️  max_portfolio_risk is very conservative (<1%). You might be limiting returns")
    
    # Check position risk
    max_position_risk = risk_config.get('max_position_risk', 0.005)
    if max_position_risk > max_portfolio_risk * 0.5:
        issues.append("⚠️  max_position_risk is too high relative to portfolio risk")
    
    # Check Kelly fraction limits
    min_kelly = risk_config.get('min_kelly_fraction', 0.01)
    max_kelly = risk_config.get('max_kelly_fraction', 0.25)
    if max_kelly > 0.3:
        issues.append("⚠️  max_kelly_fraction is very high (>30%). Consider reducing to 25%")
    
    # Check drawdown limit
    max_drawdown = risk_config.get('max_drawdown_limit', 0.05)
    if max_drawdown > 0.1:
        issues.append("⚠️  max_drawdown_limit is high (>10%). Consider reducing to 5-7%")
    
    return issues

def validate_ml_config(config: Dict[str, Any]) -> List[str]:
    """Validate ML ensemble configuration"""
    issues = []
    
    ml_config = config.get('ml_ensemble', {})
    
    # Check number of estimators
    n_estimators = ml_config.get('n_estimators', 100)
    if n_estimators < 50:
        issues.append("ℹ️  n_estimators is low (<50). Consider increasing for better performance")
    elif n_estimators > 500:
        issues.append("⚠️  n_estimators is very high (>500). This may cause overfitting")
    
    # Check learning rate
    learning_rate = ml_config.get('learning_rate', 0.1)
    if learning_rate > 0.2:
        issues.append("⚠️  learning_rate is high (>0.2). Consider reducing to 0.05-0.1")
    elif learning_rate < 0.01:
        issues.append("ℹ️  learning_rate is very low (<0.01). Training may be slow")
    
    # Check ensemble settings
    if not ml_config.get('use_voting', True):
        issues.append("ℹ️  Voting ensemble is disabled. Consider enabling for better performance")
    
    if not ml_config.get('use_stacking', True):
        issues.append("ℹ️  Stacking ensemble is disabled. Consider enabling for better performance")
    
    return issues

def validate_regime_config(config: Dict[str, Any]) -> List[str]:
    """Validate market regime detection configuration"""
    issues = []
    
    regime_config = config.get('regime_detection', {})
    
    # Check lookback period
    lookback = regime_config.get('lookback_period', 100)
    if lookback < 50:
        issues.append("⚠️  lookback_period is short (<50). Consider increasing for better regime detection")
    elif lookback > 500:
        issues.append("ℹ️  lookback_period is very long (>500). May be slow to adapt to regime changes")
    
    # Check volatility thresholds
    vol_low = regime_config.get('vol_low_threshold', 0.5)
    vol_high = regime_config.get('vol_high_threshold', 2.0)
    
    if vol_low >= vol_high:
        issues.append("❌ vol_low_threshold must be less than vol_high_threshold")
    
    if vol_high > 3.0:
        issues.append("ℹ️  vol_high_threshold is high (>3.0). Consider reducing to 2.0-2.5")
    
    return issues

def validate_trading_config(config: Dict[str, Any]) -> List[str]:
    """Validate trading strategy configuration"""
    issues = []
    
    trading_config = config.get('strategy', {})
    
    # Check signal confidence thresholds
    min_signal_conf = trading_config.get('min_signal_confidence', 0.6)
    if min_signal_conf < 0.5:
        issues.append("⚠️  min_signal_confidence is low (<0.5). May generate too many signals")
    elif min_signal_conf > 0.8:
        issues.append("ℹ️  min_signal_confidence is high (>0.8). May miss trading opportunities")
    
    # Check regime confidence
    min_regime_conf = trading_config.get('min_regime_confidence', 0.7)
    if min_regime_conf < 0.6:
        issues.append("⚠️  min_regime_confidence is low (<0.6). May trade in uncertain regimes")
    
    # Check position sizing
    position_sizing = config.get('backtesting', {}).get('position_sizing', 'kelly')
    if position_sizing not in ['fixed', 'kelly', 'volatility']:
        issues.append("❌ position_sizing must be 'fixed', 'kelly', or 'volatility'")
    
    return issues

def validate_symbols(config: Dict[str, Any]) -> List[str]:
    """Validate symbol configuration"""
    issues = []
    
    symbols = config.get('symbols', [])
    
    if not symbols:
        issues.append("❌ No symbols configured")
        return issues
    
    if len(symbols) > 10:
        issues.append("⚠️  Many symbols configured (>10). Consider focusing on fewer symbols for better performance")
    
    # Check for common symbols
    common_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
    missing_common = [s for s in common_symbols if s not in symbols]
    if missing_common:
        issues.append(f"ℹ️  Consider adding common symbols: {', '.join(missing_common)}")
    
    return issues

def get_recommendations(config: Dict[str, Any]) -> List[str]:
    """Get configuration recommendations"""
    recommendations = []
    
    # Risk management recommendations
    risk_config = config.get('risk_management', {})
    if risk_config.get('max_portfolio_risk', 0.02) <= 0.02:
        recommendations.append("✅ Portfolio risk is well-configured (≤2%)")
    
    # ML recommendations
    ml_config = config.get('ml_ensemble', {})
    if ml_config.get('use_voting', True) and ml_config.get('use_stacking', True):
        recommendations.append("✅ Ensemble methods are properly configured")
    
    # Trading recommendations
    if config.get('auto_trade', False) == False:
        recommendations.append("✅ Auto-trading is disabled (good for testing)")
    
    # Symbol recommendations
    symbols = config.get('symbols', [])
    if 3 <= len(symbols) <= 7:
        recommendations.append("✅ Good number of symbols for diversification")
    
    return recommendations

def main():
    """Main validation function"""
    try:
        from config.enhanced_trading_config import ENHANCED_TRADING_CONFIG
        
        print("Enhanced Trading Bot Configuration Validator")
        print("=" * 50)
        
        config = ENHANCED_TRADING_CONFIG
        
        # Validate different sections
        print("\n🔍 Validating Risk Management Configuration...")
        risk_issues = validate_risk_config(config)
        for issue in risk_issues:
            print(f"  {issue}")
        
        print("\n🤖 Validating ML Ensemble Configuration...")
        ml_issues = validate_ml_config(config)
        for issue in ml_issues:
            print(f"  {issue}")
        
        print("\n📊 Validating Market Regime Configuration...")
        regime_issues = validate_regime_config(config)
        for issue in regime_issues:
            print(f"  {issue}")
        
        print("\n📈 Validating Trading Strategy Configuration...")
        trading_issues = validate_trading_config(config)
        for issue in trading_issues:
            print(f"  {issue}")
        
        print("\n💱 Validating Symbol Configuration...")
        symbol_issues = validate_symbols(config)
        for issue in symbol_issues:
            print(f"  {issue}")
        
        # Get recommendations
        print("\n✅ Configuration Recommendations...")
        recommendations = get_recommendations(config)
        for rec in recommendations:
            print(f"  {rec}")
        
        # Summary
        total_issues = len(risk_issues) + len(ml_issues) + len(regime_issues) + len(trading_issues) + len(symbol_issues)
        critical_issues = sum(1 for issues in [risk_issues, ml_issues, regime_issues, trading_issues, symbol_issues] 
                            for issue in issues if issue.startswith("❌"))
        
        print(f"\n📋 Summary:")
        print(f"  Total issues: {total_issues}")
        print(f"  Critical issues: {critical_issues}")
        print(f"  Recommendations: {len(recommendations)}")
        
        if critical_issues == 0:
            print("\n🎉 Configuration looks good! You can proceed with testing.")
        else:
            print(f"\n⚠️  Please fix {critical_issues} critical issue(s) before proceeding.")
        
        return critical_issues == 0
        
    except ImportError as e:
        print(f"❌ Error importing configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
