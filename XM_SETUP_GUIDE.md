# XM Trading Account Setup Guide

## Overview
This guide explains how to configure the trading bot to use your XM trading account credentials.

## Credentials Configuration

Your XM credentials have been configured in `config/xm_credentials.py`:

- **Account Number**: 315050186
- **Server**: XMGlobal-MT5 7
- **Password**: [Configured]

## Files Modified

The following files have been updated to use your XM credentials:

1. **`config/xm_credentials.py`** - New file containing your XM account credentials
2. **`config/config.py`** - Updated to load credentials from XM configuration
3. **`connectors/mt5_connector.py`** - Updated to use XM credentials for connection
4. **`admin/app.py`** - Updated admin portal configuration

## Verification Scripts

### 1. Verify Configuration
```bash
python scripts/verify_xm_config.py
```
This script verifies that all configurations are properly loaded.

### 2. Test MT5 Connection
```bash
python scripts/test_mt5_connection.py
```
This script tests the actual connection to your XM account via MT5.

### 3. Start Trading Bot
```bash
python scripts/start_with_xm_credentials.py
```
This script starts both the trading bot and admin portal with XM credentials.

## Admin Portal Access

Once started, you can access the admin portal at:
- **URL**: http://localhost:5001
- **Features**: Trading journal, system monitoring, configuration management

## Trading Configuration

The bot is configured with the following settings:
- **Symbol**: EURUSD
- **Timeframe**: 5m
- **Risk per Trade**: 2%
- **Auto Trade**: Disabled (for safety)
- **ML Strategy**: Enabled
- **Smart Money Concepts**: Enabled

## Safety Features

- Auto-trading is **disabled** by default for safety
- Risk management is set to 2% per trade
- Maximum 3 positions per symbol
- Daily loss limit of 5%

## Troubleshooting

### Connection Issues
1. Ensure MetaTrader 5 is installed and running
2. Verify your XM account is active and funded
3. Check internet connection
4. Try logging into MT5 manually first

### Common Error Messages
- **"MetaTrader5 not available"**: MT5 is not installed
- **"Login failed"**: Check account credentials
- **"AutoTrading disabled"**: Enable AutoTrading in MT5 terminal

## Next Steps

1. **Test Connection**: Run `python scripts/test_mt5_connection.py`
2. **Verify Setup**: Run `python scripts/verify_xm_config.py`
3. **Start Bot**: Run `python scripts/start_with_xm_credentials.py`
4. **Access Admin**: Open http://localhost:5001 in your browser
5. **Monitor**: Check logs in `logs/trading_bot.log`

## Security Notes

- Your credentials are stored in `config/xm_credentials.py`
- Keep this file secure and don't share it
- Consider using environment variables in production
- Regularly update your XM account password

## Support

If you encounter issues:
1. Check the logs in `logs/trading_bot.log`
2. Verify MT5 connection manually
3. Ensure your XM account is properly configured
4. Check the troubleshooting section above
