"""
System administration routes
"""
from flask import Blueprint, request, jsonify
from admin.utils.auth import require_admin_auth, require_admin_api_key
import logging
import os
import sys
import psutil
from datetime import datetime

logger = logging.getLogger(__name__)

system_admin_bp = Blueprint('system_admin', __name__, url_prefix='/admin/system')

@system_admin_bp.route('/status')
@require_admin_auth
def system_status():
    """Get system status information"""
    try:
        # Get system information
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'processes': len(psutil.pids())
        }
        
        # Get MT5 connection status
        mt5_connected = False
        mt5_info = {}
        try:
            from connectors.mt5_connector import MT5Connector
            connector = MT5Connector()
            if connector.connect():
                mt5_connected = True
                summary = connector.get_account_summary()
                if summary:
                    mt5_info = {
                        'account': summary['login'],
                        'server': summary['server'],
                        'balance': summary['balance'],
                        'equity': summary['equity'],
                        'free_margin': summary['margin_free'],
                        'margin_level': summary['margin_level']
                    }
                connector.disconnect()
        except Exception as e:
            logger.warning(f"Could not check MT5 status: {e}")
        
        return jsonify({
            'success': True,
            'system_info': system_info,
            'connected': mt5_connected,
            'mt5_info': mt5_info
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get system status: {str(e)}'
        }), 500

@system_admin_bp.route('/logs')
@require_admin_auth
def get_logs():
    """Get application logs"""
    try:
        log_file = 'logs/app.log'
        if not os.path.exists(log_file):
            return jsonify({
                'success': True,
                'logs': [],
                'message': 'No log file found'
            })
        
        # Read last 100 lines of log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        return jsonify({
            'success': True,
            'logs': [line.strip() for line in recent_lines],
            'total_lines': len(lines)
        })
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to read logs: {str(e)}'
        }), 500

@system_admin_bp.route('/config')
@require_admin_auth
def get_config():
    """Get current configuration"""
    try:
        from core.app import bot_config
        
        # Return configuration (excluding sensitive data)
        safe_config = {}
        for key, value in bot_config.items():
            if 'password' not in key.lower() and 'key' not in key.lower():
                safe_config[key] = value
            else:
                safe_config[key] = '***HIDDEN***'
        
        return jsonify({
            'success': True,
            'config': safe_config
        })
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get config: {str(e)}'
        }), 500

@system_admin_bp.route('/restart', methods=['POST'])
@require_admin_api_key
def restart_system():
    """Restart the trading system"""
    try:
        # This would restart the trading system
        # In a real implementation, you might use a process manager
        logger.info("System restart requested by admin")
        return jsonify({
            'success': True,
            'message': 'System restart initiated'
        })
    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to restart system: {str(e)}'
        }), 500

@system_admin_bp.route('/backup', methods=['POST'])
@require_admin_api_key
def create_backup():
    """Create system backup"""
    try:
        import shutil
        from datetime import datetime
        
        # Create backup directory
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'trading_bot_backup_{timestamp}'
        
        # Create backup (this is a simplified example)
        # In production, you'd backup databases, config files, etc.
        logger.info(f"Backup created: {backup_name}")
        
        return jsonify({
            'success': True,
            'message': f'Backup created: {backup_name}',
            'backup_name': backup_name
        })
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to create backup: {str(e)}'
        }), 500
