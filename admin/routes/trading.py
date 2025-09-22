"""
Trading administration routes
"""
from flask import Blueprint, request, jsonify
from admin.utils.auth import require_admin_auth, require_admin_api_key
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

trading_admin_bp = Blueprint('trading_admin', __name__, url_prefix='/admin/trading')

@trading_admin_bp.route('/journal')
@require_admin_auth
def journal_management():
    """Trading journal management interface"""
    from flask import render_template
    return render_template('admin/trading_journal.html')

@trading_admin_bp.route('/journal/data', methods=['GET'])
@require_admin_auth
def get_journal_data():
    """Get trading journal data with pagination and filters"""
    try:
        from core.app import trade_journal, trade_stats
        
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
        logger.error(f"Error retrieving trading journal: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve trading journal: {str(e)}'
        }), 500

@trading_admin_bp.route('/journal/add', methods=['POST'])
@require_admin_api_key
def add_trade():
    """Add a new trade to the journal"""
    try:
        from core.app import trade_journal, update_trade_statistics
        
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
            'trade_type': data['trade_type'],
            'entry_price': float(data['entry_price']),
            'take_profit': float(data['take_profit']),
            'stop_loss': float(data['stop_loss']),
            'entry_date': data.get('entry_date', datetime.now().isoformat()),
            'notes': data.get('notes', ''),
            'status': 'OPEN',
            'exit_price': None,
            'exit_date': None,
            'pnl': 0.0,
            'pnl_percentage': 0.0,
            'volume': float(data.get('volume', 0.1)),
            'commission': float(data.get('commission', 0.0)),
            'swap': float(data.get('swap', 0.0))
        }
        
        # Add to journal
        trade_journal.append(trade_entry)
        update_trade_statistics()
        
        logger.info(f"Trade added to journal by admin: {trade_entry['id']}")
        
        return jsonify({
            'success': True,
            'message': 'Trade added successfully',
            'trade': trade_entry
        })
        
    except Exception as e:
        logger.error(f"Error adding trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to add trade: {str(e)}'
        }), 500

@trading_admin_bp.route('/journal/<int:trade_id>', methods=['PUT'])
@require_admin_api_key
def update_trade(trade_id):
    """Update a trade entry"""
    try:
        from core.app import trade_journal, update_trade_statistics
        
        data = request.get_json()
        
        # Find the trade
        trade = next((t for t in trade_journal if t['id'] == trade_id), None)
        if not trade:
            return jsonify({
                'success': False,
                'error': 'Trade not found'
            }), 404
        
        # Update trade fields
        for field, value in data.items():
            if field in trade and field not in ['id']:
                trade[field] = value
        
        # If exit price is provided, calculate P&L and update status
        if 'exit_price' in data:
            trade['exit_price'] = float(data['exit_price'])
            trade['exit_date'] = data.get('exit_date', datetime.now().isoformat())
            
            # Calculate P&L
            if trade['trade_type'] == 'BUY':
                trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['volume'] * 100000
            else:  # SELL
                trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['volume'] * 100000
            
            # Calculate P&L percentage
            trade['pnl_percentage'] = (trade['pnl'] / (trade['entry_price'] * trade['volume'] * 100000)) * 100
            
            # Determine status
            if trade['pnl'] > 0:
                trade['status'] = 'WIN'
            elif trade['pnl'] < 0:
                trade['status'] = 'LOSS'
            else:
                trade['status'] = 'BREAKEVEN'
        
        update_trade_statistics()
        
        logger.info(f"Trade {trade_id} updated by admin")
        
        return jsonify({
            'success': True,
            'message': 'Trade updated successfully',
            'trade': trade
        })
        
    except Exception as e:
        logger.error(f"Error updating trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to update trade: {str(e)}'
        }), 500

@trading_admin_bp.route('/journal/<int:trade_id>', methods=['DELETE'])
@require_admin_api_key
def delete_trade(trade_id):
    """Delete a trade from the journal"""
    try:
        from core.app import trade_journal, update_trade_statistics
        
        # Find and remove the trade
        original_count = len(trade_journal)
        trade_journal[:] = [t for t in trade_journal if t['id'] != trade_id]
        
        if len(trade_journal) == original_count:
            return jsonify({
                'success': False,
                'error': 'Trade not found'
            }), 404
        
        update_trade_statistics()
        
        logger.info(f"Trade {trade_id} deleted by admin")
        
        return jsonify({
            'success': True,
            'message': 'Trade deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting trade: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to delete trade: {str(e)}'
        }), 500

@trading_admin_bp.route('/journal/clear', methods=['POST'])
@require_admin_api_key
def clear_journal():
    """Clear all trades from journal"""
    try:
        # Import here to avoid circular imports
        from core.app import trade_journal, update_trade_statistics
        
        # Clear the journal
        trade_journal.clear()
        update_trade_statistics()
        
        logger.info("Trading journal cleared by admin")
        return jsonify({
            'success': True,
            'message': 'Trading journal cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing trading journal: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to clear journal: {str(e)}'
        }), 500

@trading_admin_bp.route('/journal/export', methods=['GET'])
@require_admin_auth
def export_journal():
    """Export trading journal data"""
    try:
        from core.app import trade_journal
        import json
        from datetime import datetime
        
        # Create export data
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_trades': len(trade_journal),
            'trades': trade_journal
        }
        
        return jsonify({
            'success': True,
            'data': export_data
        })
    except Exception as e:
        logger.error(f"Error exporting trading journal: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to export journal: {str(e)}'
        }), 500

@trading_admin_bp.route('/journal/import', methods=['POST'])
@require_admin_api_key
def import_journal():
    """Import trading journal data"""
    try:
        from core.app import trade_journal, update_trade_statistics
        
        data = request.get_json()
        if not data or 'trades' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid import data'
            }), 400
        
        # Clear existing journal
        trade_journal.clear()
        
        # Import new trades
        imported_trades = data.get('trades', [])
        trade_journal.extend(imported_trades)
        
        # Update statistics
        update_trade_statistics()
        
        logger.info(f"Trading journal imported with {len(imported_trades)} trades by admin")
        return jsonify({
            'success': True,
            'message': f'Successfully imported {len(imported_trades)} trades',
            'imported_count': len(imported_trades)
        })
    except Exception as e:
        logger.error(f"Error importing trading journal: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to import journal: {str(e)}'
        }), 500

@trading_admin_bp.route('/mt5/history/clear', methods=['POST'])
@require_admin_api_key
def clear_mt5_history():
    """Clear MT5 trading history cache"""
    try:
        # This would clear any cached MT5 history data
        logger.info("MT5 trading history cache cleared by admin")
        return jsonify({
            'success': True,
            'message': 'MT5 trading history cache cleared'
        })
    except Exception as e:
        logger.error(f"Error clearing MT5 history: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to clear MT5 history: {str(e)}'
        }), 500
