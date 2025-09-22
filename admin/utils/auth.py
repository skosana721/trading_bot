"""
Admin authentication utilities
"""
import hashlib
import secrets
from functools import wraps
from flask import request, jsonify, session, redirect, url_for
import logging

logger = logging.getLogger(__name__)

# Simple admin credentials (in production, use proper user management)
ADMIN_CREDENTIALS = {
    'username': 'admin',
    'password_hash': hashlib.sha256('admin123'.encode()).hexdigest()  # Default password: admin123
}

def generate_admin_token():
    """Generate a secure admin session token"""
    return secrets.token_urlsafe(32)

def verify_admin_password(username, password):
    """Verify admin credentials"""
    if username == ADMIN_CREDENTIALS['username']:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == ADMIN_CREDENTIALS['password_hash']
    return False

def require_admin_auth(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is logged in as admin
        if not session.get('admin_logged_in'):
            if request.is_json:
                return jsonify({'error': 'Admin authentication required'}), 401
            else:
                return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

def require_admin_api_key(f):
    """Decorator to require admin API key for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if admin is logged in first
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'Admin authentication required'}), 401
        
        # For now, we'll accept any API key if admin is logged in
        # In production, you'd validate the actual API key
        api_key = request.headers.get('X-Admin-API-Key')
        if not api_key:
            return jsonify({'error': 'Admin API key required'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def login_admin(username, password):
    """Login admin user"""
    if verify_admin_password(username, password):
        session['admin_logged_in'] = True
        session['admin_username'] = username
        session['admin_api_key'] = generate_admin_token()
        logger.info(f"Admin {username} logged in successfully")
        return True
    return False

def logout_admin():
    """Logout admin user"""
    username = session.get('admin_username', 'Unknown')
    session.clear()
    logger.info(f"Admin {username} logged out")

def is_admin_logged_in():
    """Check if admin is logged in"""
    return session.get('admin_logged_in', False)

def get_admin_username():
    """Get current admin username"""
    return session.get('admin_username', None)
