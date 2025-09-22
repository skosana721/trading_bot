"""
Admin authentication utilities - TEST VERSION
============================================

This version bypasses authentication for testing purposes.
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
    """Decorator to require admin authentication - TEST VERSION (bypassed)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # TESTING: Auto-login for testing purposes
        session['admin_logged_in'] = True
        session['admin_username'] = 'admin'
        session['admin_api_key'] = 'test-admin-api-key'
        return f(*args, **kwargs)
    return decorated_function

def require_admin_api_key(f):
    """Decorator to require admin API key for API endpoints - TEST VERSION (bypassed)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # TESTING: Auto-login for testing purposes
        session['admin_logged_in'] = True
        session['admin_username'] = 'admin'
        session['admin_api_key'] = 'test-admin-api-key'
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
