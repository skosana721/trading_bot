"""
Main admin routes
"""
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from admin.utils.auth import require_admin_auth, login_admin, logout_admin, is_admin_logged_in
import logging

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/')
def index():
    """Admin root - redirect to login page"""
    return redirect(url_for('admin.login'))

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        if login_admin(username, password):
            if request.is_json:
                return jsonify({'success': True, 'message': 'Login successful'})
            else:
                return redirect(url_for('admin.index'))
        else:
            if request.is_json:
                return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
            else:
                return render_template('admin/login.html', error='Invalid credentials')
    
    return render_template('admin/login.html')

@admin_bp.route('/vue-login', methods=['GET', 'POST'])
def vue_login():
    """Vue.js admin login page"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        if login_admin(username, password):
            if request.is_json:
                return jsonify({'success': True, 'message': 'Login successful'})
            else:
                return redirect(url_for('admin.vue_dashboard'))
        else:
            if request.is_json:
                return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
            else:
                return render_template('admin/vue-login.html', error='Invalid credentials')
    
    return render_template('admin/vue-login.html')

@admin_bp.route('/logout', methods=['POST'])
@require_admin_auth
def logout():
    """Admin logout"""
    logout_admin()
    if request.is_json:
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    else:
        return redirect(url_for('admin.login'))

@admin_bp.route('/status')
@require_admin_auth
def status():
    """Admin status check"""
    return jsonify({
        'success': True,
        'admin_logged_in': True,
        'username': session.get('admin_username')
    })

@admin_bp.route('/dashboard')
@require_admin_auth
def dashboard():
    """Admin dashboard with system overview"""
    return render_template('admin/dashboard.html')

@admin_bp.route('/vue-dashboard')
@require_admin_auth
def vue_dashboard():
    """Vue.js admin dashboard"""
    return render_template('admin/vue-dashboard.html')