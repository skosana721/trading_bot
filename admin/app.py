#!/usr/bin/env python3
"""
Admin Portal Flask Application
=============================

Separate Flask application for the admin portal running on a different port.
This provides better separation between the main trading bot and admin functionality.
"""

import os
import sys
import logging
from flask import Flask, render_template
from flask_cors import CORS

# Add the parent directory to the path to import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import admin routes
from admin.routes import admin_bp, trading_admin_bp, system_admin_bp

def create_admin_app():
    """Create and configure the admin Flask application"""
    
    # Create Flask app
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
                static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
                static_url_path='/static')
    
    # Enable CORS
    CORS(app)
    
    # Configuration
    # Try to get admin config from XM credentials file
    try:
        from config.xm_credentials import ADMIN_CONFIG
        app.config['SECRET_KEY'] = ADMIN_CONFIG['secret_key']
        app.config['ADMIN_PORT'] = ADMIN_CONFIG['port']
    except ImportError:
        app.config['SECRET_KEY'] = os.getenv('ADMIN_SECRET_KEY', 'admin-secret-key-change-in-production')
        app.config['ADMIN_PORT'] = int(os.getenv('ADMIN_PORT', 5001))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    app.logger.info("Admin portal starting...")
    
    # Register blueprints
    app.register_blueprint(admin_bp)
    app.register_blueprint(trading_admin_bp)
    app.register_blueprint(system_admin_bp)
    
    app.logger.info("Admin portal blueprints registered")
    
    return app

def run_admin_app():
    """Run the admin portal application"""
    app = create_admin_app()
    
    port = app.config['ADMIN_PORT']
    debug = os.getenv('ADMIN_DEBUG', 'False').lower() == 'true'
    
    app.logger.info(f"Starting admin portal on port {port}")
    app.logger.info(f"Admin portal URL: http://localhost:{port}")
    app.logger.info(f"Debug mode: {debug}")
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            use_reloader=False  # Disable reloader to avoid conflicts
        )
    except KeyboardInterrupt:
        app.logger.info("Admin portal stopped by user")
    except Exception as e:
        app.logger.error(f"Error running admin portal: {e}")

if __name__ == '__main__':
    run_admin_app()
