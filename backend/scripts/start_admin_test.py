#!/usr/bin/env python3
"""
Start Admin Portal for Testing
==============================

This script starts the admin portal with authentication bypassed for testing purposes.
"""

import sys
import os
import logging

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, session
from admin.app import create_admin_app

def create_test_admin_app():
    """Create admin app with authentication bypassed for testing"""
    
    # Create the normal admin app
    app = create_admin_app()
    
    # Override authentication for testing
    @app.before_request
    def bypass_auth_for_testing():
        # Auto-login as admin for testing
        session['admin_logged_in'] = True
        session['admin_username'] = 'admin'
        session['admin_api_key'] = 'test-admin-api-key'
    
    return app

def main():
    """Main function"""
    setup_logging()
    
    logging.info("="*60)
    logging.info("STARTING ADMIN PORTAL FOR TESTING")
    logging.info("="*60)
    
    # Create test admin app
    app = create_test_admin_app()
    
    port = 5001
    debug = True
    
    logging.info(f"Starting admin portal on port {port}")
    logging.info(f"Admin portal URL: http://localhost:{port}")
    logging.info(f"Debug mode: {debug}")
    logging.info("Authentication bypassed for testing")
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logging.info("Admin portal stopped by user")
    except Exception as e:
        logging.error(f"Error running admin portal: {e}")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

if __name__ == "__main__":
    main()
