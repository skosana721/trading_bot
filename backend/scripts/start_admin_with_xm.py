#!/usr/bin/env python3
"""
Start Admin Portal with XM Credentials and Test Authentication
==============================================================

This script starts the admin portal with XM credentials configured and authentication bypassed for testing.
"""

import sys
import os
import logging

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_test_admin_app():
    """Create admin app with test authentication"""
    
    # Temporarily replace auth module with test version
    import admin.utils.auth as auth_module
    import admin.utils.auth_test as test_auth_module
    
    # Replace the functions with test versions
    auth_module.require_admin_auth = test_auth_module.require_admin_auth
    auth_module.require_admin_api_key = test_auth_module.require_admin_api_key
    
    # Import and create the admin app
    from admin.app import create_admin_app
    app = create_admin_app()
    
    # Add test route to show XM credentials status
    @app.route('/admin/test-xm')
    def test_xm_status():
        try:
            from connectors.mt5_connector import MT5Connector
            from config.xm_credentials import XM_CREDENTIALS
            
            connector = MT5Connector()
            
            return {
                'xm_credentials_loaded': True,
                'account_number': XM_CREDENTIALS['account_number'],
                'server': XM_CREDENTIALS['server'],
                'mt5_available': connector.connect() if hasattr(connector, 'connect') else False
            }
        except Exception as e:
            return {
                'xm_credentials_loaded': False,
                'error': str(e)
            }
    
    return app

def main():
    """Main function"""
    setup_logging()
    
    logging.info("="*60)
    logging.info("STARTING ADMIN PORTAL WITH XM CREDENTIALS")
    logging.info("="*60)
    
    # Create test admin app
    app = create_test_admin_app()
    
    port = 5001
    debug = True
    
    logging.info(f"Starting admin portal on port {port}")
    logging.info(f"Admin portal URL: http://localhost:{port}")
    logging.info(f"Dashboard URL: http://localhost:{port}/admin/dashboard")
    logging.info(f"Trading Journal URL: http://localhost:{port}/admin/trading/journal")
    logging.info(f"XM Test URL: http://localhost:{port}/admin/test-xm")
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

if __name__ == "__main__":
    main()
