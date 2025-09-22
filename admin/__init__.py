"""
Admin Portal Package
"""
import os
from flask import Flask
from admin.routes import admin_bp, trading_admin_bp, system_admin_bp

def init_admin(app):
    """Initialize admin portal with Flask app"""
    
    # Get the admin directory path
    admin_dir = os.path.dirname(os.path.abspath(__file__))
    admin_templates_dir = os.path.join(admin_dir, 'templates')
    
    # Create a separate Flask app for admin with its own template folder
    admin_app = Flask(__name__, template_folder=admin_templates_dir)
    
    # Register admin blueprints with the main app
    app.register_blueprint(admin_bp)
    app.register_blueprint(trading_admin_bp)
    app.register_blueprint(system_admin_bp)
    
    # Update the main app's template folder to include admin templates
    if hasattr(app, 'template_folder'):
        # If main app has a template folder, we need to handle this differently
        # For now, we'll use the admin templates directly
        pass
    
    return app
