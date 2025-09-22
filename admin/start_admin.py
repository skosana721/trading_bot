#!/usr/bin/env python3
"""
Admin Portal Startup Script
===========================

This script starts the admin portal on a separate port.
Run this alongside the main trading bot for full functionality.
"""

import os
import sys
import subprocess
import time
import signal
import threading

def start_admin_portal():
    """Start the admin portal"""
    print("🚀 Starting Admin Portal...")
    print("=" * 50)
    
    # Change to the project directory (parent of admin folder)
    admin_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(admin_dir)
    os.chdir(project_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env['ADMIN_PORT'] = '5001'
    env['ADMIN_DEBUG'] = 'True'
    env['ADMIN_SECRET_KEY'] = 'admin-secret-key-change-in-production'
    
    try:
        # Start the admin portal
        process = subprocess.Popen([
            sys.executable, 'admin/app.py'
        ], env=env, cwd=project_dir)
        
        print(f"✅ Admin Portal started on port 5001")
        print(f"🌐 Admin Portal URL: http://localhost:5001")
        print(f"🔑 Default credentials: admin / admin123")
        print("=" * 50)
        print("Press Ctrl+C to stop the admin portal")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Admin Portal...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✅ Admin Portal stopped")
    except Exception as e:
        print(f"❌ Error starting admin portal: {e}")

if __name__ == '__main__':
    start_admin_portal()
