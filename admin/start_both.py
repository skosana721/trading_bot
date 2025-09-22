#!/usr/bin/env python3
"""
Combined Startup Script
=======================

This script starts both the main trading bot and the admin portal.
The main bot runs on port 5000 and the admin portal runs on port 5001.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from multiprocessing import Process

def start_main_app():
    """Start the main trading bot application"""
    print("🚀 Starting Main Trading Bot...")
    
    # Change to the project directory (parent of admin folder)
    admin_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(admin_dir)
    os.chdir(project_dir)
    
    try:
        # Start the main app
        subprocess.run([sys.executable, 'start_app.py'], cwd=project_dir)
    except KeyboardInterrupt:
        print("Main app stopped")

def start_admin_portal():
    """Start the admin portal"""
    print("🚀 Starting Admin Portal...")
    
    # Change to the project directory (parent of admin folder)
    admin_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(admin_dir)
    os.chdir(project_dir)
    
    # Set environment variables for admin portal
    env = os.environ.copy()
    env['ADMIN_PORT'] = '5001'
    env['ADMIN_DEBUG'] = 'True'
    env['ADMIN_SECRET_KEY'] = 'admin-secret-key-change-in-production'
    
    try:
        # Start the admin portal
        subprocess.run([sys.executable, 'admin/app.py'], env=env, cwd=project_dir)
    except KeyboardInterrupt:
        print("Admin portal stopped")

def main():
    """Main function to start both applications"""
    print("🎯 Starting Trading Bot System")
    print("=" * 60)
    print("📊 Main Trading Bot: http://localhost:5000")
    print("🔧 Admin Portal: http://localhost:5001")
    print("🔑 Admin credentials: admin / admin123")
    print("=" * 60)
    print("Press Ctrl+C to stop both applications")
    print()
    
    # Create processes for both applications
    main_process = Process(target=start_main_app)
    admin_process = Process(target=start_admin_portal)
    
    try:
        # Start both processes
        main_process.start()
        time.sleep(2)  # Give main app time to start
        admin_process.start()
        
        # Wait for both processes
        main_process.join()
        admin_process.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping both applications...")
        
        # Terminate both processes
        if main_process.is_alive():
            main_process.terminate()
        if admin_process.is_alive():
            admin_process.terminate()
        
        # Wait for processes to finish
        main_process.join()
        admin_process.join()
        
        print("✅ Both applications stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Clean up processes
        if main_process.is_alive():
            main_process.terminate()
        if admin_process.is_alive():
            admin_process.terminate()

if __name__ == '__main__':
    main()
