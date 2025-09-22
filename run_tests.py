#!/usr/bin/env python3
"""
Test Runner Script
==================

This script runs all tests with the correct PYTHONPATH set.
"""

import os
import sys
import subprocess

def run_test(test_file):
    """Run a test file with correct PYTHONPATH"""
    print(f"\n🧪 Running {test_file}...")
    print("=" * 50)
    
    # Set PYTHONPATH to current directory
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              env=env, 
                              capture_output=False, 
                              text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Trading Bot Test Suite")
    print("=" * 50)
    
    # List of test files
    test_files = [
        'tests/test_deployment_fix.py',
        'tests/test_live_trading.py', 
        'tests/test_warnings_fix.py'
    ]
    
    results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            results[test_file] = run_test(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_file}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The codebase is properly organized.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
