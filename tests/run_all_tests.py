#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner for All SOLID System Tests
Runs all test files in the correct order
"""

import sys
import os
import subprocess
from typing import List, Tuple

def run_test(test_file: str) -> Tuple[bool, str]:
    """Run a single test file and return success status and output"""
    try:
        print(f"\n{'='*60}")
        print(f"RUNNING TEST: {test_file}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        success = result.returncode == 0
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return success, result.stdout + result.stderr
        
    except Exception as e:
        print(f"ERROR running {test_file}: {e}")
        return False, str(e)

def main():
    """Run all tests in the correct order"""
    print("SOLID SYSTEM TEST RUNNER")
    print("=" * 70)
    
    # Define test files in order of execution
    test_files = [
        "test_dependencies.py",           # Check dependencies first
        "simple_test.py",                 # Basic functionality test
        "test_original_script.py",        # Original script test
        "run_training_test.py",           # Training pipeline test
        "test_fixed_solid_system.py",     # Complete SOLID system test
        "test_ml_model.py",               # ML model test
        "tcp_test_client.py"              # TCP client test
    ]
    
    results = []
    total_tests = len(test_files)
    passed_tests = 0
    
    print(f"Found {total_tests} test files to run...")
    
    for test_file in test_files:
        if os.path.exists(test_file):
            success, output = run_test(test_file)
            results.append((test_file, success, output))
            
            if success:
                passed_tests += 1
                print(f"\n✅ {test_file}: PASSED")
            else:
                print(f"\n❌ {test_file}: FAILED")
        else:
            print(f"\n⚠️  {test_file}: NOT FOUND")
            results.append((test_file, False, "File not found"))
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("The SOLID system is fully functional!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        print("Check the output above for details")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nWARNING: Test run cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

