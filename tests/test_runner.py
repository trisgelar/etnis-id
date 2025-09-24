#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner for Ethnicity Detection System
Organizes and runs all tests following SOLID principles
"""

import sys
import os
import time
sys.path.insert(0, '.')

class TestRunner:
    """Main test runner class"""
    
    def __init__(self):
        """Initialize test runner"""
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test(self, test_name: str, test_function):
        """
        Run a single test
        
        Args:
            test_name: Name of the test
            test_function: Test function to run
            
        Returns:
            Boolean indicating if test passed
        """
        print(f"\n{'='*70}")
        print(f"🧪 RUNNING TEST: {test_name}")
        print(f"{'='*70}")
        
        try:
            start_time = time.time()
            result = test_function()
            end_time = time.time()
            
            duration = end_time - start_time
            
            self.test_results[test_name] = {
                'passed': result,
                'duration': duration,
                'error': None
            }
            
            if result:
                print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.2f}s)")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if 'start_time' in locals() else 0
            
            self.test_results[test_name] = {
                'passed': False,
                'duration': duration,
                'error': str(e)
            }
            
            print(f"❌ {test_name} FAILED with exception: {e}")
            return False
    
    def run_all_tests(self):
        """Run all available tests"""
        print("🚀 ETHNICITY DETECTION SYSTEM - TEST SUITE")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Prefer pytest collection if available; otherwise run legacy tests
        try:
            import pytest
            print("🧪 Using pytest to discover and run tests (unit/integration/smoke)...")
            code = pytest.main(["-q", "tests"])
            self.test_results["pytest_suite"] = {"passed": code == 0, "duration": 0, "error": None if code == 0 else f"pytest exit code {code}"}
        except Exception as _:
            # Fallback: legacy direct runners
            self.run_test("Cross-Validation Module", self.test_cross_validation_module)
            self.run_test("Feature Scaling", self.test_feature_scaling)
            self.run_test("Cross-Validation Integration", self.test_cross_validation_integration)
            
            # Component tests
            self.run_test("Dependencies", self.test_dependencies)
            self.run_test("ML Model", self.test_ml_model)
            self.run_test("Solid Training System", self.test_solid_training)
            
            # Analysis tests
            self.run_test("Feature Analysis Diagnosis", self.test_feature_analysis)
            self.run_test("Notebook Comparison", self.test_notebook_comparison)
            
            # Demo tests
            self.run_test("Simple CV Fix Demo", self.test_simple_cv_demo)
        
        self.end_time = time.time()
        self.print_summary()
    
    def test_cross_validation_module(self):
        """Test the cross-validation module"""
        try:
            from test_cross_validation import main as test_main
            return test_main()
        except Exception as e:
            print(f"❌ Cross-validation test failed: {e}")
            return False
    
    def test_feature_scaling(self):
        """Test feature scaling functionality"""
        try:
            from test_cross_validation import test_feature_scaling
            return test_feature_scaling()
        except Exception as e:
            print(f"❌ Feature scaling test failed: {e}")
            return False
    
    def test_cross_validation_integration(self):
        """Test cross-validation integration"""
        try:
            from test_cross_validation import test_cross_validation_integration
            return test_cross_validation_integration()
        except Exception as e:
            print(f"❌ Cross-validation integration test failed: {e}")
            return False
    
    def test_dependencies(self):
        """Test dependencies"""
        try:
            from test_dependencies import main as test_main
            return test_main()
        except Exception as e:
            print(f"❌ Dependencies test failed: {e}")
            return False
    
    def test_ml_model(self):
        """Test ML model"""
        try:
            from test_ml_model import main as test_main
            return test_main()
        except Exception as e:
            print(f"❌ ML model test failed: {e}")
            return False
    
    def test_solid_training(self):
        """Test SOLID training system"""
        try:
            from test_solid_training import main as test_main
            return test_main()
        except Exception as e:
            print(f"❌ SOLID training test failed: {e}")
            return False
    
    def test_feature_analysis(self):
        """Test feature analysis"""
        try:
            from feature_analysis_diagnosis import main as test_main
            return test_main() is not None
        except Exception as e:
            print(f"❌ Feature analysis test failed: {e}")
            return False
    
    def test_notebook_comparison(self):
        """Test notebook comparison"""
        try:
            from notebook_comparison_analysis import main as test_main
            return test_main() is not None
        except Exception as e:
            print(f"❌ Notebook comparison test failed: {e}")
            return False
    
    def test_simple_cv_demo(self):
        """Test simple CV demo"""
        try:
            from simple_cv_fix_demo import main as test_main
            return test_main() is not None
        except Exception as e:
            print(f"❌ Simple CV demo test failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        total_duration = self.end_time - self.start_time
        
        print(f"\n{'='*70}")
        print("📊 TEST SUMMARY")
        print(f"{'='*70}")
        
        passed = sum(1 for result in self.test_results.values() if result['passed'])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = result['duration']
            print(f"   {test_name}: {status} ({duration:.2f}s)")
            
            if not result['passed'] and result['error']:
                print(f"      Error: {result['error']}")
        
        print(f"\n{'='*70}")
        if passed == total:
            print("🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            print("⚠️  Some tests failed. Please review and fix issues.")
        print(f"{'='*70}")

def main():
    """Main function"""
    runner = TestRunner()
    runner.run_all_tests()
    
    # Return success status
    passed = sum(1 for result in runner.test_results.values() if result['passed'])
    total = len(runner.test_results)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
