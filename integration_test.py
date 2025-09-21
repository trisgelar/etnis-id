#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test - End-to-End Testing
Comprehensive test untuk memastikan seluruh sistem bekerja dari dependencies sampai TCP communication
"""

import subprocess
import sys
import time
import threading
import os
from datetime import datetime

class IntegrationTester:
    def __init__(self):
        self.test_results = {}
        self.server_process = None
        self.server_running = False
        
    def run_test_script(self, script_name, description):
        """Jalankan test script dan return hasil"""
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING: {description}")
        print(f"Script: {script_name}")
        print(f"{'='*60}")
        
        try:
            # Jalankan script sebagai subprocess
            result = subprocess.run(
                [sys.executable, script_name], 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout
            )
            
            print("📤 STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("📥 STDERR:")
                print(result.stderr)
            
            success = result.returncode == 0
            print(f"\n🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
            print(f"Exit Code: {result.returncode}")
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            print("⏰ Test timed out!")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out'
            }
        except Exception as e:
            print(f"💥 Error running test: {e}")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def start_ml_server(self):
        """Start ML server di background"""
        print(f"\n{'='*60}")
        print("🚀 STARTING ML SERVER")
        print(f"{'='*60}")
        
        try:
            # Start server sebagai subprocess
            self.server_process = subprocess.Popen(
                [sys.executable, 'ml_server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print("🔄 Waiting for server to start...")
            time.sleep(5)  # Give server time to start
            
            # Check if server is still running
            if self.server_process.poll() is None:
                print("✅ ML Server started successfully")
                self.server_running = True
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                print(f"❌ Server failed to start")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"💥 Error starting server: {e}")
            return False
    
    def stop_ml_server(self):
        """Stop ML server"""
        if self.server_process and self.server_running:
            print("\n🛑 Stopping ML server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print("✅ ML Server stopped")
            except subprocess.TimeoutExpired:
                print("⚠️ Force killing server...")
                self.server_process.kill()
                self.server_process.wait()
            except Exception as e:
                print(f"⚠️ Error stopping server: {e}")
            
            self.server_running = False
    
    def check_server_status(self):
        """Check if server is still running"""
        if self.server_process:
            return self.server_process.poll() is None
        return False
    
    def run_comprehensive_test(self):
        """Jalankan semua test secara berurutan"""
        print("🚀 STARTING COMPREHENSIVE INTEGRATION TEST")
        print(f"⏰ Start time: {datetime.now()}")
        print("="*80)
        
        # Test 1: Dependencies
        print("\n🔍 PHASE 1: TESTING DEPENDENCIES")
        self.test_results['dependencies'] = self.run_test_script(
            'test_dependencies.py',
            'Dependency & Environment Test'
        )
        
        if not self.test_results['dependencies']['success']:
            print("❌ Dependencies test failed. Cannot continue.")
            return False
        
        # Test 2: ML Model Direct
        print("\n🔍 PHASE 2: TESTING ML MODEL DIRECTLY")
        self.test_results['ml_model'] = self.run_test_script(
            'test_ml_model.py',
            'ML Model Direct Test'
        )
        
        if not self.test_results['ml_model']['success']:
            print("❌ ML Model test failed. Cannot continue with TCP tests.")
            return False
        
        # Test 3: Start Server and Test TCP
        print("\n🔍 PHASE 3: TESTING TCP COMMUNICATION")
        
        # Start server
        server_started = self.start_ml_server()
        if not server_started:
            print("❌ Failed to start ML server. Cannot test TCP.")
            return False
        
        try:
            # Wait a bit more for server to be fully ready
            print("⏳ Waiting for server to be fully ready...")
            time.sleep(3)
            
            # Check server status
            if not self.check_server_status():
                print("❌ Server died unexpectedly")
                return False
            
            # Run TCP test
            self.test_results['tcp_communication'] = self.run_test_script(
                'tcp_test_client.py',
                'TCP Communication Test'
            )
            
        finally:
            # Always stop server
            self.stop_ml_server()
        
        return True
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"⏰ Report generated: {datetime.now()}")
        
        # Test results summary
        print(f"\n📊 TEST RESULTS SUMMARY:")
        print("-" * 40)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['success'])
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<25}: {status}")
            
            if not result['success'] and result.get('stderr'):
                print(f"  Error: {result['stderr'][:100]}...")
        
        # Overall status
        all_passed = passed_tests == total_tests
        print(f"\n🎯 OVERALL STATUS: {passed_tests}/{total_tests} tests passed")
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! SISTEM SIAP UNTUK PRODUCTION!")
            print("\n✅ VERIFICATION COMPLETE:")
            print("   ✅ Dependencies installed correctly")
            print("   ✅ ML model loading and prediction working")
            print("   ✅ TCP server communication working")
            print("   ✅ Real ML predictions (not simulation)")
            print("   ✅ End-to-end workflow functional")
            
            print(f"\n🎯 GODOT INTEGRATION STATUS: READY")
            print("   🚀 Start ml_server.py")
            print("   🎮 Run Godot project")
            print("   🖼️ Upload images for real ethnic detection")
            
        else:
            print(f"\n⚠️ {total_tests - passed_tests} TEST(S) FAILED")
            print("\n💡 RECOMMENDED ACTIONS:")
            
            if not self.test_results.get('dependencies', {}).get('success'):
                print("   🔧 Install missing dependencies: pip install -r requirements.txt")
            
            if not self.test_results.get('ml_model', {}).get('success'):
                print("   🤖 Check ML model file: model_ml/pickle_model.pkl")
                print("   🔍 Verify scikit-learn compatibility")
            
            if not self.test_results.get('tcp_communication', {}).get('success'):
                print("   🔌 Check TCP server startup")
                print("   🌐 Verify network connectivity")
                print("   🔥 Check firewall settings")
        
        # Detailed logs
        print(f"\n📋 DETAILED TEST LOGS:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            print(f"\n🧪 {test_name.upper()}:")
            print(f"   Exit Code: {result['returncode']}")
            
            if result['stdout']:
                stdout_lines = result['stdout'].split('\n')
                key_lines = [line for line in stdout_lines if any(keyword in line for keyword in ['✅', '❌', '🎯', '⚠️', 'ERROR', 'SUCCESS', 'FAIL'])]
                
                if key_lines:
                    print("   Key Output:")
                    for line in key_lines[-5:]:  # Last 5 key lines
                        print(f"     {line}")
            
            if result['stderr']:
                print(f"   Errors: {result['stderr'][:200]}...")
        
        return all_passed

def main():
    """Main integration test function"""
    print("🚀 COMPREHENSIVE INTEGRATION TEST SUITE")
    print("="*80)
    print("🎯 This will test the entire ethnic detection system:")
    print("   1. Dependencies & Environment")
    print("   2. ML Model Functionality")
    print("   3. TCP Server Communication")
    print("   4. End-to-End Workflow")
    print("="*80)
    
    # Check if test files exist
    required_files = ['test_dependencies.py', 'test_ml_model.py', 'tcp_test_client.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing test files: {missing_files}")
        print("Make sure all test scripts are in the current directory.")
        return False
    
    tester = IntegrationTester()
    
    try:
        # Run comprehensive test
        test_completed = tester.run_comprehensive_test()
        
        if test_completed:
            # Generate report
            all_passed = tester.generate_report()
            return all_passed
        else:
            print("❌ Integration test could not complete")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Integration test interrupted by user")
        tester.stop_ml_server()
        return False
    except Exception as e:
        print(f"\n💥 Fatal error during integration test: {e}")
        tester.stop_ml_server()
        return False
    finally:
        # Cleanup
        tester.stop_ml_server()

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n🚪 Integration test completed with {'SUCCESS' if success else 'FAILURES'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)