#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCP Test Client
Simulasi client Godot untuk test komunikasi TCP dengan ML Server
"""

import socket
import json
import time
import base64
import threading
import numpy as np
from PIL import Image
from io import BytesIO

class TCPTestClient:
    def __init__(self, host="127.0.0.1", port=7001):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect ke ML server"""
        try:
            print(f"🔌 Connecting to ML server {self.host}:{self.port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            print("✅ Connected to ML server!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect dari server"""
        if self.socket:
            try:
                self.socket.close()
                print("🔌 Disconnected from server")
            except:
                pass
        self.connected = False
    
    def send_message(self, message_dict):
        """Kirim message JSON ke server"""
        try:
            message_json = json.dumps(message_dict)
            message_bytes = (message_json + '\n').encode('utf-8')
            self.socket.send(message_bytes)
            return True
        except Exception as e:
            print(f"❌ Send failed: {e}")
            return False
    
    def receive_message(self, timeout=30):
        """Terima response dari server"""
        try:
            self.socket.settimeout(timeout)
            buffer = b""
            
            while True:
                chunk = self.socket.recv(1024)
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Look for newline delimiter
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if line:
                        response_text = line.decode('utf-8')
                        return json.loads(response_text)
            
            return None
            
        except socket.timeout:
            print(f"⏰ Receive timeout ({timeout}s)")
            return None
        except Exception as e:
            print(f"❌ Receive failed: {e}")
            return None
    
    def create_dummy_image_base64(self, width=200, height=200):
        """Buat dummy image dan convert ke base64"""
        print(f"🎨 Creating dummy image {width}x{height}...")
        
        # Create random image
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some patterns untuk make it more realistic
        # Add gradient
        for i in range(height):
            for j in range(width):
                image_array[i, j, 0] = min(255, int(i * 255 / height))  # Red gradient
                image_array[i, j, 1] = min(255, int(j * 255 / width))   # Green gradient
                image_array[i, j, 2] = min(255, int((i+j) * 255 / (height+width)))  # Blue gradient
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"✅ Dummy image created: {len(image_base64)} chars base64")
        return image_base64
    
    def test_connection(self):
        """Test koneksi basic"""
        print("\n" + "="*50)
        print("🧪 TEST 1: BASIC CONNECTION")
        print("="*50)
        
        if not self.connect():
            return False
        
        # Test simple ping
        test_message = {
            "type": "test_connection",
            "timestamp": time.time(),
            "client": "tcp_test_client"
        }
        
        print(f"📤 Sending test message: {test_message}")
        
        if not self.send_message(test_message):
            return False
        
        response = self.receive_message(timeout=10)
        if response:
            print(f"📥 Response received: {response}")
            return True
        else:
            print("❌ No response received")
            return False
    
    def test_ml_detection(self):
        """Test ML ethnic detection"""
        print("\n" + "="*50)
        print("🧪 TEST 2: ML ETHNIC DETECTION")
        print("="*50)
        
        # Create dummy image
        image_base64 = self.create_dummy_image_base64(300, 300)
        
        # Create detection request
        detection_request = {
            "type": "ethnic_detection",
            "image_data": image_base64,
            "timestamp": time.time(),
            "client": "tcp_test_client"
        }
        
        print(f"📤 Sending detection request...")
        print(f"   - Image size: {len(image_base64)} chars")
        print(f"   - Request type: {detection_request['type']}")
        
        if not self.send_message(detection_request):
            return False
        
        print("⏳ Waiting for ML processing...")
        response = self.receive_message(timeout=30)  # ML processing might take time
        
        if response:
            print("📥 ML Response received:")
            print(f"   - Type: {response.get('type', 'Unknown')}")
            print(f"   - Result: {response.get('result', 'None')}")
            print(f"   - Confidence: {response.get('confidence', 0)}%")
            print(f"   - ML Mode: {response.get('ml_mode', 'Unknown')}")
            print(f"   - Processing Time: {response.get('processing_time', 0)}s")
            print(f"   - Message: {response.get('message', 'No message')}")
            
            # Check if it's real ML or simulation
            ml_mode = response.get('ml_mode', 'unknown')
            if ml_mode == 'real_model':
                print("✅ REAL ML MODEL ACTIVE!")
                return True
            elif ml_mode == 'simulation':
                print("⚠️ WARNING: Running in SIMULATION mode (not real ML)")
                return False
            else:
                print(f"❓ Unknown ML mode: {ml_mode}")
                return False
        else:
            print("❌ No ML response received")
            return False
    
    def test_webcam_simulation(self):
        """Test webcam detection simulation"""
        print("\n" + "="*50)
        print("🧪 TEST 3: WEBCAM DETECTION")
        print("="*50)
        
        # Create smaller image untuk webcam simulation
        frame_base64 = self.create_dummy_image_base64(640, 480)
        
        webcam_request = {
            "type": "webcam_detection", 
            "frame_data": frame_base64,
            "timestamp": time.time(),
            "client": "tcp_test_client"
        }
        
        print(f"📤 Sending webcam frame...")
        
        if not self.send_message(webcam_request):
            return False
        
        response = self.receive_message(timeout=30)
        
        if response:
            print("📥 Webcam Response received:")
            print(f"   - Type: {response.get('type', 'Unknown')}")
            print(f"   - Result: {response.get('result', 'None')}")
            print(f"   - Confidence: {response.get('confidence', 0)}%")
            print(f"   - ML Mode: {response.get('ml_mode', 'Unknown')}")
            return response.get('ml_mode') == 'real_model'
        else:
            print("❌ No webcam response received") 
            return False
    
    def test_multiple_requests(self, count=3):
        """Test multiple requests untuk persistent connection"""
        print("\n" + "="*50)
        print(f"🧪 TEST 4: MULTIPLE REQUESTS ({count}x)")
        print("="*50)
        
        success_count = 0
        
        for i in range(count):
            print(f"\n🔄 Request {i+1}/{count}")
            
            image_base64 = self.create_dummy_image_base64(200, 200)
            
            request = {
                "type": "ethnic_detection",
                "image_data": image_base64,
                "timestamp": time.time(),
                "request_id": i+1
            }
            
            if self.send_message(request):
                response = self.receive_message(timeout=20)
                if response and response.get('ml_mode') == 'real_model':
                    print(f"✅ Request {i+1} success: {response.get('result')} ({response.get('confidence')}%)")
                    success_count += 1
                else:
                    print(f"❌ Request {i+1} failed or simulation mode")
            
            time.sleep(1)  # Small delay between requests
        
        print(f"\n📊 Multiple request summary: {success_count}/{count} successful")
        return success_count == count

def main():
    """Main testing function"""
    print("🚀 STARTING TCP COMMUNICATION TEST")
    print("="*60)
    print("⚠️ Make sure ML server (ml_server.py) is running first!")
    print("="*60)
    
    client = TCPTestClient()
    test_results = {}
    
    try:
        # Test 1: Basic connection
        test_results['connection'] = client.test_connection()
        
        if not test_results['connection']:
            print("\n❌ Basic connection failed. Check if ml_server.py is running.")
            return False
        
        # Test 2: ML Detection
        test_results['ml_detection'] = client.test_ml_detection()
        
        # Test 3: Webcam simulation  
        test_results['webcam'] = client.test_webcam_simulation()
        
        # Test 4: Multiple requests
        test_results['multiple'] = client.test_multiple_requests(3)
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
    finally:
        client.disconnect()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<20}: {status}")
    
    all_passed = all(test_results.values())
    
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 TCP Communication is working perfectly!")
        print("   - Server can receive requests")
        print("   - ML model is active (not simulation)")
        print("   - Responses are properly formatted")
        print("   - Persistent connection works")
    else:
        print("\n💡 Issues detected:")
        if not test_results.get('connection'):
            print("   - Server connection issues")
        if not test_results.get('ml_detection'):
            print("   - ML detection not working or in simulation mode")
        if not test_results.get('webcam'):
            print("   - Webcam detection issues")
        if not test_results.get('multiple'):
            print("   - Multiple request handling issues")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Test cancelled")
        exit_code = 1
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        exit_code = 1
    
    print(f"\n🚪 Exiting with code {exit_code}")
    exit(exit_code)