import socket
import json
import threading
import time
import base64
from ethnic_detector import EthnicDetector
from ml_training.core.config import get_server_config

class MLTCPServer:
    def __init__(self, host=None, port=None):
        # Get server configuration
        server_config = get_server_config()
        
        self.host = host or server_config.host
        self.port = port or server_config.port
        self.server_socket = None
        self.running = False
        self.active_clients = {}  # Track active clients
        
        # Initialize ML detector
        print("🤖 Inisialisasi ML Ethnic Detector...")
        self.ethnic_detector = EthnicDetector()
        
        if self.ethnic_detector.model is None:
            print("⚠️ Warning: Model ML tidak dapat dimuat. Server akan tetap berjalan dengan simulasi.")
            self.ml_ready = False
        else:
            print("✅ ML Model siap digunakan!")
            self.ml_ready = True

    def start_server(self):
        """Memulai ML server TCP dan mendengarkan koneksi"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"🤖 ML Server Python berjalan di {self.host}:{self.port}")
            print("🔄 Menunggu koneksi dari Godot ML client...")
            print("📸 Siap untuk deteksi etnis real-time")
            
            while self.running:
                try:
                    # Terima koneksi dari client
                    client_socket, client_address = self.server_socket.accept()
                    client_id = f"{client_address[0]}:{client_address[1]}"
                    
                    print(f"✅ ML Client terhubung: {client_address}")
                    
                    # Simpan client info
                    self.active_clients[client_id] = {
                        'socket': client_socket,
                        'address': client_address,
                        'connected_time': time.time()
                    }
                    
                    # Handle client dalam thread terpisah
                    client_thread = threading.Thread(
                        target=self.handle_ml_client, 
                        args=(client_socket, client_address, client_id)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"❌ Error server: {e}")
                    
        except Exception as e:
            print(f"❌ Gagal memulai ML server: {e}")

    def handle_ml_client(self, client_socket, client_address, client_id):
        """Menangani komunikasi dengan ML client (persistent connection)"""
        buffer = b""
        
        try:
            # Kirim welcome message
            welcome_msg = {
                "type": "welcome",
                "message": "Terhubung ke ML Ethnic Detection Server",
                "server_status": "ready",
                "ml_model_status": "ready" if self.ml_ready else "simulation_mode",
                "supported_formats": ["jpg", "png", "jpeg"],
                "supported_types": ["ethnic_detection", "webcam_detection", "test_detection"],
                "timestamp": time.time()
            }
            self.send_response(client_socket, welcome_msg)
            
            while self.running and client_id in self.active_clients:
                # Set timeout untuk detect client disconnect
                client_socket.settimeout(30.0)  # 30 detik timeout
                
                try:
                    # Terima data dari client
                    data = client_socket.recv(4096)
                    
                    if not data:
                        print(f"⚠️ ML Client {client_address} memutus koneksi")
                        break
                    
                    buffer += data
                    
                    # Cek apakah ada pesan lengkap (diakhiri dengan newline)
                    while b"\n" in buffer:
                        # Ambil satu pesan
                        message, buffer = buffer.split(b"\n", 1)
                        
                        if message:
                            self.process_ml_message(message, client_socket, client_address)
                            
                except socket.timeout:
                    # Kirim ping untuk cek koneksi
                    ping_msg = {"type": "ping", "timestamp": time.time()}
                    try:
                        self.send_response(client_socket, ping_msg)
                    except:
                        print(f"⚠️ ML Client {client_address} timeout")
                        break
                        
        except Exception as e:
            print(f"❌ Error handling ML client {client_address}: {e}")
        finally:
            # Cleanup client
            if client_id in self.active_clients:
                del self.active_clients[client_id]
            client_socket.close()
            print(f"🔌 ML Client {client_address} disconnected ({len(self.active_clients)} clients aktif)")

    def process_ml_message(self, message, client_socket, client_address):
        """Memproses pesan ML dari client"""
        try:
            # Decode pesan dari bytes ke string
            message_str = message.decode("utf-8").strip()
            print(f"📨 ML Request dari {client_address}: {message_str[:100]}...")  # Truncate untuk log
            
            # Parse JSON
            try:
                received_data = json.loads(message_str)
                print(f"📋 ML Data type: {received_data.get('type', 'unknown')}")
            except json.JSONDecodeError:
                print("⚠️ Pesan bukan JSON valid")
                error_response = {"type": "error", "message": "Invalid JSON format"}
                self.send_response(client_socket, error_response)
                return
            
            # Process berdasarkan tipe request
            response = self.process_ml_request(received_data)
            self.send_response(client_socket, response)
            
        except Exception as e:
            print(f"❌ Error processing ML message: {e}")
            error_response = {"type": "error", "message": str(e)}
            self.send_response(client_socket, error_response)

    def process_ml_request(self, request_data):
        """Memproses request ML dan return response"""
        request_type = request_data.get("type", "unknown")
        
        if request_type == "test_detection":
            # Test koneksi sederhana - BUKAN prediksi ML
            return {
                "type": "connection_test",
                "server_status": "connected",
                "ml_model_status": "ready" if self.ml_ready else "simulation_mode",
                "timestamp": time.time(),
                "message": "Koneksi ke server berhasil. Server ML siap menerima permintaan.",
                "uptime": time.time() - getattr(self, 'start_time', time.time()),
                "supported_operations": ["ethnic_detection", "webcam_detection"]
            }
            
        elif request_type == "ethnic_detection":
            # Deteksi etnis dari gambar yang diupload
            image_data = request_data.get("image_data", "")
            
            if not image_data:
                return {
                    "type": "error",
                    "message": "No image data provided"
                }
            
            if self.ml_ready:
                # Gunakan model ML asli
                try:
                    print(f"🔍 Processing uploaded image: {len(image_data)} bytes")
                    
                    ethnicity, confidence, message = self.ethnic_detector.predict_ethnicity(image_data)
                    
                    print(f"🎯 ML Result: {ethnicity} with {confidence:.1f}% confidence")
                    
                    if ethnicity:
                        return {
                            "type": "detection_result",
                            "result": ethnicity,
                            "confidence": round(confidence, 1),
                            "processing_time": round(time.time() - request_data.get("timestamp", time.time()), 3),
                            "image_size": len(image_data),
                            "timestamp": time.time(),
                            "message": f"Detected {ethnicity} with {confidence:.1f}% confidence (upload gambar)",
                            "ml_mode": "real_model"
                        }
                    else:
                        return {
                            "type": "error",
                            "message": f"ML Detection failed: {message}"
                        }
                        
                except Exception as e:
                    print(f"❌ ML Processing error: {str(e)}")
                    return {
                        "type": "error",
                        "message": f"ML Processing error: {str(e)}"
                    }
            else:
                # Mode simulasi
                import random
                simulated_results = [
                    {"ethnicity": "Jawa", "confidence": 92.3},
                    {"ethnicity": "Sunda", "confidence": 88.1},
                    {"ethnicity": "Bugis", "confidence": 85.7},
                    {"ethnicity": "Malay", "confidence": 81.2},
                    {"ethnicity": "Banjar", "confidence": 79.5}
                ]
                
                result = random.choice(simulated_results)
                
                return {
                    "type": "detection_result",
                    "result": result["ethnicity"],
                    "confidence": result["confidence"],
                    "processing_time": random.uniform(0.2, 0.8),
                    "image_size": len(image_data),
                    "timestamp": time.time(),
                    "message": f"Detected {result['ethnicity']} with {result['confidence']:.1f}% confidence (simulasi)",
                    "ml_mode": "simulation"
                }
                
        elif request_type == "webcam_detection":
            # Deteksi etnis dari webcam frame
            frame_data = request_data.get("frame_data", "")
            
            if not frame_data:
                return {
                    "type": "error",
                    "message": "No webcam frame data provided"
                }
            
            if self.ml_ready:
                # Gunakan model ML asli untuk webcam
                try:
                    ethnicity, confidence, message = self.ethnic_detector.process_webcam_frame(frame_data)
                    
                    if ethnicity:
                        return {
                            "type": "webcam_result",
                            "result": ethnicity,
                            "confidence": round(confidence, 1),
                            "processing_time": round(time.time() - request_data.get("timestamp", time.time()), 3),
                            "timestamp": time.time(),
                            "message": f"Live detection: {ethnicity} ({confidence:.1f}%)",
                            "ml_mode": "real_model"
                        }
                    else:
                        return {
                            "type": "error",
                            "message": f"Webcam ML Detection failed: {message}"
                        }
                        
                except Exception as e:
                    return {
                        "type": "error",
                        "message": f"Webcam ML Processing error: {str(e)}"
                    }
            else:
                # Mode simulasi untuk webcam
                import random
                simulated_results = [
                    {"ethnicity": "Jawa", "confidence": 88.3},
                    {"ethnicity": "Sunda", "confidence": 85.1},
                    {"ethnicity": "Bugis", "confidence": 82.7},
                    {"ethnicity": "Malay", "confidence": 79.2},
                    {"ethnicity": "Banjar", "confidence": 76.5}
                ]
                
                result = random.choice(simulated_results)
                
                return {
                    "type": "webcam_result",
                    "result": result["ethnicity"],
                    "confidence": result["confidence"],
                    "processing_time": random.uniform(0.1, 0.3),
                    "timestamp": time.time(),
                    "message": f"Live detection: {result['ethnicity']} ({result['confidence']:.1f}%) - simulasi",
                    "ml_mode": "simulation"
                }
            
        elif request_type == "ping":
            return {
                "type": "pong",
                "timestamp": time.time(),
                "server_status": "running",
                "ml_status": "ready" if self.ml_ready else "simulation_mode"
            }
            
        else:
            return {
                "type": "error", 
                "message": f"Unknown request type: {request_type}",
                "supported_types": ["test_detection", "ethnic_detection", "webcam_detection", "ping"]
            }

    def send_response(self, client_socket, response_data):
        """Mengirim response JSON ke client"""
        try:
            # Convert response ke JSON string
            response_json = json.dumps(response_data)
            
            # Encode ke bytes dan tambahkan newline delimiter
            response_bytes = response_json.encode("utf-8") + b"\n"
            
            # Kirim ke client
            client_socket.sendall(response_bytes)
            
            # Log respons (truncate jika terlalu panjang)
            log_response = response_json[:100] + "..." if len(response_json) > 100 else response_json
            print(f"📤 ML Response: {log_response}")
            
        except Exception as e:
            print(f"❌ Error sending ML response: {e}")

    def stop_server(self):
        """Menghentikan ML server"""
        self.running = False
        
        # Tutup semua koneksi client
        for client_id, client_info in self.active_clients.items():
            try:
                client_info['socket'].close()
            except:
                pass
        
        self.active_clients.clear()
        
        if self.server_socket:
            self.server_socket.close()
        print("🛑 ML Server dihentikan")

    def get_server_stats(self):
        """Get statistik server"""
        return {
            "active_clients": len(self.active_clients),
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            "total_requests": getattr(self, 'total_requests', 0)
        }

def main():
    # Buat dan jalankan ML server
    server = MLTCPServer()
    server.start_time = time.time()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 ML Server dihentikan oleh user (Ctrl+C)")
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()