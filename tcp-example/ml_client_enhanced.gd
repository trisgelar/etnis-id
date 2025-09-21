extends Node2D

# Instance dari TCP Client
var tcp_client = TcpClient.new()

# Referensi ke UI elements
@onready var connect_button = $UI/VBoxContainer/ConnectButton
@onready var disconnect_button = $UI/VBoxContainer/DisconnectButton
@onready var upload_button = $UI/VBoxContainer/UploadButton
@onready var webcam_button = $UI/VBoxContainer/WebcamButton
@onready var test_button = $UI/VBoxContainer/TestButton
@onready var status_label = $UI/VBoxContainer/StatusLabel
@onready var result_label = $UI/VBoxContainer/ResultLabel
@onready var confidence_label = $UI/VBoxContainer/ConfidenceLabel
@onready var connection_status_label = $UI/VBoxContainer/ConnectionStatusLabel
@onready var preview_rect = $UI/VBoxContainer/PreviewRect

# File dialog
var file_dialog: FileDialog

# Status koneksi
var is_connected_to_server = false
var webcam_active = false

# Camera untuk webcam
var camera: Camera3D

func _ready() -> void:
	# Tambahkan TCP client ke scene tree
	add_child(tcp_client)
	
	# Setup file dialog
	setup_file_dialog()
	
	# Setup UI
	setup_ui()
	
	# Connect sinyal dari TCP client
	tcp_client.connect("message_received", _on_message_received)
	tcp_client.connect("connection_established", _on_connection_established)
	tcp_client.connect("connection_failed", _on_connection_failed)
	tcp_client.connect("disconnected", _on_disconnected)
	
	# Connect tombol-tombol
	connect_button.connect("pressed", _on_connect_button_pressed)
	disconnect_button.connect("pressed", _on_disconnect_button_pressed)
	upload_button.connect("pressed", _on_upload_button_pressed)
	webcam_button.connect("pressed", _on_webcam_button_pressed)
	test_button.connect("pressed", _on_test_button_pressed)
	
	# Set status awal
	update_status("Tidak terhubung ke server")
	update_result("Silakan hubungkan ke server terlebih dahulu")
	update_connection_status("Belum melakukan test koneksi")

func setup_file_dialog():
	"""Setup file dialog untuk upload gambar"""
	file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	file_dialog.add_filter("*.jpg,*.jpeg,*.png", "Image Files")
	file_dialog.connect("file_selected", _on_file_selected)
	add_child(file_dialog)

func setup_ui():
	"""Setup initial UI state"""
	connect_button.text = "🔌 Hubungkan ke Server"
	disconnect_button.text = "🔌 Putuskan Koneksi"
	upload_button.text = "📁 Upload Gambar"
	webcam_button.text = "📹 Aktifkan Webcam"
	test_button.text = "🧪 Test Koneksi"
	
	# Status tombol awal
	connect_button.disabled = false
	disconnect_button.disabled = true
	upload_button.disabled = true
	webcam_button.disabled = true
	test_button.disabled = true

func _process(_delta: float) -> void:
	# Poll untuk pesan masuk hanya jika terhubung
	if is_connected_to_server:
		tcp_client.poll_messages()

# === TOMBOL KONEKSI ===
func _on_connect_button_pressed():
	"""Handler untuk tombol Connect"""
	update_status("🔄 Menghubung ke server...")
	connect_button.disabled = true
	
	var connection_success = await tcp_client.connect_to_server()
	
	if not connection_success:
		update_status("❌ Gagal terhubung ke server")
		connect_button.disabled = false

func _on_disconnect_button_pressed():
	"""Handler untuk tombol Disconnect"""
	update_status("🔄 Memutus koneksi...")
	tcp_client.disconnect_from_server()
	
	# Stop webcam jika aktif
	if webcam_active:
		stop_webcam()

# === TOMBOL UPLOAD GAMBAR ===
func _on_upload_button_pressed():
	"""Handler untuk tombol Upload Gambar"""
	if not is_connected_to_server:
		update_status("❌ Tidak terhubung ke server!")
		return
	
	file_dialog.popup_centered(Vector2i(800, 600))

func _on_file_selected(path: String):
	"""Handler saat file gambar dipilih"""
	update_status("📤 Memproses gambar...")
	upload_button.disabled = true
	
	# Load dan encode gambar
	var image = Image.new()
	var error = image.load(path)
	
	if error != OK:
		update_status("❌ Gagal memuat gambar")
		upload_button.disabled = false
		return
	
	# Convert gambar ke base64
	var image_data = encode_image_to_base64(image)
	
	if image_data == "":
		update_status("❌ Gagal mengkonversi gambar")
		upload_button.disabled = false
		return
	
	# Tampilkan preview
	show_image_preview(image)
	
	# Kirim ke server
	var ml_request = {
		"type": "ethnic_detection",
		"from": "Godot",
		"timestamp": Time.get_unix_time_from_system(),
		"image_data": image_data,
		"format": "jpg"
	}
	
	var send_success = tcp_client.send_message(ml_request)
	
	if send_success:
		update_status("📤 Gambar dikirim, memproses dengan ML...")
	else:
		update_status("❌ Gagal mengirim gambar")
		upload_button.disabled = false

# === TOMBOL WEBCAM ===
func _on_webcam_button_pressed():
	"""Handler untuk tombol Webcam"""
	if not is_connected_to_server:
		update_status("❌ Tidak terhubung ke server!")
		return
	
	if not webcam_active:
		start_webcam()
	else:
		stop_webcam()

func start_webcam():
	"""Mulai webcam untuk deteksi real-time"""
	# TODO: Implementasi webcam capture
	# Untuk sementara simulasi
	webcam_active = true
	webcam_button.text = "⏹ Stop Webcam"
	update_status("📹 Webcam aktif - mode simulasi")
	
	# Simulasi frame webcam setiap 2 detik
	simulate_webcam_frames()

func stop_webcam():
	"""Stop webcam"""
	webcam_active = false
	webcam_button.text = "📹 Aktifkan Webcam"
	update_status("📹 Webcam dihentikan")

func simulate_webcam_frames():
	"""Simulasi webcam frames untuk testing"""
	while webcam_active and is_connected_to_server:
		# Simulasi frame data
		var frame_request = {
			"type": "webcam_detection",
			"from": "Godot",
			"timestamp": Time.get_unix_time_from_system(),
			"frame_data": "simulated_webcam_frame_data"
		}
		
		tcp_client.send_message(frame_request)
		
		# Wait 2 detik sebelum frame berikutnya
		await get_tree().create_timer(2.0).timeout

# === TOMBOL TEST ===
func _on_test_button_pressed():
	"""Handler untuk tombol Test"""
	if not is_connected_to_server:
		update_status("❌ Tidak terhubung ke server!")
		return
	
	update_status("🧪 Mengirim test request...")
	test_button.disabled = true
	
	var test_data = {
		"type": "test_detection",
		"from": "Godot",
		"timestamp": Time.get_unix_time_from_system()
	}
	
	var send_success = tcp_client.send_message(test_data)
	
	if send_success:
		update_status("🧪 Test request dikirim...")
	else:
		update_status("❌ Gagal mengirim test request")
		test_button.disabled = false

# === SIGNAL HANDLERS ===
func _on_connection_established():
	"""Handler saat koneksi berhasil"""
	print("✅ Koneksi ke server berhasil")
	is_connected_to_server = true
	
	# Update UI
	connect_button.disabled = true
	disconnect_button.disabled = false
	upload_button.disabled = false
	webcam_button.disabled = false
	test_button.disabled = false
	
	update_status("✅ Terhubung ke ML Server")
	update_result("Siap menerima gambar untuk deteksi etnis")

func _on_connection_failed():
	"""Handler saat koneksi gagal"""
	print("❌ Koneksi ke server gagal")
	is_connected_to_server = false
	
	# Update UI
	connect_button.disabled = false
	disconnect_button.disabled = true
	upload_button.disabled = true
	webcam_button.disabled = true
	test_button.disabled = true
	
	update_status("❌ Gagal terhubung ke server")

func _on_disconnected():
	"""Handler saat terputus dari server"""
	print("🔌 Terputus dari server")
	is_connected_to_server = false
	
	# Stop webcam jika aktif
	if webcam_active:
		stop_webcam()
	
	# Update UI
	connect_button.disabled = false
	disconnect_button.disabled = true
	upload_button.disabled = true
	webcam_button.disabled = true
	test_button.disabled = true
	
	update_status("⚠️ Koneksi terputus dari server")
	update_result("Koneksi terputus. Silakan hubungkan kembali.")

func _on_message_received(data: Dictionary):
	"""Handler saat menerima hasil ML dari server"""
	print("📨 Hasil ML diterima: ", data)
	
	var message_type = data.get("type", "unknown")
	
	if message_type == "welcome":
		# Welcome message dari server
		var ml_status = data.get("ml_model_status", "unknown")
		if ml_status == "ready":
			update_result("🤖 Server ML siap dengan model asli")
		else:
			update_result("🤖 Server ML dalam mode simulasi")
			
	elif message_type == "connection_test":
		# Response test koneksi (bukan prediksi) - HANYA STATUS KONEKSI
		var server_status = data.get("server_status", "unknown")
		var ml_status = data.get("ml_model_status", "unknown")
		var uptime = data.get("uptime", 0.0)
		
		if server_status == "connected":
			update_status("✅ Test koneksi berhasil")
			if ml_status == "ready":
				update_connection_status("🟢 Server terhubung | Model ML: SIAP | Uptime: " + str(int(uptime)) + "s")
			else:
				update_connection_status("🟡 Server terhubung | Model ML: SIMULASI | Uptime: " + str(int(uptime)) + "s")
		else:
			update_status("❌ Test koneksi gagal")
			update_connection_status("🔴 Koneksi ke server bermasalah")
		
		# Enable tombol test lagi
		test_button.disabled = false
		# JANGAN UPDATE RESULT DAN CONFIDENCE - HANYA STATUS KONEKSI
		
	elif message_type == "detection_result":
		# Hasil deteksi gambar
		var ethnicity = data.get("result", "Unknown")
		var confidence = data.get("confidence", 0.0)
		var ml_mode = data.get("ml_mode", "unknown")
		
		update_result("🎯 Etnis: " + ethnicity)
		update_confidence("📊 Confidence: " + str(confidence) + "%")
		
		if ml_mode == "real_model":
			update_status("✅ Deteksi selesai (Model ML)")
		else:
			update_status("✅ Deteksi selesai (Simulasi)")
		
		# Enable tombol upload lagi
		upload_button.disabled = false
		
	elif message_type == "webcam_result":
		# Hasil deteksi webcam
		var ethnicity = data.get("result", "Unknown")
		var confidence = data.get("confidence", 0.0)
		
		update_result("📹 Live: " + ethnicity)
		update_confidence("📊 Confidence: " + str(confidence) + "%")
		
	elif message_type == "error":
		# Error dari server
		var error_msg = data.get("message", "Unknown error")
		update_status("❌ Error: " + error_msg)
		update_result("Error dalam pemrosesan")
		
		# Enable tombol kembali
		upload_button.disabled = false
		test_button.disabled = false
		
	else:
		# Response lainnya yang tidak dikenal
		print("⚠️ Unknown message type: ", message_type)
		update_status("⚠️ Response tidak dikenal: " + message_type)

# === UTILITY FUNCTIONS ===
func encode_image_to_base64(image: Image) -> String:
	"""Convert image ke base64 string"""
	# Convert image ke format jpg
	var buffer = image.save_jpg_to_buffer(0.8)  # Quality 80%
	
	if buffer.size() > 0:
		# Encode ke base64
		var base64_string = Marshalls.raw_to_base64(buffer)
		
		print("📸 Gambar berhasil dikonversi ke base64 (", buffer.size(), " bytes)")
		return base64_string
	else:
		print("❌ Gagal mengkonversi gambar ke base64")
		return ""

func show_image_preview(image: Image):
	"""Tampilkan preview gambar di UI"""
	# Resize image untuk preview
	var preview_size = Vector2i(200, 150)
	image.resize(preview_size.x, preview_size.y)
	
	# Create texture dari image
	var texture = ImageTexture.new()
	texture.set_image(image)
	
	# Set ke TextureRect (jika ada)
	if preview_rect:
		preview_rect.texture = texture
	
	print("📸 Preview gambar ditampilkan")

func update_status(status_text: String):
	"""Update status label"""
	status_label.text = "Status: " + status_text
	print("Status: ", status_text)

func update_result(result_text: String):
	"""Update result label"""
	result_label.text = result_text
	print("Result: ", result_text)

func update_confidence(confidence_text: String):
	"""Update confidence label"""
	confidence_label.text = confidence_text
	print("Confidence: ", confidence_text)

func update_connection_status(connection_text: String):
	"""Update connection test status - TERPISAH dari hasil ML"""
	connection_status_label.text = connection_text
	print("Connection Test: ", connection_text)
