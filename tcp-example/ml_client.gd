extends Node2D

# Instance dari TCP Client
var tcp_client = TcpClient.new()

# Referensi ke UI elements
@onready var connect_button = $UI/VBoxContainer/ConnectButton
@onready var disconnect_button = $UI/VBoxContainer/DisconnectButton
@onready var send_button = $UI/VBoxContainer/SendButton
@onready var response_label = $UI/VBoxContainer/ResponseLabel
@onready var status_label = $UI/VBoxContainer/StatusLabel

# Status koneksi
var is_connected_to_server = false

func _ready() -> void:
	# Tambahkan TCP client ke scene tree
	add_child(tcp_client)
	
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
	send_button.connect("pressed", _on_send_button_pressed)
	
	# Set status awal
	update_status("Tidak terhubung ke server")
	response_label.text = "Silakan hubungkan ke server terlebih dahulu"

func setup_ui():
	"""Setup initial UI state"""
	connect_button.text = "🔌 Hubungkan ke Server"
	disconnect_button.text = "🔌 Putuskan Koneksi"
	send_button.text = "📸 Kirim Data (Test)"
	
	# Status tombol awal
	connect_button.disabled = false
	disconnect_button.disabled = true
	send_button.disabled = true

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

# === TOMBOL KIRIM ===
func _on_send_button_pressed():
	"""Handler untuk tombol Send (test data)"""
	if not is_connected_to_server:
		update_status("❌ Tidak terhubung ke server!")
		return
	
	update_status("📤 Mengirim data test...")
	send_button.disabled = true
	
	# Kirim data test untuk ML
	var test_data = {
		"type": "test_detection",
		"from": "Godot",
		"timestamp": Time.get_unix_time_from_system(),
		"data": "test_image_data_placeholder"
	}
	
	var send_success = tcp_client.send_message(test_data)
	
	if send_success:
		update_status("📤 Data dikirim, menunggu hasil ML...")
	else:
		update_status("❌ Gagal mengirim data")
		send_button.disabled = false

# === SIGNAL HANDLERS ===
func _on_connection_established():
	"""Handler saat koneksi berhasil"""
	print("✅ Koneksi ke server berhasil")
	is_connected_to_server = true
	
	# Update UI
	connect_button.disabled = true
	disconnect_button.disabled = false
	send_button.disabled = false
	
	update_status("✅ Terhubung ke ML Server")
	response_label.text = "Siap menerima data untuk deteksi etnis"

func _on_connection_failed():
	"""Handler saat koneksi gagal"""
	print("❌ Koneksi ke server gagal")
	is_connected_to_server = false
	
	# Update UI
	connect_button.disabled = false
	disconnect_button.disabled = true
	send_button.disabled = true
	
	update_status("❌ Gagal terhubung ke server")

func _on_disconnected():
	"""Handler saat terputus dari server"""
	print("🔌 Terputus dari server")
	is_connected_to_server = false
	
	# Update UI
	connect_button.disabled = false
	disconnect_button.disabled = true
	send_button.disabled = true
	
	# Hanya tampilkan pesan disconnect jika bukan user yang memutus
	update_status("⚠️ Koneksi terputus dari server")
	response_label.text = "Koneksi terputus. Silakan hubungkan kembali."

func _on_message_received(data: Dictionary):
	"""Handler saat menerima hasil ML dari server"""
	print("📨 Hasil ML diterima: ", data)
	
	# Tampilkan hasil deteksi etnis
	if data.has("result"):
		var result = data["result"]
		if data.has("confidence"):
			response_label.text = "🎯 Hasil: " + str(result) + " (Confidence: " + str(data["confidence"]) + "%)"
		else:
			response_label.text = "🎯 Hasil: " + str(result)
	elif data.has("message"):
		response_label.text = "📝 Server: " + str(data["message"])
	else:
		response_label.text = "📨 Data: " + str(data)
	
	update_status("✅ Hasil diterima dari ML Server")
	
	# Enable tombol send lagi untuk test berikutnya
	send_button.disabled = false

func update_status(status_text: String):
	"""Update status label"""
	status_label.text = "Status: " + status_text
	print("Status: ", status_text)

# === FUNGSI UNTUK DEVELOPMENT SELANJUTNYA ===
func send_image_for_detection(image_data: PackedByteArray):
	"""Fungsi untuk mengirim gambar ke ML server (untuk development selanjutnya)"""
	if not is_connected_to_server:
		print("❌ Tidak terhubung ke server")
		return false
	
	var ml_request = {
		"type": "ethnic_detection",
		"from": "Godot",
		"timestamp": Time.get_unix_time_from_system(),
		"image_data": Marshalls.raw_to_base64(image_data),
		"format": "jpg"  # atau format lainnya
	}
	
	return tcp_client.send_message(ml_request)

func send_camera_frame(texture: ImageTexture):
	"""Fungsi untuk mengirim frame kamera (untuk development selanjutnya)"""
	if not is_connected_to_server:
		return false
	
	var image = texture.get_image()
	var image_data = image.save_jpg_to_buffer()
	
	return send_image_for_detection(image_data)