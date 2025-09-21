extends Node2D

# Instance dari TCP Client
var tcp_client = TcpClient.new()

# Referensi ke UI elements
@onready var send_button = $UI/VBoxContainer/SendButton
@onready var response_label = $UI/VBoxContainer/ResponseLabel
@onready var status_label = $UI/VBoxContainer/StatusLabel

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
	
	# Connect tombol
	send_button.connect("pressed", _on_send_button_pressed)
	
	# Set status awal
	update_status("Siap - Klik tombol untuk mengirim pesan")
	response_label.text = "Balasan dari server akan muncul di sini..."

func setup_ui():
	"""Setup initial UI state"""
	send_button.text = "Kirim Pesan"
	send_button.disabled = false

func _process(_delta: float) -> void:
	# Poll untuk pesan masuk
	tcp_client.poll_messages()

func _on_send_button_pressed():
	"""Handler saat tombol Send ditekan"""
	update_status("Sedang menghubung ke server...")
	send_button.disabled = true
	
	# Coba hubungkan ke server dulu
	var connection_success = await tcp_client.connect_to_server()
	
	if connection_success:
		update_status("Sedang mengirim pesan...")
		# Kirim pesan ke server
		var message = {"from": "Godot", "data": "ping"}
		var send_success = tcp_client.send_message(message)
		
		if send_success:
			update_status("Pesan dikirim, menunggu balasan...")
		else:
			update_status("Gagal mengirim pesan")
			send_button.disabled = false
	else:
		update_status("Gagal terhubung ke server")
		send_button.disabled = false

func _on_message_received(data: Dictionary):
	"""Handler saat menerima pesan dari server"""
	print("📨 Pesan diterima dari server: ", data)
	
	# Tampilkan balasan di label
	if data.has("message"):
		response_label.text = "Balasan: " + str(data["message"])
	else:
		response_label.text = "Balasan: " + str(data)
	
	update_status("✅ Komunikasi berhasil!")
	
	# Putus koneksi setelah menerima balasan
	tcp_client.disconnect_from_server()
	
	# Enable tombol lagi setelah delay singkat
	await get_tree().create_timer(1.5).timeout
	send_button.disabled = false
	update_status("Siap - Klik tombol untuk mengirim pesan lagi")

func _on_connection_established():
	"""Handler saat koneksi berhasil"""
	print("✅ Koneksi ke server berhasil")
	update_status("✅ Terhubung ke server")

func _on_connection_failed():
	"""Handler saat koneksi gagal"""
	print("❌ Koneksi ke server gagal")
	update_status("❌ Gagal terhubung ke server")
	send_button.disabled = false

func _on_disconnected():
	"""Handler saat terputus dari server"""
	print("🔌 Terputus dari server")
	# Tidak perlu update status di sini karena sudah di-handle di _on_message_received

func update_status(status_text: String):
	"""Update status label"""
	status_label.text = "Status: " + status_text
	print("Status: ", status_text)