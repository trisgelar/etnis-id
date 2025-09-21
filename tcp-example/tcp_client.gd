extends Node

class_name TcpClient

var tcp_client := StreamPeerTCP.new()
var connected := false
var host := "127.0.0.1"
var port := 7001

# Signals untuk komunikasi dengan script lain
@warning_ignore("unused_signal")
signal message_received(data: Dictionary)
@warning_ignore("unused_signal")
signal connection_established()
@warning_ignore("unused_signal")
signal connection_failed()
@warning_ignore("unused_signal")
signal disconnected()

func connect_to_server() -> bool:
	"""Menghubungkan ke server Python"""
	print("🔌 Mencoba terhubung ke server ", host, ":", port)
	
	var error = tcp_client.connect_to_host(host, port)
	if error != OK:
		print("❌ Gagal terhubung ke server: ", error)
		emit_signal("connection_failed")
		return false
	
	# Tunggu sampai koneksi berhasil atau gagal
	var timeout = 5.0  # 5 detik timeout
	var elapsed_time = 0.0
	
	while elapsed_time < timeout:
		tcp_client.poll()
		var status = tcp_client.get_status()
		
		if status == StreamPeerTCP.STATUS_CONNECTED:
			connected = true
			print("✅ Berhasil terhubung ke server!")
			emit_signal("connection_established")
			return true
		elif status == StreamPeerTCP.STATUS_ERROR:
			print("❌ Error saat menghubung ke server")
			emit_signal("connection_failed")
			return false
		
		await get_tree().process_frame
		elapsed_time += get_process_delta_time()
	
	print("❌ Timeout saat menghubung ke server")
	emit_signal("connection_failed")
	return false

func send_message(data: Dictionary) -> bool:
	"""Mengirim pesan JSON ke server"""
	if not connected:
		print("❌ Tidak terhubung ke server")
		return false
	
	var json_string = JSON.stringify(data)
	var message_bytes = json_string.to_utf8_buffer()
	var newline_bytes = "\n".to_utf8_buffer()
	
	# Gabungkan pesan dengan newline delimiter
	var full_message = message_bytes + newline_bytes
	
	var error = tcp_client.put_data(full_message)
	if error != OK:
		print("❌ Gagal mengirim pesan: ", error)
		return false
	
	print("📤 Pesan dikirim: ", json_string)
	return true

func poll_messages():
	"""Cek pesan masuk dari server"""
	if not connected:
		return
	
	tcp_client.poll()
	var status = tcp_client.get_status()
	
	# Cek apakah masih terhubung
	if status != StreamPeerTCP.STATUS_CONNECTED:
		if connected:  # Jika sebelumnya terhubung tapi sekarang tidak
			connected = false
			print("⚠️ Koneksi terputus dari server")
			emit_signal("disconnected")
		return
	
	# Cek apakah ada data masuk
	if tcp_client.get_available_bytes() > 0:
		var received_data = tcp_client.get_utf8_string(tcp_client.get_available_bytes())
		print("📨 Data mentah diterima: ", received_data)
		
		# Parse JSON
		var json = JSON.new()
		var parse_result = json.parse(received_data.strip_edges())
		
		if parse_result == OK:
			var parsed_data = json.data
			print("📋 Data JSON diterima: ", parsed_data)
			emit_signal("message_received", parsed_data)
		else:
			print("❌ Gagal parse JSON: ", received_data)

func disconnect_from_server():
	"""Memutus koneksi dari server"""
	if connected:
		tcp_client.disconnect_from_host()
		connected = false
		print("🔌 Terputus dari server")
		emit_signal("disconnected")

func is_client_connected() -> bool:
	"""Cek apakah masih terhubung ke server"""
	return connected
