extends Node

class_name TcpChannel

var server := TCPServer.new()
var client
var buffer := ""

func start_server(port: int = 5555) -> bool:
	if server.listen(port) != OK:
		print("❌ Failed to start TCP server on port ", port)
		return false
	print("✅ Server listening on port ", port)
	return true


signal received_json(data: Dictionary)

func poll():
	if server.is_connection_available():
		client = server.take_connection()
		print("✅ Client connected")

	if client != null:
		# Check if the client is still connected
		if client.get_status() == StreamPeerTCP.STATUS_NONE:
			print("⚠️ Client disconnected")
			client.close()
			client = null
			buffer = ""
			return  # Exit early

		if client.get_available_bytes() > 0:
			buffer += client.get_utf8_string(client.get_available_bytes())
			var json = JSON.new()
			var error = json.parse(buffer)
			if error == OK:
				var data_received = json.data
				emit_signal("received_json", data_received)
				buffer = ""  # Clear buffer after successful parse

func send_json(data: Dictionary):
	if client:
		var json = JSON.stringify(data)
		client.put_data(json.to_utf8_buffer() + "\n".to_utf8_buffer())
func close():
	if client:
		client.close()
	server.stop()

