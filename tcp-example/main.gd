extends Node2D
var tcp_channel = TcpChannel.new()

var GODOT_RL_PORT = 7001
func init_channel():
	tcp_channel.start_server(GODOT_RL_PORT)
	tcp_channel.connect("received_json",Callable(self,"_on_received_json"))

func _on_received_json(data: Dictionary):
	print(data)
	tcp_channel.send_json({"response" : "hello"})

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	init_channel()
	print("server ready")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	tcp_channel.poll()
