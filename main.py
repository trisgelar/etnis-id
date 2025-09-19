from tcp_client import TCPClient

client = TCPClient(port=7001)

client.send({"hello": "world"})
response = client.receive()
print("Response:", response)
client.close()
