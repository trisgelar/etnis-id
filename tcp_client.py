import socket
import json

class TCPClient:
    def __init__(self, host="127.0.0.1", port=5555):
        self.host = host
        self.port = port
        self.socket = self._connect()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        return s

    def send(self, data):
        message = json.dumps(data).encode("utf-8")
        self.socket.sendall(message + b"\n")  # newline delimiter

    def receive(self, timeout=5):
        self.socket.settimeout(timeout)
        buffer = b""
        try:
            while b"\n" not in buffer:
                part = self.socket.recv(4096)
                if not part:
                    raise ConnectionError("Socket connection closed by server")
                buffer += part
        except socket.timeout:
            return None
        finally:
            self.socket.settimeout(None)

        data_str = buffer.decode("utf-8").strip()
        if not data_str:
            return None

        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            print(f"(⚠️ JSON decode error): {e} | Raw: {repr(data_str)}")
            return None

    def close(self):
        self.socket.close()

