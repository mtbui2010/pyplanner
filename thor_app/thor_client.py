# thor_client.py

import json
import base64
from io import BytesIO

import zmq
from PIL import Image


class ThorClient:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host      = host
        self.port      = port
        self.connected = False
        self._ctx      = zmq.Context()
        self._socket   = self._new_socket(timeout_ms=5000)
        self._ping()

    def _new_socket(self, timeout_ms: int = 30000) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER,   0)
        sock.connect(f"tcp://{self.host}:{self.port}")
        return sock

    def _ping(self):
        try:
            self._socket.send_string(json.dumps({"cmd": "ping"}))
            raw = self._socket.recv_string()
            self.connected = json.loads(raw).get("status") == "ok"
        except Exception:
            self.connected = False

    def _send(self, msg: dict, timeout_ms: int = 30000) -> dict:
        try:
            self._socket.close()
            self._socket = self._new_socket(timeout_ms=timeout_ms)
            self._socket.send_string(json.dumps(msg))
            return json.loads(self._socket.recv_string())
        except zmq.Again:
            self._socket.close()
            self._socket = self._new_socket(timeout_ms=5000)
            self.connected = False
            return {"status": "error", "msg": "Server timeout — is thor_server.py running?"}
        except Exception as e:
            self.connected = False
            return {"status": "error", "msg": str(e)}

    def reset(self, task: str) -> dict:
        """Reset env for given task label (e.g. '☕ Make coffee')."""
        return self._send({"cmd": "reset", "task": task}, timeout_ms=60000)

    def step(self, action: str, obj: str = "", target: str = "") -> dict:
        return self._send({
            "cmd":    "step",
            "action": action,
            "object": obj,
            "target": target,
        }, timeout_ms=30000)

    def get_frame(self) -> Image.Image | None:
        resp = self._send({"cmd": "get_frame"}, timeout_ms=10000)
        if resp.get("status") != "ok":
            return None
        try:
            return Image.open(BytesIO(base64.b64decode(resp["frame"])))
        except Exception:
            return None

    def get_state(self) -> dict:
        return self._send({"cmd": "get_state"}, timeout_ms=10000)

    def reconnect(self):
        self._socket.close()
        self._socket = self._new_socket(timeout_ms=5000)
        self._ping()