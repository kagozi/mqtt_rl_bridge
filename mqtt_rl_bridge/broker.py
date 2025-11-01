import json
import threading
from queue import Queue, Empty
from typing import Callable, Optional

import paho.mqtt.client as mqtt
from .utils import safe_json_loads


class MQTTBroker:
    """
    Thin wrapper around ``paho-mqtt`` that isolates connection logic
    and gives a thread-safe ``Queue`` for incoming messages.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        keepalive: int = 60,
        client_id: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.keepalive = keepalive

        self._client = mqtt.Client(client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._msg_queue: Queue[dict] = Queue()
        self._lock = threading.Lock()
        self._connected = threading.Event()

        self._thread = threading.Thread(target=self._client.loop_forever, daemon=True)
        self._thread.start()

    # --------------------------------------------------------------------- #
    # Internal callbacks
    # --------------------------------------------------------------------- #
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected.set()
            print(f"[MQTT] Connected to {self.host}:{self.port}")
        else:
            print(f"[MQTT] Connection failed (rc={rc})")

    def _on_message(self, client, userdata, msg):
        payload = safe_json_loads(msg.payload)
        payload["_topic"] = msg.topic
        self._msg_queue.put(payload)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def wait_until_connected(self, timeout: float = 5.0) -> bool:
        return self._connected.wait(timeout)

    def subscribe(self, topic: str, qos: int = 0):
        self._client.subscribe(topic, qos=qos)

    def publish(self, topic: str, payload: dict | str, qos: int = 0):
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        self._client.publish(topic, payload, qos=qos)

    def get_message(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Non-blocking if timeout is None, otherwise blocks up to timeout."""
        try:
            return self._msg_queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self):
        self._client.loop_stop()
        self._client.disconnect()
        self._thread.join(timeout=2.0)