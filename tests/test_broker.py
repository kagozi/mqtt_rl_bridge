import pytest
from mqtt_rl_bridge import MQTTBroker

def test_broker_connect_local():
    broker = MQTTBroker(host="localhost", port=1883)
    assert broker.wait_until_connected(timeout=2.0)
    broker.close()
