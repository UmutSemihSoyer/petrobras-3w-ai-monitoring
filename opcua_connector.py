"""
Petrobras 3W AI Monitoring - Industrial OPC-UA & MQTT SCADA Connector Module

This module simulates and manages real-time industrial telemetry connections
from offshore platform SCADA/PLC systems via OPC-UA and MQTT protocols.
"""

import time
import json
import random
import threading
from typing import Callable, Optional

class IndustrialOPCUAConnector:
    """
    Industrial OPC-UA Server Connector for Offshore Oil Wells.
    Simulates or connects to OPC-UA node IDs for 3W sensor tags.
    """
    def __init__(self, server_url: str = "opc.tcp://localhost:4840/freeopcua/server/"):
        self.server_url = server_url
        self.is_connected = False
        self.is_streaming = False
        self._thread: Optional[threading.Thread] = None
        self.node_mapping = {
            "P-PDG": "ns=2;s=Well.Sensors.P_PDG",
            "T-PDG": "ns=2;s=Well.Sensors.T_PDG",
            "P-TPT": "ns=2;s=Well.Sensors.P_TPT",
            "T-TPT": "ns=2;s=Well.Sensors.T_TPT",
            "P-MON-CKP": "ns=2;s=Well.Sensors.P_MON_CKP",
            "P-JUS-CKP": "ns=2;s=Well.Sensors.P_JUS_CKP",
            "T-MON-CKP": "ns=2;s=Well.Sensors.T_MON_CKP",
            "T-JUS-CKP": "ns=2;s=Well.Sensors.T_JUS_CKP",
            "P-ANULAR": "ns=2;s=Well.Sensors.P_ANULAR",
            "QGL": "ns=2;s=Well.Sensors.QGL"
        }

    def connect(self) -> bool:
        """Establishes connection to the industrial OPC-UA server."""
        # Simulated OPC-UA handshake
        self.is_connected = True
        print(f"[OPC-UA Connector]: Connected successfully to server at {self.server_url}")
        return True

    def disconnect(self):
        """Disconnects from the OPC-UA server."""
        self.is_streaming = False
        self.is_connected = False
        print("[OPC-UA Connector]: Disconnected from server.")

    def read_telemetry_point(self) -> dict:
        """Reads instantaneous sensor values from OPC-UA nodes."""
        if not self.is_connected:
            self.connect()
            
        base_p_pdg = 2e7 + random.uniform(-50000, 50000)
        base_t_pdg = 65.0 + random.uniform(-0.5, 0.5)
        base_p_tpt = 1.5e7 + random.uniform(-40000, 40000)
        base_t_tpt = 48.0 + random.uniform(-0.3, 0.3)
        base_p_mon = 1.0e7 + random.uniform(-30000, 30000)
        base_p_jus = 5.0e6 + random.uniform(-20000, 20000)

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "P-PDG": base_p_pdg,
            "T-PDG": base_t_pdg,
            "P-TPT": base_p_tpt,
            "T-TPT": base_t_tpt,
            "P-MON-CKP": base_p_mon,
            "P-JUS-CKP": base_p_jus,
            "T-MON-CKP": 42.0 + random.uniform(-0.2, 0.2),
            "T-JUS-CKP": 36.0 + random.uniform(-0.2, 0.2),
            "P-ANULAR": 0.0,
            "QGL": 0.0,
            "protocol": "OPC-UA",
            "node_count": len(self.node_mapping)
        }

    def start_live_stream(self, callback: Callable[[dict], None], interval_sec: float = 1.0):
        """Starts asynchronous live stream reading from OPC-UA nodes."""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        def _stream_loop():
            while self.is_streaming:
                data = self.read_telemetry_point()
                try:
                    callback(data)
                except Exception as e:
                    print(f"[OPC-UA Stream Callback Error]: {e}")
                time.sleep(interval_sec)

        self._thread = threading.Thread(target=_stream_loop, daemon=True)
        self._thread.start()
        print("[OPC-UA Stream]: Started background telemetry streaming loop.")

    def stop_live_stream(self):
        self.is_streaming = False


class IndustrialMQTTConnector:
    """
    Industrial MQTT Publisher/Subscriber for IoT Sensor Telemetry.
    """
    def __init__(self, broker: str = "broker.hivemq.com", port: int = 1883, topic: str = "petrobras3w/well1/telemetry"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.is_connected = False

    def publish_point(self, telemetry_data: dict) -> bool:
        """Publishes sensor JSON telemetry payload to MQTT broker."""
        payload = json.dumps(telemetry_data)
        # Simulated publish
        print(f"[MQTT Publisher] Sent payload to {self.topic}: {len(payload)} bytes")
        return True
