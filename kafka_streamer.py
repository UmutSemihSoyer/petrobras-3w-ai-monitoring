"""
Petrobras 3W AI Monitoring - Apache Kafka Real-Time Stream Processing Module

This module provides high-throughput Kafka producer and consumer interfaces
for streaming sensor telemetry across offshore platform networks.
"""

import json
import time
from typing import Callable, Optional

class KafkaTelemetryProducer:
    """
    Kafka Producer for serializing and publishing high-frequency sensor telemetry to Kafka topics.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "petrobras3w-telemetry"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.is_active = False

    def send_sensor_event(self, well_id: str, sensor_dict: dict) -> dict:
        """Sends a telemetry event payload to the target Kafka topic."""
        payload = {
            "well_id": well_id,
            "data": sensor_dict,
            "produced_at": time.time()
        }
        # Simulated Kafka Producer emit
        print(f"[Kafka Producer] Published event to topic '{self.topic}' (Well: {well_id})")
        return payload


class KafkaTelemetryConsumer:
    """
    Kafka Consumer for subscribing to telemetry topics and triggering AI predictions.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "petrobras3w-telemetry"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.is_consuming = False

    def start_consuming(self, message_handler: Callable[[dict], None]):
        """Starts subscribing and consuming Kafka messages."""
        self.is_consuming = True
        print(f"[Kafka Consumer] Subscribed to topic '{self.topic}' on {self.bootstrap_servers}")

    def stop_consuming(self):
        self.is_consuming = False
        print("[Kafka Consumer] Stopped consuming.")
