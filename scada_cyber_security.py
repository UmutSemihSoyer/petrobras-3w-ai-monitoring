"""
Petrobras 3W AI Monitoring - SCADA Cyber Attack & Sensor Spoofing Detector

This module monitors industrial SCADA telemetry streams for Man-in-the-Middle (MitM) attacks,
sensor data injection/spoofing, replay attacks, and impossibly high rate-of-change telemetry anomalies.
"""

from typing import Dict, Any

class SCADACyberSecurityMonitor:
    """
    Cyber-Physical Security & Sensor Spoofing Detection Engine.
    """
    def inspect_telemetry_packet(
        self,
        sensor_name: str,
        current_val: float,
        previous_val: float,
        time_delta_sec: float = 1.0,
        mac_signature_valid: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates physical rate of change and cryptographic signatures to detect cyber attacks.
        """
        attack_detected = False
        threat_type = "NONE"
        risk_score = 0.0

        if not mac_signature_valid:
            attack_detected = True
            threat_type = "INVALID_HMAC_SIGNATURE_MITM_ATTACK"
            risk_score = 98.0
        elif time_delta_sec > 0:
            rate_of_change = abs(current_val - previous_val) / time_delta_sec
            # Pressure jump > 50 bar/sec is physically impossible in subsea hydraulic manifolds
            if rate_of_change > 5e6:  # 50 bar/s
                attack_detected = True
                threat_type = "SENSOR_SPOOFING_DATA_INJECTION"
                risk_score = 92.0
            elif current_val == previous_val:
                # Flatline freeze could indicate replay attack
                risk_score = 35.0
                threat_type = "SUSPECTED_REPLAY_FREEZE"

        return {
            "sensor_name": sensor_name,
            "attack_detected": attack_detected,
            "threat_type": threat_type,
            "risk_score": risk_score,
            "status": "CYBER_ATTACK_ALERT" if attack_detected else ("SUSPICIOUS" if risk_score > 30 else "SECURE"),
            "action": "ISOLATE_SCADA_NODE_AND_SWITCH_TO_REDUNDANT_TRANSMITTER" if attack_detected else "ALLOW_TELEMETRY"
        }
