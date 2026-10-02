"""
Petrobras 3W AI Monitoring - Industrial Modbus TCP/RTU Protocol Driver

This module provides low-level register decoding for Modbus TCP/RTU telemetry
from field pressure transmitters and choke valve controllers.
"""

import struct
from typing import Dict, Any

class ModbusTCPDriver:
    """
    Modbus TCP Protocol Driver for decoding 16-bit register maps
    into IEEE-754 floating point engineering units (bar, °C).
    """
    def __init__(self, host: str = "192.168.1.100", port: int = 502, slave_id: int = 1):
        self.host = host
        self.port = port
        self.slave_id = slave_id
        
        # Modbus Holding Register Memory Map
        self.register_map = {
            40001: ("P-PDG", "float32", "Pa"),
            40003: ("T-PDG", "float32", "°C"),
            40005: ("P-TPT", "float32", "Pa"),
            40007: ("T-TPT", "float32", "°C"),
            40009: ("P-MON-CKP", "float32", "Pa"),
            40011: ("P-JUS-CKP", "float32", "Pa"),
            40013: ("T-MON-CKP", "float32", "°C"),
            40015: ("T-JUS-CKP", "float32", "°C")
        }

    @staticmethod
    def decode_float32(reg1: int, reg2: int, byte_order: str = ">f") -> float:
        """Decodes two 16-bit Modbus registers into a 32-bit float."""
        raw_bytes = struct.pack(">HH", reg1, reg2)
        val = struct.unpack(byte_order, raw_bytes)[0]
        return float(val)

    @staticmethod
    def encode_float32(val: float, byte_order: str = ">f") -> tuple[int, int]:
        """Encodes a 32-bit float into two 16-bit Modbus registers."""
        raw_bytes = struct.pack(byte_order, val)
        reg1, reg2 = struct.unpack(">HH", raw_bytes)
        return reg1, reg2

    def parse_register_block(self, registers: list[int]) -> Dict[str, float]:
        """Parses a contiguous list of 16-bit holding registers into sensor readings."""
        decoded = {}
        for idx, (reg_addr, (sensor_name, data_type, unit)) in enumerate(self.register_map.items()):
            offset = (reg_addr - 40001)
            if offset + 1 < len(registers):
                r1 = registers[offset]
                r2 = registers[offset + 1]
                decoded[sensor_name] = self.decode_float32(r1, r2)
        return decoded
