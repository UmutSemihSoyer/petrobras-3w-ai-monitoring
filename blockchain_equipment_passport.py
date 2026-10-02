"""
Petrobras 3W AI Monitoring - Blockchain Subsea Equipment Passport

This module manages an immutable cryptographic ledger (Blockchain / Smart Contract format)
for logging maintenance records, component replacements, and SHAP anomaly diagnostics for subsea valves.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

class BlockchainEquipmentPassport:
    """
    Cryptographic Immutable Maintenance & Equipment Passport.
    """
    def __init__(self):
        self.chain: List[Dict[str, Any]] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            "index": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "equipment_id": "FPSO-P66-SUBSEA-TREE-001",
            "action": "GENESIS_COMMISSIONING",
            "previous_hash": "0" * 64,
            "hash": self.hash_block({"index": 0, "action": "GENESIS"})
        }
        self.chain.append(genesis_block)

    def hash_block(self, block: Dict[str, Any]) -> str:
        block_string = json.dumps(block, sort_keys=True).encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()

    def log_maintenance_event(self, equipment_id: str, technician_id: str, action: str, details: str) -> Dict[str, Any]:
        prev_block = self.chain[-1]
        block = {
            "index": len(self.chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equipment_id": equipment_id,
            "technician_id": technician_id,
            "action": action,
            "details": details,
            "previous_hash": prev_block["hash"]
        }
        block["hash"] = self.hash_block(block)
        self.chain.append(block)
        return block

    def verify_chain_integrity(self) -> bool:
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i-1]
            if curr["previous_hash"] != prev["hash"]:
                return False
        return True
