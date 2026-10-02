"""
Petrobras 3W AI Monitoring - RBAC Security & Audit Trail Manager

This module provides Role-Based Access Control (RBAC) and immutable Audit Trail Logging
for offshore platform engineering actions.
"""

import time
import sqlite3
from typing import Dict, Any, List

class SecurityRBACManager:
    """
    Role-Based Access Control (RBAC) Manager for Operator, Engineer, and Admin roles.
    """
    ROLES = {
        "operator": ["view_dashboard", "view_fleet", "stream_data"],
        "engineer": ["view_dashboard", "view_fleet", "stream_data", "export_pdf", "trigger_alert", "view_recommendations"],
        "admin": ["view_dashboard", "view_fleet", "stream_data", "export_pdf", "trigger_alert", "view_recommendations", "upload_file", "retrain_model"]
    }

    @classmethod
    def check_permission(cls, role: str, action: str) -> bool:
        """Verifies if the specified role is authorized to perform the action."""
        allowed_actions = cls.ROLES.get(role.lower(), [])
        return action in allowed_actions


class AuditTrailLogger:
    """
    Immutable Audit Log database logger for recording critical platform events.
    """
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self._shared_conn = sqlite3.connect(db_path, check_same_thread=False) if db_path == ":memory:" else None
        self._init_db()

    def _get_connection(self):
        if self._shared_conn:
            return self._shared_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT
                )
            """)
            conn.commit()

    def log_action(self, username: str, role: str, action: str, details: str = "", ip_address: str = "127.0.0.1"):
        """Records an entry in the immutable audit log table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (timestamp, username, role, action, details, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (time.strftime("%Y-%m-%d %H:%M:%S"), username, role, action, details, ip_address))
            conn.commit()

    def get_recent_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries recent audit trail records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, username, role, action, details, ip_address 
                FROM audit_logs ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
        return [
            {"id": r[0], "timestamp": r[1], "username": r[2], "role": r[3], "action": r[4], "details": r[5], "ip_address": r[6]}
            for r in rows
        ]
