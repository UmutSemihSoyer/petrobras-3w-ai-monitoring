"""
Petrobras 3W AI Monitoring - TimescaleDB / Time-Series Database Manager

This module handles persistent storage of high-frequency offshore sensor telemetry
and anomaly diagnostic logs using TimescaleDB (PostgreSQL time-series extension).
"""

import time
import sqlite3
from typing import List, Dict, Any, Optional

class TimescaleDBStore:
    """
    Time-Series Database Store providing hypertable-compatible storage
    for sensor telemetry and historical anomaly predictions.
    """
    def __init__(self, db_path: str = "petrobras3w_timeseries.db"):
        self.db_path = db_path
        self._shared_conn = sqlite3.connect(db_path, check_same_thread=False) if db_path == ":memory:" else None
        self._init_db()

    def _get_connection(self):
        if self._shared_conn:
            return self._shared_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes time-series schema tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    well_id TEXT NOT NULL,
                    p_pdg REAL,
                    t_pdg REAL,
                    p_tpt REAL,
                    t_tpt REAL,
                    p_mon_ckp REAL,
                    p_jus_ckp REAL,
                    t_mon_ckp REAL,
                    t_jus_ckp REAL,
                    created_at REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    well_id TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_used TEXT NOT NULL
                )
            """)
            conn.commit()

    def insert_telemetry(self, well_id: str, data: Dict[str, Any]):
        """Inserts a single sensor reading record into the time-series table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sensor_telemetry 
                (timestamp, well_id, p_pdg, t_pdg, p_tpt, t_tpt, p_mon_ckp, p_jus_ckp, t_mon_ckp, t_jus_ckp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                well_id,
                data.get("P-PDG"),
                data.get("T-PDG"),
                data.get("P-TPT"),
                data.get("T-TPT"),
                data.get("P-MON-CKP"),
                data.get("P-JUS-CKP"),
                data.get("T-MON-CKP"),
                data.get("T-JUS-CKP"),
                time.time()
            ))
            conn.commit()

    def log_anomaly(self, well_id: str, pred_info: Dict[str, Any]):
        """Logs diagnosed anomaly event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO anomaly_logs 
                (timestamp, well_id, class_id, class_name, severity, confidence, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                well_id,
                pred_info.get("pred_class_id", 0),
                pred_info.get("pred_class_name", "Unknown"),
                pred_info.get("severity", "Normal"),
                pred_info.get("confidence", 95.0),
                pred_info.get("model_used", "XGBoost")
            ))
            conn.commit()

    def get_recent_history(self, well_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Queries recent time-series telemetry records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, p_pdg, t_pdg, p_tpt, t_tpt, p_mon_ckp, p_jus_ckp 
                FROM sensor_telemetry 
                WHERE well_id = ? 
                ORDER BY id DESC LIMIT ?
            """, (well_id, limit))
            rows = cursor.fetchall()
            
        return [
            {
                "timestamp": r[0], "P-PDG": r[1], "T-PDG": r[2], 
                "P-TPT": r[3], "T-TPT": r[4], "P-MON-CKP": r[5], "P-JUS-CKP": r[6]
            }
            for r in reversed(rows)
        ]
