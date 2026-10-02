"""
Petrobras 3W AI Monitoring - MLOps Data Drift & Concept Drift Monitor

This module uses Kolmogorov-Smirnov (KS-Test) statistical testing to detect sensor data drift
and concept drift between historical training baseline and live operational data streams.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List

class DataDriftMonitor:
    """
    MLOps Data Drift Monitor using two-sample Kolmogorov-Smirnov statistical testing.
    """
    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold

    def detect_drift(self, baseline_sample: np.ndarray, current_sample: np.ndarray) -> Dict[str, Any]:
        """
        Calculates KS-statistic and p-value between baseline and live stream samples.
        """
        clean_base = baseline_sample[~np.isnan(baseline_sample)]
        clean_curr = current_sample[~np.isnan(current_sample)]
        
        if len(clean_base) < 5 or len(clean_curr) < 5:
            return {"drift_detected": False, "p_value": 1.0, "ks_stat": 0.0, "status": "INSUFFICIENT_DATA"}
            
        ks_stat, p_val = stats.ks_2samp(clean_base, clean_curr)
        drift_detected = bool(p_val < self.p_value_threshold)
        
        return {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": drift_detected,
            "status": "DRIFT_ALERT" if drift_detected else "NO_DRIFT"
        }
