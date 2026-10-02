"""
Petrobras 3W AI Monitoring - Predictive Maintenance & Remaining Useful Life (RUL) Engine

This module estimates equipment wear degradation and Remaining Useful Life (RUL in hours)
for choke valves, DHSVs, and wellhead transmitters using Weibull survival analysis.
"""

import numpy as np
from typing import Dict, Any

class RULPredictor:
    """
    Predictive Maintenance RUL calculator based on cumulative pressure/temperature stress.
    """
    def __init__(self, beta_shape: float = 2.5, eta_scale: float = 5000.0):
        self.beta = beta_shape  # Weibull shape parameter
        self.eta = eta_scale    # Scale parameter (rated operating hours)

    def calculate_rul(self, operating_hours: float, current_stress_factor: float = 1.0) -> Dict[str, Any]:
        """
        Calculates remaining operational hours and survival probability.
        """
        effective_hours = operating_hours * current_stress_factor
        # Weibull Cumulative Failure Probability F(t)
        failure_prob = 1.0 - np.exp(-((effective_hours / self.eta) ** self.beta))
        survival_prob = 1.0 - failure_prob
        
        # Mean Remaining Useful Life
        rul_hours = max(0.0, (self.eta - effective_hours) / (current_stress_factor + 1e-5))
        degradation_pct = min(100.0, (effective_hours / self.eta) * 100.0)
        
        status = "HEALTHY" if degradation_pct < 60 else ("MAINTENANCE_REQUIRED" if degradation_pct < 85 else "CRITICAL_REPLACEMENT")
        
        return {
            "operating_hours": operating_hours,
            "effective_hours": round(effective_hours, 1),
            "rul_hours": round(rul_hours, 1),
            "rul_days": round(rul_hours / 24.0, 1),
            "degradation_pct": round(degradation_pct, 1),
            "survival_prob": round(survival_prob * 100, 1),
            "status": status
        }
