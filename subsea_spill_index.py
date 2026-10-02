"""
Petrobras 3W AI Monitoring - Subsea Environmental Spill Risk Index Calculator

This module computes subsea oil leak risk score (0-100) based on annular pressure anomalies,
choke valve differential pressures, and DHSV seal integrity.
"""

from typing import Dict, Any

class SubseaSpillRiskCalculator:
    """
    Subsea Environmental Spill & Leak Risk Index Calculator.
    """
    @staticmethod
    def calculate_spill_risk(p_anular_pa: float, delta_p_ckp_pa: float, dhsv_status_open: bool = True) -> Dict[str, Any]:
        """
        Calculates subsea leak risk score (0-100) and environmental warning level.
        """
        risk_score = 0.0
        
        # Annular pressure elevation (>50 bar / 5e6 Pa) indicates casing leak
        if p_anular_pa > 5e6:
            risk_score += min(50.0, (p_anular_pa / 1e7) * 40.0)
            
        # High choke differential pressure (>80 bar / 8e6 Pa) indicates high blowout stress
        if delta_p_ckp_pa > 8e6:
            risk_score += min(30.0, (delta_p_ckp_pa / 1e7) * 25.0)

        # Closed or faulty DHSV increases wellhead stress
        if not dhsv_status_open:
            risk_score += 20.0

        risk_score = min(100.0, risk_score)
        
        if risk_score < 25:
            level = "SAFE"
            color = "#10B981"
        elif risk_score < 60:
            level = "MODERATE_SPILL_RISK"
            color = "#F59E0B"
        else:
            level = "CRITICAL_SUBSEA_LEAK_ALERT"
            color = "#EF4444"

        return {
            "spill_risk_score": round(risk_score, 1),
            "level": level,
            "color": color,
            "p_anular_bar": round(p_anular_pa / 1e5, 1),
            "delta_p_ckp_bar": round(delta_p_ckp_pa / 1e5, 1)
        }
