"""
Petrobras 3W AI Monitoring - Scenario A/B Root Cause Simulator

This module simulates two alternative engineering interventions (Scenario A vs Scenario B)
for well remediation during anomalies, comparing pressure stability, financial costs, and recovery times.
"""

from typing import Dict, Any

class ScenarioABSimulator:
    """
    Comparative A/B Scenario Simulator for Well Remediation.
    """
    def simulate_comparison(
        self,
        current_class_id: int,
        p_pdg_bar: float,
        p_mon_bar: float,
        t_mon_c: float
    ) -> Dict[str, Any]:
        """
        Simulates outcome of two remediation paths.
        Scenario A: Choke Valve Adjustment (+10% opening)
        Scenario B: Chemical Injection (MEG / Methanol + Choke -5%)
        """
        # Baseline simulation for Scenario A (Choke opening increase)
        scen_a_p_mon = p_mon_bar * 0.92  # 8% pressure drop
        scen_a_stability = 88.0
        scen_a_cost_usd = 120.0          # Operational valve adjust cost
        scen_a_recovery_min = 25.0
        scen_a_production_gain_bpd = 350.0

        # Baseline simulation for Scenario B (Chemical dosage + slight choke throttle)
        scen_b_p_mon = p_mon_bar * 0.96
        scen_b_stability = 95.5          # Higher flow stability
        scen_b_cost_usd = 480.0          # Chemical cost
        scen_b_recovery_min = 12.0       # Faster hydrate dissolution
        scen_b_production_gain_bpd = 520.0

        # Decide recommended scenario based on event class
        if current_class_id in [7, 8, 9]:  # Scaling / Hydrates / Slugging
            recommended = "SCENARIO_B"
            reason = "Kimyasal enjeksiyon (MEG) ve choke kısıntısı hydrate kristallerini 12 dakikada çözer ve daha yüksek kararlılık sağlar."
        else:
            recommended = "SCENARIO_A"
            reason = "Vana açıklığı ayarı maliyetsiz ve doğrudan basınç düşüşü sağlayarak 25 dakikada hedefe ulaşır."

        return {
            "current_class_id": current_class_id,
            "scenario_a": {
                "name": "Senaryo A: Choke Açıklığı +%10",
                "predicted_p_mon_bar": round(scen_a_p_mon, 1),
                "stability_score": scen_a_stability,
                "cost_usd": scen_a_cost_usd,
                "recovery_time_min": scen_a_recovery_min,
                "production_gain_bpd": scen_a_production_gain_bpd
            },
            "scenario_b": {
                "name": "Senaryo B: MEG Enjeksiyonu + Choke -%5",
                "predicted_p_mon_bar": round(scen_b_p_mon, 1),
                "stability_score": scen_b_stability,
                "cost_usd": scen_b_cost_usd,
                "recovery_time_min": scen_b_recovery_min,
                "production_gain_bpd": scen_b_production_gain_bpd
            },
            "recommended_scenario": recommended,
            "recommendation_reason": reason
        }
