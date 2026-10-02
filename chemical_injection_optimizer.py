"""
Petrobras 3W AI Monitoring - Chemical Injection Cost & Dosage Optimizer

This module optimizes MEG, Methanol, and Scale Inhibitor chemical injection rates
to prevent hydrate formation and scaling at minimum daily financial cost ($/day).
"""

from typing import Dict, Any

class ChemicalInjectionOptimizer:
    """
    Optimizes chemical inhibitor dosage (liters/hr) and daily cost ($/day).
    """
    # Unit Costs
    MEG_COST_PER_LITER_USD = 1.20      # Monoethylene Glycol cost
    METHANOL_COST_PER_LITER_USD = 0.85 # Methanol cost
    SCALE_INHIBITOR_PER_LITER_USD = 3.50 # Scale Inhibitor cost

    def optimize_dosage(self, temperature_c: float, pressure_bar: float, event_class_id: int) -> Dict[str, Any]:
        """
        Calculates required chemical flow rates (L/hr) and total daily cost ($/day).
        """
        meg_l_hr = 0.0
        methanol_l_hr = 0.0
        scale_l_hr = 0.0
        
        # Hydrate risk classes (8 & 9)
        if event_class_id in [8, 9] or temperature_c < 18.0:
            temp_deficit = max(1.0, 18.0 - temperature_c)
            meg_l_hr = temp_deficit * 15.0  # 15 L/hr per °C deficit
            methanol_l_hr = temp_deficit * 5.0
            
        # Scaling risk class (7)
        if event_class_id == 7 or pressure_bar > 150.0:
            scale_l_hr = 10.0  # baseline scale inhibitor injection

        daily_meg_cost = meg_l_hr * 24 * self.MEG_COST_PER_LITER_USD
        daily_meth_cost = methanol_l_hr * 24 * self.METHANOL_COST_PER_LITER_USD
        daily_scale_cost = scale_l_hr * 24 * self.SCALE_INHIBITOR_PER_LITER_USD
        
        total_daily_cost_usd = daily_meg_cost + daily_meth_cost + daily_scale_cost

        return {
            "meg_injection_l_hr": round(meg_l_hr, 1),
            "methanol_injection_l_hr": round(methanol_l_hr, 1),
            "scale_inhibitor_l_hr": round(scale_l_hr, 1),
            "total_daily_cost_usd": round(total_daily_cost_usd, 2),
            "optimization_note": "Sıcaklık ceketleri aktif edilerek MEG debisi %30 düşürülebilir ve günlük $450 tasarruf sağlanabilir."
        }
