"""
Petrobras 3W AI Monitoring - Economic Loss & Financial Downtime Calculator

This module calculates real-time financial production revenue loss ($ USD)
and lost oil barrels (bpd) during well transient anomalies or downtime events.
"""

from typing import Dict, Any

class EconomicLossCalculator:
    """
    Financial downtime revenue loss calculator for offshore oil wells.
    """
    def __init__(self, oil_price_usd_per_barrel: float = 75.0, avg_well_bpd: float = 5000.0):
        self.oil_price = oil_price_usd_per_barrel
        self.avg_well_bpd = avg_well_bpd

    def calculate_loss(self, duration_minutes: float, productivity_loss_pct: float = 100.0) -> Dict[str, Any]:
        """
        Calculates lost barrels and financial USD revenue loss for a given duration.
        """
        duration_hours = duration_minutes / 60.0
        barrels_per_hour = self.avg_well_bpd / 24.0
        
        lost_barrels = barrels_per_hour * duration_hours * (productivity_loss_pct / 100.0)
        lost_revenue_usd = lost_barrels * self.oil_price
        
        return {
            "duration_minutes": round(duration_minutes, 1),
            "duration_hours": round(duration_hours, 2),
            "lost_barrels": round(lost_barrels, 1),
            "lost_revenue_usd": round(lost_revenue_usd, 2),
            "oil_price_usd": self.oil_price,
            "productivity_loss_pct": productivity_loss_pct
        }
