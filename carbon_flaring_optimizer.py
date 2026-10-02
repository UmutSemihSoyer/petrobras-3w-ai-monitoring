"""
Petrobras 3W AI Monitoring - Carbon Emissions & Gas Flaring Optimizer

This module calculates $CO_2$ and $CH_4$ emissions (in metric tons) during well instability or flaring events
and optimizes gas flow rates to minimize carbon tax penalties and environmental impact.
"""

from typing import Dict, Any

class CarbonFlaringOptimizer:
    """
    Environmental ESG Carbon Emissions and Gas Flaring Optimizer for offshore platforms.
    """
    # Emission Factors (EPA / IPCC Standards)
    CO2_PER_M3_GAS = 0.00193  # metric tons of CO2 per m^3 flared gas
    CH4_SLIP_FACTOR = 0.02    # 2% unburned methane slip in flare stack
    GWP_CH4 = 28.0            # Methane Global Warming Potential (28x CO2)
    CARBON_TAX_PER_TON_USD = 85.0  # International Carbon Tax ($/ton)

    def calculate_flaring_emissions(self, flared_gas_m3: float, flaring_duration_hours: float = 1.0) -> Dict[str, Any]:
        """
        Calculates metric tons of CO2, CH4, total CO2-equivalent (CO2e), and estimated carbon tax penalty.
        """
        co2_tons = flared_gas_m3 * self.CO2_PER_M3_GAS * (1.0 - self.CH4_SLIP_FACTOR)
        ch4_unburned_tons = flared_gas_m3 * self.CO2_PER_M3_GAS * self.CH4_SLIP_FACTOR
        co2_equivalent_tons = co2_tons + (ch4_unburned_tons * self.GWP_CH4)
        
        carbon_tax_cost_usd = co2_equivalent_tons * self.CARBON_TAX_PER_TON_USD
        
        # Recommendation to reduce flaring
        optimized_gas_rate_m3 = flared_gas_m3 * 0.75  # 25% reduction target
        potential_savings_usd = (co2_equivalent_tons * 0.25) * self.CARBON_TAX_PER_TON_USD

        return {
            "flared_gas_m3": flared_gas_m3,
            "flaring_duration_hours": flaring_duration_hours,
            "co2_emissions_tons": round(co2_tons, 3),
            "ch4_unburned_tons": round(ch4_unburned_tons, 4),
            "total_co2e_tons": round(co2_equivalent_tons, 2),
            "carbon_tax_cost_usd": round(carbon_tax_cost_usd, 2),
            "potential_savings_usd": round(potential_savings_usd, 2),
            "recommendation": "Flare stack yanma verimini artırmak için Gas Lift enjeksiyon manifoldunu dengeleyin ve gaz yakma oranını %25 düşürün."
        }
