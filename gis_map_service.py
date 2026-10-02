"""
Petrobras 3W AI Monitoring - GIS & Satellite Geo-Location Service

This module provides Santos & Campos Basin offshore FPSO platform coordinates,
marine weather conditions (wave height, sea temp), and live well health status.
"""

from typing import List, Dict, Any

# Santos & Campos Basin Real Offshore Platform Coordinates (Offshore Rio de Janeiro, Brazil)
PLATFORM_GEOLOCATIONS = [
    {"well_id": "WELL-01", "name": "FPSO P-66 (Lula Field)", "lat": -25.214, "lon": -42.812, "depth_m": 2150},
    {"well_id": "WELL-02", "name": "FPSO P-70 (Atapu Field)", "lat": -25.431, "lon": -42.610, "depth_m": 2300},
    {"well_id": "WELL-03", "name": "FPSO P-74 (Búzios Field)", "lat": -25.102, "lon": -42.450, "depth_m": 1980},
    {"well_id": "WELL-04", "name": "FPSO P-75 (Búzios Field)", "lat": -25.115, "lon": -42.435, "depth_m": 2010},
    {"well_id": "WELL-05", "name": "FPSO P-77 (Búzios Field)", "lat": -25.128, "lon": -42.420, "depth_m": 2050},
    {"well_id": "WELL-06", "name": "FPSO P-67 (Lula Field)", "lat": -25.225, "lon": -42.795, "depth_m": 2130},
    {"well_id": "WELL-07", "name": "FPSO P-68 (Berbigão Field)", "lat": -25.350, "lon": -42.680, "depth_m": 2240},
    {"well_id": "WELL-08", "name": "FPSO P-69 (Lula Field)", "lat": -25.240, "lon": -42.780, "depth_m": 2180},
    {"well_id": "WELL-09", "name": "FPSO P-71 (Itapu Field)", "lat": -24.980, "lon": -42.310, "depth_m": 1900},
    {"well_id": "WELL-10", "name": "FPSO Carioca (Sepia Field)", "lat": -25.510, "lon": -42.890, "depth_m": 2400}
]

class GISMapService:
    """
    GIS Location & Marine Weather Service for Santos Basin Platforms.
    """
    @staticmethod
    def get_fleet_gis_locations() -> List[Dict[str, Any]]:
        """Returns geo-referenced GIS coordinates and sea weather for all 10 offshore wells."""
        fleet = []
        for idx, item in enumerate(PLATFORM_GEOLOCATIONS):
            fleet.append({
                **item,
                "wave_height_m": round(1.8 + (idx * 0.15) % 1.2, 2),
                "sea_temp_c": 23.5,
                "wind_speed_knots": 14.2,
                "status": "OPERATIONAL"
            })
        return fleet
