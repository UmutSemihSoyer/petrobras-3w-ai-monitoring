"""
Petrobras 3W AI Monitoring - Emergency Shutdown (ESD) Safety Lock System

This module handles automated high-integrity protection system (HIPPS / ESD) triggers,
evaluating wellhead pressures, fire/gas risk indices, and subsea integrity to issue safety interlock signals.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone

class EmergencyShutdownSystem:
    """
    Emergency Shutdown (ESD) Safety Lock Controller.
    Levels:
    - Level 0: Normal Operation
    - Level 1: Wellhead Choke Isolation
    - Level 2: Platform Subsea Safety Valve (DHSV) Shut-in
    - Level 3: Emergency Platform Facility Trip (Total Shutdown)
    """
    MAX_SAFE_P_PDG_PA = 35e7  # 350 bar
    MAX_SAFE_P_MON_PA = 25e7  # 250 bar
    CRITICAL_TEMP_C = 120.0

    def evaluate_esd_status(
        self,
        well_id: str,
        p_pdg_pa: float,
        p_mon_ckp_pa: float,
        t_mon_ckp_c: float,
        gas_detector_ppm: float = 0.0,
        manual_override_bypass: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates sensor readings against critical safety thresholds and returns ESD trip status.
        """
        triggers: List[str] = []
        esd_level = 0
        
        if p_pdg_pa > self.MAX_SAFE_P_PDG_PA:
            triggers.append(f"CATASTROPHIC DOWNHOLE PRESSURE ({p_pdg_pa / 1e5:.1f} bar > 350 bar)")
            esd_level = max(esd_level, 2)
            
        if p_mon_ckp_pa > self.MAX_SAFE_P_MON_PA:
            triggers.append(f"CRITICAL WELLHEAD PRESSURE ({p_mon_ckp_pa / 1e5:.1f} bar > 250 bar)")
            esd_level = max(esd_level, 1)

        if t_mon_ckp_c > self.CRITICAL_TEMP_C:
            triggers.append(f"HIGH TEMPERATURE SURGE ({t_mon_ckp_c:.1f}°C > 120°C)")
            esd_level = max(esd_level, 1)

        if gas_detector_ppm > 500.0:
            triggers.append(f"SUBSEA GAS LEAK DETECTED ({gas_detector_ppm:.0f} PPM)")
            esd_level = max(esd_level, 3)

        if manual_override_bypass:
            status = "BYPASSED_BY_OPERATOR"
            tripped = False
        elif esd_level > 0:
            status = f"TRIPPED_LEVEL_{esd_level}"
            tripped = True
        else:
            status = "SYSTEM_SAFE"
            tripped = False

        return {
            "well_id": well_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tripped": tripped,
            "esd_level": esd_level,
            "status": status,
            "triggers": triggers,
            "dhsv_action": "CLOSE_IMMEDIATE" if esd_level >= 2 else ("CLOSE_CHOKE" if esd_level == 1 else "OPEN"),
            "manual_override_bypass": manual_override_bypass
        }
