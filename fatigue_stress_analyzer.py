"""
Petrobras 3W AI Monitoring - Cumulative Fatigue Stress Analyzer

This module evaluates mechanical fatigue stress index on choke valves and well risers
caused by transient pressure surges and thermal expansion cycles (Rainflow counting approximation).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class FatigueStressAnalyzer:
    """
    Cumulative mechanical stress analyzer for well piping and valves.
    """
    @staticmethod
    def compute_stress_index(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes mechanical fatigue stress score based on pressure/temperature variance.
        """
        stress_score = 0.0
        details = {}
        
        if 'P-PDG' in df.columns and not df['P-PDG'].dropna().empty:
            p_diff = np.diff(df['P-PDG'].dropna().values) / 1e5  # in bar
            p_cycles = np.sum(np.abs(p_diff) > 2.0)  # pressure spikes > 2 bar
            stress_score += p_cycles * 1.5
            details['pressure_spikes'] = int(p_cycles)

        if 'T-TPT' in df.columns and not df['T-TPT'].dropna().empty:
            t_diff = np.diff(df['T-TPT'].dropna().values)  # in °C
            t_cycles = np.sum(np.abs(t_diff) > 0.5)  # thermal fluctuations
            stress_score += t_cycles * 1.0
            details['thermal_cycles'] = int(t_cycles)

        fatigue_index = min(100.0, stress_score)
        level = "LOW" if fatigue_index < 30 else ("MEDIUM" if fatigue_index < 70 else "HIGH_FATIGUE")

        return {
            "fatigue_index": round(fatigue_index, 1),
            "level": level,
            "details": details
        }
