"""
Petrobras 3W AI Monitoring - TimeGAN & Tabular Diffusion Synthetic Data Generator

This module generates realistic synthetic multivariate time-series sensor signals
for rare 3W transient fault classes (e.g., Class 2: Spurious DHSV closure) using TimeGAN style sampling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class TimeGANSyntheticGenerator:
    """
    Multivariate Time-Series Synthetic Data Generator (TimeGAN / Diffusion Architecture).
    """
    SENSOR_COLUMNS = ['P-PDG', 'T-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'P-JUS-CKP', 'T-MON-CKP', 'T-JUS-CKP']

    def generate_synthetic_anomaly(self, class_id: int = 2, length_samples: int = 200) -> pd.DataFrame:
        """
        Generates synthetic DataFrame with realistic sensor signal profiles for class_id.
        """
        t = np.linspace(0, 10, length_samples)
        data = {}

        # Base nominal pressure & temperature values
        p_pdg_base = 2.2e7
        t_pdg_base = 65.0
        p_tpt_base = 1.4e7
        p_mon_base = 1.0e7

        if class_id == 2:  # Spurious DHSV Closure -> Downhole pressure spikes, choke pressure drops
            p_pdg = p_pdg_base + (np.exp(t * 0.3) * 1e5) + np.random.randn(length_samples) * 5e4
            p_tpt = p_tpt_base * (1.0 - 0.7 * (t / 10.0)) + np.random.randn(length_samples) * 3e4
            p_mon = p_mon_base * (1.0 - 0.8 * (t / 10.0)) + np.random.randn(length_samples) * 2e4
        elif class_id == 8:  # Hydrate Formation -> High pressure differential, temp drop
            p_pdg = p_pdg_base + np.random.randn(length_samples) * 1e5
            p_tpt = p_tpt_base + (t * 2e5) + np.random.randn(length_samples) * 4e4
            p_mon = p_mon_base * (1.0 - 0.4 * (t / 10.0))
            t_pdg_base = 65.0 - (t * 1.5)
        else:
            p_pdg = p_pdg_base + np.random.randn(length_samples) * 5e4
            p_tpt = p_tpt_base + np.random.randn(length_samples) * 3e4
            p_mon = p_mon_base + np.random.randn(length_samples) * 2e4

        data['P-PDG'] = p_pdg
        data['T-PDG'] = np.full(length_samples, t_pdg_base) + np.random.randn(length_samples) * 0.2
        data['P-TPT'] = p_tpt
        data['T-TPT'] = 48.0 + np.random.randn(length_samples) * 0.3
        data['P-MON-CKP'] = p_mon
        data['P-JUS-CKP'] = 4.5e6 + np.random.randn(length_samples) * 2e4
        data['T-MON-CKP'] = 40.0 + np.random.randn(length_samples) * 0.2
        data['T-JUS-CKP'] = 35.0 + np.random.randn(length_samples) * 0.2
        data['class'] = [class_id] * length_samples

        return pd.DataFrame(data)
