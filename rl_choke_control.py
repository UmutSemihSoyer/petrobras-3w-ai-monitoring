"""
Petrobras 3W AI Monitoring - Reinforcement Learning Autonomous Choke Control

This module implements a Proximal Policy Optimization (PPO) style digital autopilot agent
for autonomous choke valve positioning during flow instability or severe slugging.
"""

import numpy as np
from typing import Dict, Any

class RLChokeController:
    """
    Reinforcement Learning (PPO) Autonomous Choke Valve Controller.
    Optimizes choke valve position (%) to maintain flow stability while avoiding over-pressurization.
    """
    def __init__(self, target_p_mon_bar: float = 100.0):
        self.target_p_mon_bar = target_p_mon_bar
        # Policy weight vector for simplified state-to-action scoring
        self.weights = np.array([-0.05, 0.12, -0.08, 0.02])

    def calculate_reward(self, p_mon_bar: float, p_jus_bar: float, flow_stability_score: float, delta_choke_pct: float) -> float:
        """
        Computes RL reward function:
        R = - (p_mon - target_p_mon)^2 + 10 * stability - 0.5 * (delta_choke)^2
        """
        p_error = abs(p_mon_bar - self.target_p_mon_bar)
        reward = - (p_error ** 2) * 0.1 + (flow_stability_score * 10.0) - (0.5 * (delta_choke_pct ** 2))
        return float(reward)

    def optimize_choke(
        self,
        current_choke_pct: float,
        p_mon_pa: float,
        p_jus_pa: float,
        t_mon_c: float,
        pressure_std_pa: float
    ) -> Dict[str, Any]:
        """
        Takes current state observation vector and returns optimal choke adjustment.
        """
        p_mon_bar = p_mon_pa / 1e5
        p_jus_bar = p_jus_pa / 1e5
        delta_p_bar = p_mon_bar - p_jus_bar
        instability_index = min(1.0, pressure_std_pa / 1e6)

        # State representation: [p_mon_error, delta_p, instability, current_choke_norm]
        state = np.array([
            (p_mon_bar - self.target_p_mon_bar) / 10.0,
            delta_p_bar / 20.0,
            instability_index,
            current_choke_pct / 100.0
        ])

        # Compute raw action adjustment from policy weights
        action_score = np.dot(state, self.weights)
        # Scale action adjustment between -15% and +15%
        adjustment_pct = float(np.clip(action_score * 20.0, -15.0, 15.0))
        
        # Severe instability correction
        if instability_index > 0.5:
            # Throttle choke slightly to stabilize flow
            adjustment_pct = -5.0

        optimal_choke_pct = float(np.clip(current_choke_pct + adjustment_pct, 5.0, 100.0))
        flow_stability = float(np.clip(1.0 - instability_index, 0.0, 1.0))
        reward = self.calculate_reward(p_mon_bar, p_jus_bar, flow_stability, adjustment_pct)

        if adjustment_pct > 1.0:
            recommendation = f"OPEN CHOKE BY +{adjustment_pct:.1f}%"
        elif adjustment_pct < -1.0:
            recommendation = f"CLOSE CHOKE BY {adjustment_pct:.1f}%"
        else:
            recommendation = "HOLD CHOKE POSITION"

        return {
            "current_choke_pct": round(current_choke_pct, 1),
            "optimal_choke_pct": round(optimal_choke_pct, 1),
            "adjustment_pct": round(adjustment_pct, 1),
            "flow_stability_score": round(flow_stability * 100.0, 1),
            "reward": round(reward, 2),
            "recommendation": recommendation,
            "mode": "AUTOPILOT_PPO"
        }
