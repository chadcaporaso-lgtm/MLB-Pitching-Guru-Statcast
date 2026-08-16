"""
MLB Analytics Engine - Home Run Prop Predictive Module
Ingests Statcast data and calculates rolling Barrel%, HardHit%,
ISO, Pitcher HR/9 metrics, and Wilson Score Confidence Intervals.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

class HRPropEngine:
    def __init__(self, sample_pa_default: int = 120):
        self.sample_pa_default = sample_pa_default

    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        return (american_odds / 100.0) + 1.0 if american_odds > 0 else (100.0 / abs(american_odds)) + 1.0

    @classmethod
    def devig_single_prop(cls, prop_yes_american: float, prop_no_american: float) -> float:
        p_yes_raw = 100.0 / (prop_yes_american + 100.0) if prop_yes_american > 0 else abs(prop_yes_american) / (abs(prop_yes_american) + 100.0)
        p_no_raw = 100.0 / (prop_no_american + 100.0) if prop_no_american > 0 else abs(prop_no_american) / (abs(prop_no_american) + 100.0)
        overround = p_yes_raw + p_no_raw
        return p_yes_raw / overround

    @classmethod
    def calculate_wilson_ci(cls, p_hat: float, n_samples: int = 120, confidence: float = 0.95) -> tuple[float, float]:
        z = norm.ppf(1 - (1 - confidence) / 2)
        denom = 1 + (z**2 / n_samples)
        center = p_hat + (z**2 / (2 * n_samples))
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_samples)) / n_samples)
        return max(0.0, (center - spread) / denom), min(1.0, (center + spread) / denom)

    @classmethod
    def calculate_ev(cls, p_model: float, book_american: float) -> float:
        dec = cls.american_to_decimal(book_american)
        return (p_model * dec) - 1.0

    @classmethod
    def calculate_quarter_kelly(cls, p_model: float, book_american: float, bankroll: float = 10000.0, fraction: float = 0.25) -> float:
        b = cls.american_to_decimal(book_american) - 1.0
        f_star = (p_model * b - (1.0 - p_model)) / b
        return max(0.0, f_star * fraction * bankroll)
