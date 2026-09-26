"""
Statcast Pitch Arsenal & Matchup Whiff Engine
---------------------------------------------
Computes weighted expected plate-appearance whiff probabilities by cross-referencing
pitcher arsenal usage by batter handedness against individual batter pitch-type whiffs.
Generates automated mismatch alerts and strikeout multiplier factors for K props and NRFIs.
"""

from typing import Dict, List, Any

PITCH_NAME_MAP = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter"
}

class ArsenalWhiffEngine:
    def __init__(self, league_avg_whiff: float = 0.245):
        self.league_avg_whiff = league_avg_whiff

    def compute_matchup_whiff(
        self,
        pitcher_name: str,
        batter_name: str,
        pitcher_hand: str,
        batter_hand: str,
        pitcher_arsenal: Dict[str, float],      # e.g., {"FF": 0.32, "ST": 0.23}
        batter_pitch_whiffs: Dict[str, float]   # e.g., {"FF": 0.30, "ST": 0.29}
    ) -> Dict[str, Any]:
        """
        Calculates the weighted expected whiff rate for a specific batter-pitcher matchup.
        """
        arsenal_breakdown = []
        weighted_whiff_sum = 0.0
        total_usage_accounted = 0.0
        alerts = []

        for pitch_code, usage in pitcher_arsenal.items():
            pitch_label = PITCH_NAME_MAP.get(pitch_code, pitch_code)
            batter_whiff = batter_pitch_whiffs.get(pitch_code, self.league_avg_whiff)
            
            weighted_whiff_sum += (usage * batter_whiff)
            total_usage_accounted += usage

            alert_tag = None
            if usage >= 0.25 and batter_whiff >= 0.28:
                alert_tag = f"High Whiff Vulnerability vs primary {pitch_label}"
                alerts.append(f"{batter_name} has significant trouble with {pitch_label.lower()} ({int(batter_whiff*100)}% whiff vs {int(usage*100)}% usage).")
            elif usage >= 0.20 and batter_whiff <= 0.14:
                alert_tag = f"Batter Advantage vs {pitch_label}"
                alerts.append(f"{batter_name} makes high contact against {pitch_label.lower()} ({int((1-batter_whiff)*100)}% contact rate).")

            arsenal_breakdown.append({
                "pitch_code": pitch_code,
                "pitch_name": pitch_label,
                "usage_pct": round(usage * 100.0, 1),
                "batter_whiff_pct": round(batter_whiff * 100.0, 1),
                "alert": alert_tag
            })

        expected_whiff_rate = (weighted_whiff_sum / total_usage_accounted) if total_usage_accounted > 0 else self.league_avg_whiff
        k_multiplier = round(expected_whiff_rate / self.league_avg_whiff, 4)

        return {
            "matchup": f"{batter_name} ({batter_hand}) vs {pitcher_name} ({pitcher_hand})",
            "expected_whiff_rate": round(expected_whiff_rate, 4),
            "league_avg_whiff": self.league_avg_whiff,
            "whiff_delta": round(expected_whiff_rate - self.league_avg_whiff, 4),
            "k_multiplier": k_multiplier,
            "alerts": alerts,
            "arsenal_details": sorted(arsenal_breakdown, key=lambda x: x["usage_pct"], reverse=True)
        }
