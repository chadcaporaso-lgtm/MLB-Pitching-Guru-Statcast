import numpy as np
import pandas as pd

def calculate_leverage_weighted_bfi(reliever_logs: pd.DataFrame, target_game_time: str) -> float:
    """
    Computes Leverage-Weighted Bullpen Fatigue Index (BFI) via rolling 72-hour
    exponential decay weighted by Game Leverage Index (gmLI).
    """
    if reliever_logs.empty:
        return 1.0  # Baseline neutral rest
        
    target_dt = pd.to_datetime(target_game_time)
    df = reliever_logs.copy()
    df["dt"] = pd.to_datetime(df["timestamp"])
    df["hours_ago"] = (target_dt - df["dt"]).dt.total_seconds() / 3600.0
    
    # Filter to rolling 72-hour window
    l3d = df[(df["hours_ago"] >= 0) & (df["hours_ago"] <= 72.0)].copy()
    if l3d.empty:
        return 0.80  # Fully rested bullpen
        
    # High-leverage stress multiplier: pitches * (1.0 + 0.65 * ln(1 + gmLI))
    gmli = l3d["avg_gmLI"].clip(lower=0.1, upper=4.0)
    stress_multiplier = 1.0 + 0.65 * np.log(1.0 + gmli)
    exertion_units = l3d["pitches_thrown"] * stress_multiplier
    
    # Exponential decay half-life ~ 24 hours
    decay_weights = np.exp(-l3d["hours_ago"] / 24.0)
    active_load = np.sum(exertion_units * decay_weights)
    
    # 70 units represents league-average 3-day load
    normalized_bfi = np.clip(active_load / 70.0, 0.40, 2.50)
    return round(float(normalized_bfi), 3)
