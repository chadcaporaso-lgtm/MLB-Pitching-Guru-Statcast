import math
import numpy as np

def calculate_october_air_density(temp_f: float, elevation_ft: float, relative_humidity: float = 0.55, pressure_inhg: float = 29.92) -> float:
    """Calculates thermodynamic air density (kg/m^3) using Tetens vapor formulation."""
    temp_c = (temp_f - 32.0) * (5.0 / 9.0)
    temp_k = temp_c + 273.15
    p_pa = (pressure_inhg * 3386.39) * math.exp(-elevation_ft / 27700.0)
    e_sat = 610.78 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    pv = relative_humidity * e_sat
    pd = p_pa - pv
    return (pd / (287.058 * temp_k)) + (pv / (461.495 * temp_k))

def calculate_postseason_hr_probability(
    batter_barrel_rate: float,
    batter_pull_air_rate: float,
    batter_ev90: float,
    batter_ev_max: float,
    pitcher_barrel_allowed: float,
    temp_f: float = 58.0,
    elevation_ft: float = 50.0,
    park_hr_mult: float = 1.0,
    bullpen_discount: float = 0.85
) -> dict:
    """
    Calibrated Multi-Stage Conditional Survival Home Run Engine (Postseason Edition).
    Applies cold air drag penalties and a late-inning high-leverage bullpen discount.
    """
    # 1. League Baseline Mean-Centering
    lg_barrel = 0.068
    lg_pull_air = 0.220
    barrel_diff = batter_barrel_rate - lg_barrel
    pull_diff = batter_pull_air_rate - lg_pull_air

    # 2. Power Bonuses
    ev90_bonus = max(0.0, batter_ev90 - 100.0) * 0.038
    evmax_bonus = max(0.0, batter_ev_max - 108.0) * 0.022

    # 3. October Air Density Drag
    rho = calculate_october_air_density(temp_f, elevation_ft)
    rho_baseline = 1.185  # Standard summer baseline
    density_penalty = (rho - rho_baseline) * 2.20  # Suppresses logit fence clearance

    # 4. Base Logit Formulation (Postseason intercept = -3.52)
    logit = (-3.52 
             + 2.85 * barrel_diff 
             + 1.65 * pull_diff 
             + ev90_bonus 
             + evmax_bonus 
             - density_penalty 
             + 0.95 * (pitcher_barrel_allowed - lg_barrel))

    p_hr_bbe = 1.0 / (1.0 + math.exp(-logit))
    bbe_per_game = 3.10

    # 5. Full-Game Probability (PAs 1-2 vs starter, PAs 3-4 vs high-leverage pen)
    p_starter_phase = 1.0 - (1.0 - p_hr_bbe * park_hr_mult) ** 2.0
    p_pen_phase = 1.0 - (1.0 - p_hr_bbe * park_hr_mult * bullpen_discount) ** 1.10
    single_game_prob = 1.0 - ((1.0 - p_starter_phase) * (1.0 - p_pen_phase))
    single_game_prob = min(0.285, max(0.025, single_game_prob))

    fair_odds = -100.0 / (1.0 - single_game_prob) if single_game_prob < 0.5 else 100.0 * single_game_prob / (1.0 - single_game_prob)

    return {
        'single_game_prob': round(single_game_prob, 4),
        'fair_american': int(round((1.0 - single_game_prob) / single_game_prob * 100.0)),
        'air_density_kg_m3': round(rho, 4),
        'drag_impact': "SUPPRESSED" if rho > 1.200 else "NEUTRAL"
    }
