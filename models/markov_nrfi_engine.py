import numpy as np

def simulate_postseason_half_inning_nrfi(
    starter_whiff_rate: float,
    starter_bb_rate: float,
    top3_obp: float,
    top3_iso: float,
    umpire_zone_factor: float = 0.985,  # Tighter horizontal zone in postseason
    max_effort_boost: float = 0.020      # Max intent in 1st inning
) -> float:
    """
    Computes single-team half-inning zero-run probability using calibrated
    1st-inning postseason transition rates.
    """
    # 1. Calibrate 1st Inning Whiff and Walk Probabilities
    effective_whiff = (starter_whiff_rate + max_effort_boost) * umpire_zone_factor
    effective_bb = starter_bb_rate / umpire_zone_factor
    
    # 2. Base-Out Transition Probabilities for First 3-4 Batters
    p_k = effective_whiff * 0.88
    p_bb = effective_bb
    p_batted_ball = max(0.05, 1.0 - (p_k + p_bb))
    
    # 3. Hit / Out distribution on contact against elite top-of-order bats
    p_hit = p_batted_ball * (top3_obp - effective_bb) / max(0.01, 1.0 - effective_bb)
    p_extra_base = p_hit * (top3_iso / max(0.01, top3_obp))
    p_single = max(0.0, p_hit - p_extra_base)
    p_field_out = max(0.0, p_batted_ball - p_hit)
    
    # Transition: Probability of 3 outs recorded before 1 run scores
    # Analytical Markov absorption approximation for half-inning:
    p_clean_3up_3down = (p_k + p_field_out) ** 3
    p_1runner_0runs = 3.0 * (p_single + p_bb) * ((p_k + p_field_out) ** 3) * 0.72
    p_2runners_0runs = 3.0 * ((p_single + p_bb) ** 2) * ((p_k + p_field_out) ** 3) * 0.28
    
    half_inning_zero_runs = p_clean_3up_3down + p_1runner_0runs + p_2runners_0runs
    return min(0.92, max(0.48, float(half_inning_zero_runs)))

def calculate_postseason_nrfi(away_half_prob: float, home_half_prob: float) -> dict:
    nrfi_prob = away_half_prob * home_half_prob
    yrfi_prob = 1.0 - nrfi_prob
    
    fair_nrfi_american = int(round(-100.0 / (1.0 - nrfi_prob))) if nrfi_prob > 0.5 else int(round((1.0 - nrfi_prob) / nrfi_prob * 100.0))
    fair_yrfi_american = int(round(-100.0 / (1.0 - yrfi_prob))) if yrfi_prob > 0.5 else int(round((1.0 - yrfi_prob) / yrfi_prob * 100.0))
    
    return {
        'nrfi_prob': round(nrfi_prob, 4),
        'yrfi_prob': round(yrfi_prob, 4),
        'fair_nrfi_american': fair_nrfi_american,
        'fair_yrfi_american': fair_yrfi_american
    }
