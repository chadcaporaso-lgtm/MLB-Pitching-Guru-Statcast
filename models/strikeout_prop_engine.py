import math
import numpy as np
from scipy.stats import nbinom

def calculate_postseason_k_distribution(
    projected_outs: float,
    pitcher_k_pct: float,
    lineup_k_pct: float,
    league_k_pct: float = 0.224,
    r_dispersion: float = 16.0
) -> dict:
    """
    Negative Binomial Postseason Strikeout Prop Engine.
    Adjusts batters faced (BF) to postseason hook thresholds and tightens right tail.
    """
    # 1. Postseason Batter Faced Estimate (BF = Outs + Baserunners)
    # Average ~0.33 baserunners per out in postseason traffic
    projected_bf = projected_outs * 1.36
    
    # 2. Log-Odds Ratio Matchup K% with Postseason Contact Adjustment (-2.0%)
    odds_p = pitcher_k_pct / (1.0 - pitcher_k_pct)
    odds_opp = lineup_k_pct / (1.0 - lineup_k_pct)
    odds_lg = league_k_pct / (1.0 - league_k_pct)
    
    combined_odds = (odds_p * odds_opp) / odds_lg
    raw_matchup_k = combined_odds / (1.0 + combined_odds)
    adj_matchup_k = max(0.12, raw_matchup_k - 0.020)  # Postseason disciplined approach penalty
    
    # 3. Mean Expected Strikeouts (mu)
    mu = projected_bf * adj_matchup_k
    
    # 4. Negative Binomial Parameterization (mean = mu, r = r_dispersion)
    # p = r / (r + mu)
    p_nbinom = r_dispersion / (r_dispersion + mu)
    
    # Discrete PMF across common prop thresholds
    prob_o3_5 = 1.0 - nbinom.cdf(3, r_dispersion, p_nbinom)
    prob_o4_5 = 1.0 - nbinom.cdf(4, r_dispersion, p_nbinom)
    prob_o5_5 = 1.0 - nbinom.cdf(5, r_dispersion, p_nbinom)
    prob_o6_5 = 1.0 - nbinom.cdf(6, r_dispersion, p_nbinom)
    prob_o7_5 = 1.0 - nbinom.cdf(7, r_dispersion, p_nbinom)
    
    return {
        'projected_bf': round(projected_bf, 1),
        'adj_k_pct': round(adj_matchup_k, 4),
        'expected_ks_mean': round(mu, 2),
        'prob_over_3_5': round(float(prob_o3_5), 4),
        'prob_over_4_5': round(float(prob_o4_5), 4),
        'prob_over_5_5': round(float(prob_o5_5), 4),
        'prob_over_6_5': round(float(prob_o6_5), 4),
        'prob_over_7_5': round(float(prob_o7_5), 4),
        'prob_under_4_5': round(float(1.0 - prob_o4_5), 4),
        'prob_under_5_5': round(float(1.0 - prob_o5_5), 4),
        'prob_under_6_5': round(float(1.0 - prob_o6_5), 4)
    }
