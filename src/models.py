# ==============================================================================
# MLB QUANTITATIVE MODELS: MONEYLINE, TOTALS & GAUSSIAN COPULA SGP
# ==============================================================================
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm

LEAGUE_AVG_RUNS = 4.45

def calculate_expected_runs(off_rating: float, opp_pitcher_era: float, is_home: bool = False) -> float:
    """Calculates expected run baseline adjusted for pitching and park/home advantage."""
    pitcher_factor = opp_pitcher_era / 4.25
    hfa = 1.04 if is_home else 1.00
    return round(LEAGUE_AVG_RUNS * off_rating * pitcher_factor * hfa, 2)

def simulate_moneyline(away_runs: float, home_runs: float, n_sims: int = 10000) -> dict:
    """Runs Poisson Monte Carlo score simulations for fair win probability."""
    sim_a = np.random.poisson(away_runs, n_sims)
    sim_h = np.random.poisson(home_runs, n_sims)
    
    h_wins = np.sum(sim_h > sim_a)
    ties = np.sum(sim_h == sim_a)
    
    home_prob = (h_wins + 0.5 * ties) / n_sims
    away_prob = 1.0 - home_prob
    
    fair_home_ml = int(-100 * (home_prob / (1 - home_prob))) if home_prob >= 0.5 else int(100 * ((1 - home_prob) / home_prob))
    fair_away_ml = int(-100 * (away_prob / (1 - away_prob))) if away_prob >= 0.5 else int(100 * ((1 - away_prob) / away_prob))
    
    return {
        "home_win_prob": round(home_prob, 3),
        "away_win_prob": round(away_prob, 3),
        "fair_home_ml": f"{'+' if fair_home_ml > 0 else ''}{fair_home_ml}",
        "fair_away_ml": f"{'+' if fair_away_ml > 0 else ''}{fair_away_ml}"
    }

def simulate_sgp_copula(team_runs: float, opp_starter_k_line: float, rho: float = -0.32, n_sims: int = 10000) -> dict:
    """Evaluates joint probability of correlated SGP legs using Gaussian Copula."""
    mean = [0, 0]
    cov = [[1.0, rho], [rho, 1.0]]
    bivariate = np.random.multivariate_normal(mean, cov, n_sims)
    
    sim_r = poisson.ppf(norm.cdf(bivariate[:, 0]), team_runs)
    sim_k = poisson.ppf(norm.cdf(bivariate[:, 1]), opp_starter_k_line)
    
    hit_rate = np.mean((sim_r > 4.5) & (sim_k < opp_starter_k_line))
    fair_odds = int((1.0 / hit_rate - 1.0) * 100) if hit_rate < 0.5 else int(-100 / (hit_rate / (1 - hit_rate)))
    
    return {
        "hit_rate": round(hit_rate, 3),
        "fair_odds": f"{'+' if fair_odds > 0 else ''}{fair_odds}"
    }
