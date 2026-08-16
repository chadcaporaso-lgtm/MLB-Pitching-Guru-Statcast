# ==============================================================================
# MLB QUANTITATIVE MODELS: MONEYLINE, TOTALS, RUN LINE & GAUSSIAN COPULA SGP
# ==============================================================================
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm

LEAGUE_AVG_RUNS = 4.45

def calculate_expected_runs(off_rating: float, opp_pitcher_era: float, is_home: bool = False) -> float:
    pitcher_factor = opp_pitcher_era / 4.25
    hfa = 1.04 if is_home else 1.00
    return round(LEAGUE_AVG_RUNS * off_rating * pitcher_factor * hfa, 2)

def simulate_moneyline(away_runs: float, home_runs: float, n_sims: int = 10000) -> dict:
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

def simulate_run_line(away_runs: float, home_runs: float, n_sims: int = 15000) -> dict:
    sim_a = np.random.poisson(away_runs, n_sims)
    sim_h = np.random.poisson(home_runs, n_sims)
    
    # Extra innings tiebreaker resolution
    tie_indices = np.where(sim_a == sim_h)[0]
    for idx in tie_indices:
        if np.random.rand() > 0.5:
            sim_h[idx] += 1
        else:
            sim_a[idx] += 1

    diff = sim_h - sim_a

    home_minus_1_5_prob = np.mean(diff >= 2)
    away_plus_1_5_prob = 1.0 - home_minus_1_5_prob

    away_minus_1_5_prob = np.mean(diff <= -2)
    home_plus_1_5_prob = 1.0 - away_minus_1_5_prob

    def to_american(prob):
        if prob <= 0.001: return "+9999"
        if prob >= 0.999: return "-9999"
        odds = int(-100 * (prob / (1 - prob))) if prob >= 0.5 else int(100 * ((1 - prob) / prob))
        return f"{'+' if odds > 0 else ''}{odds}"

    return {
        "home_minus_1_5_prob": round(home_minus_1_5_prob, 3),
        "away_plus_1_5_prob": round(away_plus_1_5_prob, 3),
        "fair_home_minus_1_5": to_american(home_minus_1_5_prob),
        "fair_away_plus_1_5": to_american(away_plus_1_5_prob),
        
        "away_minus_1_5_prob": round(away_minus_1_5_prob, 3),
        "home_plus_1_5_prob": round(home_plus_1_5_prob, 3),
        "fair_away_minus_1_5": to_american(away_minus_1_5_prob),
        "fair_home_plus_1_5": to_american(home_plus_1_5_prob)
    }

def simulate_sgp_copula(team_runs: float, opp_starter_k_line: float, rho: float = -0.32, n_sims: int = 10000) -> dict:
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


def simulate_game_totals(away_runs: float, home_runs: float, total_line: float = 8.5, n_sims: int = 20000) -> dict:
    """
    Simulates full game combined runs distribution using Poisson Monte Carlo
    and evaluates Over/Under hit rates and fair zero-vig lines.
    """
    sim_a = np.random.poisson(away_runs, n_sims)
    sim_h = np.random.poisson(home_runs, n_sims)
    sim_total = sim_a + sim_h

    # Exclude exact pushes for integer lines to compute true two-way probabilities
    if total_line % 1 == 0:
        non_pushes = sim_total[sim_total != total_line]
        over_prob = np.mean(non_pushes > total_line)
        push_prob = np.mean(sim_total == total_line)
    else:
        over_prob = np.mean(sim_total > total_line)
        push_prob = 0.0

    under_prob = 1.0 - over_prob - push_prob

    def to_american(prob):
        if prob <= 0.001: return "+9999"
        if prob >= 0.999: return "-9999"
        odds = int(-100 * (prob / (1 - prob))) if prob >= 0.5 else int(100 * ((1 - prob) / prob))
        return f"{'+' if odds > 0 else ''}{odds}"

    return {
        "projected_total": round(away_runs + home_runs, 2),
        "total_line": total_line,
        "over_prob": round(over_prob, 3),
        "under_prob": round(under_prob, 3),
        "push_prob": round(push_prob, 3),
        "fair_over_odds": to_american(over_prob),
        "fair_under_odds": to_american(under_prob)
    }


def simulate_nrfi_yrfi(away_runs: float, home_runs: float, opp_away_era: float = 4.25, opp_home_era: float = 4.25) -> dict:
    """
    Calibrated NRFI / YRFI model using Negative Binomial run clustering (alpha = 0.65)
    and empirical 1st-inning starter run expectancy (0.55x baseline per half).
    """
    alpha = 0.65
    starter_1st_inning_scale = 0.55

    # Expected runs per half-inning in the 1st
    lambda_away = (away_runs / 9.0) * starter_1st_inning_scale
    lambda_home = (home_runs / 9.0) * starter_1st_inning_scale

    # P(Scoreless Half) = (1 + alpha * lambda)^(-1 / alpha)
    prob_away_0 = (1.0 + alpha * lambda_away) ** (-1.0 / alpha)
    prob_home_0 = (1.0 + alpha * lambda_home) ** (-1.0 / alpha)

    nrfi_prob = prob_away_0 * prob_home_0
    yrfi_prob = 1.0 - nrfi_prob

    def to_american(prob):
        if prob <= 0.001: return "+9999"
        if prob >= 0.999: return "-9999"
        odds = int(-100 * (prob / (1 - prob))) if prob >= 0.5 else int(100 * ((1 - prob) / prob))
        return f"{'+' if odds > 0 else ''}{odds}"

    return {
        "nrfi_prob": round(nrfi_prob, 3),
        "yrfi_prob": round(yrfi_prob, 3),
        "fair_nrfi_odds": to_american(nrfi_prob),
        "fair_yrfi_odds": to_american(yrfi_prob),
        "exp_first_inning_runs": round(lambda_away + lambda_home, 2)
    }
