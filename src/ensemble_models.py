# ==============================================================================
# MLB HYBRID ENSEMBLE ENGINE (LightGBM ML + Monte Carlo Simulations)
# ==============================================================================
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from models import simulate_moneyline, simulate_run_line, simulate_nrfi_yrfi

DRIVE_MODELS_PATH = "/content/drive/MyDrive/MLB_Prediction_Model"

def load_ml_classifier():
    path = os.path.join(DRIVE_MODELS_PATH, "models/moneyline_classifier.pkl")
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except Exception:
            return joblib.load(path)
    return None

ml_model = load_ml_classifier()

def predict_hybrid_game(away_team: str, home_team: str, exp_away_runs: float, exp_home_runs: float, feature_row: dict = None) -> dict:
    """
    Combines high-accuracy LightGBM probability estimates (Brier: 0.2277)
    with Monte Carlo score simulation for full-distribution market pricing.
    """
    # 1. Base Monte Carlo Simulation
    sim_output = simulate_moneyline(exp_away_runs, exp_home_runs)
    sim_home_prob = sim_output["home_win_prob"]

    # 2. LightGBM ML Inference (if features available)
    ml_home_prob = sim_home_prob
    if ml_model is not None and feature_row is not None:
        try:
            df_feat = pd.DataFrame([feature_row])
            if hasattr(ml_model, "feature_names_in_"):
                req = list(ml_model.feature_names_in_)
                df_feat = df_feat.reindex(columns=req, fill_value=0.0)
            if hasattr(ml_model, "predict_proba"):
                ml_home_prob = ml_model.predict_proba(df_feat)[0, 1]
        except Exception:
            pass

    # 3. Stacked Ensemble Weighting (70% ML / 30% Simulation)
    final_home_prob = round((0.70 * ml_home_prob) + (0.30 * sim_home_prob), 4)
    final_away_prob = round(1.0 - final_home_prob, 4)

    def to_american(prob):
        if prob <= 0.001: return "+9999"
        if prob >= 0.999: return "-9999"
        odds = int(-100 * (prob / (1 - prob))) if prob >= 0.5 else int(100 * ((1 - prob) / prob))
        return f"{'+' if odds > 0 else ''}{odds}"

    return {
        "matchup": f"{away_team} @ {home_team}",
        "hybrid_home_prob": final_home_prob,
        "hybrid_away_prob": final_away_prob,
        "fair_home_ml": to_american(final_home_prob),
        "fair_away_ml": to_american(final_away_prob),
        "ml_prob": round(ml_home_prob, 4),
        "sim_prob": round(sim_home_prob, 4),
        "exp_total_runs": round(exp_away_runs + exp_home_runs, 2)
    }
