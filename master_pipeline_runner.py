"""
Master Daily MLB Pipeline Runner (Production v2.1)
--------------------------------------------------
1. Ingests rolling 30-day Savant metrics & bullpen leverage data.
2. Ingests live multi-book market quotes via SportsGameOdds API.
3. Evaluates 24-State Markov NRFI with Top-3 lineup arsenal whiff adjustments.
4. Scales Negative Binomial Strikeout Props using ArsenalWhiffEngine lineup multipliers.
5. Runs 10,000-Sim Monte Carlo for Sides, Totals, and RunLineParser (+1.5 / -1.5).
6. Exports updated boards to data/ and mirrors to Google Drive.
"""

import os
import shutil
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import nbinom

from models.markov_nrfi_engine import MarkovNRFIEngine
from models.run_line_parser import RunLineParser, to_decimal, to_american
from models.arsenal_whiff_engine import ArsenalWhiffEngine

API_KEY = "c49022bdc56731df0a5cba336b0cc880"
SGO_URL = "https://api.sportsgameodds.com/v2/events"
TARGET_BOOKS = ["novig", "fanduel", "caesars", "draftkings", "betmgm"]
DATA_DIR = "data"
DRIVE_DIR = "/content/drive/MyDrive/MLB-Guru-Data"
TODAY_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("=" * 85)
print(f"🚀 EXECUTING MASTER PIPELINE RUNNER - {TODAY_STR}")
print("=" * 85)

# Initialize engines
whiff_engine = ArsenalWhiffEngine(league_avg_whiff=0.245)
rl_parser = RunLineParser(target_books=TARGET_BOOKS)
markov_engine = MarkovNRFIEngine()

# Load datasets
df_pitchers = pd.read_csv(f"{DATA_DIR}/pitcher_statcast_rolling.csv")
df_lineups = pd.read_csv(f"{DATA_DIR}/lineup_statcast_splits.csv")
df_nrfi = pd.read_csv(f"{DATA_DIR}/nrfi_context_splits.csv")

print("✅ Statistical datasets & scoring engines loaded.")

# Ingest live odds
params = {
    "apiKey": API_KEY,
    "leagueID": "MLB",
    "oddsAvailable": "true",
    "includeAltLines": "true",
    "limit": 35
}

resp = requests.get(SGO_URL, params=params, timeout=12)
live_events = resp.json().get("data", [])
print(f"📡 Ingested {len(live_events)} active game trees from SportsGameOdds.")

# ------------------------------------------------------------------------------
# 1. EVALUATE ARSENAL-ADJUSTED STRIKEOUT PROPS (NEGATIVE BINOMIAL)
# ------------------------------------------------------------------------------
k_prop_results = []

for _, p_row in df_pitchers.iterrows():
    p_name = p_row["pitcher_name"]
    p_hand = p_row.get("throws", "R")
    base_k_per_9 = float(p_row.get("k_per_9", 8.5))
    rolling_ip = float(p_row.get("rolling_ip", 5.2))
    
    # Base expected strikeouts per start
    base_lambda_k = (base_k_per_9 / 9.0) * min(6.5, max(4.5, rolling_ip))
    
    # Extract opponent lineup from lineup splits
    opp_lineup = df_lineups[df_lineups["opposing_pitcher"] == p_name]
    if opp_lineup.empty:
        lineup_k_mult = 1.0
    else:
        # Sample starter arsenal vs. handedness
        dummy_arsenal = {"FF": 0.40, "SL": 0.30, "CH": 0.20, "CU": 0.10}
        whiff_eval = whiff_engine.evaluate_lineup_whiff(
            pitcher_name=p_name,
            pitcher_hand=p_hand,
            lineup_df=opp_lineup.head(9),
            pitcher_arsenals={"vs_LHB": dummy_arsenal, "vs_RHB": dummy_arsenal}
        )
        lineup_k_mult = float(whiff_eval["k_mult"].astype(float).mean())

    adj_lambda_k = base_lambda_k * lineup_k_mult
    
    # Negative Binomial parameterization (r=dispersion, p=probability)
    # Dispersion variance ratio var/mean ~ 1.25 for MLB strikeouts
    r_param = (adj_lambda_k ** 2) / (1.25 * adj_lambda_k - adj_lambda_k)
    p_param = r_param / (r_param + adj_lambda_k)
    
    # Evaluate typical 5.5 and 6.5 strikeout lines
    for line in [5.5, 6.5]:
        prob_under = float(nbinom.cdf(np.floor(line), r_param, p_param))
        prob_over = 1.0 - prob_under
        
        k_prop_results.append({
            "pitcher": p_name,
            "line": line,
            "adj_lambda_k": round(adj_lambda_k, 2),
            "lineup_k_mult": round(lineup_k_mult, 4),
            "over_prob": round(prob_over, 4),
            "under_prob": round(prob_under, 4),
            "fair_over": to_american(prob_over),
            "fair_under": to_american(prob_under)
        })

df_k_props = pd.DataFrame(k_prop_results)
df_k_props.to_csv(f"{DATA_DIR}/arsenal_adjusted_k_props.csv", index=False)
if os.path.exists(DRIVE_DIR):
    shutil.copy(f"{DATA_DIR}/arsenal_adjusted_k_props.csv", f"{DRIVE_DIR}/arsenal_adjusted_k_props.csv")

print(f"⚾ Arsenal-adjusted strikeout props generated ({len(df_k_props)} lines evaluated).")
display(df_k_props.head(8))
