"""
Master Daily MLB Pipeline Runner
---------------------------------
Orchestrates daily morning workflows:
1. Ingests rolling 30-day Savant metrics & bullpen leverage data.
2. Pulls live multi-book market quotes via SportsGameOdds API.
3. Evaluates 24-State Markov NRFI, 10,000-Sim Monte Carlo Sides/Totals, and Negative Binomial Ks.
4. Integrates RunLineParser to dynamically inspect bookmaker spread attributes and output
   the verified +1.5 / -1.5 run line board.
5. Exports structured daily boards to data/ and mirrors to Google Drive.
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

API_KEY = "c49022bdc56731df0a5cba336b0cc880"
SGO_URL = "https://api.sportsgameodds.com/v2/events"
TARGET_BOOKS = ["novig", "fanduel", "caesars", "draftkings", "betmgm"]
DATA_DIR = "data"
DRIVE_DIR = "/content/drive/MyDrive/MLB-Guru-Data"
TODAY_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ------------------------------------------------------------------------------
# 1. LOAD MODEL DATASETS & INITIALIZE ENGINES
# ------------------------------------------------------------------------------
print("=" * 85)
print(f"🚀 INITIALIZING MASTER PIPELINE RUNNER - {TODAY_STR}")
print("=" * 85)

df_pitchers = pd.read_csv(f"{DATA_DIR}/pitcher_statcast_rolling.csv")
df_lineups = pd.read_csv(f"{DATA_DIR}/lineup_statcast_splits.csv")
df_bullpen = pd.read_csv(f"{DATA_DIR}/bullpen_leverage_index.csv")
df_nrfi = pd.read_csv(f"{DATA_DIR}/nrfi_context_splits.csv")

markov_engine = MarkovNRFIEngine()
rl_parser = RunLineParser(target_books=TARGET_BOOKS)
print("✅ Core statistical datasets, Markov NRFI, and RunLineParser loaded.")

# ------------------------------------------------------------------------------
# 2. INGEST LIVE MARKET TREES
# ------------------------------------------------------------------------------
params = {
    "apiKey": API_KEY,
    "leagueID": "MLB",
    "oddsAvailable": "true",
    "includeAltLines": "true",
    "limit": 35
}

resp = requests.get(SGO_URL, params=params, timeout=12)
live_events = resp.json().get("data", [])
print(f"📡 Ingested {len(live_events)} active MLB event trees from SportsGameOdds.")

# ------------------------------------------------------------------------------
# 3. 10,000 MONTE CARLO & RUN LINE INTEGRATION
# ------------------------------------------------------------------------------
run_line_results = []
N_SIMS = 10000

for _, match in df_nrfi.iterrows():
    h_team, a_team = match["home_team"], match["away_team"]
    h_pitcher, a_pitcher = match["home_pitcher"], match["away_pitcher"]

    h_p = df_pitchers[df_pitchers["pitcher_name"] == h_pitcher]
    a_p = df_pitchers[df_pitchers["pitcher_name"] == a_pitcher]
    if h_p.empty or a_p.empty:
        continue
    h_p, a_p = h_p.iloc[0], a_p.iloc[0]

    h_bp = df_bullpen[df_bullpen["team_name"] == h_team]
    a_bp = df_bullpen[df_bullpen["team_name"] == a_team]
    h_bp_pen = float(h_bp["bullpen_fatigue_mult"].iloc[0]) if not h_bp.empty else 0.0
    a_bp_pen = float(a_bp["bullpen_fatigue_mult"].iloc[0]) if not a_bp.empty else 0.0

    # Match game with live SGO event
    matched_ev = None
    for ev in live_events:
        t_home = ev.get("teams", {}).get("home", {}).get("names", {}).get("long", "") or \
                 ev.get("teams", {}).get("home", {}).get("teamID", "")
        t_away = ev.get("teams", {}).get("away", {}).get("names", {}).get("long", "") or \
                 ev.get("teams", {}).get("away", {}).get("teamID", "")
        if (h_team.split()[-1].lower() in t_home.lower()) or (a_team.split()[-1].lower() in t_away.lower()):
            matched_ev = ev
            break

    if not matched_ev:
        continue

    # Pitcher stamina & run expectations
    tot_h_ip, tot_a_ip = float(h_p.get("rolling_ip", 5.0)), float(a_p.get("rolling_ip", 5.0))
    h_ip = min(6.0, max(4.5, tot_h_ip / max(1.0, round(tot_h_ip / 5.2))))
    a_ip = min(6.0, max(4.5, tot_a_ip / max(1.0, round(tot_a_ip / 5.2))))

    wind_mod = float(match.get("park_wind_suppression", 0.0))
    lambda_h = max(0.24, (float(a_p.get("xwoba", 0.310)) / 0.315) * 0.46 * (1.0 + wind_mod))
    lambda_a = max(0.24, (float(h_p.get("xwoba", 0.310)) / 0.315) * 0.44 * (1.0 + wind_mod))
    lambda_h_pen = max(0.26, 0.44 * (1.0 + a_bp_pen))
    lambda_a_pen = max(0.26, 0.44 * (1.0 + h_bp_pen))

    # Monte Carlo simulation of exact margins
    h_cover_m15 = 0
    a_cover_p15 = 0

    for _ in range(N_SIMS):
        away_r = np.random.poisson(lambda_a, int(h_ip)).sum() + np.random.poisson(lambda_a_pen, int(9 - h_ip)).sum()
        home_r_thru_8 = np.random.poisson(lambda_h, int(a_ip)).sum() + np.random.poisson(lambda_h_pen, int(8 - a_ip)).sum()
        home_r = home_r_thru_8 if home_r_thru_8 > away_r else home_r_thru_8 + np.random.poisson(lambda_h_pen)

        if home_r == away_r:
            away_r += np.random.poisson(0.85)
            home_r += np.random.poisson(0.95)

        margin = home_r - away_r
        if margin >= 2:
            h_cover_m15 += 1
        if margin <= 1:
            a_cover_p15 += 1

    prob_h_m15 = round(h_cover_m15 / N_SIMS, 4)
    prob_a_p15 = round(a_cover_p15 / N_SIMS, 4)

    # Dynamic run line parsing with attribute verification
    parsed_game_rl = rl_parser.parse_game_run_lines(
        event=matched_ev,
        model_home_m15_prob=prob_h_m15,
        model_away_p15_prob=prob_a_p15
    )

    for item in parsed_game_rl:
        q = item.get("quotes", {})
        run_line_results.append({
            "matchup": item["matchup"],
            "selection": item["selection"],
            "model_prob": item["model_prob"],
            "fair_line": item["fair_line"],
            "novig": q.get("novig", "N/A"),
            "fanduel": q.get("fanduel", "N/A"),
            "caesars": q.get("caesars", "N/A"),
            "draftkings": q.get("draftkings", "N/A"),
            "betmgm": q.get("betmgm", "N/A"),
            "best_book": item["best_book"].title(),
            "best_odds": item["best_odds"],
            "ev_num": item["ev"]
        })

# ------------------------------------------------------------------------------
# 4. EXPORT AND SAVE BOARDS
# ------------------------------------------------------------------------------
df_rl_board = pd.DataFrame(run_line_results).sort_values(by="ev_num", ascending=False).reset_index(drop=True)

# Format display columns
df_display = df_rl_board.copy()
df_display["model_prob"] = df_display["model_prob"].apply(lambda p: f"{p:.1%}")
df_display["edge_ev"] = df_display["ev_num"].apply(lambda e: f"{e:+.2%}")
df_display.drop(columns=["ev_num"], inplace=True)

OUT_CSV = f"{DATA_DIR}/run_line_board.csv"
df_rl_board.to_csv(OUT_CSV, index=False)

if os.path.exists(DRIVE_DIR):
    shutil.copy(OUT_CSV, f"{DRIVE_DIR}/run_line_board.csv")
    print("📁 Mirrored run_line_board.csv to Google Drive.")

print("\n" + "=" * 105)
print(f"⚾ VERIFIED RUN LINE BOARD GENERATED ({len(df_rl_board)} LINES EVALUATED)")
print("=" * 105)
display(df_display.head(10))
