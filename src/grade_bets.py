# ==============================================================================
# AUTOMATED POSTGAME MLB BET GRADER & PERFORMANCE EVALUATOR
# ==============================================================================
import os
import re
import datetime
import requests
import numpy as np
import pandas as pd

CLV_LOG_FILE = "data/clv_tracking_log.csv"

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=linescore,boxscore(scoringPlays)"

TEAM_NAME_MAP = {
    "AZ": "Arizona Diamondbacks", "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox", "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox", "CHW": "Chicago White Sox", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "ATH": "Oakland Athletics", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres",
    "SF": "San Francisco Giants", "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals"
}

def fetch_mlb_box_scores(target_date: str = None) -> dict:
    """Fetches final game scores, 1st inning runs, and starter strikeouts from MLB API."""
    if target_date is None:
        target_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    url = MLB_SCHEDULE_URL.format(date=target_date)
    res = requests.get(url).json()
    
    games_data = {}
    dates = res.get("dates", [])
    if not dates:
        return games_data
        
    for game in dates[0].get("games", []):
        status = game.get("status", {}).get("abstractGameState")
        detailed_status = game.get("status", {}).get("detailedState", "")
        
        home_team = game["teams"]["home"]["team"]["name"]
        away_team = game["teams"]["away"]["team"]["name"]
        
        # Match game ID key format: YYYY-MM-DD_AWAY_HOME
        linescore = game.get("linescore", {})
        home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
        away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)
        total_runs = home_score + away_score
        
        # 1st Inning Runs (NRFI / YRFI)
        innings = linescore.get("innings", [])
        first_inning_runs = 0
        first_inning_completed = False
        if len(innings) >= 1:
            inn1 = innings[0]
            away_1 = inn1.get("away", {}).get("runs", 0)
            home_1 = inn1.get("home", {}).get("runs", 0)
            first_inning_runs = away_1 + (home_1 if home_1 is not None else 0)
            first_inning_completed = True
            
        # Pitcher Strikeouts from boxscore
        pitcher_k_map = {}
        boxscore = game.get("boxscore", {})
        for side in ["away", "home"]:
            pitchers_list = boxscore.get("teams", {}).get(side, {}).get("pitchers", [])
            players = boxscore.get("teams", {}).get(side, {}).get("players", {})
            for p_id in pitchers_list:
                p_info = players.get(f"personId{p_id}", players.get(f"ID{p_id}", {}))
                p_name = p_info.get("person", {}).get("fullName", "")
                k_count = p_info.get("stats", {}).get("pitching", {}).get("strikeOuts", 0)
                if p_name:
                    pitcher_k_map[p_name.lower()] = k_count
                    # Last name mapping
                    last_name = p_name.split()[-1].lower()
                    pitcher_k_map[last_name] = k_count

        games_data[f"{away_team} @ {home_team}".lower()] = {
            "status": detailed_status if detailed_status else status,
            "is_final": (status == "Final"),
            "away_team": away_team,
            "home_team": home_team,
            "away_score": away_score,
            "home_score": home_score,
            "total_runs": total_runs,
            "winner": home_team if home_score > away_score else away_team,
            "run_diff": home_score - away_score, # Positive = Home won by X
            "first_inning_runs": first_inning_runs,
            "first_inning_completed": first_inning_completed,
            "pitcher_strikeouts": pitcher_k_map
        }
        
    return games_data

def grade_all_bets(target_date: str = "2026-08-16"):
    """Reads data/clv_tracking_log.csv and grades outcomes against MLB results."""
    if not os.path.exists(CLV_LOG_FILE):
        print("⚠️ No CLV log file found at data/clv_tracking_log.csv")
        return
        
    df = pd.read_csv(CLV_LOG_FILE)
    if df.empty:
        print("Log file is empty.")
        return
        
    box_scores = fetch_mlb_box_scores(target_date)
    if not box_scores:
        print(f"No MLB games returned for {target_date}.")
        return

    graded_count = 0
    df["profit_units"] = 0.0

    for idx, row in df.iterrows():
        market = row["market_type"]
        selection = str(row["selection"])
        game_id = str(row["game_id"])
        odds_dec = float(row["placed_odds_decimal"])
        stake = float(row["stake_units"])
        
        # Match game in box_scores
        matched_game = None
        for g_key, g_val in box_scores.items():
            # Check team names or abbreviations in selection or game_id
            parts = game_id.split("_")
            if len(parts) >= 3:
                away_abbr, home_abbr = parts[1], parts[2]
                away_full = TEAM_NAME_MAP.get(away_abbr, "").lower()
                home_full = TEAM_NAME_MAP.get(home_abbr, "").lower()
                if (away_full in g_key) and (home_full in g_key):
                    matched_game = g_val
                    break
            if not matched_game:
                if any(t.lower() in g_key for t in selection.split()):
                    matched_game = g_val
                    break

        if not matched_game:
            continue

        result = row["result"]

        # 1. MONEYLINE
        if market == "Moneyline":
            if matched_game["is_final"]:
                winner = matched_game["winner"]
                if winner.lower() in selection.lower() or any(w.lower() in winner.lower() for w in selection.split()):
                    result = "WIN"
                    profit = stake * (odds_dec - 1.0)
                else:
                    result = "LOSS"
                    profit = -stake
                df.at[idx, "result"] = result
                df.at[idx, "profit_units"] = round(profit, 3)
                graded_count += 1

        # 2. RUN LINE (+1.5 / -1.5)
        elif market == "Run Line":
            if matched_game["is_final"]:
                diff = matched_game["run_diff"] # Home - Away
                is_home = matched_game["home_team"].lower() in selection.lower()
                is_minus = "-1.5" in selection
                
                win = False
                if is_home and is_minus and diff >= 2: win = True
                elif is_home and (not is_minus) and diff >= -1: win = True
                elif (not is_home) and is_minus and diff <= -2: win = True
                elif (not is_home) and (not is_minus) and diff <= 1: win = True
                
                result = "WIN" if win else "LOSS"
                profit = (stake * (odds_dec - 1.0)) if win else -stake
                df.at[idx, "result"] = result
                df.at[idx, "profit_units"] = round(profit, 3)
                graded_count += 1

        # 3. GAME TOTALS (OVER / UNDER)
        elif market == "Game Total":
            if matched_game["is_final"]:
                tot = matched_game["total_runs"]
                line_match = re.search(r"(Over|Under)\s+(\d+\.?\d*)", selection, re.IGNORECASE)
                if line_match:
                    ou_type = line_match.group(1).capitalize()
                    line_val = float(line_match.group(2))
                    
                    if tot == line_val:
                        result = "PUSH"
                        profit = 0.0
                    elif ou_type == "Over" and tot > line_val:
                        result = "WIN"
                        profit = stake * (odds_dec - 1.0)
                    elif ou_type == "Under" and tot < line_val:
                        result = "WIN"
                        profit = stake * (odds_dec - 1.0)
                    else:
                        result = "LOSS"
                        profit = -stake
                    
                    df.at[idx, "result"] = result
                    df.at[idx, "profit_units"] = round(profit, 3)
                    graded_count += 1

        # 4. NRFI / YRFI
        elif market == "NRFI/YRFI":
            if matched_game["first_inning_completed"] or matched_game["is_final"]:
                first_runs = matched_game["first_inning_runs"]
                is_nrfi = "NRFI" in selection.upper()
                
                win = (first_runs == 0) if is_nrfi else (first_runs > 0)
                result = "WIN" if win else "LOSS"
                profit = (stake * (odds_dec - 1.0)) if win else -stake
                df.at[idx, "result"] = result
                df.at[idx, "profit_units"] = round(profit, 3)
                graded_count += 1

        # 5. PITCHER STRIKEOUTS
        elif market == "Pitcher Strikeouts":
            if matched_game["is_final"]:
                k_map = matched_game["pitcher_strikeouts"]
                prop_match = re.search(r"^(.*?)\s+(Over|Under)\s+(\d+\.?\d*)\s*K", selection, re.IGNORECASE)
                if prop_match:
                    p_name = prop_match.group(1).strip().lower()
                    ou_type = prop_match.group(2).capitalize()
                    k_line = float(prop_match.group(3))
                    
                    # Look up actual K count
                    actual_k = k_map.get(p_name)
                    if actual_k is None:
                        # Try matching last name
                        for k_key, v in k_map.items():
                            if k_key in p_name or p_name.split()[-1] in k_key:
                                actual_k = v
                                break
                                
                    if actual_k is not None:
                        if actual_k == k_line:
                            result = "PUSH"
                            profit = 0.0
                        elif ou_type == "Over" and actual_k > k_line:
                            result = "WIN"
                            profit = stake * (odds_dec - 1.0)
                        elif ou_type == "Under" and actual_k < k_line:
                            result = "WIN"
                            profit = stake * (odds_dec - 1.0)
                        else:
                            result = "LOSS"
                            profit = -stake
                            
                        df.at[idx, "result"] = result
                        df.at[idx, "profit_units"] = round(profit, 3)
                        graded_count += 1

    df.to_csv(CLV_LOG_FILE, index=False)
    print(f"📊 Evaluated log: {graded_count} wagers updated.")
    return df

def generate_performance_report() -> pd.DataFrame:
    """Computes Net Units, Win Rate, and ROI across completed bets."""
    if not os.path.exists(CLV_LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(CLV_LOG_FILE)
    completed = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
    if completed.empty:
        print("No completed bets to summarize yet.")
        return pd.DataFrame()
        
    wins = (completed["result"] == "WIN").sum()
    losses = (completed["result"] == "LOSS").sum()
    pushes = (completed["result"] == "PUSH").sum()
    total_staked = completed["stake_units"].sum()
    net_profit = completed["profit_units"].sum()
    roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0.0
    
    summary = {
        "Total Bets": len(completed),
        "Record (W-L-P)": f"{wins}-{losses}-{pushes}",
        "Win Rate %": f"{round((wins / (wins + losses) * 100), 1)}%" if (wins + losses) > 0 else "0.0%",
        "Total Staked (Units)": round(total_staked, 2),
        "Net Profit (Units)": f"{'+' if net_profit > 0 else ''}{round(net_profit, 2)}u",
        "ROI %": f"{'+' if roi > 0 else ''}{round(roi, 2)}%"
    }
    return pd.DataFrame([summary])
