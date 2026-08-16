# ==============================================================================
# LIVE SLATE & ODDS API PIPELINE
# ==============================================================================
import requests
import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson

def fetch_live_slate() -> pd.DataFrame:
    """Hydrates today's MLB probable starters and starting lineups."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,team"
    try:
        res = requests.get(url).json()
        games = []
        dates = res.get("dates", [])
        if not dates:
            return pd.DataFrame()
        for g in dates[0].get("games", []):
            h_info = g["teams"]["home"]["team"]
            a_info = g["teams"]["away"]["team"]
            h_p = g["teams"]["home"].get("probablePitcher", {})
            a_p = g["teams"]["away"].get("probablePitcher", {})
            games.append({
                "game_pk": g.get("gamePk"),
                "home_team": h_info.get("abbreviation", h_info.get("name")),
                "away_team": a_info.get("abbreviation", a_info.get("name")),
                "home_pitcher": h_p.get("fullName", "TBD"),
                "home_pitcher_id": h_p.get("id"),
                "away_pitcher": a_p.get("fullName", "TBD"),
                "away_pitcher_id": a_p.get("id")
            })
        return pd.DataFrame(games)
    except Exception:
        return pd.DataFrame()

def fetch_pitcher_baseline(pitcher_id: int) -> dict:
    """Fetches real-time season K/9 and innings workload from MLB API."""
    if not pitcher_id:
        return {"k9": 8.5, "exp_ip": 5.0, "era": 4.25}
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=[pitching],type=[season])"
    try:
        res = requests.get(url).json()
        stats = res["people"][0]["stats"][0]["splits"][0]["stat"]
        ip = float(stats.get("inningsPitched", 50))
        so = float(stats.get("strikeOuts", 45))
        era = float(stats.get("era", 4.25))
        gs = float(stats.get("gamesStarted", 10)) or 1.0
        
        k9 = (so / ip) * 9.0 if ip > 0 else 8.5
        exp_ip = min(max(ip / gs, 4.0), 6.5) if gs > 0 else 5.0
        return {"k9": round(k9, 2), "exp_ip": round(exp_ip, 2), "era": era}
    except Exception:
        return {"k9": 8.5, "exp_ip": 5.0, "era": 4.25}
