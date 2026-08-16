# ==============================================================================
# CLOSING LINE VALUE (CLV) & MARKET EFFICIENCY TRACKER
# ==============================================================================
import os
import datetime
import pandas as pd
import numpy as np

CLV_LOG_FILE = "data/clv_tracking_log.csv"

def log_bet(game_id: str, market_type: str, selection: str, bet_odds_american: int, model_prob: float, stake_units: float = 1.0):
    """Logs placed bets with timestamp, entry odds, and model probabilities."""
    os.makedirs(os.path.dirname(CLV_LOG_FILE), exist_ok=True)
    
    dec_odds = (bet_odds_american / 100 + 1.0) if bet_odds_american > 0 else (100 / abs(bet_odds_american) + 1.0)
    
    new_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "game_id": game_id,
        "market_type": market_type,
        "selection": selection,
        "placed_odds_american": bet_odds_american,
        "placed_odds_decimal": round(dec_odds, 4),
        "model_win_prob": round(model_prob, 4),
        "stake_units": stake_units,
        "closing_odds_american": np.nan,
        "closing_odds_decimal": np.nan,
        "clv_pct": np.nan,
        "result": "PENDING"
    }
    
    if os.path.exists(CLV_LOG_FILE):
        df = pd.read_csv(CLV_LOG_FILE)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
        
    df.to_csv(CLV_LOG_FILE, index=False)
    print(f"✅ Logged bet for {selection} ({bet_odds_american}) to CLV tracker.")

def update_closing_lines(game_id: str, selection: str, closing_odds_american: int, result: str = None):
    """Updates closing market price and computes exact Closing Line Value (CLV %)."""
    if not os.path.exists(CLV_LOG_FILE):
        print("⚠️ No CLV log found in data/.")
        return
        
    df = pd.read_csv(CLV_LOG_FILE)
    mask = (df["game_id"] == game_id) & (df["selection"] == selection)
    
    if not df[mask].empty:
        close_dec = (closing_odds_american / 100 + 1.0) if closing_odds_american > 0 else (100 / abs(closing_odds_american) + 1.0)
        placed_dec = df.loc[mask, "placed_odds_decimal"].values[0]
        
        # CLV Formula: (Placed Dec / Closing Dec) - 1.0
        clv_pct = round(((placed_dec / close_dec) - 1.0) * 100, 2)
        
        df.loc[mask, "closing_odds_american"] = closing_odds_american
        df.loc[mask, "closing_odds_decimal"] = round(close_dec, 4)
        df.loc[mask, "clv_pct"] = clv_pct
        if result:
            df.loc[mask, "result"] = result
            
        df.to_csv(CLV_LOG_FILE, index=False)
        print(f"📊 Updated {selection} | Placed: {placed_dec} | Closing: {closing_odds_american} ({close_dec}) | CLV: {'+' if clv_pct > 0 else ''}{clv_pct}%")
    else:
        print(f"⚠️ Bet entry not found for {selection} in {game_id}.")

def get_clv_performance_summary() -> pd.DataFrame:
    """Aggregates average CLV and win rate across closed bets."""
    if not os.path.exists(CLV_LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(CLV_LOG_FILE)
    closed = df.dropna(subset=["clv_pct"])
    if closed.empty:
        return pd.DataFrame()
    
    return pd.DataFrame([{
        "Total Bets Tracked": len(closed),
        "Avg CLV %": f"{round(closed['clv_pct'].mean(), 2)}%",
        "Positive CLV Rate": f"{round((closed['clv_pct'] > 0).mean() * 100, 1)}%",
        "Total Units": closed['stake_units'].sum()
    }])
