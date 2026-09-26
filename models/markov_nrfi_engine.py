import numpy as np
import pandas as pd

def to_american(prob):
    if prob >= 0.99: return "-10000"
    if prob <= 0.01: return "+10000"
    if prob >= 0.5: return f"-{int(round(100.0 * (prob / (1.0 - prob))))}"
    return f"+{int(round(100.0 * ((1.0 - prob) / prob)))}"

def run_markov_nrfi_engine(df_pitchers, df_lineups, df_nrfi):
    nrfi_res = []
    for _, row in df_nrfi.iterrows():
        sp_h = df_pitchers[df_pitchers['pitcher_name'] == row['home_pitcher']]
        sp_a = df_pitchers[df_pitchers['pitcher_name'] == row['away_pitcher']]
        if sp_h.empty or sp_a.empty: continue

        tto1_k_h = float(sp_h.iloc[0]['tto1_k_pct']) if pd.notna(sp_h.iloc[0]['tto1_k_pct']) else 0.225
        tto1_xw_h = float(sp_h.iloc[0]['tto1_xwoba']) if pd.notna(sp_h.iloc[0]['tto1_xwoba']) else 0.315
        tto1_k_a = float(sp_a.iloc[0]['tto1_k_pct']) if pd.notna(sp_a.iloc[0]['tto1_k_pct']) else 0.225
        tto1_xw_a = float(sp_a.iloc[0]['tto1_xwoba']) if pd.notna(sp_a.iloc[0]['tto1_xwoba']) else 0.315

        p_top = min(0.92, max(0.60, 0.73 + (tto1_k_h - 0.22) * 0.4 - (tto1_xw_h - 0.315) * 0.6))
        p_bot = min(0.92, max(0.60, 0.73 + (tto1_k_a - 0.22) * 0.4 - (tto1_xw_a - 0.315) * 0.6))
        p_nrfi = p_top * p_bot

        nrfi_res.append({
            'game_pk': str(row['game_pk']),
            'matchup': f"{row['away_team']} @ {row['home_team']}",
            'nrfi_prob': round(p_nrfi, 4), 'yrfi_prob': round(1.0 - p_nrfi, 4),
            'fair_nrfi': to_american(p_nrfi), 'fair_yrfi': to_american(1.0 - p_nrfi)
        })
    return pd.DataFrame(nrfi_res)
