import numpy as np
import pandas as pd

def to_american(prob):
    if prob >= 0.99: return "-10000"
    if prob <= 0.01: return "+10000"
    if prob >= 0.5: return f"-{int(round(100.0 * (prob / (1.0 - prob))))}"
    return f"+{int(round(100.0 * ((1.0 - prob) / prob)))}"

def run_f5_engine(df_pitchers, df_lineups, df_nrfi):
    f5_res = []
    for _, row in df_nrfi.iterrows():
        g_pk = str(row['game_pk'])
        sp_h = df_pitchers[df_pitchers['pitcher_name'] == row['home_pitcher']]
        sp_a = df_pitchers[df_pitchers['pitcher_name'] == row['away_pitcher']]
        if sp_h.empty or sp_a.empty: continue

        era_h = float(sp_h.iloc[0]['era']) if pd.notna(sp_h.iloc[0]['era']) else 4.10
        era_a = float(sp_a.iloc[0]['era']) if pd.notna(sp_a.iloc[0]['era']) else 4.10

        exp_a_5 = 2.35 * (era_h / 4.15)
        exp_h_5 = 2.44 * (era_a / 4.15)

        a_sim = np.random.poisson(exp_a_5 / 5.0, size=(10000, 5)).sum(axis=1)
        h_sim = np.random.poisson(exp_h_5 / 5.0, size=(10000, 5)).sum(axis=1)
        margin = h_sim - a_sim
        h_w, a_w = float(np.mean(margin > 0)), float(np.mean(margin < 0))
        h_ml = h_w / (h_w + a_w) if (h_w + a_w) > 0 else 0.50

        f5_res.append({
            'game_pk': g_pk, 'matchup': f"{row['away_team']} @ {row['home_team']}",
            'exp_a5': round(exp_a_5, 2), 'exp_h5': round(exp_h_5, 2),
            'tie_prob': round(float(np.mean(margin == 0)), 4),
            'home_f5_ml': to_american(h_ml), 'away_f5_ml': to_american(1.0 - h_ml),
            'home_cover_m05': round(h_w, 4)
        })
    return pd.DataFrame(f5_res)
