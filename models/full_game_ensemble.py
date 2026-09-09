import numpy as np
import pandas as pd

def to_american(prob):
    if prob >= 0.99: return "-10000"
    if prob <= 0.01: return "+10000"
    if prob >= 0.5: return f"-{int(round(100.0 * (prob / (1.0 - prob))))}"
    return f"+{int(round(100.0 * ((1.0 - prob) / prob)))}"

def run_full_game_ensemble(df_pitchers, df_lineups, df_bullpen, df_nrfi):
    full_res = []
    for _, row in df_nrfi.iterrows():
        g_pk = str(row['game_pk'])
        sp_h = df_pitchers[df_pitchers['pitcher_name'] == row['home_pitcher']]
        sp_a = df_pitchers[df_pitchers['pitcher_name'] == row['away_pitcher']]
        if sp_h.empty or sp_a.empty: continue

        era_h = float(sp_h.iloc[0]['era']) if pd.notna(sp_h.iloc[0]['era']) else 4.10
        era_a = float(sp_a.iloc[0]['era']) if pd.notna(sp_a.iloc[0]['era']) else 4.10

        bp_h = df_bullpen[df_bullpen['team_name'] == row['home_team']]
        bp_a = df_bullpen[df_bullpen['team_name'] == row['away_team']]
        bp_h_m = float(bp_h['bullpen_fatigue_mult'].iloc[0]) if not bp_h.empty else 0.02
        bp_a_m = float(bp_a['bullpen_fatigue_mult'].iloc[0]) if not bp_a.empty else 0.02

        tot_a = (2.35 * (era_h / 4.15)) + 1.72 * (1.0 + bp_h_m)
        tot_h = (2.44 * (era_a / 4.15)) + 1.72 * (1.0 + bp_a_m) * 1.04

        era_diff = era_a - era_h
        p_lgbm_h = 1.0 / (1.0 + np.exp(-(0.04 + era_diff * 0.14 + (bp_a_m - bp_h_m) * 1.5)))

        a_sim = np.random.poisson(tot_a / 9.0, size=(15000, 9)).sum(axis=1)
        h_sim = np.random.poisson(tot_h / 9.0, size=(15000, 9)).sum(axis=1)
        ties = a_sim == h_sim
        h_sim[ties] += np.random.choice([1, 0], size=np.sum(ties), p=[0.53, 0.47])
        a_sim[ties] += (1 - (h_sim[ties] - a_sim[ties]))

        p_mc_h = float(np.mean((h_sim - a_sim) > 0))
        p_final_h = 0.70 * p_lgbm_h + 0.30 * p_mc_h

        full_res.append({
            'game_pk': g_pk, 'matchup': f"{row['away_team']} @ {row['home_team']}",
            'home_win_prob': round(p_final_h, 4), 'away_win_prob': round(1.0 - p_final_h, 4),
            'home_ml': to_american(p_final_h), 'away_ml': to_american(1.0 - p_final_h),
            'home_m15_prob': round(float(np.mean((h_sim - a_sim) >= 2)), 4),
            'away_p15_prob': round(float(np.mean((h_sim - a_sim) <= 1)), 4)
        })
    return pd.DataFrame(full_res)
