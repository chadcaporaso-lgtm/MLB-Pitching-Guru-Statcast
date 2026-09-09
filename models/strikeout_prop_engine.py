import numpy as np
import pandas as pd
from scipy.stats import nbinom

def to_american(prob):
    if prob >= 0.99: return "-10000"
    if prob <= 0.01: return "+10000"
    if prob >= 0.5: return f"-{int(round(100.0 * (prob / (1.0 - prob))))}"
    return f"+{int(round(100.0 * ((1.0 - prob) / prob)))}"

def run_strikeout_prop_engine(df_pitchers, df_lineups, df_nrfi):
    k_res = []
    for _, row in df_nrfi.iterrows():
        g_pk = str(row['game_pk'])
        configs = [
            (row['home_pitcher'], row['home_team'], row['away_team']),
            (row['away_pitcher'], row['away_team'], row['home_team'])
        ]
        for sp_name, team, opp in configs:
            sp_m = df_pitchers[df_pitchers['pitcher_name'] == sp_name]
            opp_l = df_lineups[(df_lineups['game_pk'].astype(str) == g_pk) & (df_lineups['team_name'] == opp)]
            if sp_m.empty or opp_l.empty: continue
            sp = sp_m.iloc[0]

            csw = float(sp['csw_pct']) if pd.notna(sp['csw_pct']) else 0.260
            base_k = max(0.12, min(0.38, 1.40 * csw - 0.10))
            k_mult = 0.60 * (float(opp_l['k_pct_vs_hand'].mean()) / 0.225) + 0.40 * (float(opp_l['whiff_pct_vs_hand'].mean()) / 0.245)

            era = float(sp['era']) if pd.notna(sp['era']) else 3.85
            lam = (min(6.3, max(4.7, 6.2 - era * 0.25)) * 4.15) * (base_k * k_mult)
            r = (lam ** 2) / (1.22 * lam - lam)
            p = r / (r + lam)

            for line in [3.5, 4.5, 5.5, 6.5]:
                p_u = float(nbinom.cdf(np.floor(line), r, p))
                p_o = 1.0 - p_u
                k_res.append({
                    'game_pk': g_pk, 'pitcher': sp_name, 'team': team, 'opp': opp,
                    'exp_k': round(lam, 2), 'line': line,
                    'over_prob': round(p_o, 4), 'under_prob': round(p_u, 4),
                    'fair_over': to_american(p_o), 'fair_under': to_american(p_u)
                })
    return pd.DataFrame(k_res)
