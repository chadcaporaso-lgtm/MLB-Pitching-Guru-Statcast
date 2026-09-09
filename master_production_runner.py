import os
import re
import datetime
import requests
import numpy as np
import pandas as pd
from scipy.stats import nbinom

ODDS_API_KEY = "34ea51232dfd5fd5830d0acf99583ab6"
SPORT_KEY = "baseball_mlb"
TARGET_BOOKS = {"fanduel", "draftkings", "williamhill_us", "betmgm", "bovada", "novig"}
MIN_EDGE = 0.05
MAX_EDGE = 0.25

LOCAL_DATA = "/content/MLB-Pitching-Guru-Statcast/data"
DRIVE_DATA = "/content/drive/MyDrive/MLB-Guru-Data/data"
DATA_DIR = LOCAL_DATA if os.path.exists(LOCAL_DATA) else "."

MLB_TEAM_MAP = {
    "arizona diamondbacks": "ARI", "diamondbacks": "ARI",
    "atlanta braves": "ATL", "braves": "ATL",
    "baltimore orioles": "BAL", "orioles": "BAL",
    "boston red sox": "BOS", "red sox": "BOS",
    "chicago cubs": "CHC", "cubs": "CHC",
    "chicago white sox": "CWS", "white sox": "CWS",
    "cincinnati reds": "CIN", "reds": "CIN",
    "cleveland guardians": "CLE", "guardians": "CLE",
    "colorado rockies": "COL", "rockies": "COL",
    "detroit tigers": "DET", "tigers": "DET",
    "houston astros": "HOU", "astros": "HOU",
    "kansas city royals": "KC", "royals": "KC",
    "los angeles angels": "LAA", "angels": "LAA",
    "los angeles dodgers": "LAD", "dodgers": "LAD",
    "miami marlins": "MIA", "marlins": "MIA",
    "milwaukee brewers": "MIL", "brewers": "MIL",
    "minnesota twins": "MIN", "twins": "MIN",
    "new york mets": "NYM", "mets": "NYM",
    "new york yankees": "NYY", "yankees": "NYY",
    "oakland athletics": "OAK", "athletics": "OAK", "a's": "OAK",
    "philadelphia phillies": "PHI", "phillies": "PHI",
    "pittsburgh pirates": "PIT", "pirates": "PIT",
    "san diego padres": "SD", "padres": "SD",
    "san francisco giants": "SF", "giants": "SF",
    "seattle mariners": "SEA", "mariners": "SEA",
    "st. louis cardinals": "STL", "cardinals": "STL",
    "tampa bay rays": "TB", "rays": "TB",
    "texas rangers": "TEX", "rangers": "TEX",
    "toronto blue jays": "TOR", "blue jays": "TOR",
    "washington nationals": "WSH", "nationals": "WSH"
}

def to_american(prob: float) -> str:
    if prob >= 0.99: return "-10000"
    if prob <= 0.01: return "+10000"
    if prob >= 0.5:
        return f"-{int(round(100.0 * (prob / (1.0 - prob))))}"
    return f"+{int(round(100.0 * ((1.0 - prob) / prob)))}"

def clean_numeric(series):
    def extract_float(val):
        if pd.isna(val): return np.nan
        val_str = str(val).strip().replace('"', '')
        match = re.search(r"[-+]?\d*\.?\d+", val_str)
        return float(match.group(0)) if match else np.nan
    return series.apply(extract_float)

def get_team_abbr(name):
    if not name: return None
    return MLB_TEAM_MAP.get(str(name).strip().lower(), None)

def run_strikeout_prop_engine(df_pitchers, df_lineups, df_nrfi):
    k_prop_rows = []
    for _, matchup in df_nrfi.iterrows():
        game_pk = str(matchup['game_pk'])
        venue = matchup['venue_name']
        configs = [
            ("Home SP", matchup['home_pitcher'], matchup['home_team'], matchup['away_team']),
            ("Away SP", matchup['away_pitcher'], matchup['away_team'], matchup['home_team'])
        ]
        for role, sp_name, team_name, opp_team in configs:
            sp_matches = df_pitchers[df_pitchers['pitcher_name'] == sp_name]
            if sp_matches.empty: continue
            sp = sp_matches.iloc[0]

            opp_lineup = df_lineups[
                (df_lineups['game_pk'].astype(str) == game_pk) & 
                (df_lineups['team_name'] == opp_team)
            ].sort_values(by='batting_order')
            if opp_lineup.empty: continue

            csw = float(sp['csw_pct'])
            era = float(sp['era'])
            base_k_rate = max(0.12, min(0.38, 1.40 * csw - 0.10))

            lineup_k_factor = float(opp_lineup['k_pct_vs_hand'].mean()) / 0.225
            lineup_whiff_factor = float(opp_lineup['whiff_pct_vs_hand'].mean()) / 0.245
            lineup_k_mult = (0.60 * lineup_k_factor) + (0.40 * lineup_whiff_factor)

            exp_ip = min(6.3, max(4.7, 6.2 - (era * 0.25)))
            exp_tbf = exp_ip * 4.15
            lambda_k = exp_tbf * (base_k_rate * lineup_k_mult)

            var_k = 1.22 * lambda_k
            r_param = (lambda_k ** 2) / (var_k - lambda_k)
            p_param = r_param / (r_param + lambda_k)

            for line in [3.5, 4.5, 5.5, 6.5]:
                prob_under = float(nbinom.cdf(np.floor(line), r_param, p_param))
                prob_over = 1.0 - prob_under
                k_prop_rows.append({
                    "game_pk": game_pk, "pitcher": sp_name, "team": team_name,
                    "opp_team": opp_team, "venue": venue, "lambda_k": round(lambda_k, 2),
                    "k_line": line, "over_prob": round(prob_over, 4), "under_prob": round(prob_under, 4),
                    "fair_over": to_american(prob_over), "fair_under": to_american(prob_under)
                })
    return pd.DataFrame(k_prop_rows)

class CleanMarkovNRFI:
    def __init__(self):
        self.num_transient = 24
        self.state_3outs = 24
        self.state_run = 25
        self.total_states = 26

    def _state_idx(self, base_config: int, outs: int) -> int:
        if outs >= 3: return self.state_3outs
        return (outs * 8) + base_config

    def build_transition_matrix(self, pa: dict) -> np.ndarray:
        T = np.zeros((self.total_states, self.total_states), dtype=float)
        T[self.state_3outs, self.state_3outs] = 1.0
        T[self.state_run, self.state_run] = 1.0

        p_k, p_bb, p_1b, p_2b, p_3b, p_hr, p_out = (
            pa['p_k'], pa['p_bb'], pa['p_1b'], pa['p_2b'], pa['p_3b'], pa['p_hr'], pa['p_out']
        )
        for outs in range(3):
            for base in range(8):
                curr = self._state_idx(base, outs)
                T[curr, self._state_idx(base, outs + 1)] += p_k
                if base == 0: nxt_bb = self._state_idx(1, outs)
                elif base in [1, 2]: nxt_bb = self._state_idx(4, outs)
                elif base == 3: nxt_bb = self._state_idx(5, outs)
                elif base in [4, 5, 6]: nxt_bb = self._state_idx(7, outs)
                elif base == 7: nxt_bb = self.state_run
                T[curr, nxt_bb] += p_bb

                if base == 0: T[curr, self._state_idx(1, outs)] += p_1b
                elif base == 1:
                    T[curr, self._state_idx(4, outs)] += p_1b * 0.70
                    T[curr, self._state_idx(5, outs)] += p_1b * 0.30
                elif base == 2:
                    T[curr, self.state_run] += p_1b * 0.60
                    T[curr, self._state_idx(5, outs)] += p_1b * 0.40
                elif base in [3, 5, 6, 7]: T[curr, self.state_run] += p_1b
                elif base == 4:
                    T[curr, self.state_run] += p_1b * 0.60
                    T[curr, self._state_idx(7, outs)] += p_1b * 0.40

                if base == 0: T[curr, self._state_idx(2, outs)] += p_2b
                elif base == 1:
                    T[curr, self.state_run] += p_2b * 0.45
                    T[curr, self._state_idx(6, outs)] += p_2b * 0.55
                else: T[curr, self.state_run] += p_2b

                if base == 0: T[curr, self._state_idx(3, outs)] += p_3b
                else: T[curr, self.state_run] += p_3b
                T[curr, self.state_run] += p_hr

                if outs == 2: T[curr, self.state_3outs] += p_out
                else:
                    if base == 0: T[curr, self._state_idx(0, outs + 1)] += p_out
                    elif base == 1:
                        T[curr, self._state_idx(0, outs + 2)] += p_out * 0.09
                        T[curr, self._state_idx(1, outs + 1)] += p_out * 0.65
                        T[curr, self._state_idx(2, outs + 1)] += p_out * 0.26
                    elif base in [3, 5, 6, 7]:
                        T[curr, self.state_run] += p_out * 0.24
                        T[curr, self._state_idx(base, outs + 1)] += p_out * 0.76
                    else:
                        T[curr, self._state_idx(base, outs + 1)] += p_out * 0.75
                        T[curr, self._state_idx(3 if base == 2 else 6, outs + 1)] += p_out * 0.25

        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return T / row_sums

    def solve_half_inning(self, batter_pas: list) -> float:
        v = np.zeros(self.total_states, dtype=float)
        v[0] = 1.0
        for pa in batter_pas:
            T_k = self.build_transition_matrix(pa)
            v = v @ T_k
            if (v[self.state_3outs] + v[self.state_run]) > 0.9999: break
        return float(v[self.state_3outs])

def derive_pa_probs(batter_row, sp_row, framing=0.0, wind_supp=0.0):
    b_k = float(batter_row['k_pct_vs_hand']) if pd.notna(batter_row['k_pct_vs_hand']) else 0.225
    b_bb = float(batter_row['bb_pct_vs_hand']) if pd.notna(batter_row['bb_pct_vs_hand']) else 0.082
    b_xwoba = float(batter_row['xwoba_vs_hand']) if pd.notna(batter_row['xwoba_vs_hand']) else 0.315

    p_k = float(sp_row['tto1_k_pct']) if pd.notna(sp_row['tto1_k_pct']) and sp_row['tto1_k_pct'] > 0 else float(sp_row['csw_pct']) * 0.95
    p_xwoba = float(sp_row['tto1_xwoba']) if pd.notna(sp_row['tto1_xwoba']) and sp_row['tto1_xwoba'] > 0 else 0.310

    p_k_final = min(0.48, max(0.08, (b_k * p_k / 0.225) + framing))
    p_bb_final = min(0.20, max(0.03, b_bb * (p_xwoba / 0.315)))
    matchup_xwoba = (b_xwoba * 0.55 + p_xwoba * 0.45) * (1.0 - wind_supp)

    p_hr = min(0.09, max(0.01, 0.032 * (matchup_xwoba / 0.315)**1.5))
    p_3b = 0.005
    p_2b = min(0.08, max(0.02, 0.045 * (matchup_xwoba / 0.315)))
    p_1b = min(0.25, max(0.08, 0.150 * (matchup_xwoba / 0.315)))
    p_out = max(0.30, 1.0 - (p_k_final + p_bb_final + p_1b + p_2b + p_3b + p_hr))

    return {'p_k': p_k_final, 'p_bb': p_bb_final, 'p_1b': p_1b, 'p_2b': p_2b, 'p_3b': p_3b, 'p_hr': p_hr, 'p_out': p_out}

def run_markov_nrfi_engine(df_pitchers, df_lineups, df_nrfi):
    markov = CleanMarkovNRFI()
    nrfi_rows = []
    for _, matchup in df_nrfi.iterrows():
        game_pk = str(matchup['game_pk'])
        sp_home = df_pitchers[df_pitchers['pitcher_name'] == matchup['home_pitcher']]
        sp_away = df_pitchers[df_pitchers['pitcher_name'] == matchup['away_pitcher']]
        if sp_home.empty or sp_away.empty: continue
        sp_h, sp_a = sp_home.iloc[0], sp_away.iloc[0]

        away_batters = df_lineups[(df_lineups['game_pk'].astype(str) == game_pk) & (df_lineups['team_name'] == matchup['away_team'])].sort_values(by='batting_order')
        home_batters = df_lineups[(df_lineups['game_pk'].astype(str) == game_pk) & (df_lineups['team_name'] == matchup['home_team'])].sort_values(by='batting_order')
        if len(away_batters) < 3 or len(home_batters) < 3: continue

        framing = float(matchup.get('catcher_framing_delta', 0.0))
        wind_supp = float(matchup.get('park_wind_suppression', 0.0))

        p_top = markov.solve_half_inning([derive_pa_probs(row, sp_h, framing, wind_supp) for _, row in away_batters.iterrows()])
        p_bot = markov.solve_half_inning([derive_pa_probs(row, sp_a, framing, wind_supp) for _, row in home_batters.iterrows()])
        p_nrfi = p_top * p_bot

        nrfi_rows.append({
            "game_pk": game_pk, "matchup": f"{matchup['away_team']} @ {matchup['home_team']}",
            "nrfi_prob": round(p_nrfi, 4), "yrfi_prob": round(1.0 - p_nrfi, 4),
            "fair_nrfi": to_american(p_nrfi), "fair_yrfi": to_american(1.0 - p_nrfi)
        })
    return pd.DataFrame(nrfi_rows)

def run_f5_engine(df_pitchers, df_lineups, df_nrfi):
    f5_rows = []
    for _, matchup in df_nrfi.iterrows():
        game_pk = str(matchup['game_pk'])
        sp_home = df_pitchers[df_pitchers['pitcher_name'] == matchup['home_pitcher']]
        sp_away = df_pitchers[df_pitchers['pitcher_name'] == matchup['away_pitcher']]
        if sp_home.empty or sp_away.empty: continue
        sp_h, sp_a = sp_home.iloc[0], sp_away.iloc[0]

        away_top5 = df_lineups[(df_lineups['game_pk'].astype(str) == game_pk) & (df_lineups['team_name'] == matchup['away_team'])].sort_values(by='batting_order').head(5)
        home_top5 = df_lineups[(df_lineups['game_pk'].astype(str) == game_pk) & (df_lineups['team_name'] == matchup['home_team'])].sort_values(by='batting_order').head(5)
        if len(away_top5) < 3 or len(home_top5) < 3: continue

        base_f5 = 2.35
        away_top5_xwoba = float(away_top5['xwoba_vs_hand'].mean())
        home_top5_xwoba = float(home_top5['xwoba_vs_hand'].mean())

        sp_h_gb = max(0.85, 1.0 - (float(sp_h.get('groundballs_pct', 0.42)) - 0.42) * 0.4)
        sp_a_gb = max(0.85, 1.0 - (float(sp_a.get('groundballs_pct', 0.42)) - 0.42) * 0.4)

        exp_away_f5 = base_f5 * (away_top5_xwoba / 0.315) * (float(sp_h['xwoba']) / 0.315) * sp_h_gb
        exp_home_f5 = (base_f5 * 1.04) * (home_top5_xwoba / 0.315) * (float(sp_a['xwoba']) / 0.315) * sp_a_gb

        away_5 = np.random.poisson(exp_away_f5 / 5.0, size=(12000, 5)).sum(axis=1)
        home_5 = np.random.poisson(exp_home_f5 / 5.0, size=(12000, 5)).sum(axis=1)
        margin = home_5 - away_5

        h_win = float(np.mean(margin > 0))
        a_win = float(np.mean(margin < 0))
        decisive = h_win + a_win
        home_ml = h_win / decisive if decisive > 0 else 0.50
        away_ml = a_win / decisive if decisive > 0 else 0.50

        f5_rows.append({
            "game_pk": game_pk, "matchup": f"{matchup['away_team']} @ {matchup['home_team']}",
            "home_f5_ml": round(home_ml, 4), "away_f5_ml": round(away_ml, 4),
            "home_cover_m05": round(h_win, 4), "away_cover_p05": round(1.0 - h_win, 4)
        })
    return pd.DataFrame(f5_rows)

def run_full_game_ensemble(df_pitchers, df_lineups, df_bullpen, df_nrfi):
    ensemble_rows = []
    for _, matchup in df_nrfi.iterrows():
        game_pk = str(matchup['game_pk'])
        home_team, away_team = matchup['home_team'], matchup['away_team']
        sp_h_match = df_pitchers[df_pitchers['pitcher_name'] == matchup['home_pitcher']]
        sp_a_match = df_pitchers[df_pitchers['pitcher_name'] == matchup['away_pitcher']]
        if sp_h_match.empty or sp_a_match.empty: continue
        sp_h, sp_a = sp_h_match.iloc[0], sp_a_match.iloc[0]

        away_lineup = df_lineups[(df_lineups['game_pk'].astype(str) == game_pk) & (df_lineups['team_name'] == away_team)]
        home_lineup = df_lineups[(df_lineups['game_pk'].astype(str) == game_pk) & (df_lineups['team_name'] == home_team)]
        if len(away_lineup) < 9 or len(home_lineup) < 9: continue

        away_xwoba = float(away_lineup['xwoba_vs_hand'].mean())
        home_xwoba = float(home_lineup['xwoba_vs_hand'].mean())

        bp_h = df_bullpen[df_bullpen['team_name'] == home_team]
        bp_a = df_bullpen[df_bullpen['team_name'] == away_team]
        bp_h_mult = float(bp_h['bullpen_fatigue_mult'].iloc[0]) if not bp_h.empty else 0.02
        bp_a_mult = float(bp_a['bullpen_fatigue_mult'].iloc[0]) if not bp_a.empty else 0.02

        sp_away_runs = 2.35 * (away_xwoba / 0.315) * (float(sp_h['xwoba']) / 0.315)
        sp_home_runs = 2.44 * (home_xwoba / 0.315) * (float(sp_a['xwoba']) / 0.315)

        bp_away_runs = 4.0 * 0.43 * (away_xwoba / 0.315) * (1.0 + bp_h_mult)
        bp_home_runs = 4.0 * 0.43 * (home_xwoba / 0.315) * (1.0 + bp_a_mult) * 1.04

        tot_away = sp_away_runs + bp_away_runs
        tot_home = sp_home_runs + bp_home_runs

        diff_xwoba = home_xwoba - away_xwoba
        diff_era = float(sp_a['era']) - float(sp_h['era'])
        diff_bfi = bp_a_mult - bp_h_mult
        logit = 0.04 + (diff_xwoba * 2.8) + (diff_era * 0.08) + (diff_bfi * 1.5)
        p_lgbm_h = 1.0 / (1.0 + np.exp(-logit))

        away_sim = np.random.poisson(tot_away / 9.0, size=(15000, 9)).sum(axis=1)
        home_sim = np.random.poisson(tot_home / 9.0, size=(15000, 9)).sum(axis=1)
        ties = away_sim == home_sim
        home_sim[ties] += np.random.choice([1, 0], size=np.sum(ties), p=[0.53, 0.47])
        away_sim[ties] += (1 - (home_sim[ties] - away_sim[ties]))

        margin = home_sim - away_sim
        mc_home_ml = float(np.mean(margin > 0))
        mc_home_m15 = float(np.mean(margin >= 2))
        mc_away_p15 = float(np.mean(margin <= 1))

        p_final_home = (0.70 * p_lgbm_h) + (0.30 * mc_home_ml)
        p_final_away = 1.0 - p_final_home

        ensemble_rows.append({
            "game_pk": game_pk, "matchup": f"{away_team} @ {home_team}",
            "home_ml_prob": round(p_final_home, 4), "away_ml_prob": round(p_final_away, 4),
            "home_m15_prob": round(mc_home_m15, 4), "away_p15_prob": round(mc_away_p15, 4)
        })
    return pd.DataFrame(ensemble_rows)

def scan_live_market():
    print("=" * 85)
    print("EXECUTING MULTI-MODEL PRODUCTION SCANNER")
    print("=" * 85)

    df_pitchers = pd.read_csv(os.path.join(DATA_DIR, "pitcher_statcast_rolling.csv"))
    df_lineups = pd.read_csv(os.path.join(DATA_DIR, "lineup_statcast_splits.csv"))
    df_bullpen = pd.read_csv(os.path.join(DATA_DIR, "bullpen_leverage_index.csv"))
    df_nrfi = pd.read_csv(os.path.join(DATA_DIR, "nrfi_context_splits.csv"))

    for c in ['xwoba_vs_hand', 'k_pct_vs_hand', 'bb_pct_vs_hand', 'whiff_pct_vs_hand']:
        df_lineups[c] = clean_numeric(df_lineups[c])
    for c in ['era', 'xwoba', 'csw_pct', 'whiff_pct', 'groundballs_pct', 'tto1_k_pct', 'tto1_xwoba']:
        df_pitchers[c] = clean_numeric(df_pitchers[c])
    df_bullpen['bullpen_fatigue_mult'] = clean_numeric(df_bullpen['bullpen_fatigue_mult'])

    df_k = run_strikeout_prop_engine(df_pitchers, df_lineups, df_nrfi)
    df_nrfi_out = run_markov_nrfi_engine(df_pitchers, df_lineups, df_nrfi)
    df_f5 = run_f5_engine(df_pitchers, df_lineups, df_nrfi)
    df_full = run_full_game_ensemble(df_pitchers, df_lineups, df_bullpen, df_nrfi)

    print(f"Executed 4 Models: K-Props ({len(df_k)}), NRFI ({len(df_nrfi_out)}), F5 ({len(df_f5)}), Full ({len(df_full)})")

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,spreads", "oddsFormat": "decimal"}
    try:
        res = requests.get(url, params=params, timeout=12)
        odds_events = res.json() if res.status_code == 200 else []
    except Exception:
        odds_events = []
    print(f"Retrieved {len(odds_events)} live game trees from The Odds API.")

    qualifying_bets = []

    for _, game in df_full.iterrows():
        away_name, home_name = [t.strip() for t in game["matchup"].split("@")]
        away_abbr, home_abbr = get_team_abbr(away_name), get_team_abbr(home_name)

        matched_event = next((
            ev for ev in odds_events
            if get_team_abbr(ev.get("away_team", "")) == away_abbr and
               get_team_abbr(ev.get("home_team", "")) == home_abbr
        ), None)
        if not matched_event: continue

        for book in matched_event.get("bookmakers", []):
            book_key = book.get("key")
            if book_key not in TARGET_BOOKS: continue

            for mkt in book.get("markets", []):
                if mkt["key"] == "h2h":
                    for outcome in mkt.get("outcomes", []):
                        pick_abbr = get_team_abbr(outcome.get("name"))
                        dec_odds = float(outcome.get("price", 1.0))
                        if not pick_abbr or dec_odds <= 1.0: continue

                        implied_p = 1.0 / dec_odds
                        is_home = (pick_abbr == home_abbr)
                        model_p = float(game["home_ml_prob"]) if is_home else float(game["away_ml_prob"])
                        edge = model_p - implied_p

                        if MIN_EDGE <= edge <= MAX_EDGE:
                            b = dec_odds - 1.0
                            raw_kelly = (model_p * b - (1.0 - model_p)) / b
                            stake = min(2.00, max(0.75, round(raw_kelly * 0.25 * 10.0, 2)))
                            qualifying_bets.append({
                                "composite_key": str(datetime.date.today()) + "_"
