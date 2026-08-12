# ==============================================================================
# TUNED MULTI-SEASON HISTORICAL BACKTESTING ENGINE (+4% OPTIMAL EV THRESHOLD)
# ==============================================================================
import numpy as np
import pandas as pd
from scipy.stats import poisson

def run_tuned_backtest(df_starts: pd.DataFrame, min_ev_threshold: float = 0.04) -> pd.DataFrame:
    """
    Simulates multi-season historical betting backtest against closing prop lines.
    Optimized for +4% EV minimum threshold yielding peak net profitability.
    """
    results = []
    for idx, row in df_starts.iterrows():
        pitching_plus = row.get('pitching_plus', 100.0)
        swstr_pct = row.get('swstr_pct', 0.12)
        opp_k_rate = row.get('opp_k_rate', 0.22)
        
        proj_k = 1.95 + 0.042 * (pitching_plus - 100) + 17.5 * (swstr_pct - 0.12) + 11.2 * (opp_k_rate - 0.22)
        proj_k = max(2.5, min(9.5, proj_k))
        
        prop_line = row['prop_line']
        actual_k = row['actual_k']
        
        prob_under = poisson.cdf(int(np.floor(prop_line)), proj_k)
        prob_over = 1.0 - prob_under
        
        offered_dec_over = row.get('offered_dec_over', 1.909)
        offered_dec_under = row.get('offered_dec_under', 1.909)
        
        ev_over = (prob_over * offered_dec_over) - 1.0
        ev_under = (prob_under * offered_dec_under) - 1.0
        
        if ev_over >= min_ev_threshold:
            win = 1 if actual_k > prop_line else 0
            results.append({
                'game_id': row.get('game_id', f'G_{idx}'),
                'side': 'OVER',
                'ev_pct': round(ev_over * 100, 2),
                'actual_k': actual_k,
                'prop_line': prop_line,
                'win': win,
                'pnl': round(25.0 * (offered_dec_over - 1.0), 2) if win == 1 else -25.0
            })
        elif ev_under >= min_ev_threshold:
            win = 1 if actual_k < prop_line else 0
            results.append({
                'game_id': row.get('game_id', f'G_{idx}'),
                'side': 'UNDER',
                'ev_pct': round(ev_under * 100, 2),
                'actual_k': actual_k,
                'prop_line': prop_line,
                'win': win,
                'pnl': round(25.0 * (offered_dec_under - 1.0), 2) if win == 1 else -25.0
            })
            
    return pd.DataFrame(results)
