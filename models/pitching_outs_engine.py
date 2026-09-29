import os
import json
import numpy as np
import pandas as pd

BASE_DIR = '/content/drive/MyDrive/MLB-Playoffs-Data'
OVERRIDE_PATH = os.path.join(BASE_DIR, 'data', 'manager_context_overrides.json')

def load_manager_overrides() -> dict:
    if os.path.exists(OVERRIDE_PATH):
        try:
            with open(OVERRIDE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[-] Warning: Failed to parse manager overrides: {e}")
    return {}

def simulate_pitcher_outing(
    pitcher_name: str,
    baseline_pitch_cap: int = 85,      # Postseason baseline default reduced from 95
    csw_rate: float = 0.285,
    bb_rate: float = 0.082,
    lineup_obp: float = 0.315,
    tto_limit: float = 2.2,            # Postseason TTO default compressed from 3.0
    is_tandem: bool = False,
    tandem_max_pitches: int = 60,
    n_sims: int = 10000
) -> dict:
    overrides = load_manager_overrides()
    
    if pitcher_name in overrides:
        effective_cap = overrides[pitcher_name].get('pitch_cap', baseline_pitch_cap)
        effective_tto = overrides[pitcher_name].get('tto_limit', tto_limit)
        tag = overrides[pitcher_name].get('leash_type', 'BEAT_OVERRIDE')
    else:
        effective_cap = baseline_pitch_cap
        effective_tto = tto_limit
        tag = 'POSTSEASON_DEFAULT'

    if is_tandem:
        effective_cap = min(effective_cap, tandem_max_pitches)
        effective_tto = min(effective_tto, 1.8)
        tag = 'TANDEM_PIGGYBACK'

    # Expected pitches per PA: conditioned on CSW% and BB%
    expected_p_pa = 3.90 - 1.85 * (csw_rate - 0.285) + 1.20 * (bb_rate - 0.082)
    expected_p_pa = max(3.30, min(expected_p_pa, 4.40))

    simulated_outs = np.zeros(n_sims, dtype=int)
    simulated_pitches = np.zeros(n_sims, dtype=int)

    for i in range(n_sims):
        total_outs = 0
        total_pitches = 0
        batters_faced = 0
        inning = 1

        while inning <= 9:
            inning_outs = 0
            inning_pitches = 0
            inning_baserunners = 0

            while inning_outs < 3:
                pa_pitches = np.random.geometric(p=1.0 / expected_p_pa)
                inning_pitches += pa_pitches
                total_pitches += pa_pitches
                batters_faced += 1

                # Postseason Emergency Hook 1: Total Pitch Cap breached
                if total_pitches >= effective_cap:
                    break
                
                # Postseason Emergency Hook 2: In-Inning High Stress (compressed from 32 down to 26 pitches)
                if inning_pitches >= 26 and inning_baserunners >= 2:
                    break

                is_on_base = np.random.rand() < lineup_obp
                if is_on_base:
                    inning_baserunners += 1
                else:
                    inning_outs += 1
                    total_outs += 1

            # Check if pitcher failed to finish the inning
            if total_pitches >= effective_cap or inning_outs < 3:
                break

            # Postseason Between-Inning Managerial Hook:
            # Hooks trigger earlier: within 12 pitches of cap or crossing postseason TTO threshold
            times_through = batters_faced / 9.0
            if total_pitches >= (effective_cap - 12) or times_through >= effective_tto:
                break

            inning += 1

        simulated_outs[i] = total_outs
        simulated_pitches[i] = total_pitches

    return {
        'pitcher': pitcher_name,
        'effective_pitch_cap': effective_cap,
        'effective_tto': effective_tto,
        'leash_profile': tag,
        'projected_outs_mean': round(float(np.mean(simulated_outs)), 2),
        'prob_under_13_5': round(float(np.mean(simulated_outs < 13.5)), 4),
        'prob_under_14_5': round(float(np.mean(simulated_outs < 14.5)), 4),
        'prob_under_15_5': round(float(np.mean(simulated_outs < 15.5)), 4),
        'prob_over_14_5': round(float(np.mean(simulated_outs > 14.5)), 4),
        'prob_over_15_5': round(float(np.mean(simulated_outs > 15.5)), 4),
        'prob_over_17_5': round(float(np.mean(simulated_outs > 17.5)), 4),
        'sim_p25_outs': int(np.percentile(simulated_outs, 25)),
        'sim_p50_outs': int(np.percentile(simulated_outs, 50)),
        'sim_p75_outs': int(np.percentile(simulated_outs, 75))
    }
