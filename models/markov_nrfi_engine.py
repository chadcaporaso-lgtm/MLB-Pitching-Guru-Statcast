"""
24-State Discrete Base-Out Markov Chain for 1st-Inning NRFI/YRFI Pricing.
Evaluates 8 base configurations x 3 out states + 2 absorbing states:
- State 24: 3 Outs, 0 Runs Scored (NRFI Win)
- State 25: Run Scored (YRFI Triggered)
"""

import numpy as np
import pandas as pd

class MarkovNRFIEngine:
    BASE_STATES = {
        0: "---", 1: "1--", 2: "-2-", 3: "--3",
        4: "12-", 5: "1-3", 6: "-23", 7: "123"
    }

    def __init__(self):
        self.num_transient = 24  # 8 base configs * 3 outs
        self.state_3outs = 24    # Absorbing state: 3 outs, 0 runs
        self.state_run = 25      # Absorbing state: >= 1 run scored
        self.total_states = 26

    def _state_idx(self, base_config: int, outs: int) -> int:
        if outs >= 3:
            return self.state_3outs
        return (outs * 8) + base_config

    def build_transition_matrix(self, pa_probs: dict) -> np.ndarray:
        T = np.zeros((self.total_states, self.total_states), dtype=float)
        T[self.state_3outs, self.state_3outs] = 1.0
        T[self.state_run, self.state_run] = 1.0

        for outs in range(3):
            for base in range(8):
                curr = self._state_idx(base, outs)

                # 1. Strikeout (K)
                T[curr, self._state_idx(base, outs + 1)] += pa_probs['p_k']

                # 2. Walk / HBP (BB)
                if base == 0:    nxt_bb = self._state_idx(1, outs)
                elif base in [1, 2]: nxt_bb = self._state_idx(4, outs)
                elif base == 3:  nxt_bb = self._state_idx(5, outs)
                elif base in [4, 5, 6]: nxt_bb = self._state_idx(7, outs)
                elif base == 7:  nxt_bb = self.state_run
                T[curr, nxt_bb] += pa_probs['p_bb']

                # 3. Single (1B)
                if base == 0:
                    T[curr, self._state_idx(1, outs)] += pa_probs['p_1b']
                elif base == 1:
                    T[curr, self._state_idx(4, outs)] += pa_probs['p_1b'] * 0.70
                    T[curr, self._state_idx(5, outs)] += pa_probs['p_1b'] * 0.30
                elif base == 2:
                    T[curr, self.state_run] += pa_probs['p_1b'] * 0.60
                    T[curr, self._state_idx(5, outs)] += pa_probs['p_1b'] * 0.40
                elif base in [3, 5, 6, 7]:
                    T[curr, self.state_run] += pa_probs['p_1b']
                elif base == 4:
                    T[curr, self.state_run] += pa_probs['p_1b'] * 0.60
                    T[curr, self._state_idx(7, outs)] += pa_probs['p_1b'] * 0.40

                # 4. Double (2B)
                if base == 0:
                    T[curr, self._state_idx(2, outs)] += pa_probs['p_2b']
                elif base == 1:
                    T[curr, self.state_run] += pa_probs['p_2b'] * 0.45
                    T[curr, self._state_idx(6, outs)] += pa_probs['p_2b'] * 0.55
                else:
                    T[curr, self.state_run] += pa_probs['p_2b']

                # 5. Triple (3B) & HR
                if base == 0:
                    T[curr, self._state_idx(3, outs)] += pa_probs['p_3b']
                else:
                    T[curr, self.state_run] += pa_probs['p_3b']
                T[curr, self.state_run] += pa_probs['p_hr']

                # 6. Field Out (p_out)
                if outs == 2:
                    T[curr, self.state_3outs] += pa_probs['p_out']
                else:
                    if base == 0:
                        T[curr, self._state_idx(0, outs + 1)] += pa_probs['p_out']
                    elif base == 1:
                        T[curr, self._state_idx(0, outs + 2)] += pa_probs['p_out'] * 0.09  # GDP
                        T[curr, self._state_idx(1, outs + 1)] += pa_probs['p_out'] * 0.65
                        T[curr, self._state_idx(2, outs + 1)] += pa_probs['p_out'] * 0.26
                    elif base in [3, 5, 6, 7]:
                        T[curr, self.state_run] += pa_probs['p_out'] * 0.24  # Sac Fly
                        T[curr, self._state_idx(base, outs + 1)] += pa_probs['p_out'] * 0.76
                    else:
                        T[curr, self._state_idx(base, outs + 1)] += pa_probs['p_out'] * 0.75
                        T[curr, self._state_idx(3 if base == 2 else 6, outs + 1)] += pa_probs['p_out'] * 0.25

        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return T / row_sums

    def solve_sequential_lineup(self, lineup_pas: list) -> float:
        v = np.zeros(self.total_states, dtype=float)
        v[0] = 1.0  # Bases empty, 0 outs
        for pa in lineup_pas:
            T_k = self.build_transition_matrix(pa)
            v = v @ T_k
            if (v[self.state_3outs] + v[self.state_run]) > 0.9999:
                break
        return float(v[self.state_3outs])