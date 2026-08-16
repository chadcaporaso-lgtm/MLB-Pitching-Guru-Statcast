# ⚾ MLB-Pitching-Guru-Statcast

An end-to-end quantitative MLB analytics and predictive betting pipeline powered by Statcast metrics, automated market efficiency tracking, and live odds integration.

---

## 🚀 Repository Architecture

* **`src/models.py`**: Quantitative simulation and pricing models covering:
  * **Moneyline & Team Totals**: Poisson score simulations with regressed ERA/bullpen shrinkage.
  * **Run Line (+1.5 / -1.5)**: Discrete margin distribution engine factoring extra innings and bottom-of-the-9th home lead asymmetries.
  * **Game Totals (Over/Under)**: Push-adjusted Poisson Monte Carlo scoring distribution engine.
  * **NRFI / YRFI (1st Inning)**: Negative Binomial overdispersion model (alpha = 0.65) capturing empirical MLB scoreless half-inning clustering.
  * **Correlated Same Game Parlays (SGP)**: Gaussian Copula simulation modeling negative correlation (rho ≈ -0.32) between starter strikeouts and opposing team runs.
* **`src/live_pipeline.py`**: Dynamic data hydrator connecting directly to the MLB Stats API (for live probable starters, K/9 baselines, workload limits) and The Odds API for real-time market prices.
* **`src/clv_tracker.py`**: Closing Line Value (CLV) logging and market efficiency tracking module to benchmark opening wagers against sharp closing lines.
* **`src/grade_bets.py`**: Automated postgame grader querying official MLB box scores to grade W/L outcomes, compute net unit profit, and generate ROI reports.
* **`src/backtest_engine.py`**: Historical simulation framework calibrated for +EV threshold optimization.

---

## 📊 Core Betting Engines & Mathematical Formulations

### 1. Volumetric Pitcher Strikeout Model
$$\text{Proj K} = \left(\frac{\text{K/9}}{9}\right) \times \text{Exp IP} \times \text{Matchup Factors}$$

### 2. Moneyline, Run Line & Game Totals
* **Regressed Pitching Baseline:** Composite ERA blends raw starter ERA (60%) with league baseline (4.25) and bullpen baselines (3.95 across 3.9 IP).
* **15,000 Monte Carlo Poisson Simulations:** Derives fair zero-vig American lines for outright win rates and cover probabilities:
  * **Favorite -1.5:** Run Differential >= 2
  * **Underdog +1.5:** Run Differential >= -1

### 3. NRFI / YRFI First Inning Expectancy
Uses Negative Binomial run clustering (alpha = 0.65) to accurately reflect MLB half-inning scoreless rates (~72.5% base):
$$P(\text{Scoreless Half}) = (1 + \alpha \lambda)^{-1 / \alpha}$$
$$P(\text{NRFI}) = P(\text{Away } 0) \times P(\text{Home } 0), \quad P(\text{YRFI}) = 1.0 - P(\text{NRFI})$$

### 4. Closing Line Value (CLV) Tracker
Measures edge retention against sharp market closing lines:
$$\text{CLV \%} = \left(\frac{\text{Placed Decimal Odds}}{\text{Closing Decimal Odds}} - 1.0\right) \times 100$$

---

## 🛠️ Usage Example

```python
from live_pipeline import fetch_live_slate, fetch_pitcher_baseline
from models import simulate_moneyline, simulate_run_line, simulate_nrfi_yrfi, simulate_game_totals
from clv_tracker import log_bet
from grade_bets import grade_all_bets, generate_performance_report

# 1. Pull Live Slate & Run Projections
slate = fetch_live_slate()

# 2. Log Placed Bets to CLV Tracker
log_bet(
    game_id="2026-08-16_MIL_LAD",
    market_type="NRFI/YRFI",
    selection="MIL @ LAD NRFI",
    bet_odds_american=-110,
    model_prob=0.646,
    stake_units=1.25
)

# 3. Grade Results Postgame
grade_all_bets(target_date="2026-08-16")
display(generate_performance_report())
```
