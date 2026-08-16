# ⚾ MLB-Pitching-Guru-Statcast

An end-to-end quantitative MLB analytics and predictive betting pipeline powered by Statcast metrics, automated market efficiency tracking, and live odds integration.

---

## 🚀 Repository Architecture

* **`src/models.py`**: Quantitative modeling engines covering:
  * **Moneyline & Team Totals**: Poisson score simulations with regressed ERA/bullpen shrinkage.
  * **Run Line (+1.5 / -1.5)**: Discrete margin distribution engine factoring extra innings and bottom-of-the-9th home lead asymmetries.
  * **Correlated Same Game Parlays (SGP)**: Gaussian Copula simulation modeling negative correlation (rho ≈ -0.32) between starter strikeouts and opposing team run totals.
* **`src/live_pipeline.py`**: Dynamic data hydrator connecting directly to the MLB Stats API (for live probable starters, K/9 baselines, workload limits) and The Odds API for real-time market prices.
* **`src/clv_tracker.py`**: Closing Line Value (CLV) logging and market efficiency tracking module to benchmark opening wagers against sharp closing lines.
* **`src/backtest_engine.py`**: Historical simulation framework calibrated for +EV threshold optimization.

---

## 📊 Core Betting Engines & Formulas

### 1. Volumetric Pitcher Strikeout Model
Proj K = (K/9 / 9) * Exp IP * Matchup Factors

### 2. Moneyline & Run Line (+1.5 / -1.5) Expectancy
* **Regressed Pitching Factor:** Blends raw starter ERA (60%) with league baseline (4.25) and bullpen baselines (3.95 across 3.9 IP).
* **15,000 Monte Carlo Poisson Simulations:** Derives fair zero-vig American lines for outright win rates and cover probabilities:
  * **Favorite -1.5:** Run Differential >= 2
  * **Underdog +1.5:** Run Differential >= -1

### 3. Closing Line Value (CLV) Tracker
Measures edge retention against market closing lines:
CLV % = ((Placed Decimal Odds / Closing Decimal Odds) - 1.0) * 100

---

## 🛠️ Usage Example

```python
from live_pipeline import fetch_live_slate, fetch_pitcher_baseline
from models import simulate_moneyline, simulate_run_line, simulate_sgp_copula
from clv_tracker import log_bet, update_closing_lines

# 1. Pull Live Slate & Run Projections
slate = fetch_live_slate()

# 2. Log Placed Bets to CLV Tracker
log_bet(
    game_id="2026-08-16_STL_CHC",
    market_type="Moneyline",
    selection="St. Louis Cardinals",
    bet_odds_american=+162,
    model_prob=0.548,
    stake_units=1.0
)
```
