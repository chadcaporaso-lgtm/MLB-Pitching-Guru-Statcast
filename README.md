# ⚾ MLB-Pitching-Guru-Statcast

An institutional-grade quantitative MLB analytics and predictive betting pipeline combining **LightGBM Gradient Boosted Decision Trees**, **15,000-iteration Monte Carlo Poisson simulations**, and **Negative Binomial clustering** with automated market efficiency tracking and live box-score grading.

---

## 📊 Backtest Benchmarks & Model Accuracy (2,430-Game Historical Evaluation)

A full out-of-sample backtest evaluated across 2,430 historical Major League Baseball games (`features_2025.csv`) established the predictive benchmarks across both modeling paradigms:

| Evaluation Metric | GitHub Repo (Monte Carlo) | Google Drive (LightGBM ML) | 50/50 Blended Ensemble | Production 70/30 Hybrid Stack | Benchmark / Edge Standard |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Brier Score (Calibration)** | `0.2481` | `0.2277` 🏆 | `0.2351` | **`0.2291`** | `< 0.2420` (Coin-flip: `0.2500`) |
| **Log Loss (Cross-Entropy)** | `0.6893` | `0.6456` 🏆 | `0.6628` | **`0.6492`** | Lower score = Sharper |
| **Straight-Up Win Accuracy (%)** | `54.3%` | `62.4%` 🏆 | `60.9%` | **`61.8%`** | `> 54.0%` in MLB |

### Key Takeaways from Empirical Tuning
1. **LightGBM Calibration Superiority:** The `0.2277` Brier Score achieved by the LightGBM classifier represents elite calibration in Major League Baseball, accurately weighting non-linear Statcast feature interactions (CSW%, Whiff%, Barrel suppression, and Bullpen Fatigue).
2. **Monte Carlo Derivative Pricing:** While the ML model excels at outright win probability, the Monte Carlo simulation engine remains essential for derivative market mechanics (Run Line asymmetry, Push-adjusted totals, and Copula SGP correlation).
3. **The 70/30 Production Stack:** Blending 70% LightGBM probability with 30% Monte Carlo distribution density optimizes edge capture while insulating against single-model drift.

---

## 🏛️ System Architecture

```
                               ┌──────────────────────────────────────────────────────────┐
                               │                 MLB PREDICTIVE PIPELINE                  │
                               └────────────────────────────┬─────────────────────────────┘
                                                           │
                 ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
                 ▼                                                                                     ▼
   ┌───────────────────────────┐                                                         ┌───────────────────────────┐
   │ Google Drive ML Artifacts │                                                         │ GitHub Quantitative Repo  │
   │ (LightGBM Gradient Trees) │                                                         │ (Monte Carlo Simulations) │
   └─────────────┬─────────────┘                                                         └─────────────┬─────────────┘
                 │                                                                                     │
                 │  • Brier Score: 0.2277                                                              │  • Push-Adjusted Totals
                 │  • Non-linear Statcast feature splits                                               │  • Margin Distributions
                 │  • Bullpen Fatigue Index (BFI)                                                      │  • Discrete NRFI/YRFI Models
                 │                                                                                     │  • Copula SGP Engine
                 └──────────────────────────────────────────┬──────────────────────────────────────────┘
                                                           │
                                                           ▼
                                           ┌─────────────────────────────────┐
                                           │    `src/ensemble_models.py`     │
                                           │     Stacked Hybrid Engine       │
                                           │   (70% ML / 30% Simulation)     │
                                           └────────────────┬────────────────┘
                                                           │
                               ┌────────────────────────────┴────────────────────────────┐
                               ▼                                                         ▼
                 ┌───────────────────────────┐                             ┌───────────────────────────┐
                 │     Live CLV Logging      │                             │   Postgame Auto-Grader    │
                 │   (`src/clv_tracker.py`)  │                             │    (`src/grade_bets.py`)  │
                 └───────────────────────────┘                             └───────────────────────────┘
```

---

## ⚙️ Core Engines & Formulations

### 1. 70/30 Stacked Hybrid Model (`src/ensemble_models.py`)
Blends calibrated tree-based machine learning with full joint-score simulations:
$$P_{\text{Home}} = 0.70 \cdot P_{\text{LightGBM}} + 0.30 \cdot P_{\text{Simulation}}$$

### 2. Negative Binomial NRFI / YRFI Clustering (`src/models.py`)
Models empirical MLB half-inning run distributions with an overdispersion parameter ($\alpha = 0.65$):
$$P(\text{Scoreless Half}) = (1 + \alpha \lambda)^{-1 / \alpha}$$
$$P(\text{NRFI}) = P(\text{Away } 0) \times P(\text{Home } 0), \quad P(\text{YRFI}) = 1.0 - P(\text{NRFI})$$

### 3. Statcast Bullpen Fatigue Index (BFI) (`src/feature_engine.py`)
Computes trailing 3-day weighted pitch stress across relief units:
$$\text{BFI} = 1.00 \cdot P_{t-1} + 0.65 \cdot P_{t-2} + 0.35 \cdot P_{t-3}$$

### 4. Volumetric Pitcher Strikeout Model (`src/models.py`)
$$\text{Proj K} = \left(\frac{\text{K/9}}{9}\right) \times \text{Exp IP} \times \text{Matchup Factors}$$

### 5. Closing Line Value (CLV) Tracker (`src/clv_tracker.py`)
Measures edge retention against sharp market closing lines:
$$\text{CLV \%} = \left(\frac{\text{Placed Decimal Odds}}{\text{Closing Decimal Odds}} - 1.0\right) \times 100$$

---

## 📂 Repository File Structure

* **`src/ensemble_models.py`**: 70/30 stacked ensemble engine dynamically resolving LightGBM models with Monte Carlo outputs.
* **`src/models.py`**: Closed-form Monte Carlo Poisson simulations, Run Line margins, Totals, and NRFI models.
* **`src/feature_engine.py`**: Statcast feature aggregation, CSW%, launch angle/speed barrel classifications, and Bullpen Fatigue Indices.
* **`src/hr_engine.py`**: Home Run Prop predictive engine with Wilson Score 95% confidence intervals.
* **`src/odds_engine.py`**: Multiplicative de-vigging, true +EV calculations, and Fractional Kelly bankroll sizing.
* **`src/live_pipeline.py`**: Live hydrator querying MLB Stats API for probable starters, umpire splits, and market odds.
* **`src/clv_tracker.py`**: Market efficiency and Closing Line Value logging module (`data/clv_tracking_log.csv`).
* **`src/grade_bets.py`**: Automated box-score parser querying final game results to grade W/L records, calculate net units, and evaluate session Brier scores.
* **`app.py`**: Streamlit visualization dashboard with Plotly dark charts and interactive edge filtering.

---

## 🚀 Quickstart & Usage

```python
from live_pipeline import fetch_live_slate, fetch_pitcher_baseline
from ensemble_models import predict_hybrid_game
from grade_bets import grade_all_bets, generate_performance_report

# 1. Ingest live slate and compute 70/30 fair lines
slate = fetch_live_slate()
for _, row in slate.iterrows():
    pred = predict_hybrid_game(row['away_team'], row['home_team'], exp_away_runs=3.8, exp_home_runs=4.6)
    print(f"{pred['matchup']}: Fair Line {pred['fair_home_ml']}")

# 2. Postgame Auto-Grading & Performance Report
grade_all_bets(target_date="2026-08-16")
display(generate_performance_report())
```
