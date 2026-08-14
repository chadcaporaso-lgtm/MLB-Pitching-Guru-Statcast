# ⚾ MLB-Pitching-Guru-Statcast

An end-to-end quantitative MLB pitching analytics and betting model pipeline powered by Statcast pitch-tracking data, machine learning, and exchange-based market efficiency tracking.

## 🚀 Key Features
* **Statcast Pitching+ Engine:** XGBoost models predicting pitch quality using physical movement/velocity (Stuff+) and detailed zone execution/location error (Command+).
* **LightGBM Strikeout Model:** Time-lagged 3-start rolling features predicting starter strikeout distributions.
* **Gaussian Copula SGP Evaluator:** Monte Carlo simulation engine calculating true joint probabilities for correlated Same Game Parlays and profit boost tokens.
* **Novig Zero-Vig CLV Tracker:** Automated closing line value tracking benchmarked against Novig exchange order book midpoints.
* **Live Odds API Connector:** Real-time multi-book scanner (FanDuel, Caesars, Novig) for +EV line discovery.

## 📈 Multi-Season Backtest Grid Summary (+4% Optimal EV Threshold)
| EV Threshold | Total Bets | Win Rate % | Total Staked | Net PnL | ROI % | Mean CLV % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| +2% | 727 | 50.34% | $18,175.00 | +$1,459.88 | +8.03% | +0.46% |
| +3% | 615 | 51.22% | $15,375.00 | +$1,686.00 | +10.97% | +0.42% |
| **+4% (Optimal)** | **515** | **52.23%** | **$12,875.00** | **+$1,778.39** | **+13.81%** | **+0.46%** |
| +6% | 354 | 53.11% | $8,850.00 | +$1,583.80 | +17.90% | +0.42% |
| **+8% (Peak ROI)** | **249** | **53.41%** | **$6,225.00** | **+$1,362.90** | **+21.89%** | **+0.50%** |
## Developer Notes
* **Automated Reviews:** This repository uses CodeRabbit AI for real-time security scans and pull request validations.
* **Security Scans:** Code checks run automatically on every open pull request to catch vulnerabilities before merging.
* 
