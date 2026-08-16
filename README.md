# ⚾ MLB-Pitching-Guru-Statcast

An end-to-end quantitative MLB analytics and predictive betting pipeline powered by Statcast metrics, automated market efficiency tracking, and live odds integration.

## 🚀 Repository Architecture
* **`src/models.py`**: Quantitative engines including Poisson score simulations for Moneyline/Totals and Gaussian Copula simulations for correlated Same Game Parlays (SGP).
* **`src/live_pipeline.py`**: Dynamic hydrator connecting the MLB Stats API to Open-Meteo weather and The Odds API.
* **`src/backtest_engine.py`**: Historical simulation framework calibrated for +4.0% EV threshold optimization.

## 📊 Betting Engines
1. **Volumetric Strikeout Projection:** Proj K = (K/9 / 9) * Exp IP * Matchup Factors
2. **Moneyline Win Expectancy:** 10,000 Monte Carlo Poisson simulations generating fair, zero-vig American lines.
3. **Correlated SGP Engine:** Bivariate normal Gaussian Copula modeling inverse correlation (rho ≈ -0.32) between opponent starter strikeouts and team run totals.
