# MLB-Pitching-Guru-Statcast

Quantitative MLB proposition betting engine (Ks, Outs, NRFI, F5) integrating 3D pitch kinematics, spatial umpire zones, 24-state Markov chains, and rolling 72-hr bullpen fatigue (BFI).

## 1. Quantitative Core & Formulas

- **Kinematics Engine (`kinematics_engine.py`):**
  - $v_{y_f} = -\sqrt{v_{y_0}^2 - 2 a_y (y_0 - y_f)}$
  - $\text{VAA} = -\arctan(v_{z_f} / v_{y_f}) \cdot (180/\pi)$
  - $\text{HAA} = -\arctan(v_{x_f} / v_{y_f}) \cdot (180/\pi)$

- **CSW & Strikeouts (`csw_pricing.py`):**
  - $\text{CSW\%}_{\text{adj}} = \text{CSW\%}_{\text{base}} + \beta_1 \Delta x_{\text{ump}} + \beta_2 \Delta z_{\text{ump}} - \beta_3 \text{Z-Contact}$
  - $\text{Projected K\%} = 1.62 \cdot \text{CSW\%}_{\text{adj}} - 0.224$
  - $\text{Batters Faced (BF)} = \text{Outs} / (1 - \text{Lineup OBP})$

- **Dynamic Scoring (`dynamic_negbin.py`):**
  - Environmentally conditioned $\text{NegBin}(\mu, r)$ for F5/Totals, where $r = f(\Delta\text{ADI}, \text{Wind}, \text{BFI})$.

- **1st Inning NRFI Engine (`markov_nrfi.py`):**
  - 24-state base-out Markov chain transitions ($3 \times 8$ configurations) modeling $P(\text{NRFI})$.

- **Bullpen Leverage (`bullpen_leverage.py`):**
  - $\text{BFI}_{72} = \sum_{k=1}^N \text{Pitches}_k \cdot \text{gmLI}_k \cdot e^{-\lambda (t - t_k)}$. High BFI (>1.35) extends starter survival.

## 2. Directory Layout

```
MLB-Pitching-Guru-Statcast/
├── data/
│   ├── clv_tracking_log.csv
│   ├── umpire_spatial_database.csv
│   └── park_environmental_index.csv
├── modules/
│   ├── kinematics_engine.py
│   ├── csw_pricing.py
│   ├── dynamic_negbin.py
│   ├── markov_nrfi.py
│   └── bullpen_leverage.py
├── scripts/
│   ├── live_slate_tracker.py
│   ├── github_sync.py
│   └── statcast_arsenal_harvester.py
└── README.md
```

## 3. Master Ledger Schema (`clv_tracking_log.csv`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | ISO-8601 | UTC generation timestamp |
| `date` | String | Slate date (`YYYY-MM-DD`) |
| `game_id` | String | Matchup key (`YYYY-MM-DD_AWAY_HOME`) |
| `market_type` | String | Category (`Pitcher Strikeouts`, `Outs`, `NRFI`, `F5`) |
| `selection` | String | Wager description |
| `line` | Float | Target line (0.5, 2.5, 3.5, 11.5, etc.) |
| `placed_odds_american` | Int | American line |
| `placed_odds_decimal` | Float | Payout multiplier |
| `model_win_prob` | Float | Modeled conversion probability |
| `stake_units` | Float | Risked stake units/dollars |
| `book` | String | Bookmaker / Tracked |
| `result` | String | Status (`PENDING`, `WON`, `LOST`, `PUSH`) |
| `pnl_dollars` | Float | Net profit/loss in dollars |
| `pnl_units` | Float | Net units |
| `clv_percent` | Float | Closing line value vs Pinnacle de-vigged line |

## 4. Execution Workflow

```bash
# Install the scanner's dependencies in a local virtual environment
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy requests

# Run all four models using the CSVs in data/ (no network or credentials)
.venv/bin/python master_production_runner.py

# Optionally scan live moneyline odds after refreshing the input CSVs
ODDS_API_KEY=your_key .venv/bin/python master_production_runner.py --live
```

Run from any directory; the default input directory is the repository's `data/` folder.
Use `--data-dir PATH` to select another set of input CSVs. Offline mode previews
the four projection boards but does not generate betting recommendations. Live mode prints
qualifying moneyline opportunities; it does not modify the ledger. Keep the input
slate current before comparing projections to live odds: the bundled CSVs are snapshots,
not a live feed.
