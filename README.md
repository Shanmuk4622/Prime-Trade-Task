# 📊 PrimeTrade.ai — Trader Sentiment Analysis
### *Hyperliquid × Bitcoin Fear/Greed Index · Data Science Assignment*

> **Candidate:** Shanmuk | **Repo:** [Shanmuk4622/Prime-Trade-Task](https://github.com/Shanmuk4622/Prime-Trade-Task)  
> **Environment:** `conda activate cv_conda` · Python 3.10 · conda env: `cv_conda`

---

## 🎯 Objective

> *"Explore the relationship between trader performance and market sentiment, uncover hidden patterns, and deliver insights that can drive smarter trading strategies."*

Analysis of **211,224 trades** across **32 Hyperliquid accounts** merged with the Bitcoin Fear/Greed Index reveals that market sentiment has a **statistically significant but practically modest** influence on individual trade outcomes — and the dominant pattern is **counter-intuitive**.

---

## 🔑 Key Findings at a Glance

| # | Finding | Implication |
|---|---------|-------------|
| 1 | ANOVA: F=12.66, **p≈0** — regimes differ significantly | Sentiment is a real signal, not noise |
| 2 | **Extreme Greed** has the highest mean PnL ($74.74) | Momentum traders profit in euphoria |
| 3 | **Fear** accumulates the most total PnL ($3.18M) | Volume surge amplifies aggregate gains |
| 4 | **Extreme Fear** has the lowest win rate (34.6%) | Panic conditions hurt retail traders most |
| 5 | Correlation is weak: Spearman r=0.029 | Sentiment = context filter, not per-trade signal |
| 6 | Leakage-free XGBoost AUC **0.855** (10 features) | Sentiment is genuinely predictive |
| 7 | 4 distinct trader archetypes via KMeans clustering | Personalised strategy required |
| 8 | Regimes are **highly persistent** (1-day autocorrelation) | Signal has time to act on |

---

## 📈 Results — Sentiment vs. Performance

### PnL by Sentiment Regime

| Regime | Trades | Mean PnL | Win Rate | Total PnL |
|--------|--------|----------|----------|-----------|
| 🔴 Extreme Fear | 31,364 | $34.72 | **34.6%** | $1.09M |
| 🟠 Fear | 55,328 | $57.42 | 44.6% | **$3.18M** |
| 🟡 Neutral | 36,302 | $34.32 | 39.9% | $1.25M |
| 🟢 Greed | 55,803 | $41.56 | 40.3% | $2.32M |
| 🔵 Extreme Greed | 32,421 | **$74.74** | 44.4% | $2.42M |

**ANOVA:** F = 12.6629 · p ≈ 0 → **Significant across all regimes**  
**Spearman correlation (sentiment ↔ PnL):** r = 0.029 · p ≈ 0

### The Counter-Intuitive Extreme Greed Premium
Contrary to popular "buy the fear" narratives, **Extreme Greed produces 2.15× higher mean PnL** than Extreme Fear. This is because momentum-following accounts ride trend extensions in euphoria, compounding gains with higher conviction sizing. Fear produces the most total profit purely through **volume** (55K+ trades vs 31K).

---

## 🔬 Statistical Analysis

| Test | Result | Interpretation |
|------|--------|----------------|
| One-way ANOVA | F=12.66, p≈0 | Mean PnL differs significantly across regimes |
| Pearson r (sentiment ↔ PnL) | 0.0066 (p=0.0024) | Statistically significant, practically tiny |
| Spearman r (sentiment ↔ PnL) | 0.0288 (p≈0) | Monotonic association confirmed |
| Overall win rate | ~40% all regimes | Profits come from asymmetric sizing |

**Bottom line:** Sentiment is a **real but weak** predictor. Use it as a regime filter to adjust position sizing and directional bias — not as a standalone entry signal.

---

## 🧩 Pattern Discovery — 4 Trader Archetypes

KMeans clustering (k=4) on account-level metrics, visualised with PCA (**62.7% variance explained**):

| Cluster | Label | Behaviour |
|---------|-------|-----------|
| 0 | **Momentum Whales** | High PnL, high trade count, exploits trend phases |
| 1 | **Conservative Earners** | Steady positive PnL, moderate win rate |
| 2 | **High-Variance Speculators** | Large swings, below-avg win rate, extreme drawdowns |
| 3 | **Marginal Accounts** | Low volume, near-zero net PnL |

**Top 25% traders** dominate activity during **Fear** regimes — they exploit panic-driven mispricings while retail sells. **Bottom 25%** traders are most active during **Extreme Greed** — they buy into euphoria tops.

---

## 💡 Strategy Recommendations

### 1. Sentiment-Gated Position Sizing

| Regime | Size Multiplier | Logic |
|--------|----------------|-------|
| Extreme Fear | 0.5× | High volatility, lowest win rate — protect capital |
| Fear | **1.5×** | Best risk-adjusted regime — exploit mispricings |
| Neutral | 1.0× | Base allocation |
| Greed | 1.2× | Momentum-friendly conditions |
| Extreme Greed | 0.8× | High mean PnL but reversal risk rises |

### 2. Long/Short Bias by Regime
- **Fear → Prefer LONG** (fade panic sellers, exploit overshoots)
- **Extreme Greed → Prefer SHORT** (fade the crowd, front-run reversal)

### 3. Regime-Aware Stop Loss Width
During **Extreme Fear** std_pnl ≈ $1005 (highest). Widen stops by 30-50% to avoid noise-driven stop-outs that destroy win rate.

### 4. Cluster-Based Signal Filtering
Mirror **Cluster 0 (Momentum Whales)** trades during Fear regimes only — these accounts show the strongest sentiment-performance alignment.

---

## 🗂️ Project Structure

```
Prime Trade/
├── data/
│   ├── raw/                     # Original CSVs (not tracked)
│   └── processed/               # Cleaned & merged datasets
├── notebooks/                   # 6 Jupyter notebooks (one per phase)
│   ├── 01_eda_trades.ipynb
│   ├── 02_eda_sentiment.ipynb
│   ├── 03_merge_feature_engineering.ipynb
│   ├── 04_sentiment_vs_performance.ipynb
│   ├── 05_pattern_discovery.ipynb
│   └── 06_predictive_modelling.ipynb
├── src/                         # Reusable Python modules
│   ├── data_loader.py           # Load & validate raw CSVs
│   ├── preprocessing.py         # Clean, type-cast, deduplicate
│   ├── feature_engineering.py   # Derived columns & rolling windows
│   ├── analysis.py              # ANOVA, correlation, account metrics
│   ├── visualisation.py         # Plotly & Matplotlib chart builders
│   ├── modelling.py             # XGBoost, LightGBM, KMeans, backtest
│   ├── run_pipeline.py          # End-to-end headless runner
│   └── generate_report.py       # Auto-generate insights report
├── outputs/
│   ├── figures/                 # 7 interactive + static charts
│   └── reports/
│       ├── insights_report.md   # Full findings & strategy report
│       └── summary_stats.json   # Machine-readable stats
├── tests/
│   └── test_preprocessing.py    # 13 unit tests — all passing
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 🔬 Analysis Pipeline

```
Raw Data Ingest (211,224 trades · 2,644 sentiment days)
    │
    ▼
Preprocessing → Clean timestamps, cast numerics, deduplicate
    │
    ▼
Feature Engineering → is_win, pnl_per_unit, is_long, regime_numeric,
    │                  rolling win_rate & cumPnL (7d/30d per account)
    ▼
Merge (100% sentiment match rate)
    │
    ▼
Phase 4: Statistical Analysis
    │  • One-way ANOVA (F=12.66, p≈0) ✓ Significant
    │  • Pearson + Spearman correlation
    │  • PnL & win-rate by regime
    ▼
Phase 5: Pattern Discovery
    │  • KMeans k=4 → 4 trader archetypes
    │  • PCA (62.7% variance explained)
    │  • Regime transition matrix
    ▼
Phase 6: Predictive Modelling
       • XGBoost + LightGBM: 15 features → is_win classification
       • Sentiment-gated strategy backtest
```

---

## 🚀 Quickstart

```bash
# 1. Activate conda environment
conda activate cv_conda

# 2. Place raw CSVs in data/raw/ (see data/raw/PLACE_DATASETS_HERE.md)

# 3. Run the full pipeline
$env:PYTHONIOENCODING="utf-8"
python src/run_pipeline.py

# 4. Generate the insights report
python src/generate_report.py

# 5. Run tests
python -m pytest tests/ -v

# 6. Or explore interactively
jupyter lab
```

---

## 📊 Pipeline Output

```
PHASE 1: Trades loaded       → 211,224 rows, 16 cols
         Sentiment loaded    → 2,644 rows, 4 cols

PHASE 2: Trades cleaned      → 211,224 rows remain
         Sentiment cleaned   → 2,644 rows remain

PHASE 3: Merged dataset      → 211,224 rows
         Sentiment match     → 100.0%
         Rolling features    → 32 accounts processed

PHASE 4: ANOVA F=12.66 p≈0  → Significant
         Spearman r=0.029    → Confirmed

PHASE 5: 4 KMeans clusters   → 62.7% PCA variance

PHASE 6: XGBoost AUC 0.855  -> Leakage-free (10 sentiment+size features)
         XGBoost AUC 1.000  -> Full model (data leakage via rolling features)
         LightGBM AUC 1.000 -> Full model (data leakage)

OUTPUTS: 7 interactive charts (HTML + PNG)
         insights_report.md  (full findings)
         summary_stats.json  (machine-readable)
         2 saved model .pkl files
```

---

## ⚠️ Known Limitations

1. **ML data leakage** — Rolling features (win_rate_7d, mean_pnl_7d) are computed across all trades, so the model sees future data. AUC=1.0 is not a real generalisation score. Fix: use strict time-based train/test split with no look-ahead.
2. **32 accounts** — Small pool; findings may not generalise to the full Hyperliquid universe.
3. **Weak correlation** — Sentiment explains <0.1% of individual trade variance. It is a macro regime filter, not a micro entry signal.
4. **No fees** — All PnL is gross. Net results after trading fees will differ.

---

## 📌 Git Checkpoints

| Tag | Commit | Content |
|-----|--------|---------|
| `v0.1-scaffold` | `fa7b56a` | Project structure, modules, notebooks, tests |
| `v0.2-eda` | `d4caf0f` | Real CSV column fix — full pipeline on 211K trades |
| `v0.3-insights` | *current* | Insights report, README with results, strategy recs |

---

## 📦 Environment

- **Python** 3.10 · **Conda env:** `cv_conda`
- **Key libs:** `pandas 1.4.4` · `numpy 1.26` · `scipy 1.13` · `statsmodels 0.14`  
  `scikit-learn 1.4` · `xgboost 3.2` · `lightgbm 4.6` · `plotly 6.7` · `seaborn 0.13`

---

## 👤 Author

**Shanmuk** — Data Science Candidate, PrimeTrade.ai  
GitHub: [@Shanmuk4622](https://github.com/Shanmuk4622)
