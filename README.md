# 📊 Prime Trade — Trader Sentiment Analysis

> **PrimeTrade.ai Hiring Assignment** · Data Science Track  
> *Exploring the nexus between Bitcoin market sentiment and Hyperliquid trader behaviour*

---

## 🗺️ Project Architecture

```
Prime Trade/
├── data/
│   ├── raw/                     # Original unmodified CSVs (gitignored)
│   │   ├── historical_trades.csv
│   │   └── fear_greed_index.csv
│   └── processed/               # Cleaned, merged, feature-engineered datasets
│       ├── trades_clean.csv
│       ├── fear_greed_clean.csv
│       └── merged_dataset.csv
│
├── notebooks/                   # Jupyter notebooks — one per analysis phase
│   ├── 01_eda_trades.ipynb          # EDA on raw trader data
│   ├── 02_eda_sentiment.ipynb       # EDA on fear/greed index
│   ├── 03_merge_feature_engineering.ipynb  # Merge + derived features
│   ├── 04_sentiment_vs_performance.ipynb   # Core relationship analysis
│   ├── 05_pattern_discovery.ipynb          # Hidden pattern mining
│   └── 06_predictive_modelling.ipynb       # ML models & strategy signals
│
├── src/                         # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py           # Load & validate raw CSVs
│   ├── preprocessing.py         # Cleaning, type casting, dedup
│   ├── feature_engineering.py   # Derived columns & window features
│   ├── analysis.py              # Statistical tests & metrics
│   ├── visualisation.py         # Plotly & matplotlib chart helpers
│   └── modelling.py             # ML pipeline utilities
│
├── outputs/
│   ├── figures/                 # All exported charts (PNG / HTML)
│   ├── reports/                 # Final insight summary (Markdown / PDF)
│   └── models/                  # Saved model artefacts (.pkl / .json)
│
├── tests/
│   └── test_preprocessing.py    # Unit tests for data pipeline
│
├── environment.yml              # Conda environment spec
├── requirements.txt             # pip requirements
├── .gitignore
└── README.md
```

---

## 🎯 Objective

Uncover the statistical & predictive relationship between **Bitcoin market sentiment** (Fear/Greed Index) and **Hyperliquid trader performance**, then distil actionable trading-strategy insights.

### Research Questions
| # | Question |
|---|----------|
| 1 | Does extreme fear/greed predict higher or lower trader profitability? |
| 2 | Do winning vs. losing traders react differently to sentiment shifts? |
| 3 | Are there hidden clustering patterns in trader behaviour by sentiment regime? |
| 4 | Can sentiment be used as a feature to predict next-day PnL sign? |
| 5 | Which symbols / leverage levels amplify sentiment-driven risk? |

---

## 🔬 Analysis Pipeline

```
Raw Data Ingest
    │
    ▼
01 EDA Trades ──────────────────────── distribution, outliers, top accounts
02 EDA Sentiment ───────────────────── time-series of fear/greed, regime stats
    │
    ▼
03 Merge + Feature Engineering
    │  • Date-join trades ↔ sentiment
    │  • Rolling PnL windows (7d / 30d)
    │  • Sentiment regime labels (Extreme Fear / Fear / Neutral / Greed / Extreme Greed)
    │  • Win-rate, Sharpe-proxy, max drawdown per account
    │
    ▼
04 Sentiment vs. Performance
    │  • Correlation matrix (Pearson + Spearman)
    │  • ANOVA across sentiment regimes
    │  • Grouped box plots & violin plots
    │
    ▼
05 Pattern Discovery
    │  • K-Means clustering of trader profiles
    │  • PCA visualisation of clusters
    │  • Sentiment-regime transition analysis
    │
    ▼
06 Predictive Modelling
       • XGBoost / LightGBM: PnL direction prediction
       • Feature importance → top sentiment-driven signals
       • Strategy back-test: sentiment-gated long/short bias
```

---

## 🚀 Quickstart

```bash
# 1. Activate conda environment
conda activate cv_conda

# 2. Install any missing packages
pip install -r requirements.txt

# 3. Place raw CSVs
#    data/raw/historical_trades.csv
#    data/raw/fear_greed_index.csv

# 4. Run the full pipeline (headless)
python src/run_pipeline.py

# 5. Or open notebooks interactively
jupyter lab
```

---

## 📌 Checkpoints & Git Tags

| Tag | Content |
|-----|---------|
| `v0.1-scaffold` | Project structure, README, environment |
| `v0.2-eda` | EDA notebooks completed |
| `v0.3-features` | Merge + feature engineering |
| `v0.4-analysis` | Sentiment vs. performance analysis |
| `v0.5-patterns` | Clustering & pattern discovery |
| `v0.6-models` | Predictive modelling & strategy |
| `v1.0-final` | Final report + polished outputs |

---

## 📦 Environment

- **Python** 3.10  
- **Conda env**: `cv_conda`  
- Key libs: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`, `plotly`, `seaborn`, `matplotlib`

---

## 👤 Author

**Shanmuk** — Data Science Candidate, PrimeTrade.ai  
GitHub: [@Shanmuk4622](https://github.com/Shanmuk4622)
