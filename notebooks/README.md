# notebooks/README.md

# 📓 Notebook Guide

Run notebooks in order using `jupyter lab` from the project root.

| Notebook | Phase | Purpose |
|----------|-------|---------|
| `01_eda_trades.ipynb`               | EDA       | Explore raw Hyperliquid trader data |
| `02_eda_sentiment.ipynb`            | EDA       | Explore Fear/Greed Index timeline   |
| `03_merge_feature_engineering.ipynb`| Processing| Join datasets + derive features     |
| `04_sentiment_vs_performance.ipynb` | Analysis  | Correlation, ANOVA, grouped plots   |
| `05_pattern_discovery.ipynb`        | Patterns  | KMeans clustering + PCA             |
| `06_predictive_modelling.ipynb`     | Models    | XGBoost, LightGBM, strategy backtest|

> **Before running:** Place raw CSVs in `data/raw/` as described in the main README.
