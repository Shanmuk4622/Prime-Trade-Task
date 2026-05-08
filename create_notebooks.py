"""
create_notebooks.py
===================
Generates the six Jupyter notebooks programmatically.
Run once to bootstrap the notebook files:

    conda run -n cv_conda python create_notebooks.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def nb(cells):
    """Create a minimal notebook JSON structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "cv_conda",
                "language": "python",
                "name": "cv_conda",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": "md_" + source[:20].replace(" ", "_")}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
        "id": "code_" + source[:20].replace(" ", "_").replace("\\", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 01 — EDA: Trades
# ──────────────────────────────────────────────────────────────────────────────

nb01 = nb([
    md("# 01 · Exploratory Data Analysis — Hyperliquid Trader Data\n"
       "> **Goal:** Understand the shape, quality, and distribution of the raw trader dataset."),

    code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from data_loader import load_trades
from preprocessing import clean_trades

%matplotlib inline
sns.set_theme(style="darkgrid")
"""),

    md("## 1. Load Raw Data"),
    code("""\
trades_raw = load_trades()
trades_raw.head()
"""),

    md("## 2. Schema & Data Types"),
    code("""\
print(trades_raw.dtypes)
print(f"\\nShape: {trades_raw.shape}")
"""),

    md("## 3. Missing Values"),
    code("""\
null_pct = trades_raw.isnull().mean().sort_values(ascending=False) * 100
null_pct[null_pct > 0].plot(kind="bar", title="% Missing per Column", figsize=(12, 4))
plt.ylabel("% Null")
plt.tight_layout()
plt.show()
print(null_pct[null_pct > 0].to_string())
"""),

    md("## 4. Clean & Basic Stats"),
    code("""\
trades = clean_trades(trades_raw)
trades.describe()
"""),

    md("## 5. PnL Distribution"),
    code("""\
fig = px.histogram(
    trades, x="closedpnl", nbins=100,
    title="Distribution of Closed PnL",
    color_discrete_sequence=["#3B82F6"],
    template="plotly_dark",
)
fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
fig.show()
"""),

    md("## 6. Trades by Symbol"),
    code("""\
sym_counts = trades["symbol"].value_counts().head(20)
px.bar(sym_counts, title="Top 20 Traded Symbols", template="plotly_dark",
       labels={"value": "Trade Count", "index": "Symbol"}).show()
"""),

    md("## 7. Trade Volume Over Time"),
    code("""\
daily = trades.groupby("trade_date").size().reset_index(name="trade_count")
px.line(daily, x="trade_date", y="trade_count",
        title="Daily Trade Count Over Time", template="plotly_dark").show()
"""),

    md("## 8. Side Distribution (Long vs. Short)"),
    code("""\
if "side" in trades.columns:
    px.pie(trades, names="side", title="Long vs Short Trade Distribution",
           template="plotly_dark").show()
"""),

    md("## 9. Leverage Distribution"),
    code("""\
if "leverage" in trades.columns:
    px.histogram(trades, x="leverage", nbins=50,
                 title="Leverage Distribution", template="plotly_dark").show()
"""),

    md("## 10. Top 20 Accounts by Trade Count"),
    code("""\
top_traders = trades["account"].value_counts().head(20).reset_index()
top_traders.columns = ["account", "trade_count"]
top_traders["account"] = top_traders["account"].str[:12] + "…"
px.bar(top_traders, x="trade_count", y="account", orientation="h",
       title="Top 20 Accounts by Trade Count", template="plotly_dark").show()
"""),

    md("## ✅ EDA Complete — proceed to `02_eda_sentiment.ipynb`"),
])

# ──────────────────────────────────────────────────────────────────────────────
# 02 — EDA: Sentiment
# ──────────────────────────────────────────────────────────────────────────────

nb02 = nb([
    md("# 02 · Exploratory Data Analysis — Fear/Greed Index"),
    code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent / "src"))

import pandas as pd
import plotly.express as px
from data_loader import load_sentiment
from preprocessing import clean_sentiment
from feature_engineering import add_sentiment_features
from visualisation import plot_sentiment_timeline, REGIME_COLORS, REGIME_ORDER
"""),

    md("## 1. Load & Clean"),
    code("""\
sentiment_raw = load_sentiment()
sentiment_raw.head()
"""),
    code("""\
sentiment = clean_sentiment(sentiment_raw)
sentiment = add_sentiment_features(sentiment)
sentiment.head()
"""),

    md("## 2. Timeline"),
    code("""\
fig = plot_sentiment_timeline(sentiment, save=True)
fig.show()
"""),

    md("## 3. Regime Distribution"),
    code("""\
regime_counts = sentiment["regime"].value_counts().reindex(REGIME_ORDER)
px.bar(regime_counts, color=regime_counts.index,
       color_discrete_map=REGIME_COLORS,
       title="Days per Sentiment Regime",
       labels={"value": "Days", "index": "Regime"},
       template="plotly_dark").show()
"""),

    md("## 4. Sentiment Score Over Time (7-day MA)"),
    code("""\
if "value" in sentiment.columns:
    fig = px.line(sentiment, x="trade_date",
                  y=["value", "sentiment_shift_7d_mean"],
                  title="Sentiment Score & 7-day Moving Average",
                  template="plotly_dark")
    fig.show()
"""),

    md("## 5. Volatility of Sentiment"),
    code("""\
if "sentiment_volatility_7d" in sentiment.columns:
    px.line(sentiment, x="trade_date", y="sentiment_volatility_7d",
            title="7-day Rolling Sentiment Volatility",
            template="plotly_dark").show()
"""),

    md("## ✅ Sentiment EDA complete → `03_merge_feature_engineering.ipynb`"),
])

# ──────────────────────────────────────────────────────────────────────────────
# 03 — Merge & Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

nb03 = nb([
    md("# 03 · Merge & Feature Engineering"),
    code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent / "src"))

from data_loader import load_trades, load_sentiment
from preprocessing import clean_trades, clean_sentiment, save_processed
from feature_engineering import (
    add_trade_level_features, add_sentiment_features,
    merge_datasets, add_rolling_account_features,
)
import pandas as pd
"""),

    md("## 1. Load & Clean Both Datasets"),
    code("""\
trades    = clean_trades(load_trades())
sentiment = add_sentiment_features(clean_sentiment(load_sentiment()))
trades    = add_trade_level_features(trades)
"""),

    md("## 2. Merge on Trade Date"),
    code("""\
merged = merge_datasets(trades, sentiment)
merged.head()
"""),

    md("## 3. Add Rolling Account Features"),
    code("""\
merged = add_rolling_account_features(merged, windows=[7, 30])
merged.describe()
"""),

    md("## 4. Save Processed Datasets"),
    code("""\
save_processed(trades,    "trades_clean.csv")
save_processed(sentiment, "fear_greed_clean.csv")
save_processed(merged,    "merged_dataset.csv")
print("All processed datasets saved ✓")
"""),

    md("## 5. Inspect New Features"),
    code("""\
print("New columns added:", [c for c in merged.columns if c not in trades.columns])
merged[["account", "trade_date", "closedpnl", "is_win", "regime", "regime_numeric",
        "win_rate_7d", "cum_pnl_7d"]].head(15)
"""),

    md("## ✅ Feature engineering complete → `04_sentiment_vs_performance.ipynb`"),
])

# ──────────────────────────────────────────────────────────────────────────────
# 04 — Sentiment vs. Performance
# ──────────────────────────────────────────────────────────────────────────────

nb04 = nb([
    md("# 04 · Sentiment vs. Trader Performance\n"
       "> Core relationship analysis — ANOVA, correlation, grouped visuals"),

    code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent / "src"))

import pandas as pd
import plotly.express as px
from preprocessing import PROCESSED_DIR  # noqa
from analysis import pnl_by_regime, run_anova, pearson_spearman
from visualisation import (
    plot_pnl_by_regime, plot_winrate_by_regime, plot_correlation_heatmap
)

PROCESSED_DIR = pathlib.Path().resolve().parent / "data" / "processed"
merged = pd.read_csv(PROCESSED_DIR / "merged_dataset.csv")
print(merged.shape)
"""),

    md("## 1. PnL Statistics by Regime"),
    code("""\
regime_stats = pnl_by_regime(merged)
regime_stats
"""),

    md("## 2. ANOVA — Does Regime Significantly Affect PnL?"),
    code("""\
anova = run_anova(merged)
print(f"F = {anova['F_statistic']}  |  p = {anova['p_value']}")
print(f"→ {anova['interpretation']}")
"""),

    md("## 3. Pearson & Spearman Correlations"),
    code("""\
corr = pearson_spearman(merged.dropna(subset=["regime_numeric", "closedpnl"]),
                         "regime_numeric", "closedpnl")
print("Sentiment ↔ PnL Correlation:")
for k, v in corr.items():
    print(f"  {k}: {v}")
"""),

    md("## 4. Violin Plot — PnL by Regime"),
    code("""\
fig = plot_pnl_by_regime(merged, save=True)
fig.show()
"""),

    md("## 5. Win Rate by Regime"),
    code("""\
fig = plot_winrate_by_regime(regime_stats, save=True)
fig.show()
"""),

    md("## 6. Correlation Heatmap"),
    code("""\
import matplotlib.pyplot as plt
fig = plot_correlation_heatmap(merged, save=True)
plt.show()
"""),

    md("## 7. Long vs. Short PnL by Regime"),
    code("""\
if "is_long" in merged.columns:
    side_regime = merged.groupby(["regime", "is_long"])["closedpnl"].mean().reset_index()
    side_regime["direction"] = side_regime["is_long"].map({1: "Long", 0: "Short"})
    px.bar(side_regime, x="regime", y="closedpnl", color="direction",
           barmode="group", template="plotly_dark",
           title="Mean PnL: Long vs Short by Regime").show()
"""),

    md("## ✅ Analysis complete → `05_pattern_discovery.ipynb`"),
])

# ──────────────────────────────────────────────────────────────────────────────
# 05 — Pattern Discovery
# ──────────────────────────────────────────────────────────────────────────────

nb05 = nb([
    md("# 05 · Pattern Discovery — Trader Clustering & Regime Transitions"),
    code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent / "src"))

import pandas as pd
import plotly.express as px
from analysis import account_performance_summary, trader_segments
from modelling import cluster_traders
from visualisation import plot_top_accounts, plot_pca_clusters, REGIME_COLORS, REGIME_ORDER

PROCESSED_DIR = pathlib.Path().resolve().parent / "data" / "processed"
merged = pd.read_csv(PROCESSED_DIR / "merged_dataset.csv")
"""),

    md("## 1. Account Performance Summary"),
    code("""\
acct_summary = account_performance_summary(merged)
acct_summary = trader_segments(acct_summary)
acct_summary.head(10)
"""),

    md("## 2. Top & Bottom Accounts"),
    code("""\
fig = plot_top_accounts(acct_summary, n=20, save=True)
fig.show()
"""),

    md("## 3. KMeans Clustering of Trader Profiles"),
    code("""\
acct_summary, pca_df = cluster_traders(acct_summary, n_clusters=4)
pca_df.head()
"""),

    md("## 4. PCA Visualisation of Clusters"),
    code("""\
fig = plot_pca_clusters(pca_df, save=True)
fig.show()
"""),

    md("## 5. Cluster Profiles"),
    code("""\
cluster_profile = acct_summary.groupby("cluster")[
    ["total_pnl", "win_rate", "trade_count", "sharpe_proxy"]
].mean().round(3)
cluster_profile
"""),

    md("## 6. Sentiment Regime Transition Matrix"),
    code("""\
sentiment_df = pd.read_csv(PROCESSED_DIR / "fear_greed_clean.csv")
sentiment_df["next_regime"] = sentiment_df["regime"].shift(-1)
transition = pd.crosstab(sentiment_df["regime"], sentiment_df["next_regime"], normalize="index")
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(transition, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Sentiment Regime Transition Probability Matrix")
plt.tight_layout()
plt.show()
"""),

    md("## ✅ Pattern discovery complete → `06_predictive_modelling.ipynb`"),
])

# ──────────────────────────────────────────────────────────────────────────────
# 06 — Predictive Modelling
# ──────────────────────────────────────────────────────────────────────────────

nb06 = nb([
    md("# 06 · Predictive Modelling — Win/Loss Prediction & Strategy Backtest\n"
       "> Can sentiment predict trade outcomes? Can a sentiment-gated strategy beat the baseline?"),

    code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent / "src"))

import pandas as pd
import plotly.express as px
from modelling import prepare_ml_data, train_xgboost, train_lightgbm, sentiment_gated_strategy_backtest
from visualisation import plot_feature_importance

PROCESSED_DIR = pathlib.Path().resolve().parent / "data" / "processed"
merged = pd.read_csv(PROCESSED_DIR / "merged_dataset.csv")
"""),

    md("## 1. Prepare ML Features & Target"),
    code("""\
X, y = prepare_ml_data(merged)
print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")
X.describe()
"""),

    md("## 2. Train XGBoost"),
    code("""\
xgb_results = train_xgboost(X, y, cv_folds=5)
print(f"XGBoost  AUC: {xgb_results['cv_auc_mean']} ± {xgb_results['cv_auc_std']}")
print(f"XGBoost  Acc: {xgb_results['cv_acc_mean']}")
"""),

    md("## 3. Train LightGBM"),
    code("""\
lgb_results = train_lightgbm(X, y, cv_folds=5)
print(f"LightGBM AUC: {lgb_results['cv_auc_mean']} ± {lgb_results['cv_auc_std']}")
print(f"LightGBM Acc: {lgb_results['cv_acc_mean']}")
"""),

    md("## 4. Feature Importance"),
    code("""\
fig = plot_feature_importance(xgb_results["feature_importance"], save=True)
fig.show()
"""),

    md("## 5. Sentiment-Gated Strategy Backtest"),
    code("""\
# Aggregate to daily PnL
daily_pnl = merged.groupby(["trade_date", "regime"])["closedpnl"].sum().reset_index()
daily_pnl.columns = ["trade_date", "regime", "daily_pnl"]
daily_pnl = daily_pnl.sort_values("trade_date")

backtest_df = sentiment_gated_strategy_backtest(daily_pnl)
backtest_df.head()
"""),

    md("## 6. Equity Curve — Strategy vs Baseline"),
    code("""\
fig = px.line(backtest_df, x="trade_date",
              y=["cum_strategy_pnl", "cum_baseline_pnl"],
              title="Equity Curve: Sentiment-Gated Strategy vs Baseline",
              labels={"value": "Cumulative PnL", "variable": "Strategy"},
              template="plotly_dark")
fig.show()
"""),

    md("## 7. Key Insights Summary"),
    code("""\
print("=" * 60)
print("  PRIME TRADE — KEY FINDINGS")
print("=" * 60)
print(f"  XGBoost AUC   : {xgb_results['cv_auc_mean']:.3f}")
print(f"  LightGBM AUC  : {lgb_results['cv_auc_mean']:.3f}")
print(f"  Top sentiment feature importance:")
print(xgb_results['feature_importance'].head(5).to_string(index=False))
print()
print(f"  Strategy vs Baseline:")
print(f"  Strategy total PnL  : {backtest_df['cum_strategy_pnl'].iloc[-1]:,.2f}")
print(f"  Baseline total PnL  : {backtest_df['cum_baseline_pnl'].iloc[-1]:,.2f}")
"""),

    md("## ✅ Analysis Complete!\n"
       "> See `outputs/figures/` for all charts and `outputs/models/` for saved models."),
])

# ──────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ──────────────────────────────────────────────────────────────────────────────

notebooks = {
    "01_eda_trades.ipynb":               nb01,
    "02_eda_sentiment.ipynb":            nb02,
    "03_merge_feature_engineering.ipynb": nb03,
    "04_sentiment_vs_performance.ipynb": nb04,
    "05_pattern_discovery.ipynb":        nb05,
    "06_predictive_modelling.ipynb":     nb06,
}

for filename, notebook in notebooks.items():
    path = NB_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"[✓] Created → {path}")

print("\n✅ All 6 notebooks generated. Open with: jupyter lab")
