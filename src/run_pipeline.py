"""
run_pipeline.py
===============
End-to-end headless pipeline runner.
Executes all phases without Jupyter and saves all outputs.

Usage
-----
    conda activate cv_conda
    python src/run_pipeline.py

Checkpoints
-----------
The script saves processed CSVs and figures at each phase so partial runs
still produce useful artefacts.
"""

import sys
import pathlib
import pandas as pd
import warnings

# Force UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# Ensure src/ is on the path when run from project root
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader        import load_trades, load_sentiment
from preprocessing      import clean_trades, clean_sentiment, save_processed
from feature_engineering import (
    add_trade_level_features,
    add_sentiment_features,
    merge_datasets,
    add_rolling_account_features,
)
from analysis import (
    pnl_by_regime,
    run_anova,
    pearson_spearman,
    account_performance_summary,
    trader_segments,
)
from visualisation import (
    plot_sentiment_timeline,
    plot_pnl_by_regime,
    plot_winrate_by_regime,
    plot_correlation_heatmap,
    plot_top_accounts,
    plot_pca_clusters,
    plot_feature_importance,
)
from modelling import (
    prepare_ml_data,
    train_xgboost,
    train_lightgbm,
    cluster_traders,
)


def banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    # ──────────────────────────────────────────────────────────────
    # PHASE 1 — INGEST
    # ──────────────────────────────────────────────────────────────
    banner("PHASE 1 · Data Ingest")
    trades_raw    = load_trades()
    sentiment_raw = load_sentiment()

    # ──────────────────────────────────────────────────────────────
    # PHASE 2 — PREPROCESSING
    # ──────────────────────────────────────────────────────────────
    banner("PHASE 2 · Preprocessing")
    trades_clean    = clean_trades(trades_raw)
    sentiment_clean = clean_sentiment(sentiment_raw)

    save_processed(trades_clean,    "trades_clean.csv")
    save_processed(sentiment_clean, "fear_greed_clean.csv")

    # ──────────────────────────────────────────────────────────────
    # PHASE 3 — FEATURE ENGINEERING & MERGE
    # ──────────────────────────────────────────────────────────────
    banner("PHASE 3 · Feature Engineering")
    sentiment_enriched = add_sentiment_features(sentiment_clean)
    trades_enriched    = add_trade_level_features(trades_clean)
    merged             = merge_datasets(trades_enriched, sentiment_enriched)
    merged             = add_rolling_account_features(merged)

    save_processed(merged, "merged_dataset.csv")

    # ──────────────────────────────────────────────────────────────
    # PHASE 4 — SENTIMENT vs. PERFORMANCE ANALYSIS
    # ──────────────────────────────────────────────────────────────
    banner("PHASE 4 · Sentiment vs. Performance Analysis")

    regime_stats = pnl_by_regime(merged)
    print("\nPnL by Regime:")
    print(regime_stats.to_string())

    anova = run_anova(merged)
    print(f"\nANOVA → F={anova['F_statistic']}, p={anova['p_value']} — {anova['interpretation']}")

    if "regime_numeric" in merged.columns and "closed_pnl" in merged.columns:
        corr = pearson_spearman(merged, "regime_numeric", "closed_pnl")
        print(f"\nCorrelation (regime_numeric ↔ closedPnL):")
        print(f"  Pearson  r={corr['pearson_r']}, p={corr['pearson_p']}")
        print(f"  Spearman r={corr['spearman_r']}, p={corr['spearman_p']}")

    # Charts
    plot_sentiment_timeline(sentiment_enriched)
    plot_pnl_by_regime(merged)
    plot_winrate_by_regime(regime_stats)
    plot_correlation_heatmap(merged)

    # ──────────────────────────────────────────────────────────────
    # PHASE 5 — PATTERN DISCOVERY
    # ──────────────────────────────────────────────────────────────
    banner("PHASE 5 · Pattern Discovery")

    account_summary = account_performance_summary(merged)
    account_summary = trader_segments(account_summary)
    print(f"\nTrader segments:\n{account_summary['segment'].value_counts().to_string()}")

    plot_top_accounts(account_summary)

    account_summary, pca_df = cluster_traders(account_summary)
    plot_pca_clusters(pca_df)

    # ──────────────────────────────────────────────────────────────
    # PHASE 6 — PREDICTIVE MODELLING
    # ──────────────────────────────────────────────────────────────
    banner("PHASE 6 · Predictive Modelling")

    X, y = prepare_ml_data(merged)

    if len(X) > 100:
        xgb_results = train_xgboost(X, y)
        lgb_results = train_lightgbm(X, y)

        plot_feature_importance(xgb_results["feature_importance"])

        print(f"\n{'─'*40}")
        print(f"  XGBoost  → AUC {xgb_results['cv_auc_mean']} ± {xgb_results['cv_auc_std']}")
        print(f"  LightGBM → AUC {lgb_results['cv_auc_mean']} ± {lgb_results['cv_auc_std']}")
    else:
        print("[!] Not enough labelled samples for ML — skipping modelling phase")

    # ──────────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────────
    banner("✅ Pipeline Complete")
    print(f"  Figures   → {ROOT / 'outputs' / 'figures'}")
    print(f"  Models    → {ROOT / 'outputs' / 'models'}")
    print(f"  Processed → {ROOT / 'data' / 'processed'}")


if __name__ == "__main__":
    main()
