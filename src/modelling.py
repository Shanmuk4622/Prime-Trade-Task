"""
modelling.py
============
Machine-learning pipeline for the PrimeTrade.ai assignment.

Goals
-----
1. Predict the **direction** of a trade's PnL (binary: win/loss) using sentiment
   features, trade features, and rolling account statistics.
2. Rank feature importances to quantify how much market sentiment contributes.
3. Simulate a simple sentiment-gated trading strategy and report performance.
"""

import pathlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

ROOT    = pathlib.Path(__file__).resolve().parents[1]
MDL_DIR = ROOT / "outputs" / "models"
MDL_DIR.mkdir(parents=True, exist_ok=True)

# ── Default feature sets ─────────────────────────────────────────────────────
# Sentiment-ONLY (no rolling account stats) — leakage-free for honest CV
SENTIMENT_ONLY_FEATURES = [
    "regime_numeric", "sentiment_shift", "sentiment_shift_7d_mean",
    "sentiment_volatility_7d", "is_extreme", "is_fear", "is_greed",
    "size_usd", "size_tokens", "is_long",
]
# Full set incl. rolling stats — WARNING: data leakage unless time-split CV used
SENTIMENT_FEATURES = [
    "regime_numeric", "sentiment_shift", "sentiment_shift_7d_mean",
    "sentiment_volatility_7d", "is_extreme", "is_fear", "is_greed",
]
TRADE_FEATURES = [
    "size_usd", "size_tokens", "is_long", "pnl_per_unit",
    "win_rate_7d", "win_rate_30d", "mean_pnl_7d", "trade_count_7d",
]
TARGET_COL = "is_win"


def prepare_ml_data(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    target_col: str = TARGET_COL,
    leakage_free: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select features and target, drop rows with NaN, return X, y.

    Parameters
    ----------
    df : pd.DataFrame
        Fully merged and feature-engineered dataframe.
    feature_cols : list[str]
        Columns to use as features. If None, uses SENTIMENT_ONLY_FEATURES
        when leakage_free=True, or SENTIMENT_FEATURES+TRADE_FEATURES otherwise.
    target_col : str
        Binary target column name.
    leakage_free : bool
        If True, use only SENTIMENT_ONLY_FEATURES (no rolling stats that leak).
    """
    if feature_cols is None:
        if leakage_free:
            feature_cols = SENTIMENT_ONLY_FEATURES
            print("[INFO] Using leakage-free sentiment-only features")
        else:
            feature_cols = [c for c in SENTIMENT_FEATURES + TRADE_FEATURES if c in df.columns]
            print("[WARN] Rolling features included — use leakage_free=True for real CV")

    available = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        print(f"  [!] Missing feature columns (skipped): {missing}")

    data = df[available + [target_col]].dropna()
    X = data[available].copy()
    # Cast all features to float (avoids XGBoost 'category' dtype issue)
    for col in X.columns:
        if X[col].dtype.name == "category" or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float)
    y = data[target_col].astype(int)
    print(f"[OK] ML data ready -> {X.shape[0]:,} samples x {X.shape[1]} features")
    return X, y


def train_xgboost(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """
    Train an XGBoost classifier with cross-validation.

    Returns
    -------
    dict with keys: model, feature_importance, cv_auc, cv_accuracy
    """
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    acc_scores  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Refit on full data for feature importance
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save model
    mdl_path = MDL_DIR / "xgboost_model.pkl"
    joblib.dump(model, mdl_path)
    print(f"[✓] XGBoost trained → AUC {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"    Saved → {mdl_path}")

    return {
        "model": model,
        "feature_importance": importance,
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std":  round(auc_scores.std(), 4),
        "cv_acc_mean": round(acc_scores.mean(), 4),
    }


def train_lightgbm(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """
    Train a LightGBM classifier with cross-validation.

    Returns
    -------
    dict with keys: model, feature_importance, cv_auc, cv_accuracy
    """
    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    acc_scores  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    mdl_path = MDL_DIR / "lightgbm_model.pkl"
    joblib.dump(model, mdl_path)
    print(f"[✓] LightGBM trained → AUC {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"    Saved → {mdl_path}")

    return {
        "model": model,
        "feature_importance": importance,
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std":  round(auc_scores.std(), 4),
        "cv_acc_mean": round(acc_scores.mean(), 4),
    }


def cluster_traders(
    account_summary: pd.DataFrame,
    feature_cols: list[str] = None,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply KMeans clustering to account-level performance features.
    Then run PCA(2) for visualisation.

    Parameters
    ----------
    account_summary : pd.DataFrame
        Output of analysis.account_performance_summary().
    feature_cols : list[str]
        Columns to cluster on.
    n_clusters : int
        Number of KMeans clusters.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (account_summary with 'cluster' column, pca_df for plotting)
    """
    if feature_cols is None:
        feature_cols = [c for c in ["total_pnl", "win_rate", "trade_count", "sharpe_proxy", "mean_pnl"]
                        if c in account_summary.columns]

    data = account_summary[feature_cols].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    account_summary = account_summary.copy()
    account_summary.loc[data.index, "cluster"] = labels.astype(str)

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(coords, index=data.index, columns=["PC1", "PC2"])
    pca_df["cluster"] = labels.astype(str)
    pca_df["total_pnl"] = account_summary.loc[data.index, "total_pnl"]

    print(f"[✓] KMeans clustering → {n_clusters} clusters, "
          f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return account_summary, pca_df


def sentiment_gated_strategy_backtest(
    daily_df: pd.DataFrame,
    pnl_col: str = "daily_pnl",
    regime_col: str = "regime",
) -> pd.DataFrame:
    """
    Simulate a simple sentiment-gated strategy:
      • Greed / Extreme Greed → go SHORT (fade the crowd)
      • Fear  / Extreme Fear  → go LONG  (buy the dip)
      • Neutral               → sit out

    Parameters
    ----------
    daily_df : pd.DataFrame
        Aggregated daily PnL with a 'regime' column.
    pnl_col : str
        Column representing raw daily market PnL or benchmark returns.

    Returns
    -------
    pd.DataFrame
        daily_df with 'strategy_signal', 'strategy_pnl', 'cum_strategy_pnl' appended.
    """
    df = daily_df.copy()

    signal_map = {
        "Extreme Fear":  1,   # Long
        "Fear":          1,
        "Neutral":       0,   # Flat
        "Greed":        -1,   # Short
        "Extreme Greed": -1,
    }
    df["strategy_signal"] = df[regime_col].map(signal_map).fillna(0)
    df["strategy_pnl"]    = df["strategy_signal"] * df[pnl_col]
    df["cum_strategy_pnl"]= df["strategy_pnl"].cumsum()
    df["cum_closed_pnl"]  = df[pnl_col].cumsum()

    total_strategy  = df["strategy_pnl"].sum()
    total_baseline  = df[pnl_col].sum()
    improvement_pct = (total_strategy - total_baseline) / abs(total_baseline) * 100 if total_baseline != 0 else 0

    print(f"[📈] Strategy backtest:")
    print(f"     Baseline total PnL  : {total_baseline:,.2f}")
    print(f"     Strategy total PnL  : {total_strategy:,.2f}")
    print(f"     Improvement         : {improvement_pct:+.1f}%")

    return df
