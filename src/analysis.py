"""
analysis.py
===========
Statistical analysis utilities:
  • Correlation (Pearson + Spearman)
  • One-way ANOVA across sentiment regimes
  • Account-level performance aggregation
  • Drawdown / Sharpe-proxy calculation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


# ── Regime ordering for display ────────────────────────────────────────────────
REGIME_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def pnl_by_regime(df: pd.DataFrame, pnl_col: str = "closed_pnl") -> pd.DataFrame:
    """
    Aggregate PnL statistics grouped by sentiment regime.

    Returns
    -------
    pd.DataFrame
        Index = regime, columns = [count, mean_pnl, median_pnl, std_pnl, win_rate, total_pnl]
    """
    if "regime" not in df.columns or pnl_col not in df.columns:
        raise ValueError("DataFrame must have 'regime' and pnl_col columns.")

    grp = df.groupby("regime", observed=True)[pnl_col]
    result = pd.DataFrame({
        "count":      grp.count(),
        "mean_pnl":   grp.mean(),
        "median_pnl": grp.median(),
        "std_pnl":    grp.std(),
        "total_pnl":  grp.sum(),
    })

    if "is_win" in df.columns:
        result["win_rate"] = df.groupby("regime", observed=True)["is_win"].mean()

    # Reindex to preserve logical regime order
    available = [r for r in REGIME_ORDER if r in result.index]
    return result.reindex(available)


def run_anova(df: pd.DataFrame, pnl_col: str = "closed_pnl") -> dict:
    """
    One-way ANOVA: test whether mean PnL differs across sentiment regimes.

    Returns
    -------
    dict with keys: F_statistic, p_value, interpretation
    """
    groups = [
        grp[pnl_col].dropna().values
        for _, grp in df.groupby("regime", observed=True)
        if len(grp) > 1
    ]

    if len(groups) < 2:
        return {"error": "Not enough groups for ANOVA"}

    f_stat, p_value = stats.f_oneway(*groups)
    interp = "Significant (p<0.05)" if p_value < 0.05 else "Not significant (p≥0.05)"

    return {
        "F_statistic": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "interpretation": interp,
    }


def pearson_spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    """
    Compute Pearson and Spearman correlation between two numeric columns.

    Returns
    -------
    dict with pearson_r, pearson_p, spearman_r, spearman_p
    """
    pair = df[[x, y]].dropna()
    pr, pp = stats.pearsonr(pair[x], pair[y])
    sr, sp = stats.spearmanr(pair[x], pair[y])
    return {
        "pearson_r": round(pr, 4),
        "pearson_p": round(pp, 6),
        "spearman_r": round(sr, 4),
        "spearman_p": round(sp, 6),
    }


def account_performance_summary(df: pd.DataFrame, pnl_col: str = "closed_pnl") -> pd.DataFrame:
    """
    Aggregate per-account performance metrics.

    Columns returned
    ----------------
    total_pnl, mean_pnl, win_rate, trade_count, sharpe_proxy,
    max_single_loss, max_single_gain, dominant_regime
    """
    grp = df.groupby("account")

    pnl_stats = grp[pnl_col].agg(
        total_pnl="sum",
        mean_pnl="mean",
        std_pnl="std",
        trade_count="count",
        max_single_gain="max",
        max_single_loss="min",
    )

    if "is_win" in df.columns:
        pnl_stats["win_rate"] = grp["is_win"].mean()

    # Sharpe proxy: mean / std (ignores risk-free rate for simplicity)
    pnl_stats["sharpe_proxy"] = pnl_stats["mean_pnl"] / pnl_stats["std_pnl"].replace(0, np.nan)

    # Most common sentiment regime for each account's trades
    if "regime" in df.columns:
        pnl_stats["dominant_regime"] = (
            df.groupby("account")["regime"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        )

    return pnl_stats.sort_values("total_pnl", ascending=False)


def trader_segments(account_summary: pd.DataFrame, quantile_threshold: float = 0.25) -> pd.DataFrame:
    """
    Label accounts as 'Top', 'Middle', or 'Bottom' performers by total_pnl.

    Parameters
    ----------
    account_summary : pd.DataFrame
        Output of account_performance_summary()
    quantile_threshold : float
        Fraction used to define top/bottom buckets (default 25%).

    Returns
    -------
    pd.DataFrame
        account_summary with a 'segment' column appended.
    """
    df = account_summary.copy()
    low_q  = df["total_pnl"].quantile(quantile_threshold)
    high_q = df["total_pnl"].quantile(1 - quantile_threshold)

    def _label(v):
        if v >= high_q:
            return "Top"
        if v <= low_q:
            return "Bottom"
        return "Middle"

    df["segment"] = df["total_pnl"].apply(_label)
    return df
