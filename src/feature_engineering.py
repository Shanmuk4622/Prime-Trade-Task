"""
feature_engineering.py
=======================
Build derived features on the merged dataset that power the analysis and ML models.

Key derived features
--------------------
Per-trade:
  • is_win              — bool: closedPnL > 0
  • pnl_per_unit        — closedPnL / size (risk-normalised PnL)
  • leverage_bin        — low / medium / high / ultra

Per-account per-day window:
  • win_rate_7d / 30d
  • cum_pnl_7d / 30d
  • trade_count_7d

Sentiment:
  • regime_numeric      — ordinal encoding of regime (0–4)
  • sentiment_shift     — daily change in Fear/Greed value
  • is_extreme          — bool: Extreme Fear or Extreme Greed
"""

import numpy as np
import pandas as pd

# ── Regime ordinal map ─────────────────────────────────────────────────────────
REGIME_ORDER = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


def add_trade_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-trade engineered columns to the merged dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe from merge_datasets().

    Returns
    -------
    pd.DataFrame
        Same dataframe with additional columns.
    """
    df = df.copy()

    pnl_col = "closed_pnl"

    # ── Win flag ─────────────────────────────────────────────────────────────
    if pnl_col in df.columns:
        df["is_win"] = (df[pnl_col] > 0).astype(int)

        # PnL per unit of size traded (risk-adjusted, use size_usd)
        size_col = "size_usd" if "size_usd" in df.columns else "size_tokens"
        if size_col in df.columns:
            df["pnl_per_unit"] = df[pnl_col] / df[size_col].replace(0, np.nan)

    # ── Direction binary (real data uses 'Side': BUY=long, SELL=short) ────────────
    if "side" in df.columns:
        df["is_long"] = df["side"].isin(["BUY", "LONG", "B", "L"]).astype(int)

    return df


def add_sentiment_features(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the sentiment dataframe with derived temporal features.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Cleaned sentiment dataframe from preprocessing.clean_sentiment()

    Returns
    -------
    pd.DataFrame
        Enriched sentiment dataframe.
    """
    df = sentiment_df.copy().sort_values("trade_date")

    # Ordinal encoding
    df["regime_numeric"] = df["regime"].map(REGIME_ORDER).fillna(2)

    if "value" in df.columns:
        # Daily shift in sentiment score
        df["sentiment_shift"] = df["value"].diff()
        df["sentiment_shift_7d_mean"] = df["value"].rolling(7, min_periods=1).mean()
        df["sentiment_volatility_7d"] = df["value"].rolling(7, min_periods=1).std()

    # Extreme market conditions flag
    df["is_extreme"] = df["regime"].isin(["Extreme Fear", "Extreme Greed"]).astype(int)
    df["is_fear"] = df["regime"].isin(["Extreme Fear", "Fear"]).astype(int)
    df["is_greed"] = df["regime"].isin(["Greed", "Extreme Greed"]).astype(int)

    return df


def add_rolling_account_features(df: pd.DataFrame, windows: list[int] = [7, 30]) -> pd.DataFrame:
    """
    Add rolling window statistics per account (win rate, cumulative PnL, trade count).

    NOTE: This function is expensive for large datasets — it groups by account
    and applies rolling computations. Progress is printed every 500 accounts.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level merged dataframe sorted by (account, trade_date).
    windows : list[int]
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional rolling columns.
    """
    df = df.copy().sort_values(["account", "trade_date"])
    pnl_col = "closed_pnl"

    if pnl_col not in df.columns:
        print("[!] closed_pnl not found — skipping rolling features")
        return df

    results = []
    accounts = df["account"].unique()
    print(f"[⚙️] Computing rolling features for {len(accounts):,} accounts …")

    for i, acct in enumerate(accounts):
        acct_df = df[df["account"] == acct].copy()

        for w in windows:
            # Rolling mean PnL
            acct_df[f"mean_pnl_{w}d"] = (
                acct_df[pnl_col].rolling(window=w, min_periods=1).mean()
            )
            # Rolling win rate
            acct_df[f"win_rate_{w}d"] = (
                acct_df["is_win"].rolling(window=w, min_periods=1).mean()
                if "is_win" in acct_df.columns
                else np.nan
            )
            # Cumulative sum
            acct_df[f"cum_pnl_{w}d"] = (
                acct_df[pnl_col].rolling(window=w, min_periods=1).sum()
            )
            # Trade count
            acct_df[f"trade_count_{w}d"] = (
                acct_df[pnl_col].rolling(window=w, min_periods=1).count()
            )

        results.append(acct_df)

        if (i + 1) % 500 == 0:
            print(f"  → processed {i + 1:,} / {len(accounts):,} accounts")

    print(f"[✓] Rolling features complete")
    return pd.concat(results, ignore_index=True)


def merge_datasets(trades_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join trades onto sentiment by trade_date.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Cleaned trades (from preprocessing.clean_trades)
    sentiment_df : pd.DataFrame
        Enriched sentiment (from add_sentiment_features)

    Returns
    -------
    pd.DataFrame
        Merged dataframe — one row per trade, with sentiment columns appended.
    """
    # Ensure trade_date is date (not datetime) in both
    trades_df = trades_df.copy()
    sentiment_df = sentiment_df.copy()

    # Strip timezone (UTC) so both columns are naive datetime64[ns] for the merge
    trades_td = pd.to_datetime(trades_df["trade_date"])
    if trades_td.dt.tz is not None:
        trades_td = trades_td.dt.tz_convert(None)
    trades_df["trade_date"] = trades_td.dt.normalize()

    sent_td = pd.to_datetime(sentiment_df["trade_date"])
    if sent_td.dt.tz is not None:
        sent_td = sent_td.dt.tz_convert(None)
    sentiment_df["trade_date"] = sent_td.dt.normalize()

    merged = trades_df.merge(sentiment_df, on="trade_date", how="left")

    sentiment_match = merged["regime"].notna().mean() * 100
    print(f"[✓] Merged dataset      → {len(merged):,} rows")
    print(f"    Sentiment match rate → {sentiment_match:.1f}%")

    return merged
