"""
preprocessing.py
================
Clean, type-cast, and deduplicate the raw datasets.
Outputs go to data/processed/.
"""

import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Sentiment regime labels ────────────────────────────────────────────────────
REGIME_BINS   = [0, 25, 45, 55, 75, 100]
REGIME_LABELS = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Hyperliquid trades dataframe.

    Steps
    -----
    1. Parse timestamps → datetime (UTC)
    2. Cast numeric columns
    3. Drop full duplicates
    4. Remove rows with null PnL (unclosed positions)
    5. Add a 'trade_date' column (date only, for join)

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_trades()

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # ── 1. Timestamps ──────────────────────────────────────────────────────────
    time_cols = [c for c in df.columns if "time" in c]
    for col in time_cols:
        try:
            # Handle millisecond epoch integers or ISO strings
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
            else:
                df[col] = pd.to_datetime(df[col], utc=True, infer_datetime_format=True)
        except Exception:
            pass  # Leave unparseable columns as-is

    # Identify primary time column
    primary_time = next((c for c in ["time", "timestamp", "created_at"] if c in df.columns), time_cols[0] if time_cols else None)
    if primary_time:
        df["trade_date"] = df[primary_time].dt.normalize()

    # ── 2. Numeric casting ─────────────────────────────────────────────────────
    numeric_candidates = ["closedpnl", "size", "execution_price", "leverage", "start_position"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3. Deduplicate ─────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"  [!] Dropped {before - after:,} duplicate trade rows")

    # ── 4. Drop null PnL ───────────────────────────────────────────────────────
    pnl_col = "closedpnl" if "closedpnl" in df.columns else None
    if pnl_col:
        null_count = df[pnl_col].isna().sum()
        df = df.dropna(subset=[pnl_col])
        if null_count:
            print(f"  [!] Dropped {null_count:,} rows with null closedPnL")

    # ── 5. Side normalisation ──────────────────────────────────────────────────
    if "side" in df.columns:
        df["side"] = df["side"].str.strip().str.upper()  # → 'BUY' / 'SELL' or 'LONG' / 'SHORT'

    print(f"[✓] Trades cleaned      → {len(df):,} rows remain")
    return df.reset_index(drop=True)


def clean_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Fear/Greed Index dataframe.

    Steps
    -----
    1. Parse 'date' column
    2. Cast 'value' to numeric (if present)
    3. Map classification to ordered categorical
    4. Add 'regime' bin column from numeric value
    5. Deduplicate by date

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_sentiment()

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 'trade_date' as date key.
    """
    df = df.copy()

    # ── 1. Parse date ──────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["trade_date"] = df["date"].dt.normalize()

    # ── 2. Numeric value ───────────────────────────────────────────────────────
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # Build regime from numeric score
        df["regime"] = pd.cut(df["value"], bins=REGIME_BINS, labels=REGIME_LABELS, right=True)
    else:
        # Fallback: derive regime from classification string
        classification_map = {
            "extreme fear": "Extreme Fear",
            "fear": "Fear",
            "neutral": "Neutral",
            "greed": "Greed",
            "extreme greed": "Extreme Greed",
        }
        df["regime"] = df["classification"].str.lower().map(classification_map)

    # ── 3. Ordered categorical ─────────────────────────────────────────────────
    cat_type = pd.CategoricalDtype(categories=REGIME_LABELS, ordered=True)
    df["regime"] = df["regime"].astype(cat_type)

    # ── 4. Deduplicate by date ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["trade_date"])
    after = len(df)
    if before != after:
        print(f"  [!] Dropped {before - after:,} duplicate sentiment rows")

    print(f"[✓] Sentiment cleaned   → {len(df):,} rows remain")
    return df.sort_values("trade_date").reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str) -> pathlib.Path:
    """Save a processed dataframe to data/processed/<name>.csv."""
    out_path = PROCESSED_DIR / name
    df.to_csv(out_path, index=False)
    print(f"[💾] Saved → {out_path}")
    return out_path
