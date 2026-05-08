"""
test_preprocessing.py
=====================
Unit tests for the data preprocessing pipeline.
Run with:  conda run -n cv_conda python -m pytest tests/ -v
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import clean_trades, clean_sentiment, REGIME_LABELS
from feature_engineering import (
    add_trade_level_features,
    add_sentiment_features,
    merge_datasets,
)
from analysis import pnl_by_regime, run_anova, pearson_spearman, trader_segments, account_performance_summary


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_trades():
    """Minimal fake trades dataframe."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "account":        [f"acct_{i % 10}" for i in range(n)],
        "symbol":         np.random.choice(["BTC", "ETH", "SOL"], n),
        "side":           np.random.choice(["BUY", "SELL"], n),
        "size":           np.abs(np.random.randn(n)) * 100 + 10,
        "closedpnl":      np.random.randn(n) * 500,
        "leverage":       np.random.choice([1, 5, 10, 20, 50], n).astype(float),
        "time":           pd.date_range("2023-01-01", periods=n, freq="6H").astype(np.int64) // 10**6,
    })


@pytest.fixture
def raw_sentiment():
    """Minimal fake sentiment dataframe."""
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    values = np.clip(np.random.randint(0, 100, 365), 1, 99)
    return pd.DataFrame({
        "date":           dates.astype(str),
        "value":          values,
        "classification": np.random.choice(["Fear", "Greed", "Neutral", "Extreme Fear", "Extreme Greed"], 365),
    })


# ── Tests: clean_trades ─────────────────────────────────────────────────────────

def test_clean_trades_shape(raw_trades):
    cleaned = clean_trades(raw_trades)
    assert len(cleaned) <= len(raw_trades)
    assert len(cleaned) > 0


def test_clean_trades_no_null_pnl(raw_trades):
    cleaned = clean_trades(raw_trades)
    assert cleaned["closedpnl"].isna().sum() == 0


def test_clean_trades_has_trade_date(raw_trades):
    cleaned = clean_trades(raw_trades)
    assert "trade_date" in cleaned.columns


# ── Tests: clean_sentiment ──────────────────────────────────────────────────────

def test_clean_sentiment_regime_column(raw_sentiment):
    cleaned = clean_sentiment(raw_sentiment)
    assert "regime" in cleaned.columns


def test_clean_sentiment_no_duplicate_dates(raw_sentiment):
    cleaned = clean_sentiment(raw_sentiment)
    assert cleaned["trade_date"].duplicated().sum() == 0


def test_clean_sentiment_valid_regime_labels(raw_sentiment):
    cleaned = clean_sentiment(raw_sentiment)
    valid = set(REGIME_LABELS)
    actual = set(cleaned["regime"].dropna().unique())
    assert actual.issubset(valid)


# ── Tests: feature engineering ──────────────────────────────────────────────────

def test_add_trade_features_creates_is_win(raw_trades):
    cleaned = clean_trades(raw_trades)
    enriched = add_trade_level_features(cleaned)
    assert "is_win" in enriched.columns
    assert enriched["is_win"].isin([0, 1]).all()


def test_add_sentiment_features_regime_numeric(raw_sentiment):
    cleaned   = clean_sentiment(raw_sentiment)
    enriched  = add_sentiment_features(cleaned)
    assert "regime_numeric" in enriched.columns
    assert enriched["regime_numeric"].between(0, 4).all()


def test_merge_datasets_no_extra_rows(raw_trades, raw_sentiment):
    tc = clean_trades(raw_trades)
    sc = clean_sentiment(raw_sentiment)
    se = add_sentiment_features(sc)
    te = add_trade_level_features(tc)
    merged = merge_datasets(te, se)
    assert len(merged) == len(te)


# ── Tests: analysis ─────────────────────────────────────────────────────────────

def test_pnl_by_regime_returns_dataframe(raw_trades, raw_sentiment):
    tc = add_trade_level_features(clean_trades(raw_trades))
    sc = add_sentiment_features(clean_sentiment(raw_sentiment))
    merged = merge_datasets(tc, sc)
    result = pnl_by_regime(merged)
    assert isinstance(result, pd.DataFrame)
    assert "mean_pnl" in result.columns


def test_anova_returns_dict(raw_trades, raw_sentiment):
    tc     = add_trade_level_features(clean_trades(raw_trades))
    sc     = add_sentiment_features(clean_sentiment(raw_sentiment))
    merged = merge_datasets(tc, sc)
    result = run_anova(merged)
    assert "F_statistic" in result
    assert "p_value" in result


def test_pearson_spearman_keys(raw_trades, raw_sentiment):
    tc     = add_trade_level_features(clean_trades(raw_trades))
    sc     = add_sentiment_features(clean_sentiment(raw_sentiment))
    merged = merge_datasets(tc, sc)
    merged = merged.dropna(subset=["regime_numeric", "closedpnl"])
    result = pearson_spearman(merged, "regime_numeric", "closedpnl")
    assert all(k in result for k in ["pearson_r", "spearman_r"])


def test_trader_segments_labels(raw_trades, raw_sentiment):
    tc     = add_trade_level_features(clean_trades(raw_trades))
    sc     = add_sentiment_features(clean_sentiment(raw_sentiment))
    merged = merge_datasets(tc, sc)
    summary   = account_performance_summary(merged)
    segmented = trader_segments(summary)
    assert set(segmented["segment"].unique()).issubset({"Top", "Middle", "Bottom"})
