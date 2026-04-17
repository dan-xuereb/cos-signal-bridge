"""Evaluator unit tests — per-operator correctness and look-ahead bias validation.

Look-ahead bias tests cover ONLY temporal operators (_SINGLE_SERIES_TEMPORAL,
_TWO_SERIES_TEMPORAL, lag, diff, pct_change, crossover, crossunder). Normalization
ops (zscore, rank_norm, etc.) are intentionally full-window in the Phase 1 evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sdl.models.ir import ExpressionNode
from sdl.types import OperatorTag, TypeTag

from signal_bridge.evaluator import compute_lookback, evaluate

# conftest.py provides: make_leaf, make_unary, make_binary, sample_ohlcv_df
from tests.conftest import make_binary, make_leaf, make_unary

# ---------------------------------------------------------------------------
# Primitive operator tests
# ---------------------------------------------------------------------------


def test_primitive_close(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(close_leaf, df) == df['close']."""
    close_leaf = make_leaf(OperatorTag.close)
    result = evaluate(close_leaf, sample_ohlcv_df)
    pd.testing.assert_series_equal(result, sample_ohlcv_df["close"])


def test_primitive_missing_column(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(vwap_leaf, df_without_vwap) returns all-NaN series."""
    # sample_ohlcv_df has no 'vwap' column
    vwap_leaf = make_leaf(OperatorTag.vwap)
    result = evaluate(vwap_leaf, sample_ohlcv_df)
    assert result.isna().all(), "Expected all-NaN for missing column 'vwap'"


# ---------------------------------------------------------------------------
# Look-ahead bias tests — temporal operators
# ---------------------------------------------------------------------------


def test_lag_no_lookahead(sample_ohlcv_df: pd.DataFrame) -> None:
    """lag(close, 1): iloc[0] is NaN, iloc[1] == close[0], iloc[2] == close[1]."""
    close_leaf = make_leaf(OperatorTag.close)
    lag1 = make_unary(OperatorTag.lag, close_leaf, n=1)
    result = evaluate(lag1, sample_ohlcv_df)

    assert pd.isna(result.iloc[0]), "First bar must be NaN (no prior bar)"
    assert result.iloc[1] == pytest.approx(
        sample_ohlcv_df["close"].iloc[0]
    ), "Bar 1 must equal close[0] — lag(1) shifts backward, not forward"
    assert result.iloc[2] == pytest.approx(
        sample_ohlcv_df["close"].iloc[1]
    ), "Bar 2 must equal close[1]"


def test_diff_no_lookahead(sample_ohlcv_df: pd.DataFrame) -> None:
    """diff(close, 1): iloc[0] is NaN, iloc[1] == close[1] - close[0]."""
    close_leaf = make_leaf(OperatorTag.close)
    diff1 = make_unary(OperatorTag.diff, close_leaf, n=1)
    result = evaluate(diff1, sample_ohlcv_df)

    assert pd.isna(result.iloc[0]), "First bar must be NaN"
    expected = sample_ohlcv_df["close"].iloc[1] - sample_ohlcv_df["close"].iloc[0]
    assert result.iloc[1] == pytest.approx(expected), "Bar 1 must equal close[1] - close[0]"


def test_pct_change_no_lookahead(sample_ohlcv_df: pd.DataFrame) -> None:
    """pct_change(close, 1): iloc[0] is NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    pct1 = make_unary(OperatorTag.pct_change, close_leaf, n=1)
    result = evaluate(pct1, sample_ohlcv_df)

    assert pd.isna(result.iloc[0]), "First bar must be NaN"


def test_roll_mean(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_mean(close, 3): first 2 rows NaN, row[2] == mean of first 3 closes."""
    close_leaf = make_leaf(OperatorTag.close)
    rm3 = make_unary(OperatorTag.roll_mean, close_leaf, n=3)
    result = evaluate(rm3, sample_ohlcv_df)

    assert result.isna().sum() == 2, "roll_mean(n=3) must have exactly 2 leading NaNs"
    expected_row2 = sample_ohlcv_df["close"].iloc[:3].mean()
    assert result.iloc[2] == pytest.approx(expected_row2), "Row 2 must equal mean of first 3 closes"


def test_roll_std(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_std(close, 3): first 2 rows NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    rs3 = make_unary(OperatorTag.roll_std, close_leaf, n=3)
    result = evaluate(rs3, sample_ohlcv_df)

    assert result.isna().sum() == 2, "roll_std(n=3) must have exactly 2 leading NaNs"


def test_roll_zscore(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_zscore(close, 5): first 4 rows NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    rz5 = make_unary(OperatorTag.roll_zscore, close_leaf, n=5)
    result = evaluate(rz5, sample_ohlcv_df)

    assert result.isna().sum() == 4, "roll_zscore(n=5) must have exactly 4 leading NaNs"


def test_roll_autocorr(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_autocorr(close, n=10, lag=1): first 9 rows NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    ra = ExpressionNode(
        op=OperatorTag.roll_autocorr,
        children=[close_leaf],
        params={"n": 10, "lag": 1},
        inferred_type=TypeTag.Series,
    )
    result = evaluate(ra, sample_ohlcv_df)

    assert result.isna().sum() == 9, "roll_autocorr(n=10, lag=1) must have exactly 9 leading NaNs"


def test_roll_corr_two_series(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_corr(close, volume, n=5): first 4 rows NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    rc5 = make_binary(OperatorTag.roll_corr, close_leaf, volume_leaf, n=5)
    result = evaluate(rc5, sample_ohlcv_df)

    assert result.isna().sum() == 4, "roll_corr(n=5) must have exactly 4 leading NaNs"


def test_roll_beta_two_series(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_beta(close, volume, n=5): first 4 rows NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    rb5 = make_binary(OperatorTag.roll_beta, close_leaf, volume_leaf, n=5)
    result = evaluate(rb5, sample_ohlcv_df)

    assert result.isna().sum() == 4, "roll_beta(n=5) must have exactly 4 leading NaNs"


# ---------------------------------------------------------------------------
# Unary arithmetic tests
# ---------------------------------------------------------------------------


def test_unary_neg(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(neg(close), df) == -df['close']."""
    close_leaf = make_leaf(OperatorTag.close)
    neg_node = make_unary(OperatorTag.neg, close_leaf)
    result = evaluate(neg_node, sample_ohlcv_df)

    pd.testing.assert_series_equal(result, -sample_ohlcv_df["close"])


def test_unary_abs(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(abs(neg(close)), df) == df['close'].abs()."""
    close_leaf = make_leaf(OperatorTag.close)
    neg_node = make_unary(OperatorTag.neg, close_leaf)
    abs_node = make_unary(OperatorTag.abs, neg_node)
    result = evaluate(abs_node, sample_ohlcv_df)

    pd.testing.assert_series_equal(result, sample_ohlcv_df["close"].abs())


def test_unary_log1p(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(log1p(close), df) == np.log1p(df['close'])."""
    close_leaf = make_leaf(OperatorTag.close)
    log_node = make_unary(OperatorTag.log1p, close_leaf)
    result = evaluate(log_node, sample_ohlcv_df)

    expected = pd.Series(np.log1p(sample_ohlcv_df["close"]), index=sample_ohlcv_df.index)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Binary arithmetic tests
# ---------------------------------------------------------------------------


def test_binary_add(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(add(close, volume), df) == df['close'] + df['volume']."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    add_node = make_binary(OperatorTag.add, close_leaf, volume_leaf)
    result = evaluate(add_node, sample_ohlcv_df)

    pd.testing.assert_series_equal(result, sample_ohlcv_df["close"] + sample_ohlcv_df["volume"])


def test_binary_clip(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(clip(close, lo=99, hi=101), df) — all values in [99, 101]."""
    close_leaf = make_leaf(OperatorTag.close)
    clip_node = make_unary(OperatorTag.clip, close_leaf, lo=99.0, hi=101.0)
    result = evaluate(clip_node, sample_ohlcv_df)

    assert (result >= 99.0).all(), "All clipped values must be >= 99"
    assert (result <= 101.0).all(), "All clipped values must be <= 101"


# ---------------------------------------------------------------------------
# Decay / EWM tests
# ---------------------------------------------------------------------------


def test_ewm(sample_ohlcv_df: pd.DataFrame) -> None:
    """ewm(close, span=5): no NaN after the first row (ewm is expanding by default)."""
    close_leaf = make_leaf(OperatorTag.close)
    ewm_node = make_unary(OperatorTag.ewm, close_leaf, span=5)
    result = evaluate(ewm_node, sample_ohlcv_df)

    # EWM starts immediately from row 0 — no leading NaN
    assert not result.isna().any(), "ewm() must produce no NaN values"


def test_decay_linear(sample_ohlcv_df: pd.DataFrame) -> None:
    """decay_linear(close, n=3): first 2 rows NaN."""
    close_leaf = make_leaf(OperatorTag.close)
    dl_node = make_unary(OperatorTag.decay_linear, close_leaf, n=3)
    result = evaluate(dl_node, sample_ohlcv_df)

    assert result.isna().sum() == 2, "decay_linear(n=3) must have exactly 2 leading NaNs"


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_zscore_full_series(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(zscore(close), df): mean approx 0, std approx 1."""
    close_leaf = make_leaf(OperatorTag.close)
    zs_node = make_unary(OperatorTag.zscore, close_leaf)
    result = evaluate(zs_node, sample_ohlcv_df)

    assert result.mean() == pytest.approx(0.0, abs=1e-10), "z-scored series mean must be 0"
    assert result.std() == pytest.approx(1.0, abs=1e-10), "z-scored series std must be 1"


def test_rank_norm(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(rank_norm(close), df): all values in (0, 1]."""
    close_leaf = make_leaf(OperatorTag.close)
    rn_node = make_unary(OperatorTag.rank_norm, close_leaf)
    result = evaluate(rn_node, sample_ohlcv_df)

    assert (result > 0.0).all(), "rank_norm values must be > 0"
    assert (result <= 1.0).all(), "rank_norm values must be <= 1"


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------


def test_comparison_gt(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(gt(close, volume), df) returns 1.0 where close > volume, 0.0 elsewhere."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    gt_node = make_binary(OperatorTag.gt, close_leaf, volume_leaf)
    result = evaluate(gt_node, sample_ohlcv_df)

    expected = (sample_ohlcv_df["close"] > sample_ohlcv_df["volume"]).astype(float)
    pd.testing.assert_series_equal(result, expected)


def test_crossover(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(crossover(close, volume), df): returns 1.0 only at crossover points."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    co_node = make_binary(OperatorTag.crossover, close_leaf, volume_leaf)
    result = evaluate(co_node, sample_ohlcv_df)

    # All values must be 0.0 or 1.0
    assert set(result.unique()).issubset({0.0, 1.0}), "crossover must return only 0.0 or 1.0"
    # Manual check: crossover = (c[0] > c[1]) & (c[0].shift(1) <= c[1].shift(1))
    c0 = sample_ohlcv_df["close"]
    c1 = sample_ohlcv_df["volume"]
    expected = ((c0 > c1) & (c0.shift(1) <= c1.shift(1))).astype(float)
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Logical tests
# ---------------------------------------------------------------------------


def test_logical_and(sample_ohlcv_df: pd.DataFrame) -> None:
    """and(above_zero(close), above_zero(volume)) == all 1.0 (both positive)."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    az_close = make_unary(OperatorTag.above_zero, close_leaf)
    az_volume = make_unary(OperatorTag.above_zero, volume_leaf)
    and_node = make_binary(OperatorTag.and_, az_close, az_volume)
    result = evaluate(and_node, sample_ohlcv_df)

    # close and volume are always positive in sample_ohlcv_df
    assert (result == 1.0).all(), "Both close and volume positive: and() must be all 1.0"


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------


def test_to_signal(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(to_signal(zscore(close)), df): values in (-1, 1) after z-scoring.

    to_signal applies tanh. Raw close values (~99-101) saturate tanh to ±1.0,
    so we z-score first to get values near zero where tanh is strictly in (-1, 1).
    """
    close_leaf = make_leaf(OperatorTag.close)
    zs_node = make_unary(OperatorTag.zscore, close_leaf)
    sig_node = make_unary(OperatorTag.to_signal, zs_node)
    result = evaluate(sig_node, sample_ohlcv_df)

    assert (result > -1.0).all(), "tanh(zscore) output must be > -1"
    assert (result < 1.0).all(), "tanh(zscore) output must be < 1"


def test_scale(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(scale(close, factor=2.0), df) == df['close'] * 2.0."""
    close_leaf = make_leaf(OperatorTag.close)
    scale_node = make_unary(OperatorTag.scale, close_leaf, factor=2.0)
    result = evaluate(scale_node, sample_ohlcv_df)

    pd.testing.assert_series_equal(result, sample_ohlcv_df["close"] * 2.0)


# ---------------------------------------------------------------------------
# Regime stub tests
# ---------------------------------------------------------------------------


def test_regime_raises(sample_ohlcv_df: pd.DataFrame) -> None:
    """evaluate(regime_gate(...), df) raises NotImplementedError."""
    close_leaf = make_leaf(OperatorTag.close)
    regime_node = ExpressionNode(
        op=OperatorTag.regime_gate,
        children=[close_leaf],
        inferred_type=TypeTag.Series,
    )
    with pytest.raises(NotImplementedError, match="regime_gate"):
        evaluate(regime_node, sample_ohlcv_df)


# ---------------------------------------------------------------------------
# compute_lookback tests
# ---------------------------------------------------------------------------


def test_compute_lookback_primitive() -> None:
    """compute_lookback(close) == 0 (no warmup needed for primitives)."""
    close_leaf = make_leaf(OperatorTag.close)
    assert compute_lookback(close_leaf) == 0


def test_compute_lookback_lag() -> None:
    """compute_lookback(lag(close, 5)) == 5."""
    close_leaf = make_leaf(OperatorTag.close)
    lag5 = make_unary(OperatorTag.lag, close_leaf, n=5)
    assert compute_lookback(lag5) == 5


def test_compute_lookback_nested() -> None:
    """compute_lookback(roll_mean(lag(close, 5), n=14)) == 5 + 13 = 18."""
    close_leaf = make_leaf(OperatorTag.close)
    lag5 = make_unary(OperatorTag.lag, close_leaf, n=5)
    rm14 = make_unary(OperatorTag.roll_mean, lag5, n=14)
    # Additive: lag contributes 5, roll_mean contributes n-1=13
    assert compute_lookback(rm14) == 18


def test_compute_lookback_binary() -> None:
    """compute_lookback(roll_corr(lag(close, 2), volume, n=10)) == 2 + 0 + 9 = 11."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    lag2 = make_unary(OperatorTag.lag, close_leaf, n=2)
    rc10 = make_binary(OperatorTag.roll_corr, lag2, volume_leaf, n=10)
    # lag contributes 2, volume primitive contributes 0, roll_corr contributes 9
    assert compute_lookback(rc10) == 11


def test_compute_lookback_crossover() -> None:
    """compute_lookback(crossover(close, volume)) == 1 (needs 1 bar of history)."""
    close_leaf = make_leaf(OperatorTag.close)
    volume_leaf = make_leaf(OperatorTag.volume)
    co = make_binary(OperatorTag.crossover, close_leaf, volume_leaf)
    assert compute_lookback(co) == 1


def test_compute_lookback_roll_autocorr_includes_lag() -> None:
    """compute_lookback(roll_autocorr(close, n=10, lag=3)) == (10 - 1) + 3 == 12.

    roll_autocorr needs an n-sized window PLUS lag offset bars before the
    inner _linear_autocorr produces a non-NaN value; its lookback is not
    simply n - 1.
    """
    close_leaf = make_leaf(OperatorTag.close)
    ra = ExpressionNode(
        op=OperatorTag.roll_autocorr,
        children=[close_leaf],
        params={"n": 10, "lag": 3},
        inferred_type=TypeTag.Series,
    )
    assert compute_lookback(ra) == 12


def test_compute_lookback_roll_autocorr_default_lag() -> None:
    """compute_lookback(roll_autocorr(close, n=10)) == (10 - 1) + 1 == 10 (default lag=1)."""
    close_leaf = make_leaf(OperatorTag.close)
    ra = ExpressionNode(
        op=OperatorTag.roll_autocorr,
        children=[close_leaf],
        params={"n": 10},
        inferred_type=TypeTag.Series,
    )
    assert compute_lookback(ra) == 10


# ---------------------------------------------------------------------------
# Cost gate smoke
# ---------------------------------------------------------------------------


def test_evaluate_triggers_cost_gate_before_dispatch() -> None:
    """evaluate() pre-flight gate fires before dispatch for over-budget trees."""
    from signal_bridge.evaluator import ExpressionCostExceeded

    leaf = make_leaf(OperatorTag.close)
    node = make_unary(OperatorTag.roll_mean, leaf, n=2048)
    with pytest.raises(ExpressionCostExceeded):
        evaluate(node, pd.DataFrame({"close": [1.0]}))
