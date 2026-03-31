"""Unit tests for normalization closure factory (normalization.py).

Tests cover all 4 NormMethod variants, clip, invert, warmup NaN mask,
combined_lookback arithmetic, and config=None passthrough.
"""

import numpy as np
import pandas as pd
from sdl.models.config import NormalizationConfig
from sdl.types import NormMethod, OperatorTag

from signal_bridge.evaluator import compute_lookback, evaluate
from signal_bridge.normalization import make_normalized_callable
from tests.conftest import make_leaf, make_unary

# ---------------------------------------------------------------------------
# Helper — build test data and lag(close, 1) callable
# ---------------------------------------------------------------------------


def _make_test_data(n: int = 50):
    """Build a deterministic n-bar OHLCV DataFrame and a lag(close,1) raw fn."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    close_leaf = make_leaf(OperatorTag.close)
    lag_node = make_unary(OperatorTag.lag, close_leaf, n=1)

    def raw_fn(df_: pd.DataFrame) -> pd.Series:
        return evaluate(lag_node, df_)

    expr_lb = compute_lookback(lag_node)  # returns 1
    return df, raw_fn, expr_lb, lag_node


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tanh_zscore_basic():
    """tanh_zscore(window=5): first combined_lb=6 bars NaN, rest finite in (-1,1)."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    config = NormalizationConfig(method=NormMethod.tanh_zscore, window=5, clip_lo=-1.0, clip_hi=1.0)
    combined_lb = expr_lb + config.window  # 1 + 5 = 6
    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)

    assert isinstance(result, pd.Series)
    assert result.iloc[:combined_lb].isna().all(), "First combined_lb bars must be NaN"
    valid = result.iloc[combined_lb:].dropna()
    assert len(valid) > 0, "Should have some valid (non-NaN) values after warmup"
    assert (valid.abs() < 1.0).all(), "tanh output must be in (-1, 1)"


def test_minmax_basic():
    """minmax: valid values are in [0, 1]."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    config = NormalizationConfig(method=NormMethod.minmax, window=0, clip_lo=0.0, clip_hi=1.0)
    combined_lb = expr_lb + config.window  # 1 + 0 = 1
    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)

    assert result.iloc[:combined_lb].isna().all()
    valid = result.iloc[combined_lb:].dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all() and (valid <= 1.0).all(), "minmax values must be in [0, 1]"


def test_minmax_constant():
    """minmax with constant input returns 0.0 for valid bars (not NaN or inf)."""
    n = 20
    close = np.full(n, 100.0)  # constant
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    close_leaf = make_leaf(OperatorTag.close)
    lag_node = make_unary(OperatorTag.lag, close_leaf, n=1)

    def raw_fn(df_: pd.DataFrame) -> pd.Series:
        return evaluate(lag_node, df_)

    expr_lb = compute_lookback(lag_node)  # 1

    config = NormalizationConfig(method=NormMethod.minmax, window=0, clip_lo=0.0, clip_hi=1.0)
    combined_lb = expr_lb + config.window  # 1
    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)

    valid = result.iloc[combined_lb:].dropna()
    assert len(valid) > 0
    assert (valid == 0.0).all(), "Constant input must produce 0.0 (not NaN/inf)"
    assert not valid.isna().any()
    assert np.isfinite(valid).all()


def test_rank_basic():
    """rank normalization: valid values in (0, 1]."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    config = NormalizationConfig(method=NormMethod.rank, window=0, clip_lo=0.0, clip_hi=1.0)
    combined_lb = expr_lb + config.window  # 1
    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)

    assert result.iloc[:combined_lb].isna().all()
    valid = result.iloc[combined_lb:].dropna()
    assert len(valid) > 0
    assert (valid > 0.0).all() and (valid <= 1.0).all(), "rank(pct=True) values must be in (0, 1]"


def test_passthrough():
    """passthrough: result equals raw fn output (after warmup mask)."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    config = NormalizationConfig(
        method=NormMethod.passthrough, window=0, clip_lo=-999.0, clip_hi=999.0
    )
    combined_lb = expr_lb + config.window  # 1
    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)
    raw = raw_fn(df)

    # First bar forced NaN by warmup mask
    assert result.iloc[:combined_lb].isna().all()
    # After warmup, passthrough preserves raw values
    pd.testing.assert_series_equal(
        result.iloc[combined_lb:],
        raw.iloc[combined_lb:],
        check_names=False,
    )


def test_clip():
    """clip_lo/clip_hi applied after normalization — no value outside bounds."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    config = NormalizationConfig(method=NormMethod.tanh_zscore, window=5, clip_lo=-0.5, clip_hi=0.5)
    combined_lb = expr_lb + config.window  # 6
    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)

    valid = result.iloc[combined_lb:].dropna()
    assert len(valid) > 0
    assert (valid >= -0.5).all() and (valid <= 0.5).all(), "All valid values must be in [-0.5, 0.5]"


def test_invert():
    """invert=True negates output vs invert=False (sign flipped on valid bars)."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    config = NormalizationConfig(method=NormMethod.tanh_zscore, window=5, clip_lo=-1.0, clip_hi=1.0)
    combined_lb = expr_lb + config.window  # 6

    fn_normal = make_normalized_callable(raw_fn, config, combined_lb, invert=False)
    fn_inverted = make_normalized_callable(raw_fn, config, combined_lb, invert=True)

    result_normal = fn_normal(df)
    result_inverted = fn_inverted(df)

    # Both warmup regions must be NaN
    assert result_normal.iloc[:combined_lb].isna().all()
    assert result_inverted.iloc[:combined_lb].isna().all()

    # Valid bars: inverted == -normal
    valid_mask = ~result_normal.isna()
    pd.testing.assert_series_equal(
        result_inverted[valid_mask],
        -result_normal[valid_mask],
        check_names=False,
    )


def test_warmup_nan_invariant():
    """First combined_lb bars are NaN for ALL 4 NormMethods."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    window = 5

    for method in [
        NormMethod.tanh_zscore,
        NormMethod.minmax,
        NormMethod.rank,
        NormMethod.passthrough,
    ]:
        # tanh_zscore uses window, others window=0 for combined_lb math (but we set window=5 for tanh)
        norm_window = window if method == NormMethod.tanh_zscore else 0
        config = NormalizationConfig(
            method=method, window=norm_window, clip_lo=-999.0, clip_hi=999.0
        )
        combined_lb = expr_lb + config.window
        fn = make_normalized_callable(raw_fn, config, combined_lb)
        result = fn(df)

        assert (
            result.iloc[:combined_lb].isna().all()
        ), f"First {combined_lb} bars must be NaN for method={method.value}"


def test_combined_lookback():
    """combined_lookback = expr_lb + norm_window; first combined_lb bars NaN, bar at index combined_lb is not NaN."""
    df, raw_fn, _, lag_node = _make_test_data(50)

    expr_lb = compute_lookback(lag_node)
    assert expr_lb == 1, f"lag(close,1) lookback must be 1, got {expr_lb}"

    window = 5
    config = NormalizationConfig(
        method=NormMethod.tanh_zscore, window=window, clip_lo=-1.0, clip_hi=1.0
    )
    combined_lb = expr_lb + window  # = 6
    assert combined_lb == 6

    fn = make_normalized_callable(raw_fn, config, combined_lb)
    result = fn(df)

    # First 6 bars: NaN
    assert result.iloc[:combined_lb].isna().all()
    # Bar at index 6: should have a valid value (tanh_zscore on 50-bar series with window=5)
    assert not pd.isna(result.iloc[combined_lb]), f"Bar at index {combined_lb} should not be NaN"


def test_config_none():
    """config=None behaves as passthrough with warmup mask applied."""
    df, raw_fn, expr_lb, _ = _make_test_data(50)
    combined_lb = 1  # just expr_lb for lag(close,1)

    fn = make_normalized_callable(raw_fn, None, combined_lb)
    result = fn(df)
    raw = raw_fn(df)

    # Warmup bars must be NaN
    assert result.iloc[:combined_lb].isna().all()
    # After warmup: raw values preserved
    pd.testing.assert_series_equal(
        result.iloc[combined_lb:],
        raw.iloc[combined_lb:],
        check_names=False,
    )
