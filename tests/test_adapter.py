"""Tests for the SDL-to-SGL adapter (factor_to_indicator_spec)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sdl.models.config import NormalizationConfig, SglIntegration
from sdl.models.factor import FactorRecord
from sdl.types import NormMethod
from signal_bridge.adapter import factor_to_indicator_spec
from signal_bridge.evaluator import compute_lookback
from xuer_sgl.models import IndicatorSpec, NaNReport, SignalManifest, StalenessReport, TimeGrid
from xuer_sgl.signal_frame import SignalFrame
from xuer_sgl.types import BarAvailabilityState, GapMode, WindowSemantics


# ---------------------------------------------------------------------------
# Helper — build a SignalFrame from a spec + df (kept in test file, not adapter)
# ---------------------------------------------------------------------------


def apply_indicator_spec_to_df(spec: IndicatorSpec, df: pd.DataFrame) -> SignalFrame:
    """Apply an SDL-derived IndicatorSpec to a DataFrame, producing a valid SignalFrame."""
    result_series = spec.func(df)
    col = spec.outputs[0]
    avail_values = [
        BarAvailabilityState.INSUFFICIENT_DATA.value
        if i < spec.lookback
        else (
            BarAvailabilityState.MISSING_NATIVE.value
            if pd.isna(v)
            else BarAvailabilityState.VALID.value
        )
        for i, v in enumerate(result_series)
    ]
    data = pd.DataFrame({col: result_series}, index=df.index, dtype=float)
    avail = pd.DataFrame({col: avail_values}, index=df.index)
    return SignalFrame(
        data=data,
        availability=avail,
        manifest=SignalManifest(columns=[col]),
        time_grid=TimeGrid(freq="1h", gap_mode=GapMode.GAPLESS, clock="UTC", index=df.index),
        nan_report=NaNReport.from_frame(data),
        staleness_report=StalenessReport(),
    )


# ---------------------------------------------------------------------------
# 50-bar fixture (for tanh_zscore tests — needs more bars than combined_lb=6)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_df_50() -> pd.DataFrame:
    """50-bar OHLCV DataFrame for tests requiring more history after warmup."""
    n = 50
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0.1, 0.5, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_spec_fields(lag1_factor_with_sgl: FactorRecord, sample_ohlcv_df: pd.DataFrame) -> None:
    """IndicatorSpec has correct name, source, version, lookback, outputs."""
    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    assert spec.name == "sdl_lag_close_1_v1"
    assert spec.source == "custom"
    assert spec.version == "v1"
    assert spec.lookback == 1  # lag_lb=1 + norm_window=0
    assert spec.outputs == ["sdl_lag_close_1_v1"]
    assert spec.window_semantics == WindowSemantics.BAR_COUNT
    assert spec.permitted_availability == {BarAvailabilityState.VALID}


def test_column_naming(lag1_factor_with_sgl: FactorRecord) -> None:
    """Column name follows sdl_{signal_name}_{version} pattern."""
    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    assert spec.name == "sdl_lag_close_1_v1"
    assert spec.outputs[0] == "sdl_lag_close_1_v1"


def test_combined_lookback_with_norm(lag1_factor_with_tanh: FactorRecord) -> None:
    """lookback = compute_lookback(expr_ir) + normalization.window."""
    spec = factor_to_indicator_spec(lag1_factor_with_tanh)
    # expr_lb = compute_lookback(lag(close, 1)) = 1; norm_window = 5 → combined = 6
    expr_lb = compute_lookback(lag1_factor_with_tanh.expr_ir)
    assert expr_lb == 1
    assert spec.lookback == 6


def test_missing_sgl_integration(lag1_factor: FactorRecord) -> None:
    """ValueError when sgl_integration is None."""
    lag1_factor.sgl_integration = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="missing sgl_integration"):
        factor_to_indicator_spec(lag1_factor)


def test_missing_signal_name(lag1_factor: FactorRecord) -> None:
    """ValueError when signal_name is None."""
    lag1_factor.sgl_integration = SglIntegration()  # signal_name defaults to None
    with pytest.raises(ValueError, match="missing sgl_integration.signal_name"):
        factor_to_indicator_spec(lag1_factor)


def test_func_produces_series(
    lag1_factor_with_sgl: FactorRecord, sample_ohlcv_df: pd.DataFrame
) -> None:
    """spec.func(df) returns pd.Series with len == len(df)."""
    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    result = spec.func(sample_ohlcv_df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlcv_df)


def test_warmup_nan_in_func_output(
    lag1_factor_with_tanh: FactorRecord, sample_ohlcv_df_50: pd.DataFrame
) -> None:
    """First combined_lookback bars of spec.func(df) are NaN."""
    spec = factor_to_indicator_spec(lag1_factor_with_tanh)
    result = spec.func(sample_ohlcv_df_50)
    # combined_lb = 1 + 5 = 6
    assert result.iloc[:6].isna().all()


def test_signal_frame_invariants(
    lag1_factor_with_sgl: FactorRecord, sample_ohlcv_df: pd.DataFrame
) -> None:
    """Build SignalFrame from spec output — passes __post_init__ with no errors."""
    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df)
    assert sf.data.columns.tolist() == ["sdl_lag_close_1_v1"]


def test_signal_frame_invariants_with_tanh(
    lag1_factor_with_tanh: FactorRecord, sample_ohlcv_df_50: pd.DataFrame
) -> None:
    """Round-trip with tanh_zscore normalization must not raise."""
    spec = factor_to_indicator_spec(lag1_factor_with_tanh)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df_50)
    assert sf.data.columns.tolist() == ["sdl_lag_close_1_v1"]


def test_availability_assignment(
    lag1_factor_with_sgl: FactorRecord, sample_ohlcv_df: pd.DataFrame
) -> None:
    """First combined_lb bars are INSUFFICIENT_DATA, rest are VALID or MISSING_NATIVE."""
    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df)
    col = "sdl_lag_close_1_v1"
    avail = sf.availability[col]
    # First lookback (=1) bars must be INSUFFICIENT_DATA
    assert (avail.iloc[: spec.lookback] == BarAvailabilityState.INSUFFICIENT_DATA.value).all()
    # Remaining rows: VALID or MISSING_NATIVE only
    remaining = avail.iloc[spec.lookback :]
    valid_states = {BarAvailabilityState.VALID.value, BarAvailabilityState.MISSING_NATIVE.value}
    assert remaining.isin(valid_states).all()
