"""Tests for signal_bridge.provider — extract_signal_dict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xuer_sgl.models import NaNReport, SignalManifest, StalenessReport, TimeGrid
from xuer_sgl.signal_frame import SignalFrame
from xuer_sgl.types import BarAvailabilityState, GapMode

from signal_bridge.provider import extract_signal_dict

# Re-use the apply_indicator_spec_to_df helper from test_adapter
from test_adapter import apply_indicator_spec_to_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_frame(
    n: int = 20,
    n_invalid: int = 1,
    col: str = "sdl_test_v1",
    invalid_state: str = BarAvailabilityState.INSUFFICIENT_DATA.value,
) -> SignalFrame:
    """Build a synthetic SignalFrame with n bars, first n_invalid bars non-VALID.

    VALID bars have deterministic float values; non-VALID bars have NaN data.
    """
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n)
    avail_list = [invalid_state] * n_invalid + [BarAvailabilityState.VALID.value] * (n - n_invalid)
    data_list = [float("nan")] * n_invalid + list(values[n_invalid:])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    data = pd.DataFrame({col: data_list}, index=idx, dtype=float)
    avail = pd.DataFrame({col: avail_list}, index=idx)
    return SignalFrame(
        data=data,
        availability=avail,
        manifest=SignalManifest(columns=[col]),
        time_grid=TimeGrid(freq="1h", gap_mode=GapMode.GAPLESS, clock="UTC", index=idx),
        nan_report=NaNReport.from_frame(data),
        staleness_report=StalenessReport(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_bars_only(lag1_factor_with_sgl, sample_ohlcv_df) -> None:
    """extract_signal_dict returns dict with exactly 19 entries (1 INSUFFICIENT_DATA, 19 VALID)."""
    from signal_bridge.adapter import factor_to_indicator_spec

    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df)
    col = spec.outputs[0]
    result = extract_signal_dict(sf, col)
    # spec.lookback == 1 → 1 INSUFFICIENT_DATA bar, 19 VALID bars
    assert len(result) == 19


def test_keys_are_nanoseconds(lag1_factor_with_sgl, sample_ohlcv_df) -> None:
    """All dict keys are int nanosecond timestamps from sf.data.index.asi8."""
    from signal_bridge.adapter import factor_to_indicator_spec

    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df)
    col = spec.outputs[0]
    result = extract_signal_dict(sf, col)
    assert all(isinstance(k, int) for k in result)
    # All returned keys must be in the full index nanoseconds set
    all_ns = set(sf.data.index.asi8.tolist())
    assert set(result.keys()).issubset(all_ns)


def test_values_match_data(lag1_factor_with_sgl, sample_ohlcv_df) -> None:
    """Dict values are float and match sf.data[col] at the corresponding index positions."""
    from signal_bridge.adapter import factor_to_indicator_spec

    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df)
    col = spec.outputs[0]
    result = extract_signal_dict(sf, col)
    assert all(isinstance(v, float) for v in result.values())
    # Build expected: map nanosecond timestamp → data value for VALID bars
    avail = sf.availability[col]
    valid_mask = avail == BarAvailabilityState.VALID.value
    expected_series = sf.data[col][valid_mask]
    for ts_ns, expected_val in zip(expected_series.index.asi8.tolist(), expected_series.tolist()):
        assert ts_ns in result
        assert result[ts_ns] == pytest.approx(expected_val)


def test_missing_native_excluded() -> None:
    """Bars with MISSING_NATIVE availability are excluded from the dict."""
    col = "sdl_test_v1"
    sf = _make_signal_frame(
        n=20,
        n_invalid=3,
        col=col,
        invalid_state=BarAvailabilityState.MISSING_NATIVE.value,
    )
    result = extract_signal_dict(sf, col)
    assert len(result) == 17  # 3 MISSING_NATIVE excluded


def test_insufficient_data_excluded() -> None:
    """Bars with INSUFFICIENT_DATA availability are excluded from the dict."""
    col = "sdl_test_v1"
    sf = _make_signal_frame(
        n=20,
        n_invalid=5,
        col=col,
        invalid_state=BarAvailabilityState.INSUFFICIENT_DATA.value,
    )
    result = extract_signal_dict(sf, col)
    assert len(result) == 15  # 5 INSUFFICIENT_DATA excluded


def test_empty_frame() -> None:
    """SignalFrame where ALL bars are INSUFFICIENT_DATA returns empty dict."""
    col = "sdl_test_v1"
    sf = _make_signal_frame(
        n=20,
        n_invalid=20,
        col=col,
        invalid_state=BarAvailabilityState.INSUFFICIENT_DATA.value,
    )
    result = extract_signal_dict(sf, col)
    assert result == {}


def test_missing_column(lag1_factor_with_sgl, sample_ohlcv_df) -> None:
    """Column not found raises KeyError."""
    from signal_bridge.adapter import factor_to_indicator_spec

    spec = factor_to_indicator_spec(lag1_factor_with_sgl)
    sf = apply_indicator_spec_to_df(spec, sample_ohlcv_df)
    with pytest.raises(KeyError):
        extract_signal_dict(sf, "nonexistent_column")
