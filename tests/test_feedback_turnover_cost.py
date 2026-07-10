"""
Tests for the turnover-scaled cost haircut in feedback.py (Step 6).

NOTE: distinct from tests/test_cost_gate.py, which covers expression
compute-cost limits — unrelated to trading-cost governance.

Covers:
- Default coefficient (0.0) preserves legacy raw-IC verdicts exactly
- Positive coefficient flags a high-turnover factor whose raw IC clears ic_floor
- Positive coefficient invalidates when effective invalidation threshold is crossed,
  and invalidation_reason reports the effective threshold
- Zero turnover (constant signal) is neutral: verdict identical to raw thresholds
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sdl.models.factor import FactorRecord
from sdl.models.governance import GovernancePolicy
from sdl.types import (
    DATA_SOURCE_BTC_FORGE,
    DiscoveryMethod,
    FactorStatus,
    OperatorTag,
    TypeTag,
)
from xuer_sgl.models import NaNReport, SignalManifest, StalenessReport, TimeGrid
from xuer_sgl.signal_frame import SignalFrame
from xuer_sgl.types import BarAvailabilityState, GapMode

from signal_bridge.feedback import compute_monitoring_update
from signal_bridge.registry import load_factor, save_factor

WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2024, 1, 2, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_feedback.py fixture patterns)
# ---------------------------------------------------------------------------


def _make_lag1_record(status: FactorStatus = FactorStatus.active) -> FactorRecord:
    from sdl.models.ir import ExpressionNode

    close_leaf = ExpressionNode(op=OperatorTag.close, inferred_type=TypeTag.Series)
    lag_node = ExpressionNode(
        op=OperatorTag.lag,
        children=[close_leaf],
        params={"n": 1},
        inferred_type=TypeTag.Series,
    )
    kwargs = {
        "canonical_expr": "lag(close, n=1)",
        "expr_ir": lag_node,
        "source_expr": "lag(close, n=1)",
        "description": "Lagged close by 1 bar.",
        "output_type": TypeTag.Series,
        "input_primitives": ["close"],
        "data_sources": [DATA_SOURCE_BTC_FORGE],
        "lookback_bars": 1,
        "complexity_score": lag_node.complexity,
        "discovery_method": DiscoveryMethod.hand_crafted,
        "discovery_ts": datetime.now(UTC),
        "author": "test-suite",
        "status": status,
    }
    if status == FactorStatus.active:
        kwargs["activation_date"] = datetime.now(UTC)
    return FactorRecord(**kwargs)


def _make_signal_frame(
    signal_values: np.ndarray,
    close_values: np.ndarray,
    col: str = "sdl_lag_close_1_v1",
) -> SignalFrame:
    n = len(signal_values)
    index = pd.date_range("2024-01-01", periods=n, freq="1h")
    data = pd.DataFrame(
        {col: signal_values, "close": close_values}, index=index, dtype=float
    )
    valid = [BarAvailabilityState.VALID.value] * n
    avail = pd.DataFrame({col: valid, "close": valid}, index=index)
    return SignalFrame(
        data=data,
        availability=avail,
        manifest=SignalManifest(columns=[col, "close"]),
        time_grid=TimeGrid(freq="1h", gap_mode=GapMode.GAPLESS, clock="UTC", index=index),
        nan_report=NaNReport.from_frame(data),
        staleness_report=StalenessReport(),
    )


def _high_ic_high_turnover_frame(col: str = "sdl_lag_close_1_v1") -> SignalFrame:
    """Positively predictive signal (high raw IC) with substantial turnover."""
    n = 40
    signal_vals = np.arange(1.0, n + 1.0)
    # close = 100 + cumsum(signal) → returns increase with lagged signal → high IC
    close_vals = 100.0 + np.cumsum(signal_vals)
    return _make_signal_frame(signal_vals, close_vals, col=col)


def _metrics(sf: SignalFrame, col: str) -> tuple[float, float]:
    """Compute (realized_ic, realized_turnover) exactly as feedback.py does."""
    signal_lagged = sf.data[col].shift(1)
    returns = sf.data["close"].pct_change()
    valid_mask = sf.availability[col] == BarAvailabilityState.VALID.value
    combined = pd.DataFrame({"s": signal_lagged, "r": returns})[valid_mask].dropna()
    ic = float(combined["s"].corr(combined["r"], method="spearman"))
    turnover = float(sf.data[col][valid_mask].diff().abs().mean())
    return ic, turnover


# ---------------------------------------------------------------------------
# Test 1: default coefficient (omitted) preserves legacy behavior
# ---------------------------------------------------------------------------


def test_default_coefficient_preserves_legacy_verdict(tmp_path: Path) -> None:
    """Coefficient omitted → raw-IC comparison; huge-turnover factor NOT flagged."""
    col = "sdl_lag_close_1_v1"
    sf = _high_ic_high_turnover_frame(col)
    actual_ic, actual_turnover = _metrics(sf, col)
    assert actual_turnover > 0.5  # sanity: this really is a high-turnover signal

    record = _make_lag1_record(FactorStatus.active)
    registry_dir = tmp_path / "registry"
    save_factor(record, registry_dir)

    # ic_floor just below realized IC → raw comparison passes despite huge turnover
    policy = GovernancePolicy(ic_floor=actual_ic - 0.01, ic_invalidation=-2.0)

    update = compute_monitoring_update(
        sf=sf,
        col=col,
        factor_id=record.factor_id,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        registry_dir=registry_dir,
        policy=policy,
    )

    assert update.flagged is False
    assert update.invalidate is False
    assert load_factor(record.factor_id, registry_dir).status == FactorStatus.active


# ---------------------------------------------------------------------------
# Test 2: positive coefficient flags marginal-IC / high-turnover factor
# ---------------------------------------------------------------------------


def test_positive_coefficient_flags_high_turnover_factor(tmp_path: Path) -> None:
    """realized_ic > ic_floor but < effective floor → flagged, active → monitoring."""
    col = "sdl_lag_close_1_v1"
    sf = _high_ic_high_turnover_frame(col)
    actual_ic, actual_turnover = _metrics(sf, col)

    record = _make_lag1_record(FactorStatus.active)
    registry_dir = tmp_path / "registry"
    save_factor(record, registry_dir)

    ic_floor = actual_ic - 0.01  # raw IC clears the floor
    policy = GovernancePolicy(ic_floor=ic_floor, ic_invalidation=-2.0)
    # coefficient chosen so effective floor exceeds realized IC
    coeff = (actual_ic - ic_floor + 0.05) / actual_turnover

    update = compute_monitoring_update(
        sf=sf,
        col=col,
        factor_id=record.factor_id,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        registry_dir=registry_dir,
        policy=policy,
        cost_ic_penalty_per_unit_turnover=coeff,
    )

    assert update.flagged is True
    assert update.invalidate is False
    assert load_factor(record.factor_id, registry_dir).status == FactorStatus.monitoring


# ---------------------------------------------------------------------------
# Test 3: positive coefficient invalidates; reason reports effective threshold
# ---------------------------------------------------------------------------


def test_positive_coefficient_invalidates_and_reports_effective_threshold(
    tmp_path: Path,
) -> None:
    """realized_ic < effective invalidation → invalidated; reason mentions 'effective'."""
    col = "sdl_lag_close_1_v1"
    sf = _high_ic_high_turnover_frame(col)
    actual_ic, actual_turnover = _metrics(sf, col)

    record = _make_lag1_record(FactorStatus.active)
    registry_dir = tmp_path / "registry"
    save_factor(record, registry_dir)

    ic_invalidation = actual_ic - 0.01  # raw IC clears the invalidation threshold
    policy = GovernancePolicy(ic_floor=ic_invalidation, ic_invalidation=ic_invalidation)
    coeff = (actual_ic - ic_invalidation + 0.05) / actual_turnover

    update = compute_monitoring_update(
        sf=sf,
        col=col,
        factor_id=record.factor_id,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        registry_dir=registry_dir,
        policy=policy,
        cost_ic_penalty_per_unit_turnover=coeff,
    )

    assert update.invalidate is True
    updated = load_factor(record.factor_id, registry_dir)
    assert updated.status == FactorStatus.invalidated
    assert updated.invalidation_reason is not None
    assert "effective" in updated.invalidation_reason


# ---------------------------------------------------------------------------
# Test 4: zero turnover is neutral — verdict identical to raw thresholds
# ---------------------------------------------------------------------------


def test_zero_turnover_is_neutral(tmp_path: Path) -> None:
    """Constant signal (turnover 0.0) with coefficient > 0 → raw-threshold verdict."""
    col = "sdl_lag_close_1_v1"
    n = 40
    signal_vals = np.full(n, 5.0)  # constant → turnover 0, IC 0.0 (constant series)
    rng = np.random.default_rng(7)
    close_vals = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    sf = _make_signal_frame(signal_vals, close_vals, col=col)

    record = _make_lag1_record(FactorStatus.active)
    registry_dir = tmp_path / "registry"
    save_factor(record, registry_dir)

    # IC will be 0.0 (constant signal); floor below it → no flag even with huge coeff
    policy = GovernancePolicy(ic_floor=-0.5, ic_invalidation=-1.0)

    update = compute_monitoring_update(
        sf=sf,
        col=col,
        factor_id=record.factor_id,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        registry_dir=registry_dir,
        policy=policy,
        cost_ic_penalty_per_unit_turnover=100.0,
    )

    assert update.realized_turnover == 0.0
    assert update.flagged is False
    assert update.invalidate is False
    assert load_factor(record.factor_id, registry_dir).status == FactorStatus.active
