"""
BTE-to-SDL monitoring feedback: computes realized IC and applies status transitions.

After a backtest completes, this module closes the monitoring loop by:
1. Computing signal quality metrics (realized IC, rank IC, turnover) from a SignalFrame
2. Applying status transition logic based on OosMonitoringConfig thresholds
3. Atomically writing the updated FactorRecord back to the registry
4. Returning an SglFactorMonitoringUpdate to the caller
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd
from sdl.models.factor import FactorRecord
from sdl.models.search import SglFactorMonitoringUpdate
from sdl.types import FactorStatus

from signal_bridge.ic import compute_ic
from signal_bridge.registry import load_factor, save_factor

try:
    from xuer_sgl.signal_frame import SignalFrame
    from xuer_sgl.types import BarAvailabilityState
except ImportError:
    SignalFrame = None  # noqa: N816
    BarAvailabilityState = None  # noqa: N816


def compute_monitoring_update(
    sf: SignalFrame,
    col: str,
    factor_id: UUID,
    window_start: datetime,
    window_end: datetime,
    registry_dir: Path,
) -> SglFactorMonitoringUpdate:
    """
    Compute post-backtest realized IC and apply factor status transitions.

    Loads the FactorRecord from registry, computes realized IC/rank IC/turnover
    from the SignalFrame using only VALID bars, applies OOS monitoring status
    transition logic, writes the updated record atomically, and returns an
    SglFactorMonitoringUpdate.

    Args:
        sf: SignalFrame containing the factor signal column and 'close' prices.
        col: Column name in sf.data / sf.availability for the factor signal.
        factor_id: UUID of the FactorRecord in the registry.
        window_start: Start of the evaluation window.
        window_end: End of the evaluation window.
        registry_dir: Directory where FactorRecord JSON files are stored.

    Returns:
        SglFactorMonitoringUpdate with realized metrics and status flags.

    Raises:
        RuntimeError: If xuer_sgl is not installed.
        FileNotFoundError: If factor_id does not exist in registry_dir.
    """
    if SignalFrame is None:
        raise RuntimeError(
            "xuer_sgl is not installed — install with: pip install cos-signal-bridge[sgl]"
        )

    # Step 1: Load the factor record from registry
    record: FactorRecord = load_factor(factor_id, registry_dir)

    # Step 2: Build VALID mask (exclude INSUFFICIENT_DATA and other non-VALID states)
    valid_mask = sf.availability[col] == BarAvailabilityState.VALID.value

    # Step 3: Compute realized IC (Spearman correlation of lagged signal vs forward returns)
    # Lag the signal by 1 bar to prevent look-ahead bias (per D-04)
    signal_lagged = sf.data[col].shift(1)
    returns = sf.data["close"].pct_change()
    realized_ic: float = compute_ic(signal_lagged, returns, valid_mask)

    # Rebuild combined for rank IC (Step 4) — same filter as compute_ic applies internally
    combined = pd.DataFrame({"signal": signal_lagged, "returns": returns})
    combined_valid = combined[valid_mask].dropna()

    # Step 4: Compute realized rank IC (Pearson on pct-ranked signal and returns, per D-05)
    if len(combined_valid) >= 2:
        rank_signal = combined_valid["signal"].rank(pct=True)
        rank_returns = combined_valid["returns"].rank(pct=True)
        realized_rank_ic: float = float(rank_signal.corr(rank_returns))
        if pd.isna(realized_rank_ic):
            realized_rank_ic = 0.0
    else:
        realized_rank_ic = 0.0

    # Step 5: Compute realized turnover on VALID rows (per D-06)
    valid_col = sf.data[col][valid_mask]
    realized_turnover: float = float(valid_col.diff().abs().mean())
    if pd.isna(realized_turnover):
        realized_turnover = 0.0

    # Step 6: Determine flagged/invalidate status based on IC thresholds
    ic_floor = record.oos_monitoring.ic_floor
    ic_invalidation = record.oos_monitoring.ic_invalidation

    flagged: bool = realized_ic < ic_floor
    invalidate: bool = realized_ic < ic_invalidation

    # Step 7: Apply status transitions (per D-02)
    if invalidate and record.status in {FactorStatus.active, FactorStatus.monitoring}:
        record.status = FactorStatus.invalidated
        record.invalidation_reason = (
            f"Realized IC {realized_ic:.4f} < ic_invalidation {ic_invalidation}"
        )
    elif flagged and record.status == FactorStatus.active:
        record.status = FactorStatus.monitoring

    # Step 8: Update OOS monitoring state
    record.oos_monitoring.last_checked = window_end
    record.oos_monitoring.current_rolling_ic = realized_ic

    # Step 9: Atomic write-back to registry (tmp -> rename)
    save_factor(record, registry_dir)

    # Step 10: Return SglFactorMonitoringUpdate (signal_id == factor_id per D-03)
    return SglFactorMonitoringUpdate(
        signal_id=factor_id,
        factor_id=factor_id,
        window_start=window_start,
        window_end=window_end,
        realized_ic=realized_ic,
        realized_rank_ic=realized_rank_ic,
        realized_turnover=realized_turnover,
        realized_regime_breakdown=[],  # Deferred per D-07
        flagged=flagged,
        invalidate=invalidate,
    )
