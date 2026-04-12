"""Bridge orchestration of CIE composition (per D-13).

Wires CIE's compose() function into the bridge pipeline by extracting
signal values and IC weights from bridge-side data structures and passing
them to the pure CIE composition engine.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    from cos_cie.composite import CompositeScore, compose
    from cos_cie.library import SignalLibrary
    from cos_cie.types import HorizonCategory
except ImportError:
    CompositeScore = None  # noqa: N816
    compose = None  # noqa: N816
    SignalLibrary = None  # noqa: N816
    HorizonCategory = None  # noqa: N816


def compose_signals(
    signal_values: dict[str, float],
    factor_ics: dict[str, Optional[float]],
    library: "SignalLibrary",
    timestamp: datetime,
    horizons: "list[HorizonCategory] | None" = None,
) -> "dict[HorizonCategory, CompositeScore]":
    """Orchestrate CIE composition across all horizons.

    Args:
        signal_values: {signal_name: float_value} — pre-extracted normalized values.
        factor_ics: {signal_name: current_rolling_ic} — IC from FactorRecord.oos_monitoring.
            Values of None are treated as missing and excluded by CIE compose().
        library: CIE SignalLibrary with registered SignalMeta entries.
        timestamp: Timestamp for CompositeScore snapshots.
        horizons: Which horizons to compose for. Defaults to all four HorizonCategory values.

    Returns:
        Dict mapping HorizonCategory to CompositeScore. Horizons with no valid
        signals are omitted (D-06).

    Raises:
        RuntimeError: If cos-cie is not installed.
    """
    if compose is None:
        raise RuntimeError(
            "cos-cie is not installed — install with: pip install cos-signal-bridge[cie]"
        )

    if horizons is None:
        horizons = list(HorizonCategory)

    results: dict = {}
    for h in horizons:
        score = compose(
            values=signal_values,
            ic_weights=factor_ics,
            library=library,
            horizon=h,
            timestamp=timestamp,
        )
        if score is not None:
            results[h] = score

    return results
