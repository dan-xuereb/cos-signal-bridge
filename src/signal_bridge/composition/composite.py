"""CompositeScore output contract and IC-weighted composition engine.

IC values are CONSUMED here, not computed. The canonical IC implementation
lives at ``signal_bridge.ic.compute_ic``. Per GOV-02 (v3.0 milestone), bridge
feedback is the single source of truth for IC; the composition engine treats
IC weights as a pre-computed input dict supplied by the caller. Do not
re-implement IC computation here.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from .library import SignalLibrary
from .polarity import apply_polarity
from .types import HorizonCategory


class SignalContribution(BaseModel):
    """A single signal's contribution to a CompositeScore."""

    model_config = ConfigDict(frozen=True)

    signal_name: str
    weight: float  # normalized IC weight (sums to 1.0 within horizon)


class CompositeScore(BaseModel):
    """Frozen typed output of the composition engine (per D-01, D-02, D-03).

    score: final composite score after hierarchical IC-weighted aggregation
    horizon_category: horizon bucket this score was computed for
    timestamp: snapshot timestamp
    contributions: per-signal normalized IC weights (sums to 1.0)
    """

    model_config = ConfigDict(frozen=True)

    score: float
    horizon_category: HorizonCategory
    timestamp: datetime
    contributions: tuple[SignalContribution, ...]


def compose(
    values: dict[str, float],
    ic_weights: dict[str, Optional[float]],
    library: SignalLibrary,
    horizon: HorizonCategory,
    timestamp: datetime,
) -> Optional[CompositeScore]:
    """IC-weighted hierarchical composition per D-04, D-05, D-06, D-07, D-12.

    Algorithm:
    1. Filter library to signals present in both values and ic_weights for this horizon
    2. Exclude signals with ic_weight None or <= 0
    3. Group remaining signals by theme (D-05)
    4. Per theme: compute IC-weighted mean of polarity-adjusted values (D-04, CIE-03)
    5. Across themes: equal-weight mean of theme scores (D-05)
    6. Return None if no valid signals for this horizon (D-06)

    Args:
        values: {signal_name: normalized_value} — pre-extracted signal values (D-12)
        ic_weights: {signal_name: current_rolling_ic} — IC from oos_monitoring (D-07);
            values of None are treated as missing and excluded.
        library: SignalLibrary providing SignalMeta for polarity and theme
        horizon: Which HorizonCategory to compose for
        timestamp: Timestamp for the CompositeScore snapshot

    Returns:
        CompositeScore or None if no valid signals exist for this horizon.
    """
    # Step 1: filter to signals with valid values and positive IC for this horizon
    candidates = [
        m for m in library.filter(horizon=horizon)
        if m.name in values
        and m.name in ic_weights
        and ic_weights[m.name] is not None
        and ic_weights[m.name] > 0  # type: ignore[operator]
    ]
    if not candidates:
        return None

    # Step 2: group by theme (D-05)
    by_theme: dict[str, list] = defaultdict(list)
    for m in candidates:
        by_theme[m.theme].append(m)

    # Step 3: per-theme IC-weighted mean with polarity adjustment (D-04, CIE-03)
    theme_scores: dict[str, float] = {}
    for theme, metas in by_theme.items():
        total_ic = sum(ic_weights[m.name] for m in metas)  # type: ignore[misc]
        if total_ic == 0:
            continue
        theme_scores[theme] = sum(
            float(apply_polarity(values[m.name], m)) * ic_weights[m.name]  # type: ignore[operator]
            for m in metas
        ) / total_ic

    if not theme_scores:
        return None

    # Step 4: equal-weight mean across themes (D-05)
    final_score = sum(theme_scores.values()) / len(theme_scores)

    # Step 5: build contributions list with normalized weights
    total_ic_all = sum(ic_weights[m.name] for m in candidates)  # type: ignore[misc]
    contributions = tuple(
        SignalContribution(
            signal_name=m.name,
            weight=ic_weights[m.name] / total_ic_all if total_ic_all > 0 else 0.0,  # type: ignore[operator]
        )
        for m in candidates
    )

    return CompositeScore(
        score=final_score,
        horizon_category=horizon,
        timestamp=timestamp,
        contributions=contributions,
    )
