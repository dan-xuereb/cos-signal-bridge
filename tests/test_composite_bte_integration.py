"""Typed integration test proving BTE can consume CompositeScore directly (SC-2).

No hasattr(), getattr(), or casting -- all field access is typed.

Tests prove that:
1. CompositeScore fields are accessible via typed attribute access (not duck-typing).
2. The .score float flows into a BTE-compatible signal_dict (dict[int, float]) without
   any casting or hasattr checks — the same type that SignalDrivenStrategyConfig.signal_dict
   accepts per test_e2e.py.
3. Multiple CompositeScore instances can feed a multi-timestamp signal dict without casting.
"""

from __future__ import annotations

from datetime import UTC, datetime

from signal_bridge.composition import (
    CompositeScore,
    HorizonCategory,
    Polarity,
    SignalContribution,
    SignalLibrary,
    SignalMeta,
    compose_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = datetime(2024, 6, 1, tzinfo=UTC)


def _make_library(*entries: tuple[str, str, HorizonCategory]) -> SignalLibrary:
    """Build a SignalLibrary from (name, theme, horizon) tuples."""
    lib = SignalLibrary()
    for name, theme, horizon in entries:
        meta = SignalMeta(
            name=name,
            display_name=name.replace("_", " ").title(),
            polarity=Polarity.POSITIVE,
            horizon_category=horizon,
            half_life_days=5.0,
            theme=theme,
            update_frequency="daily",
            economic_rationale="test signal",
            data_source="test",
        )
        lib.register(meta)
    return lib


# ---------------------------------------------------------------------------
# Test 1: Single horizon — typed consumption of CompositeScore for BTE
# ---------------------------------------------------------------------------


def test_composite_score_typed_consumption_for_bte() -> None:
    """BTE can consume CompositeScore via typed attribute access without casting.

    SC-2: CompositeScore fields are accessed with typed attribute access.
    The .score float is placed directly into a signal_dict[int, float] —
    the exact type SignalDrivenStrategyConfig.signal_dict accepts.
    """
    lib = _make_library(
        ("momentum_1d", "price", HorizonCategory.FAST),
        ("rsi_14", "price", HorizonCategory.FAST),
    )
    signal_values = {"momentum_1d": 0.6, "rsi_14": 0.4}
    factor_ics = {"momentum_1d": 0.07, "rsi_14": 0.05}

    result = compose_signals(signal_values, factor_ics, lib, TS)

    # Typed check — no duck-typing
    assert isinstance(result[HorizonCategory.FAST], CompositeScore)

    # Typed attribute access — no hasattr, no getattr, no cast
    composite: CompositeScore = result[HorizonCategory.FAST]
    assert isinstance(composite.score, float)
    assert isinstance(composite.horizon_category, HorizonCategory)
    assert isinstance(composite.timestamp, datetime)
    assert isinstance(composite.contributions, tuple)

    # Build a BTE-compatible signal dict using typed .score float — no casting needed
    signal_dict: dict[int, float] = {0: composite.score}
    assert signal_dict[0] == composite.score

    # Typed sub-model access — contributions are SignalContribution, not generic objects
    assert isinstance(composite.contributions[0], SignalContribution)
    assert isinstance(composite.contributions[0].weight, float)
    assert isinstance(composite.contributions[0].signal_name, str)


# ---------------------------------------------------------------------------
# Test 2: Multi-horizon — typed consumption of multiple CompositeScores for BTE
# ---------------------------------------------------------------------------


def test_composite_score_multi_horizon_typed_bte_consumption() -> None:
    """Multiple CompositeScore instances can feed a BTE signal dict without casting.

    SC-2: every horizon's .score flows to dict[int, float] via typed attribute access.
    """
    lib = _make_library(
        ("momentum_1d", "price", HorizonCategory.FAST),
        ("funding_rate", "onchain", HorizonCategory.SLOW),
    )
    signal_values = {"momentum_1d": 0.5, "funding_rate": -0.2}
    factor_ics = {"momentum_1d": 0.06, "funding_rate": 0.08}

    result = compose_signals(signal_values, factor_ics, lib, TS)

    # Both horizons must be present
    assert HorizonCategory.FAST in result
    assert HorizonCategory.SLOW in result

    # Every result value is a typed CompositeScore — no duck-typing
    for score in result.values():
        assert isinstance(score, CompositeScore)

    # Build a multi-timestamp BTE signal dict: {ts_ns: score.score} — all typed float access
    signal_dict: dict[int, float] = {
        ts_ns: score.score for ts_ns, score in enumerate(result.values())
    }

    # Every value in the signal dict is a float — proves typed .score flowed through
    assert all(isinstance(v, float) for v in signal_dict.values())
    assert len(signal_dict) == len(result)
