"""
Tests for composition.py — bridge orchestration of CIE compose().

Tests cover:
- compose_signals() returns a dict of {HorizonCategory: CompositeScore} for horizons with signals
- compose_signals() omits horizons where compose() returns None (D-06)
- compose_signals() correctly extracts values from signal_values dict and IC from factor_ics dict
- compose_signals() returns empty dict when no signals have valid IC
- compose_signals() raises RuntimeError if cos-cie is not installed (optional import guard)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from cos_cie.composite import CompositeScore
from cos_cie.library import SignalLibrary
from cos_cie.models import SignalMeta
from cos_cie.types import HorizonCategory, Polarity

from signal_bridge.composition import compose_signals


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
# Test 1: compose_signals returns {HorizonCategory: CompositeScore} for matching horizons
# ---------------------------------------------------------------------------


def test_compose_signals_returns_scores_for_valid_horizons() -> None:
    """compose_signals() returns a CompositeScore for horizons that have valid signals."""
    lib = _make_library(
        ("momentum_1d", "price", HorizonCategory.FAST),
        ("momentum_7d", "price", HorizonCategory.MEDIUM),
    )
    signal_values = {"momentum_1d": 0.5, "momentum_7d": 0.3}
    factor_ics = {"momentum_1d": 0.05, "momentum_7d": 0.04}

    result = compose_signals(signal_values, factor_ics, lib, TS)

    assert isinstance(result, dict)
    assert HorizonCategory.FAST in result
    assert HorizonCategory.MEDIUM in result
    # Score objects have expected typed fields — no duck-typing
    fast_score = result[HorizonCategory.FAST]
    assert isinstance(fast_score, CompositeScore)
    assert isinstance(fast_score.score, float)
    assert isinstance(fast_score.horizon_category, HorizonCategory)
    assert fast_score.horizon_category == HorizonCategory.FAST
    assert fast_score.timestamp == TS


# ---------------------------------------------------------------------------
# Test 2: compose_signals omits horizons where compose() returns None (D-06)
# ---------------------------------------------------------------------------


def test_compose_signals_omits_horizons_with_no_valid_signals() -> None:
    """Horizons with no matching signals are omitted from result dict."""
    lib = _make_library(
        ("momentum_1d", "price", HorizonCategory.FAST),
    )
    # Only FAST horizon has a registered signal; MEDIUM/SLOW/CYCLE have none
    signal_values = {"momentum_1d": 0.5}
    factor_ics = {"momentum_1d": 0.05}

    result = compose_signals(signal_values, factor_ics, lib, TS)

    assert HorizonCategory.FAST in result
    assert HorizonCategory.MEDIUM not in result
    assert HorizonCategory.SLOW not in result
    assert HorizonCategory.CYCLE not in result


# ---------------------------------------------------------------------------
# Test 3: compose_signals correctly routes values and IC weights to compose()
# ---------------------------------------------------------------------------


def test_compose_signals_routes_values_and_ic_correctly() -> None:
    """Values from signal_values dict and IC from factor_ics dict reach compose()."""
    lib = _make_library(
        ("sig_a", "price", HorizonCategory.SLOW),
        ("sig_b", "macro", HorizonCategory.SLOW),
    )
    signal_values = {"sig_a": 1.0, "sig_b": -0.5}
    factor_ics = {"sig_a": 0.10, "sig_b": 0.08}

    result = compose_signals(signal_values, factor_ics, lib, TS)

    assert HorizonCategory.SLOW in result
    score = result[HorizonCategory.SLOW]
    # Contributions should reference the registered signals
    contrib_names = {c.signal_name for c in score.contributions}
    assert "sig_a" in contrib_names
    assert "sig_b" in contrib_names


# ---------------------------------------------------------------------------
# Test 4: compose_signals returns empty dict when no signals have valid IC
# ---------------------------------------------------------------------------


def test_compose_signals_returns_empty_when_no_valid_ic() -> None:
    """Empty dict returned when all factor_ics are None or <= 0."""
    lib = _make_library(
        ("momentum_1d", "price", HorizonCategory.FAST),
        ("momentum_7d", "price", HorizonCategory.MEDIUM),
    )
    signal_values = {"momentum_1d": 0.5, "momentum_7d": 0.3}
    # All IC weights are zero or negative → compose() returns None for every horizon
    factor_ics = {"momentum_1d": 0.0, "momentum_7d": -0.01}

    result = compose_signals(signal_values, factor_ics, lib, TS)

    assert result == {}


# ---------------------------------------------------------------------------
# Test 5: compose_signals raises RuntimeError if cos-cie is not installed
# ---------------------------------------------------------------------------


def test_compose_signals_raises_if_cie_not_installed() -> None:
    """RuntimeError raised with helpful message if cos-cie is not installed."""
    lib = _make_library(("momentum_1d", "price", HorizonCategory.FAST))
    signal_values = {"momentum_1d": 0.5}
    factor_ics = {"momentum_1d": 0.05}

    with patch("signal_bridge.composition.compose", None):
        with pytest.raises(RuntimeError, match="cos-cie"):
            compose_signals(signal_values, factor_ics, lib, TS)
