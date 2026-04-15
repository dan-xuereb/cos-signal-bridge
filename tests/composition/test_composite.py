"""Unit tests for CompositeScore, SignalContribution models and compose() engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from signal_bridge.composition.library import SignalLibrary
from signal_bridge.composition.models import SignalMeta
from signal_bridge.composition.types import HorizonCategory, Polarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(
    name: str,
    theme: str,
    horizon: HorizonCategory = HorizonCategory.MEDIUM,
    polarity: Polarity = Polarity.POSITIVE,
) -> SignalMeta:
    """Build a minimal SignalMeta for testing."""
    return SignalMeta(
        name=name,
        display_name=name.replace("_", " ").title(),
        polarity=polarity,
        horizon_category=horizon,
        half_life_days=10.0,
        theme=theme,
        update_frequency="1D",
        economic_rationale="Test signal.",
        data_source="test",
    )


def _make_library(*metas: SignalMeta) -> SignalLibrary:
    lib = SignalLibrary()
    for m in metas:
        lib.register(m)
    return lib


_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests for CompositeScore model (CIE-01)
# ---------------------------------------------------------------------------

class TestCompositeScoreModel:

    def test_composite_score_is_frozen(self) -> None:
        """CompositeScore is frozen — assignment raises TypeError or ValidationError."""
        from signal_bridge.composition.composite import CompositeScore, SignalContribution

        cs = CompositeScore(
            score=0.5,
            horizon_category=HorizonCategory.MEDIUM,
            timestamp=_TS,
            contributions=(),
        )
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            cs.score = 0.9  # type: ignore[misc]

    def test_composite_score_contributions_is_tuple(self) -> None:
        """CompositeScore contributions is stored as tuple."""
        from signal_bridge.composition.composite import CompositeScore, SignalContribution

        contrib = SignalContribution(signal_name="sig_a", weight=1.0)
        cs = CompositeScore(
            score=0.5,
            horizon_category=HorizonCategory.MEDIUM,
            timestamp=_TS,
            contributions=(contrib,),
        )
        assert isinstance(cs.contributions, tuple)

    def test_composite_score_fields(self) -> None:
        """CompositeScore has score (float), horizon_category, timestamp, contributions."""
        from signal_bridge.composition.composite import CompositeScore

        cs = CompositeScore(
            score=0.42,
            horizon_category=HorizonCategory.FAST,
            timestamp=_TS,
            contributions=(),
        )
        assert cs.score == 0.42
        assert cs.horizon_category == HorizonCategory.FAST
        assert cs.timestamp == _TS
        assert cs.contributions == ()

    def test_signal_contribution_is_frozen(self) -> None:
        """SignalContribution is frozen — assignment raises TypeError or ValidationError."""
        from signal_bridge.composition.composite import SignalContribution

        sc = SignalContribution(signal_name="foo", weight=0.5)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            sc.weight = 0.9  # type: ignore[misc]

    def test_signal_contribution_fields(self) -> None:
        """SignalContribution has signal_name (str) and weight (float)."""
        from signal_bridge.composition.composite import SignalContribution

        sc = SignalContribution(signal_name="rsi", weight=0.75)
        assert sc.signal_name == "rsi"
        assert sc.weight == 0.75


# ---------------------------------------------------------------------------
# Tests for compose() engine (CIE-02, CIE-03)
# ---------------------------------------------------------------------------

class TestCompose:

    def test_single_theme_single_horizon_two_signals(self) -> None:
        """compose() with two same-theme signals returns IC-weighted mean with polarity applied."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.MEDIUM)
        meta_b = _make_meta("sig_b", theme="momentum", horizon=HorizonCategory.MEDIUM)
        lib = _make_library(meta_a, meta_b)

        values = {"sig_a": 0.6, "sig_b": 0.4}
        ic_weights = {"sig_a": 0.8, "sig_b": 0.2}

        result = compose(values, ic_weights, lib, HorizonCategory.MEDIUM, _TS)

        assert result is not None
        # Expected: IC-weighted mean = (0.6*0.8 + 0.4*0.2) / (0.8+0.2) = (0.48+0.08)/1.0 = 0.56
        assert abs(result.score - 0.56) < 1e-9
        assert result.horizon_category == HorizonCategory.MEDIUM
        assert result.timestamp == _TS

    def test_two_themes_equal_weight_cross_theme(self) -> None:
        """compose() with two themes: per-theme IC-weighted mean then equal-weight cross-theme average."""
        from signal_bridge.composition.composite import compose

        # Theme A: one signal
        meta_a1 = _make_meta("a1", theme="themeA", horizon=HorizonCategory.FAST)
        # Theme B: one signal
        meta_b1 = _make_meta("b1", theme="themeB", horizon=HorizonCategory.FAST)
        lib = _make_library(meta_a1, meta_b1)

        values = {"a1": 1.0, "b1": 0.0}
        ic_weights = {"a1": 0.9, "b1": 0.5}

        result = compose(values, ic_weights, lib, HorizonCategory.FAST, _TS)

        assert result is not None
        # themeA: 1.0 * 0.9 / 0.9 = 1.0
        # themeB: 0.0 * 0.5 / 0.5 = 0.0
        # cross-theme equal weight: (1.0 + 0.0) / 2 = 0.5
        assert abs(result.score - 0.5) < 1e-9

    def test_returns_none_for_empty_horizon(self) -> None:
        """compose() returns None when no signals match the horizon (D-06)."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.SLOW)
        lib = _make_library(meta_a)

        # Ask for FAST but only SLOW signal registered
        result = compose(
            {"sig_a": 0.5},
            {"sig_a": 0.8},
            lib,
            HorizonCategory.FAST,
            _TS,
        )
        assert result is None

    def test_returns_none_when_all_ic_weights_zero_or_none(self) -> None:
        """compose() returns None when all IC weights are 0 or None."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.MEDIUM)
        lib = _make_library(meta_a)

        result = compose(
            {"sig_a": 0.5},
            {"sig_a": 0.0},  # IC weight is 0
            lib,
            HorizonCategory.MEDIUM,
            _TS,
        )
        assert result is None

    def test_excludes_signals_with_nonpositive_ic(self) -> None:
        """compose() excludes signals with ic_weight <= 0 from computation."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.MEDIUM)
        meta_b = _make_meta("sig_b", theme="momentum", horizon=HorizonCategory.MEDIUM)
        lib = _make_library(meta_a, meta_b)

        values = {"sig_a": 0.9, "sig_b": 0.1}
        # sig_b IC <= 0, should be excluded
        ic_weights = {"sig_a": 0.7, "sig_b": -0.3}

        result = compose(values, ic_weights, lib, HorizonCategory.MEDIUM, _TS)

        assert result is not None
        # Only sig_a contributes; IC-weighted mean is just sig_a's value
        assert abs(result.score - 0.9) < 1e-9
        # sig_b must not appear in contributions
        names_in_contributions = {c.signal_name for c in result.contributions}
        assert "sig_b" not in names_in_contributions

    def test_polarity_negative_negates_value_before_weighting(self) -> None:
        """compose() applies polarity — NEGATIVE signal value is negated before IC weighting."""
        from signal_bridge.composition.composite import compose

        meta_pos = _make_meta("pos_sig", theme="momentum", polarity=Polarity.POSITIVE)
        meta_neg = _make_meta("neg_sig", theme="momentum", polarity=Polarity.NEGATIVE)
        lib = _make_library(meta_pos, meta_neg)

        values = {"pos_sig": 0.6, "neg_sig": 0.4}  # neg_sig should become -0.4
        ic_weights = {"pos_sig": 0.5, "neg_sig": 0.5}

        result = compose(values, ic_weights, lib, HorizonCategory.MEDIUM, _TS)

        assert result is not None
        # Both in same theme with equal IC weights:
        # IC-weighted mean = (0.6*0.5 + (-0.4)*0.5) / 1.0 = (0.3 - 0.2) = 0.1
        assert abs(result.score - 0.1) < 1e-9

    def test_contributions_weights_sum_to_one(self) -> None:
        """compose() contributions weights sum to 1.0 within the horizon."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.SLOW)
        meta_b = _make_meta("sig_b", theme="momentum", horizon=HorizonCategory.SLOW)
        meta_c = _make_meta("sig_c", theme="onchain", horizon=HorizonCategory.SLOW)
        lib = _make_library(meta_a, meta_b, meta_c)

        values = {"sig_a": 0.5, "sig_b": 0.3, "sig_c": 0.7}
        ic_weights = {"sig_a": 0.6, "sig_b": 0.3, "sig_c": 0.5}

        result = compose(values, ic_weights, lib, HorizonCategory.SLOW, _TS)

        assert result is not None
        total_weight = sum(c.weight for c in result.contributions)
        assert abs(total_weight - 1.0) < 1e-9

    def test_single_signal_returns_polarity_adjusted_value(self) -> None:
        """compose() with single signal returns that signal's polarity-adjusted value as score."""
        from signal_bridge.composition.composite import compose

        meta = _make_meta("solo", theme="macro", horizon=HorizonCategory.CYCLE, polarity=Polarity.NEGATIVE)
        lib = _make_library(meta)

        values = {"solo": 0.8}
        ic_weights = {"solo": 0.6}

        result = compose(values, ic_weights, lib, HorizonCategory.CYCLE, _TS)

        assert result is not None
        # NEGATIVE polarity: 0.8 * -1 = -0.8; single signal, so score == -0.8
        assert abs(result.score - (-0.8)) < 1e-9

    def test_returns_none_when_no_ic_weights_provided(self) -> None:
        """compose() returns None when ic_weights is empty for all horizon signals."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.MEDIUM)
        lib = _make_library(meta_a)

        # Signal not in ic_weights at all
        result = compose(
            {"sig_a": 0.5},
            {},  # empty ic_weights
            lib,
            HorizonCategory.MEDIUM,
            _TS,
        )
        assert result is None

    def test_none_ic_weight_excludes_signal(self) -> None:
        """compose() excludes signals where ic_weight value is None."""
        from signal_bridge.composition.composite import compose

        meta_a = _make_meta("sig_a", theme="momentum", horizon=HorizonCategory.FAST)
        meta_b = _make_meta("sig_b", theme="momentum", horizon=HorizonCategory.FAST)
        lib = _make_library(meta_a, meta_b)

        values = {"sig_a": 0.7, "sig_b": 0.3}
        ic_weights: dict = {"sig_a": None, "sig_b": 0.5}  # sig_a has None IC

        result = compose(values, ic_weights, lib, HorizonCategory.FAST, _TS)

        assert result is not None
        # Only sig_b should contribute; score = 0.3
        assert abs(result.score - 0.3) < 1e-9

    def test_two_themes_multiple_signals_per_theme(self) -> None:
        """compose() with two themes and multiple signals per theme."""
        from signal_bridge.composition.composite import compose

        # Theme A: two signals
        meta_a1 = _make_meta("a1", theme="momentum", horizon=HorizonCategory.MEDIUM)
        meta_a2 = _make_meta("a2", theme="momentum", horizon=HorizonCategory.MEDIUM)
        # Theme B: one signal
        meta_b1 = _make_meta("b1", theme="onchain", horizon=HorizonCategory.MEDIUM)
        lib = _make_library(meta_a1, meta_a2, meta_b1)

        values = {"a1": 1.0, "a2": 0.0, "b1": 0.5}
        ic_weights = {"a1": 1.0, "a2": 1.0, "b1": 1.0}

        result = compose(values, ic_weights, lib, HorizonCategory.MEDIUM, _TS)

        assert result is not None
        # momentum IC-weighted mean: (1.0*1 + 0.0*1)/2 = 0.5
        # onchain IC-weighted mean: 0.5*1/1 = 0.5
        # cross-theme equal weight: (0.5 + 0.5) / 2 = 0.5
        assert abs(result.score - 0.5) < 1e-9

    def test_no_sgl_imports_in_composite_module(self) -> None:
        """composite.py must not import from xuer_sgl or SignalFrame (D-12)."""
        import inspect
        from signal_bridge.composition import composite

        source = inspect.getsource(composite)
        assert "xuer_sgl" not in source
        assert "SignalFrame" not in source
        assert "import xuer_sgl" not in source
