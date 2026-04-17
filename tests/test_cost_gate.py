"""Tests for evaluate()'s BRIDGE-01 pre-flight cost gate."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from sdl.types import OperatorTag

from signal_bridge.evaluator import (
    _MAX_NODE_COUNT,
    _MAX_TOTAL_LOOKBACK,
    _MAX_WINDOW,
    ExpressionCostExceeded,
    evaluate,
)
from tests.conftest import make_leaf, make_unary

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Valid (non-violating) expressions still pass
# ---------------------------------------------------------------------------


def test_valid_expression_passes(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_mean(lag(close, n=5), n=14): lookback=18, 3 nodes, max_window=14 — within limits."""
    close = make_leaf(OperatorTag.close)
    lag5 = make_unary(OperatorTag.lag, close, n=5)
    rm14 = make_unary(OperatorTag.roll_mean, lag5, n=14)
    result = evaluate(rm14, sample_ohlcv_df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlcv_df)


# ---------------------------------------------------------------------------
# total_lookback violation
# ---------------------------------------------------------------------------


def test_total_lookback_exceeded_raises() -> None:
    """21 × lag(n=100) on close: cumulative lookback 2100 > 2000."""
    node = make_leaf(OperatorTag.close)
    outermost = None
    for _ in range(21):
        node = make_unary(OperatorTag.lag, node, n=100)
        outermost = node

    assert outermost is not None

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(node, pd.DataFrame({"close": [1.0]}))

    exc = excinfo.value
    assert exc.limit_name == "total_lookback"
    assert exc.observed_value == 2100
    assert exc.limit_value == _MAX_TOTAL_LOOKBACK == 2000
    # Deepest offender = the 21st (outermost) lag where cumulative breaches 2000.
    assert exc.node_id == str(outermost.node_id)


# ---------------------------------------------------------------------------
# node_count violation
# ---------------------------------------------------------------------------


def test_node_count_exceeded_raises() -> None:
    """Chain of 256 neg wrapping close = 257 nodes > 256."""
    node = make_leaf(OperatorTag.close)
    for _ in range(256):
        node = make_unary(OperatorTag.neg, node)

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(node, pd.DataFrame({"close": [1.0]}))

    exc = excinfo.value
    assert exc.limit_name == "node_count"
    assert exc.observed_value == 257
    assert exc.limit_value == _MAX_NODE_COUNT == 256


# ---------------------------------------------------------------------------
# max_window violation
# ---------------------------------------------------------------------------


def test_max_window_exceeded_raises() -> None:
    """roll_mean(close, n=1025): single-node window > 1024."""
    close = make_leaf(OperatorTag.close)
    rm = make_unary(OperatorTag.roll_mean, close, n=1025)

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(rm, pd.DataFrame({"close": [1.0]}))

    exc = excinfo.value
    assert exc.limit_name == "max_window"
    assert exc.observed_value == 1025
    assert exc.limit_value == _MAX_WINDOW == 1024
    assert exc.node_id == str(rm.node_id)


def test_ewm_alpha_passes_cost_gate() -> None:
    """ewm(close, alpha=0.99): alpha is not cost-gated — gate lets it through.

    alpha ∈ (0, 1) has no integer-window analog; SDL validation bounds it.
    Contract: _node_window returns 0 for alpha-only ewm, so the gate passes.
    """
    close = make_leaf(OperatorTag.close)
    ewm_alpha = make_unary(OperatorTag.ewm, close, alpha=0.99)

    # Does not raise — alpha is not cost-gated.
    result = evaluate(ewm_alpha, pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    assert isinstance(result, pd.Series)


def test_ewm_halflife_above_window_raises() -> None:
    """ewm(close, halflife=2000): integer halflife > _MAX_WINDOW triggers gate."""
    close = make_leaf(OperatorTag.close)
    ewm_hl = make_unary(OperatorTag.ewm, close, halflife=2000)

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(ewm_hl, pd.DataFrame({"close": [1.0]}))

    exc = excinfo.value
    assert exc.limit_name == "max_window"
    assert exc.observed_value == 2000
    assert exc.limit_value == _MAX_WINDOW == 1024
    assert exc.node_id == str(ewm_hl.node_id)


def test_ewm_fractional_halflife_contributes_at_least_one() -> None:
    """ewm(close, halflife=0.01): sub-1 halflife must NOT truncate to 0.

    Prior int(float(0.01)) coerced to 0, giving the impression of "no window"
    cost for aggressive decays. _node_window now ceils halflife so fractional
    values still contribute to the max_window check.
    """
    from signal_bridge.evaluator import _node_window

    close = make_leaf(OperatorTag.close)
    ewm_tiny = make_unary(OperatorTag.ewm, close, halflife=0.01)

    # _node_window contract: halflife=0.01 -> ceil(0.01) == 1, not 0.
    assert _node_window(ewm_tiny) == 1

    # Evaluation passes the gate (1 <= _MAX_WINDOW) but the contract is that
    # halflife now participates in max_window tracking at the low end.
    result = evaluate(ewm_tiny, pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Structured-field contract: node_id is str, message format
# ---------------------------------------------------------------------------


def test_exception_node_id_is_string_not_uuid_object() -> None:
    """ExpressionCostExceeded.node_id must be a string matching the UUID regex."""
    close = make_leaf(OperatorTag.close)
    rm = make_unary(OperatorTag.roll_mean, close, n=1025)

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(rm, pd.DataFrame({"close": [1.0]}))

    exc = excinfo.value
    assert isinstance(exc.node_id, str)
    assert UUID_RE.match(exc.node_id), f"node_id not a UUID string: {exc.node_id!r}"


def test_exception_message_format() -> None:
    """str(exc) matches template 'Expression cost exceeded: {name}={obs} > {lim} at node_id={id!r}'."""
    close = make_leaf(OperatorTag.close)
    rm = make_unary(OperatorTag.roll_mean, close, n=1025)

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(rm, pd.DataFrame({"close": [1.0]}))

    exc = excinfo.value
    expected = (
        f"Expression cost exceeded: {exc.limit_name}={exc.observed_value} "
        f"> {exc.limit_value} at node_id={exc.node_id!r}"
    )
    assert str(exc) == expected


# ---------------------------------------------------------------------------
# Gate runs BEFORE dispatch — regime_gate would raise NotImplementedError,
# but cost violation trumps it.
# ---------------------------------------------------------------------------


def test_gate_runs_before_dispatch() -> None:
    """regime_gate(roll_mean(close, n=2048)) raises ExpressionCostExceeded, not NotImplementedError."""
    close = make_leaf(OperatorTag.close)
    inner = make_unary(OperatorTag.roll_mean, close, n=2048)
    node = make_unary(OperatorTag.regime_gate, inner)

    with pytest.raises(ExpressionCostExceeded) as excinfo:
        evaluate(node, pd.DataFrame({"close": [1.0]}))

    # Gate wins — make sure it's specifically our cost exception type, not bare
    # NotImplementedError from the regime stub dispatch.
    assert isinstance(excinfo.value, ExpressionCostExceeded)
    assert not isinstance(excinfo.value, NotImplementedError)


# ---------------------------------------------------------------------------
# Boundary: exactly at limits is still safe (inclusive on safe side)
# ---------------------------------------------------------------------------


def test_valid_tree_at_boundary_passes(sample_ohlcv_df: pd.DataFrame) -> None:
    """roll_mean(close, n=1024): max_window=1024, lookback=1023, 2 nodes — all within limits."""
    close = make_leaf(OperatorTag.close)
    rm = make_unary(OperatorTag.roll_mean, close, n=1024)

    # Does not raise; value correctness is not under test here (20-bar fixture
    # cannot produce a valid 1024-bar rolling mean — leading NaNs only).
    result = evaluate(rm, sample_ohlcv_df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlcv_df)
