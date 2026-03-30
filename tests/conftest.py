"""Shared test fixtures for cos-signal-bridge."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from uuid import uuid4

from sdl.models.ir import ExpressionNode
from sdl.models.factor import FactorRecord
from sdl.types import (
    DiscoveryMethod,
    OperatorTag,
    TypeTag,
    DATA_SOURCE_BTC_FORGE,
)


def make_leaf(op: OperatorTag = OperatorTag.close) -> ExpressionNode:
    """Create a primitive (leaf) ExpressionNode."""
    return ExpressionNode(op=op, inferred_type=TypeTag.Series)


def make_unary(op: OperatorTag, child: ExpressionNode, **params) -> ExpressionNode:
    """Create a unary operator ExpressionNode."""
    return ExpressionNode(
        op=op, children=[child], params=params, inferred_type=TypeTag.Series
    )


def make_binary(
    op: OperatorTag, left: ExpressionNode, right: ExpressionNode, **params
) -> ExpressionNode:
    """Create a binary operator ExpressionNode."""
    return ExpressionNode(
        op=op, children=[left, right], params=params, inferred_type=TypeTag.Series
    )


@pytest.fixture
def lag1_factor() -> FactorRecord:
    """Minimal FactorRecord: lag(close, 1)."""
    close_leaf = make_leaf(OperatorTag.close)
    lag_node = make_unary(OperatorTag.lag, close_leaf, n=1)
    return FactorRecord(
        canonical_expr="lag(close, n=1)",
        expr_ir=lag_node,
        source_expr="lag(close, n=1)",
        description="Lagged close by 1 bar.",
        output_type=TypeTag.Series,
        input_primitives=["close"],
        data_sources=[DATA_SOURCE_BTC_FORGE],
        lookback_bars=1,
        complexity_score=lag_node.complexity,
        discovery_method=DiscoveryMethod.hand_crafted,
        discovery_ts=datetime.now(timezone.utc),
        author="test-suite",
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """20-bar OHLCV DataFrame with deterministic values for evaluator tests."""
    n = 20
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
