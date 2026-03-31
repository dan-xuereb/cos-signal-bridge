"""Shared test fixtures for cos-signal-bridge."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sdl.models.config import NormalizationConfig, SglIntegration
from sdl.models.factor import FactorRecord
from sdl.models.ir import ExpressionNode
from sdl.types import (
    DATA_SOURCE_BTC_FORGE,
    DiscoveryMethod,
    NormMethod,
    OperatorTag,
    TypeTag,
)


def make_leaf(op: OperatorTag = OperatorTag.close) -> ExpressionNode:
    """Create a primitive (leaf) ExpressionNode."""
    return ExpressionNode(op=op, inferred_type=TypeTag.Series)


def make_unary(op: OperatorTag, child: ExpressionNode, **params) -> ExpressionNode:
    """Create a unary operator ExpressionNode."""
    return ExpressionNode(op=op, children=[child], params=params, inferred_type=TypeTag.Series)


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
        discovery_ts=datetime.now(UTC),
        author="test-suite",
    )


@pytest.fixture
def lag1_factor_with_sgl(lag1_factor: FactorRecord) -> FactorRecord:
    """FactorRecord for lag(close, 1) with SglIntegration populated (passthrough norm)."""
    lag1_factor.sgl_integration = SglIntegration(
        signal_name="lag_close_1",
        signal_version="v1",
        normalization=NormalizationConfig(
            method=NormMethod.passthrough,
            window=0,
            clip_lo=-999.0,
            clip_hi=999.0,
        ),
        invert=False,
    )
    return lag1_factor


@pytest.fixture
def lag1_factor_with_tanh(lag1_factor: FactorRecord) -> FactorRecord:
    """FactorRecord for lag(close, 1) with tanh_zscore normalization (window=5)."""
    lag1_factor.sgl_integration = SglIntegration(
        signal_name="lag_close_1",
        signal_version="v1",
        normalization=NormalizationConfig(
            method=NormMethod.tanh_zscore,
            window=5,
            clip_lo=-1.0,
            clip_hi=1.0,
        ),
        invert=False,
    )
    return lag1_factor


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
