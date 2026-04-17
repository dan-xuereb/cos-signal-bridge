"""
Dispatch-table expression evaluator for SDL ExpressionNode DAGs.

Walks an ExpressionNode tree bottom-up and maps each OperatorTag to its
pandas Series implementation. All operator categories from XUER-SDL-SPEC §4
are covered; regime operators raise NotImplementedError (v1 stub).

Param key conventions (D-01/D-02/D-03):
  - D-01: All window/period/count params use key "n"
  - D-02: Multi-param ops use descriptive keys (lo/hi, span/alpha/halflife, lag)
  - D-03: Backend-agnostic — params are bridge-defined, not pandas kwargs
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sdl.models.ir import ExpressionNode
from sdl.types import (
    _PRIMITIVE_OPS,
    OperatorTag,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_Handler = Callable[[ExpressionNode, list[pd.Series], pd.DataFrame], pd.Series]
_DISPATCH: dict[OperatorTag, _Handler] = {}

_LookbackFn = Callable[[ExpressionNode], int]
_LOOKBACK: dict[OperatorTag, _LookbackFn] = {}


# ---------------------------------------------------------------------------
# Cost model (BRIDGE-01)
# ---------------------------------------------------------------------------

_MAX_TOTAL_LOOKBACK: int = 2000
_MAX_NODE_COUNT: int = 256
_MAX_WINDOW: int = 1024

_LimitName = Literal["total_lookback", "node_count", "max_window"]


class ExpressionCostExceeded(ValueError):
    """Raised by evaluate() pre-flight gate when a cost limit is exceeded."""

    def __init__(
        self,
        *,
        node_id: str,
        limit_name: _LimitName,
        limit_value: int,
        observed_value: int,
    ) -> None:
        self.node_id = node_id
        self.limit_name = limit_name
        self.limit_value = limit_value
        self.observed_value = observed_value
        super().__init__(
            f"Expression cost exceeded: {limit_name}={observed_value} "
            f"> {limit_value} at node_id={node_id!r}"
        )


def _node_window(node: ExpressionNode) -> int:
    """Return the single-node window param for cost tracking.

    Probes `params["n"]`, `params["span"]`, `params["halflife"]` in order and
    returns the first numeric value found, coerced to int. Non-numeric or
    missing params yield 0 (no window contribution).
    """
    for key in ("n", "span", "halflife"):
        if key in node.params:
            try:
                return int(float(node.params[key]))
            except (TypeError, ValueError):
                return 0
    return 0


def _check_cost(root: ExpressionNode) -> None:
    """Pre-flight gate: walk the tree once and raise on first BRIDGE-01 violation.

    Tracks three cost metrics in one pass:
      * node_count — raised when > _MAX_NODE_COUNT; emits the deepest visited node.
      * max_window — raised when any single node's window > _MAX_WINDOW; emits that node.
      * total_lookback — raised when any subtree's cumulative lookback > _MAX_TOTAL_LOOKBACK;
        emits the deepest subtree root at which the budget blew.

    Lookback semantics match `compute_lookback` exactly (additive via `_LOOKBACK` table).
    """
    state: dict[str, int] = {"count": 0}

    def _walk(node: ExpressionNode) -> int:
        # node_count: pre-order increment; deepest-visited node on overflow.
        state["count"] += 1
        if state["count"] > _MAX_NODE_COUNT:
            raise ExpressionCostExceeded(
                node_id=str(node.node_id),
                limit_name="node_count",
                limit_value=_MAX_NODE_COUNT,
                observed_value=state["count"],
            )

        # max_window: single-node check, this node is the offender.
        own_window = _node_window(node)
        if own_window > _MAX_WINDOW:
            raise ExpressionCostExceeded(
                node_id=str(node.node_id),
                limit_name="max_window",
                limit_value=_MAX_WINDOW,
                observed_value=own_window,
            )

        # Recurse children first (bottom-up aggregation), summing subtree lookbacks.
        children_lookback = 0
        for child in node.children:
            children_lookback += _walk(child)

        # total_lookback: subtree aggregate — this subtree root is the offender when
        # cumulative additive lookback first breaches the budget.
        own_lookback_fn = _LOOKBACK.get(node.op)
        own_lookback = own_lookback_fn(node) if own_lookback_fn else 0
        subtree_lookback = children_lookback + own_lookback
        if subtree_lookback > _MAX_TOTAL_LOOKBACK:
            raise ExpressionCostExceeded(
                node_id=str(node.node_id),
                limit_name="total_lookback",
                limit_value=_MAX_TOTAL_LOOKBACK,
                observed_value=subtree_lookback,
            )
        return subtree_lookback

    _walk(root)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(node: ExpressionNode, df: pd.DataFrame) -> pd.Series:
    """
    Evaluate an ExpressionNode tree bottom-up against a DataFrame.

    Recursively evaluates each child before evaluating the current node.
    Returns a pd.Series aligned to df.index.

    Args:
        node: The expression node (root of a subtree) to evaluate.
        df:   DataFrame containing primitive columns keyed by OperatorTag value.

    Returns:
        pd.Series with the same index as df.

    Raises:
        ExpressionCostExceeded: If the expression tree exceeds any BRIDGE-01
            cost limit (total_lookback <= 2000, node_count <= 256,
            max_window <= 1024).
        NotImplementedError: For regime operators (regime_gate, regime_blend,
            regime_switch) which are stubbed in v1.
        NotImplementedError: For any operator not registered in the dispatch table.
    """
    _check_cost(node)
    return _evaluate_inner(node, df)


def _evaluate_inner(node: ExpressionNode, df: pd.DataFrame) -> pd.Series:
    child_results: list[pd.Series] = [_evaluate_inner(c, df) for c in node.children]
    handler = _DISPATCH.get(node.op)
    if handler is None:
        raise NotImplementedError(f"No handler registered for operator: {node.op.value!r}")
    return handler(node, child_results, df)


def compute_lookback(node: ExpressionNode) -> int:
    """
    Compute the total lookback (warmup bars) needed for an ExpressionNode tree.

    Uses additive combination: for roll_mean(lag(close, 5), 14), returns 5 + 13 = 18.
    This is the conservative choice — guarantees the outer window always has n
    fully-valid inner values, ensuring Phase 2 availability never under-counts
    warmup bars.

    Args:
        node: The expression node to compute lookback for.

    Returns:
        Total number of warmup bars required before the first valid output.
    """
    children_lookback = sum(compute_lookback(child) for child in node.children)
    own_lookback_fn = _LOOKBACK.get(node.op)
    own = own_lookback_fn(node) if own_lookback_fn else 0
    return children_lookback + own


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _linear_autocorr(arr: np.ndarray, lag_: int = 1) -> float:
    """Compute autocorrelation at lag_ for a window array."""
    if len(arr) <= lag_:
        return np.nan
    return float(pd.Series(arr).autocorr(lag=lag_))


def _ewm_kwargs(params: dict) -> dict:
    """
    Map bridge params to pandas ewm kwargs.

    Checks span, alpha, halflife in order. Exactly one must be present.
    """
    if "span" in params:
        return {"span": int(params["span"])}
    if "alpha" in params:
        return {"alpha": float(params["alpha"])}
    if "halflife" in params:
        return {"halflife": float(params["halflife"])}
    raise ValueError("ewm requires one of: span, alpha, halflife in params")


def _linear_decay(arr: np.ndarray) -> float:
    """Linear decay weights [1, 2, ..., n] normalized to sum=1."""
    w = np.arange(1, len(arr) + 1, dtype=float)
    w /= w.sum()
    return float(np.dot(arr, w))


# ---------------------------------------------------------------------------
# Primitive handlers — return df column or NaN series if column absent
# ---------------------------------------------------------------------------


def _make_primitive_handler(op: OperatorTag) -> _Handler:
    col = op.value

    def _handler(node: ExpressionNode, children: list[pd.Series], df: pd.DataFrame) -> pd.Series:
        if col in df.columns:
            result: pd.Series = df[col].copy()
            return result
        return pd.Series(np.nan, index=df.index, dtype=float)

    _handler.__name__ = f"_primitive_{col}"
    return _handler


for _op in _PRIMITIVE_OPS:
    _DISPATCH[_op] = _make_primitive_handler(_op)


# ---------------------------------------------------------------------------
# Unary arithmetic handlers
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.neg] = lambda node, c, df: -c[0]
_DISPATCH[OperatorTag.abs] = lambda node, c, df: c[0].abs()
_DISPATCH[OperatorTag.sign] = lambda node, c, df: pd.Series(np.sign(c[0]), index=c[0].index)
_DISPATCH[OperatorTag.log1p] = lambda node, c, df: pd.Series(np.log1p(c[0]), index=c[0].index)
_DISPATCH[OperatorTag.sqrt] = lambda node, c, df: pd.Series(np.sqrt(c[0]), index=c[0].index)
_DISPATCH[OperatorTag.tanh] = lambda node, c, df: pd.Series(np.tanh(c[0]), index=c[0].index)
_DISPATCH[OperatorTag.sigmoid] = lambda node, c, df: pd.Series(
    1.0 / (1.0 + np.exp(-c[0])), index=c[0].index
)
_DISPATCH[OperatorTag.reciprocal] = lambda node, c, df: 1.0 / c[0]
_DISPATCH[OperatorTag.square] = lambda node, c, df: c[0] ** 2


# ---------------------------------------------------------------------------
# Binary arithmetic handlers
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.add] = lambda node, c, df: c[0] + c[1]
_DISPATCH[OperatorTag.sub] = lambda node, c, df: c[0] - c[1]
_DISPATCH[OperatorTag.mul] = lambda node, c, df: c[0] * c[1]
_DISPATCH[OperatorTag.div] = lambda node, c, df: c[0] / c[1]
_DISPATCH[OperatorTag.pow] = lambda node, c, df: c[0] ** c[1]


def _clip_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    return c[0].clip(lower=node.params["lo"], upper=node.params["hi"])


_DISPATCH[OperatorTag.clip] = _clip_handler


# ---------------------------------------------------------------------------
# Single-series temporal handlers (D-01: all use params["n"])
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.lag] = lambda node, c, df: c[0].shift(int(node.params["n"]))
_DISPATCH[OperatorTag.diff] = lambda node, c, df: c[0].diff(int(node.params["n"]))
_DISPATCH[OperatorTag.pct_change] = lambda node, c, df: c[0].pct_change(int(node.params["n"]))

_DISPATCH[OperatorTag.roll_mean] = lambda node, c, df: c[0].rolling(int(node.params["n"])).mean()
_DISPATCH[OperatorTag.roll_std] = lambda node, c, df: c[0].rolling(int(node.params["n"])).std()
_DISPATCH[OperatorTag.roll_var] = lambda node, c, df: c[0].rolling(int(node.params["n"])).var()
_DISPATCH[OperatorTag.roll_skew] = lambda node, c, df: c[0].rolling(int(node.params["n"])).skew()
_DISPATCH[OperatorTag.roll_kurt] = lambda node, c, df: c[0].rolling(int(node.params["n"])).kurt()
_DISPATCH[OperatorTag.roll_min] = lambda node, c, df: c[0].rolling(int(node.params["n"])).min()
_DISPATCH[OperatorTag.roll_max] = lambda node, c, df: c[0].rolling(int(node.params["n"])).max()
_DISPATCH[OperatorTag.roll_sum] = lambda node, c, df: c[0].rolling(int(node.params["n"])).sum()
_DISPATCH[OperatorTag.roll_median] = (
    lambda node, c, df: c[0].rolling(int(node.params["n"])).median()
)
_DISPATCH[OperatorTag.roll_rank] = (
    lambda node, c, df: c[0].rolling(int(node.params["n"])).rank(pct=True)
)


def _roll_zscore_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    n = int(node.params["n"])
    s = c[0]
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


_DISPATCH[OperatorTag.roll_zscore] = _roll_zscore_handler


def _roll_autocorr_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    # D-02: roll_autocorr uses both params["n"] (window) and params["lag"] (lag offset)
    n = int(node.params["n"])
    lag_ = int(node.params["lag"])
    s = c[0]
    return s.rolling(n).apply(_linear_autocorr, raw=True, kwargs={"lag_": lag_})


_DISPATCH[OperatorTag.roll_autocorr] = _roll_autocorr_handler


# ---------------------------------------------------------------------------
# Two-series temporal handlers (D-01: all use params["n"])
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.roll_corr] = (
    lambda node, c, df: c[0].rolling(int(node.params["n"])).corr(c[1])
)
_DISPATCH[OperatorTag.roll_cov] = lambda node, c, df: c[0].rolling(int(node.params["n"])).cov(c[1])


def _roll_beta_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    n = int(node.params["n"])
    return c[0].rolling(n).cov(c[1]) / c[1].rolling(n).var()


_DISPATCH[OperatorTag.roll_beta] = _roll_beta_handler


# ---------------------------------------------------------------------------
# Normalization handlers (full-series in Phase 1; rolling if "n" in params)
# ---------------------------------------------------------------------------


def _zscore_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    s = c[0]
    if "n" in node.params:
        n = int(node.params["n"])
        return (s - s.rolling(n).mean()) / s.rolling(n).std()
    return (s - s.mean()) / s.std()


def _rank_norm_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    return c[0].rank(pct=True)


def _winsorize_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    s = c[0]
    q_lo = float(node.params.get("q_lo", 0.05))
    q_hi = float(node.params.get("q_hi", 0.95))
    return s.clip(lower=s.quantile(q_lo), upper=s.quantile(q_hi))


def _minmax_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    s = c[0]
    s_min = s.min()
    s_max = s.max()
    denom = s_max - s_min
    if denom == 0:
        return pd.Series(0.0, index=s.index)
    result: pd.Series = (s - s_min) / denom
    return result


def _demean_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    s = c[0]
    return s - s.mean()


def _standardize_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    # Alias of zscore
    s = c[0]
    if "n" in node.params:
        n = int(node.params["n"])
        return (s - s.rolling(n).mean()) / s.rolling(n).std()
    return (s - s.mean()) / s.std()


_DISPATCH[OperatorTag.zscore] = _zscore_handler
_DISPATCH[OperatorTag.rank_norm] = _rank_norm_handler
_DISPATCH[OperatorTag.winsorize] = _winsorize_handler
_DISPATCH[OperatorTag.minmax] = _minmax_handler
_DISPATCH[OperatorTag.demean] = _demean_handler
_DISPATCH[OperatorTag.standardize] = _standardize_handler


# ---------------------------------------------------------------------------
# Decay handlers
# ---------------------------------------------------------------------------


def _ewm_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    return c[0].ewm(**_ewm_kwargs(node.params)).mean()


def _ewm_std_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    return c[0].ewm(**_ewm_kwargs(node.params)).std()


def _decay_linear_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    n = int(node.params["n"])
    return c[0].rolling(n).apply(_linear_decay, raw=True)


_DISPATCH[OperatorTag.ewm] = _ewm_handler
_DISPATCH[OperatorTag.ewm_std] = _ewm_std_handler
_DISPATCH[OperatorTag.decay_linear] = _decay_linear_handler


# ---------------------------------------------------------------------------
# Binary comparison handlers — return float (1.0/0.0) for downstream composition
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.gt] = lambda node, c, df: (c[0] > c[1]).astype(float)
_DISPATCH[OperatorTag.lt] = lambda node, c, df: (c[0] < c[1]).astype(float)
_DISPATCH[OperatorTag.gte] = lambda node, c, df: (c[0] >= c[1]).astype(float)
_DISPATCH[OperatorTag.lte] = lambda node, c, df: (c[0] <= c[1]).astype(float)
_DISPATCH[OperatorTag.eq] = lambda node, c, df: (c[0] == c[1]).astype(float)
_DISPATCH[OperatorTag.crossover] = lambda node, c, df: (
    (c[0] > c[1]) & (c[0].shift(1) <= c[1].shift(1))
).astype(float)
_DISPATCH[OperatorTag.crossunder] = lambda node, c, df: (
    (c[0] < c[1]) & (c[0].shift(1) >= c[1].shift(1))
).astype(float)


# ---------------------------------------------------------------------------
# Unary comparison handlers
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.above_zero] = lambda node, c, df: (c[0] > 0).astype(float)
_DISPATCH[OperatorTag.below_zero] = lambda node, c, df: (c[0] < 0).astype(float)
_DISPATCH[OperatorTag.not_] = lambda node, c, df: (~c[0].astype(bool)).astype(float)


# ---------------------------------------------------------------------------
# Logical handlers
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.and_] = lambda node, c, df: (c[0].astype(bool) & c[1].astype(bool)).astype(
    float
)
_DISPATCH[OperatorTag.or_] = lambda node, c, df: (c[0].astype(bool) | c[1].astype(bool)).astype(
    float
)


# ---------------------------------------------------------------------------
# Composition handlers
# ---------------------------------------------------------------------------

_DISPATCH[OperatorTag.to_signal] = lambda node, c, df: pd.Series(np.tanh(c[0]), index=c[0].index)
_DISPATCH[OperatorTag.scale] = lambda node, c, df: c[0] * float(node.params.get("factor", 1.0))


def _combine_handler(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
    if "w" in node.params:
        w = float(node.params["w"])
        return c[0] * w + c[1] * (1.0 - w)
    return (c[0] + c[1]) / 2.0


_DISPATCH[OperatorTag.combine] = _combine_handler


# ---------------------------------------------------------------------------
# Regime stubs — NotImplementedError in v1
# ---------------------------------------------------------------------------


def _make_regime_stub(op: OperatorTag) -> _Handler:
    def _stub(node: ExpressionNode, c: list[pd.Series], df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(
            f"{node.op.value!r} is not implemented in v1 — regime operators are stubbed"
        )

    _stub.__name__ = f"_regime_stub_{op.value}"
    return _stub


for _op in (OperatorTag.regime_gate, OperatorTag.regime_blend, OperatorTag.regime_switch):
    _DISPATCH[_op] = _make_regime_stub(_op)


# ---------------------------------------------------------------------------
# Lookback rules — additive bottom-up (conservative for Phase 2)
# ---------------------------------------------------------------------------


def _lookback_n(node: ExpressionNode) -> int:
    """Lookback = n (for lag, diff, pct_change)."""
    return int(node.params["n"])


def _lookback_n_minus_1(node: ExpressionNode) -> int:
    """Lookback = n - 1 (for rolling window operators)."""
    return int(node.params["n"]) - 1


def _lookback_ewm(node: ExpressionNode) -> int:
    """Lookback = span - 1 (for ewm operators)."""
    return int(node.params.get("span", 1)) - 1


def _lookback_norm(node: ExpressionNode) -> int:
    """Lookback = n - 1 if windowed, else 0 (for normalization operators)."""
    return int(node.params["n"]) - 1 if "n" in node.params else 0


def _lookback_one(_node: ExpressionNode) -> int:
    """Lookback = 1 (for crossover/crossunder)."""
    return 1


# lag, diff, pct_change: own lookback = n
for _op in (OperatorTag.lag, OperatorTag.diff, OperatorTag.pct_change):
    _LOOKBACK[_op] = _lookback_n

# All single-series rolling ops: own lookback = n - 1
for _op in (
    OperatorTag.roll_mean,
    OperatorTag.roll_std,
    OperatorTag.roll_var,
    OperatorTag.roll_skew,
    OperatorTag.roll_kurt,
    OperatorTag.roll_min,
    OperatorTag.roll_max,
    OperatorTag.roll_sum,
    OperatorTag.roll_median,
    OperatorTag.roll_rank,
    OperatorTag.roll_zscore,
    OperatorTag.roll_autocorr,
):
    _LOOKBACK[_op] = _lookback_n_minus_1

# Two-series temporal: own lookback = n - 1
for _op in (OperatorTag.roll_corr, OperatorTag.roll_cov, OperatorTag.roll_beta):
    _LOOKBACK[_op] = _lookback_n_minus_1

# Decay
_LOOKBACK[OperatorTag.decay_linear] = _lookback_n_minus_1
for _op in (OperatorTag.ewm, OperatorTag.ewm_std):
    _LOOKBACK[_op] = _lookback_ewm

# Normalization (windowed if "n" in params, else full-series = 0)
for _op in (
    OperatorTag.zscore,
    OperatorTag.rank_norm,
    OperatorTag.winsorize,
    OperatorTag.minmax,
    OperatorTag.demean,
    OperatorTag.standardize,
):
    _LOOKBACK[_op] = _lookback_norm

# Crossover/crossunder: need 1 bar of history for .shift(1)
for _op in (OperatorTag.crossover, OperatorTag.crossunder):
    _LOOKBACK[_op] = _lookback_one

# All other operators: 0 (not registered = defaults to 0 in compute_lookback)
