"""
Hand-declared operator signature registry for SDL OperatorTag (BRIDGE-03).

Provides O(1) shape/param/window metadata lookup for every OperatorTag enum
value, independent of the evaluator dispatch handler. Marimo authoring cells
consume this registry to validate operand shapes and param types at
expression-composition time — before attempting evaluation.

Relationship to evaluator.py:
  * `_DISPATCH` is the RUNTIME source of truth (80/80 operator handlers,
    including regime stubs that raise NotImplementedError).
  * `_LOOKBACK` documents WHICH operators are windowed (29/80, the rest
    default to 0 via `compute_lookback`'s `dict.get`).
  * This module is the DECLARATIVE source of truth for static contract
    metadata: arity, operand shape, param types, and window bounds.

Rationale for hand-declaration (per CONTEXT.md decision): the 80 entries are
finite and auditable by a human reviewer; metaprogramming over `_DISPATCH`
would couple the static contract to the runtime implementation shape,
defeating the purpose of a separate signature layer.

This module has ZERO runtime dependency on `signal_bridge.evaluator` —
keeping the contract surface isolated from dispatch-table evolution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from sdl.types import OperatorTag

# ---------------------------------------------------------------------------
# Window bound constants
# ---------------------------------------------------------------------------
# DO NOT import from signal_bridge.evaluator — keep this module standalone.
# The 1024 bound intentionally mirrors evaluator._MAX_WINDOW (BRIDGE-01 spec).

_SIG_MIN_WINDOW: int = 1
_SIG_MAX_WINDOW: int = 1024  # matches evaluator._MAX_WINDOW (intentional duplication)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

OperandShape = Literal["Scalar", "Series", "OHLC", "OHLCV"]
ParamType = Literal["int", "float", "bool", "str"]


# ---------------------------------------------------------------------------
# Signature model
# ---------------------------------------------------------------------------


class OperatorSignature(BaseModel):
    """Declarative metadata for a single OperatorTag.

    Fields:
        operand_arity: number of child operands the operator consumes.
        operand_shape: expected shape of each operand input.
        param_types: mapping of param name -> primitive type literal.
        min_window: lower bound on window param (None if not windowed).
        max_window: upper bound on window param (None if not windowed).
    """

    model_config = ConfigDict(frozen=True)

    operand_arity: int
    operand_shape: OperandShape
    param_types: dict[str, ParamType]
    min_window: int | None
    max_window: int | None


# ---------------------------------------------------------------------------
# Registry — hand-declared; one entry per OperatorTag enum value (80 total).
# Categories mirror evaluator.py's section banners for cross-referencing.
# ---------------------------------------------------------------------------


OPERATOR_SIGNATURES: dict[OperatorTag, OperatorSignature] = {
    # ------------------------------------------------------------------
    # § 4.1 Primitives (20) — arity 0, Series, no params, no window.
    # ------------------------------------------------------------------
    OperatorTag.open: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.high: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.low: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.close: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.vwap: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.mid: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.volume: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.dollar_volume: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.num_trades: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.nupl: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.sopr: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.mvrv: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.nvt: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.rhodl: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.puell: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.hash_ribbon: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.funding: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.oi: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.oi_delta: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.liquidations: OperatorSignature(
        operand_arity=0,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.2 Unary arithmetic (9) — arity 1, Series, no params, no window.
    # ------------------------------------------------------------------
    OperatorTag.neg: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.abs: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.sign: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.log1p: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.sqrt: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.tanh: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.sigmoid: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.reciprocal: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.square: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.2 Binary arithmetic (6)
    # add/sub/mul/div/pow: arity 2, no params. clip: arity 1 with lo/hi params.
    # ------------------------------------------------------------------
    OperatorTag.add: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.sub: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.mul: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.div: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.pow: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.clip: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"lo": "float", "hi": "float"},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.3 Single-series temporal (15) — arity 1, windowed [1, 1024].
    # lag/diff/pct_change take only params["n"]; rolling ops same; roll_autocorr
    # additionally takes params["lag"] (per evaluator._roll_autocorr_handler).
    # ------------------------------------------------------------------
    OperatorTag.lag: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.diff: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.pct_change: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_mean: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_std: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_var: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_skew: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_kurt: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_min: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_max: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_sum: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_median: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_rank: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_zscore: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_autocorr: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int", "lag": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    # ------------------------------------------------------------------
    # § 4.3 Two-series temporal (3) — arity 2, windowed [1, 1024].
    # ------------------------------------------------------------------
    OperatorTag.roll_corr: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_cov: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.roll_beta: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    # ------------------------------------------------------------------
    # § 4.4 Normalization (6) — arity 1.
    # zscore / standardize: handler treats "n" as optional (full-series if
    # absent); declare "n" as the canonical parameter with windowed bounds.
    # rank_norm / minmax / demean: full-series only (no window).
    # winsorize: q_lo/q_hi quantile clips (no window).
    # ------------------------------------------------------------------
    OperatorTag.zscore: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.rank_norm: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.winsorize: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"q_lo": "float", "q_hi": "float"},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.minmax: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.demean: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.standardize: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    # ------------------------------------------------------------------
    # § 4.5 Decay (3) — arity 1, windowed [1, 1024].
    # ewm / ewm_std: handler accepts any ONE of span/alpha/halflife via
    # `_ewm_kwargs`; all three are declared as accepted param keys.
    # decay_linear: linear-weighted rolling via params["n"].
    # ------------------------------------------------------------------
    OperatorTag.ewm: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"span": "int", "alpha": "float", "halflife": "float"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.ewm_std: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"span": "int", "alpha": "float", "halflife": "float"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    OperatorTag.decay_linear: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"n": "int"},
        min_window=_SIG_MIN_WINDOW,
        max_window=_SIG_MAX_WINDOW,
    ),
    # ------------------------------------------------------------------
    # § 4.6 Regime (3) — dispatch stubs (NotImplementedError) in v1; signatures
    # describe intended contract per RESEARCH.md Pitfall #3.
    # regime_gate: mask(signal, regime)     -> signal gated by regime match.
    # regime_blend: blend(signal_a, regime) -> signal weighted by regime.
    # regime_switch: switch(a, b, regime)   -> choose a vs b per regime label.
    # ------------------------------------------------------------------
    OperatorTag.regime_gate: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.regime_blend: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.regime_switch: OperatorSignature(
        operand_arity=3,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.7 Unary comparison (3) — arity 1, no params, no window.
    # ------------------------------------------------------------------
    OperatorTag.above_zero: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.below_zero: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.not_: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.7 Binary comparison (7) — arity 2, no params, no window.
    # ------------------------------------------------------------------
    OperatorTag.gt: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.lt: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.gte: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.lte: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.eq: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.crossover: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.crossunder: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.7 Logical (2) — arity 2, no params, no window.
    # Note: Python-name vs value caveat — OperatorTag.and_ has value "and";
    # OperatorTag.or_ has value "or". Keys use the enum member, not the str.
    # ------------------------------------------------------------------
    OperatorTag.and_: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.or_: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    # ------------------------------------------------------------------
    # § 4.8 Composition (3)
    # to_signal: unary tanh wrapper.
    # scale: arity 1 with `factor` float multiplier.
    # combine: arity 2 with optional `w` weight (defaults to 0.5 in handler).
    # ------------------------------------------------------------------
    OperatorTag.to_signal: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.scale: OperatorSignature(
        operand_arity=1,
        operand_shape="Series",
        param_types={"factor": "float"},
        min_window=None,
        max_window=None,
    ),
    OperatorTag.combine: OperatorSignature(
        operand_arity=2,
        operand_shape="Series",
        param_types={"w": "float"},
        min_window=None,
        max_window=None,
    ),
}


# ---------------------------------------------------------------------------
# Public re-export (BRIDGE-03)
# ---------------------------------------------------------------------------
# `OperatorSignatureRegistry` is the stable notebook-facing name. Exporting
# the raw dict (vs a Mapping-subclass accessor) keeps Marimo cell rendering
# native and readable; see RESEARCH.md Open Question #1.

OperatorSignatureRegistry = OPERATOR_SIGNATURES


__all__ = [
    "OperandShape",
    "ParamType",
    "OperatorSignature",
    "OPERATOR_SIGNATURES",
    "OperatorSignatureRegistry",
]
