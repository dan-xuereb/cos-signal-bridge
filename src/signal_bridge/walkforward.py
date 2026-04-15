"""
Factor-centric walk-forward harness (Phase 11 D-01..D-14).

Orchestrates N OOS-only folds by calling
``cos_bte.runners.factor_backtest._execute_backtest`` once per fold and
computing realized IC / rank IC / turnover on the fold's test window.
Returns a typed ``WalkForwardResult`` with per-fold entries and aggregate
summary stats.

``cos_bte`` is imported lazily inside function bodies (D-04) so the module
is importable in the 3.11 bridge venv without NautilusTrader installed.
The harness performs NO governance write-back (D-13) — callers decide
whether to feed results into the SDL monitoring-update flow themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sdl.models.factor import FactorRecord

from signal_bridge.evaluator import evaluate
from signal_bridge.ic import compute_ic

if TYPE_CHECKING:
    from cos_bte.walkforward.windows import WindowSpec  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Result types (D-05, D-06, D-07, D-08 — single module, frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Per-fold walk-forward result (D-06).

    Carries the three primary factor-quality metrics (IC, rank IC, turnover),
    fold window metadata, raw signal stats for diagnosing dead signals, and
    Sharpe/PnL passthrough from the underlying backtest. On per-fold failure
    (D-14), ``error`` is populated with the exception message and all numeric
    metrics are NaN.
    """

    fold_index: int
    test_start: date
    test_end: date
    n_bars: int
    ic: float
    rank_ic: float
    turnover: float
    signal_mean: float
    signal_std: float
    signal_nan_count: int
    sharpe: float | None
    pnl: float | None
    error: str | None


@dataclass(frozen=True)
class WalkForwardResult:
    """Top-level walk-forward result (D-05, D-07).

    Aggregates over folds where ``error is None``. When no folds succeed,
    every aggregate field is ``math.nan`` (raising is explicitly avoided per
    D-14). ``t_stat_ic`` is ``0.0`` when N<2 or std==0.
    """

    factor_id: str
    factor_name: str
    n_folds: int
    n_folds_successful: int
    folds: tuple[FoldResult, ...]
    # Aggregates over successful folds (NaN if n_folds_successful == 0)
    mean_ic: float
    std_ic: float
    mean_rank_ic: float
    std_rank_ic: float
    mean_turnover: float
    std_turnover: float
    hit_rate: float
    min_ic: float
    max_ic: float
    min_rank_ic: float
    max_rank_ic: float
    min_turnover: float
    max_turnover: float
    t_stat_ic: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_factor_folds(
    start: date,
    end: date,
    test_days: int,
) -> list[WindowSpec]:
    """OOS-only folds per D-09. Wraps ``generate_windows`` with ``train_days=0`` (D-10).

    Lazy-imports ``cos_bte`` inside the function body (D-04) so this module
    stays importable without NautilusTrader.
    """
    from cos_bte.walkforward.windows import generate_windows

    windows: list[WindowSpec] = generate_windows(start, end, train_days=0, test_days=test_days)
    return windows


def _compute_fold_metrics(
    signal: pd.Series,
    close: pd.Series,
    fold_index: int,
    test_start: date,
    test_end: date,
) -> tuple[float, float, float, int, float, float, int]:
    """Compute per-fold (ic, rank_ic, turnover, n_bars, signal_mean, signal_std, signal_nan_count).

    Mirrors the IC / rank-IC / turnover arithmetic used in
    ``signal_bridge.feedback`` (feedback.py lines 82-106) without invoking
    that module (D-13 — no governance write-back from the harness).
    """
    start_ts = pd.Timestamp(test_start)
    end_ts = pd.Timestamp(test_end)
    idx_mask = (signal.index >= start_ts) & (signal.index < end_ts)
    sig_slice = signal[idx_mask]
    close_slice = close[idx_mask]

    # Lag the signal by 1 bar to prevent look-ahead bias (mirrors feedback.py)
    signal_lagged = sig_slice.shift(1)
    returns = close_slice.pct_change()
    valid_mask = pd.Series(True, index=sig_slice.index)

    ic = compute_ic(signal_lagged, returns, valid_mask)

    # Rank IC — Pearson correlation on pct-ranked signal/returns
    combined = pd.DataFrame({"signal": signal_lagged, "returns": returns})
    combined_valid = combined[valid_mask].dropna()
    if len(combined_valid) >= 2:
        rank_signal = combined_valid["signal"].rank(pct=True)
        rank_returns = combined_valid["returns"].rank(pct=True)
        rank_ic_val = float(rank_signal.corr(rank_returns))
        if pd.isna(rank_ic_val):
            rank_ic_val = 0.0
    else:
        rank_ic_val = 0.0

    turnover = float(sig_slice.diff().abs().mean())
    if pd.isna(turnover):
        turnover = 0.0

    n_bars = int(len(sig_slice))
    signal_nan_count = int(sig_slice.isna().sum())

    sig_nonnan = sig_slice.dropna()
    if n_bars > 0 and not sig_nonnan.empty:
        signal_mean = float(sig_slice.mean())
    else:
        signal_mean = 0.0
    if n_bars > 1 and not sig_nonnan.empty:
        std_val = sig_slice.std()
        signal_std = float(std_val) if not pd.isna(std_val) else 0.0
    else:
        signal_std = 0.0

    return ic, rank_ic_val, turnover, n_bars, signal_mean, signal_std, signal_nan_count


def _aggregate(folds: tuple[FoldResult, ...], ic_floor: float) -> dict[str, float]:
    """Compute aggregate stats over successful folds (D-07).

    Uses sample std (``ddof=1``) to match quant convention. If no folds
    succeeded, every aggregate is ``math.nan``. With exactly one successful
    fold, std values are ``0.0`` and ``t_stat_ic`` is ``0.0``.
    """
    ok = [f for f in folds if f.error is None]
    n = len(ok)

    if n == 0:
        nan = math.nan
        return {
            "mean_ic": nan,
            "std_ic": nan,
            "mean_rank_ic": nan,
            "std_rank_ic": nan,
            "mean_turnover": nan,
            "std_turnover": nan,
            "hit_rate": nan,
            "min_ic": nan,
            "max_ic": nan,
            "min_rank_ic": nan,
            "max_rank_ic": nan,
            "min_turnover": nan,
            "max_turnover": nan,
            "t_stat_ic": nan,
        }

    ics = pd.Series([f.ic for f in ok], dtype=float)
    rank_ics = pd.Series([f.rank_ic for f in ok], dtype=float)
    turnovers = pd.Series([f.turnover for f in ok], dtype=float)

    mean_ic = float(ics.mean())
    mean_rank_ic = float(rank_ics.mean())
    mean_turnover = float(turnovers.mean())

    if n >= 2:
        std_ic_val = ics.std(ddof=1)
        std_rank_ic_val = rank_ics.std(ddof=1)
        std_turnover_val = turnovers.std(ddof=1)
        std_ic = float(std_ic_val) if not pd.isna(std_ic_val) else 0.0
        std_rank_ic = float(std_rank_ic_val) if not pd.isna(std_rank_ic_val) else 0.0
        std_turnover = float(std_turnover_val) if not pd.isna(std_turnover_val) else 0.0
    else:
        std_ic = 0.0
        std_rank_ic = 0.0
        std_turnover = 0.0

    hits = sum(1 for f in ok if f.ic >= ic_floor)
    hit_rate = hits / n

    if n >= 2 and std_ic > 0:
        t_stat_ic = mean_ic / (std_ic / math.sqrt(n))
    else:
        t_stat_ic = 0.0

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "mean_turnover": mean_turnover,
        "std_turnover": std_turnover,
        "hit_rate": hit_rate,
        "min_ic": float(ics.min()),
        "max_ic": float(ics.max()),
        "min_rank_ic": float(rank_ics.min()),
        "max_rank_ic": float(rank_ics.max()),
        "min_turnover": float(turnovers.min()),
        "max_turnover": float(turnovers.max()),
        "t_stat_ic": t_stat_ic,
    }


# ---------------------------------------------------------------------------
# Main orchestrator (D-01, D-03, D-11, D-12, D-14)
# ---------------------------------------------------------------------------


def run_walk_forward(
    factor_record: FactorRecord,
    folds: list[WindowSpec],
    data: pd.DataFrame,
    *,
    ic_floor: float = 0.02,
    spread_bps: int = 10,
    starting_balance: float = 1_000_000.0,
    strategy_id: str = "ema_cross",
    strategy_params: dict[str, Any] | None = None,
    instruments_spec: list[dict[str, Any]] | None = None,
    results_root: Path | None = None,
) -> WalkForwardResult:
    """Run the factor-centric walk-forward harness (D-01).

    Evaluates the factor signal once globally (D-11) over the full input
    DataFrame, then iterates the caller-supplied folds (D-10). For each fold
    it lazily invokes ``cos_bte.runners.factor_backtest._execute_backtest``
    (D-04) over the test window and computes realized IC / rank IC /
    turnover on that window's bars. Per-fold exceptions are wrapped into
    ``FoldResult.error`` (D-14) — subsequent folds still run. Callers
    provide the OHLCV DataFrame (D-12); no BTC-Forge path lookups happen
    inside the harness.

    No governance write-back occurs (D-13). Returns a typed
    ``WalkForwardResult`` with per-fold entries plus aggregate mean/std,
    hit rate, min/max per metric, and t-stat for ``IC > 0``.

    Args:
        factor_record: SDL FactorRecord providing the expression tree.
        folds: Caller-built list of WindowSpec (use generate_factor_folds
            for contiguous OOS-only folds with train_days=0).
        data: OHLCV pandas DataFrame keyed by DatetimeIndex; must contain
            a ``close`` column and all primitive columns referenced by
            ``factor_record.expr_ir``.
        ic_floor: IC threshold for the aggregate hit rate (default 0.02,
            matching GovernancePolicy.default()).
        spread_bps: Simulated slippage passed to ``_execute_backtest``.
        starting_balance: Initial account balance per fold.
        strategy_id: BTE strategy to run (default "ema_cross").
        strategy_params: Strategy-specific params (default: empty dict).
        instruments_spec: List of instrument specs for BTE (default: a
            single daily BTC/USD Coinbase instrument).
        results_root: Optional results directory override for BTE output.

    Returns:
        WalkForwardResult with per-fold IC / rank IC / turnover plus
        aggregate summary stats.
    """
    # D-04: lazy import inside function body — keeps this module importable
    # in the 3.11 bridge venv with no cos_bte / nautilus_trader installed.
    from cos_bte.runners.factor_backtest import (  # type: ignore[import-not-found]
        _execute_backtest,
    )

    if strategy_params is None:
        strategy_params = {}
    if instruments_spec is None:
        instruments_spec = [
            {
                "source": "btc_forge",
                "exchange": "coinbase",
                "symbol": "BTC/USD",
                "granularity": "1d",
                "trade_size": "0.01",
            },
        ]

    # D-11: evaluate the factor signal once globally over the full data range.
    # Per-fold work slices the pre-computed series.
    signal = evaluate(factor_record.expr_ir, data)
    close = data["close"]

    # Best-effort factor identity/name — FactorRecord uses factor_id/human_name,
    # but upstream code/tests sometimes pass simple stand-ins. Fall back safely.
    factor_id_attr = getattr(factor_record, "factor_id", None)
    if factor_id_attr is None:
        factor_id_attr = getattr(factor_record, "id", "unknown")
    factor_id_str = str(factor_id_attr)

    factor_name = getattr(factor_record, "human_name", None) or getattr(factor_record, "name", None)
    if not factor_name:
        factor_name = factor_id_str

    results: list[FoldResult] = []
    for fold in folds:
        try:
            # D-03: per-fold backtest via _execute_backtest — NOT BTE's
            # strategy-optimization run_walk_forward.
            bt_summary = _execute_backtest(
                strategy_id=strategy_id,
                instruments_spec=instruments_spec,
                start=fold.test_start.isoformat(),
                end=fold.test_end.isoformat(),
                strategy_params=strategy_params,
                starting_balance=starting_balance,
                run_id=f"wf-{factor_id_str}-{fold.fold_index}",
                spread_bps=spread_bps,
                results_root=results_root,
            )
            ic, rank_ic, turnover, n_bars, s_mean, s_std, s_nan = _compute_fold_metrics(
                signal,
                close,
                fold.fold_index,
                fold.test_start,
                fold.test_end,
            )
            sharpe_val = bt_summary.get("sharpe_ratio", 0.0)
            pnl_val = bt_summary.get("realized_pnl", 0.0)
            results.append(
                FoldResult(
                    fold_index=fold.fold_index,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    n_bars=n_bars,
                    ic=ic,
                    rank_ic=rank_ic,
                    turnover=turnover,
                    signal_mean=s_mean,
                    signal_std=s_std,
                    signal_nan_count=s_nan,
                    sharpe=float(sharpe_val) if sharpe_val is not None else 0.0,
                    pnl=float(pnl_val) if pnl_val is not None else 0.0,
                    error=None,
                )
            )
        except Exception as exc:  # D-14: wrap per-fold failure, continue
            results.append(
                FoldResult(
                    fold_index=fold.fold_index,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    n_bars=0,
                    ic=math.nan,
                    rank_ic=math.nan,
                    turnover=math.nan,
                    signal_mean=math.nan,
                    signal_std=math.nan,
                    signal_nan_count=0,
                    sharpe=None,
                    pnl=None,
                    error=str(exc),
                )
            )

    folds_tup = tuple(results)
    n_ok = sum(1 for f in folds_tup if f.error is None)
    agg = _aggregate(folds_tup, ic_floor=ic_floor)

    return WalkForwardResult(
        factor_id=factor_id_str,
        factor_name=str(factor_name),
        n_folds=len(folds_tup),
        n_folds_successful=n_ok,
        folds=folds_tup,
        **agg,
    )
