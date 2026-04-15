"""Unit tests for signal_bridge.walkforward (Phase 11 — WFV-01/02/03).

These tests run in the 3.11 bridge venv WITHOUT cos_bte / NautilusTrader
installed. The lazy-imported ``cos_bte.runners.factor_backtest._execute_backtest``
and ``cos_bte.walkforward.windows`` symbols are stubbed into ``sys.modules``
so the harness orchestration can be exercised in isolation (WFV-02).

Covers:
    - Module imports cleanly without cos_bte (D-04, WFV-02).
    - FoldResult / WalkForwardResult are frozen dataclasses (D-05).
    - generate_factor_folds yields contiguous OOS-only folds (D-09, D-10).
    - run_walk_forward calls _execute_backtest once per fold (WFV-01).
    - Signal is evaluated once globally, not per fold (D-11).
    - Per-fold exceptions are wrapped into FoldResult.error; siblings survive (D-14).
    - Aggregate summary fields are populated and internally consistent (D-07, WFV-03).
    - All-failed folds yield NaN aggregates without raising (D-14).
    - t_stat_ic matches mean / (std / sqrt(N)) on known inputs.
    - WalkForwardResult carries factor identity.
    - No module-top Flask import (WFV-02 source inspection).
    - No governance-writeback symbol references in the module source (D-13).
"""

from __future__ import annotations

import importlib
import inspect
import math
import sys
import types
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sdl.models.factor import FactorRecord
from sdl.models.ir import ExpressionNode
from sdl.types import (
    DATA_SOURCE_BTC_FORGE,
    DiscoveryMethod,
    OperatorTag,
    TypeTag,
)

import signal_bridge.walkforward as wf_module
from signal_bridge.walkforward import (
    FoldResult,
    WalkForwardResult,
    _aggregate,
    generate_factor_folds,
    run_walk_forward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_execute_backtest(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install stub cos_bte.* modules in sys.modules.

    Returns the MagicMock standing in for ``_execute_backtest`` so tests can
    assert call counts, call arguments, and program custom side effects.
    Also installs a minimal ``cos_bte.walkforward.windows`` module with a
    local ``WindowSpec`` / ``generate_windows`` that matches the real BTE
    contract closely enough for the bridge harness to iterate.
    """
    mock = MagicMock(
        return_value={
            "sharpe_ratio": 1.23,
            "realized_pnl": 500.0,
            "pnl_pct": 0.5,
            "total_trades": 10,
        }
    )

    @dataclass(frozen=True)
    class StubWindowSpec:
        train_start: date
        train_end: date
        test_start: date
        test_end: date
        fold_index: int

    def stub_generate_windows(
        start: date, end: date, train_days: int, test_days: int
    ) -> list[StubWindowSpec]:
        out: list[StubWindowSpec] = []
        fold = 0
        t = start
        while True:
            tr_end = t + timedelta(days=train_days)
            te_start = tr_end
            te_end = te_start + timedelta(days=test_days)
            if te_end > end:
                break
            out.append(StubWindowSpec(t, tr_end, te_start, te_end, fold))
            fold += 1
            t = t + timedelta(days=test_days)
        return out

    stub_root = types.ModuleType("cos_bte")
    stub_runners = types.ModuleType("cos_bte.runners")
    stub_factor_bt = types.ModuleType("cos_bte.runners.factor_backtest")
    stub_factor_bt._execute_backtest = mock  # type: ignore[attr-defined]
    stub_wf = types.ModuleType("cos_bte.walkforward")
    stub_windows = types.ModuleType("cos_bte.walkforward.windows")
    stub_windows.WindowSpec = StubWindowSpec  # type: ignore[attr-defined]
    stub_windows.generate_windows = stub_generate_windows  # type: ignore[attr-defined]

    stub_wf.windows = stub_windows  # type: ignore[attr-defined]
    stub_runners.factor_backtest = stub_factor_bt  # type: ignore[attr-defined]
    stub_root.runners = stub_runners  # type: ignore[attr-defined]
    stub_root.walkforward = stub_wf  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cos_bte", stub_root)
    monkeypatch.setitem(sys.modules, "cos_bte.runners", stub_runners)
    monkeypatch.setitem(sys.modules, "cos_bte.runners.factor_backtest", stub_factor_bt)
    monkeypatch.setitem(sys.modules, "cos_bte.walkforward", stub_wf)
    monkeypatch.setitem(sys.modules, "cos_bte.walkforward.windows", stub_windows)

    return mock


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """180 daily bars starting 2024-01-01 with a trending close."""
    idx = pd.date_range("2024-01-01", periods=180, freq="D")
    close = pd.Series(np.linspace(100.0, 200.0, 180), index=idx)
    return pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": pd.Series(1000.0, index=idx),
        },
        index=idx,
    )


@pytest.fixture
def simple_factor_record() -> FactorRecord:
    """FactorRecord whose expr evaluates to the ``close`` primitive Series."""
    close_leaf = ExpressionNode(op=OperatorTag.close, inferred_type=TypeTag.Series)
    return FactorRecord(
        canonical_expr="close",
        expr_ir=close_leaf,
        source_expr="close",
        description="Primitive close-price factor for harness tests.",
        output_type=TypeTag.Series,
        input_primitives=["close"],
        data_sources=[DATA_SOURCE_BTC_FORGE],
        lookback_bars=0,
        complexity_score=close_leaf.complexity,
        discovery_method=DiscoveryMethod.hand_crafted,
        discovery_ts=datetime.now(UTC),
        author="test-suite",
        human_name="close_probe",
    )


def _make_folds(
    stub_execute_backtest: MagicMock, n: int = 3, test_days: int = 30
) -> list:
    """Build ``n`` contiguous WindowSpec folds using the stub windows module."""
    # Lazy local import — resolves against the sys.modules stub installed by
    # ``stub_execute_backtest``; mypy can't see the stub so ignore the miss.
    from cos_bte.walkforward.windows import generate_windows  # type: ignore[import-not-found]

    start = date(2024, 1, 1)
    end = start + timedelta(days=test_days * n)
    folds: list = generate_windows(start, end, train_days=0, test_days=test_days)
    assert len(folds) == n, f"expected {n} folds, got {len(folds)}"
    return folds


# ---------------------------------------------------------------------------
# Test 1: Module imports without cos_bte
# ---------------------------------------------------------------------------


def test_module_imports_without_cos_bte(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reloading signal_bridge.walkforward with no cos_bte in sys.modules must not raise."""
    for key in [
        "cos_bte",
        "cos_bte.runners",
        "cos_bte.runners.factor_backtest",
        "cos_bte.walkforward",
        "cos_bte.walkforward.windows",
    ]:
        monkeypatch.delitem(sys.modules, key, raising=False)

    # Reload the module — module-level code must not touch cos_bte (D-04).
    reloaded = importlib.reload(wf_module)
    assert hasattr(reloaded, "run_walk_forward")
    assert hasattr(reloaded, "generate_factor_folds")
    assert hasattr(reloaded, "FoldResult")
    assert hasattr(reloaded, "WalkForwardResult")


# ---------------------------------------------------------------------------
# Test 2: FoldResult is a frozen dataclass
# ---------------------------------------------------------------------------


def test_fold_result_is_frozen() -> None:
    """FoldResult rejects attribute mutation (frozen dataclass per D-05)."""
    fold = FoldResult(
        fold_index=0,
        test_start=date(2024, 1, 1),
        test_end=date(2024, 2, 1),
        n_bars=30,
        ic=0.1,
        rank_ic=0.09,
        turnover=0.5,
        signal_mean=150.0,
        signal_std=10.0,
        signal_nan_count=0,
        sharpe=1.0,
        pnl=100.0,
        error=None,
    )
    with pytest.raises(FrozenInstanceError):
        fold.ic = 0.99


def test_walk_forward_result_is_frozen() -> None:
    """WalkForwardResult rejects attribute mutation (frozen dataclass per D-05)."""
    result = WalkForwardResult(
        factor_id="fid",
        factor_name="fname",
        n_folds=0,
        n_folds_successful=0,
        folds=(),
        mean_ic=math.nan,
        std_ic=math.nan,
        mean_rank_ic=math.nan,
        std_rank_ic=math.nan,
        mean_turnover=math.nan,
        std_turnover=math.nan,
        hit_rate=math.nan,
        min_ic=math.nan,
        max_ic=math.nan,
        min_rank_ic=math.nan,
        max_rank_ic=math.nan,
        min_turnover=math.nan,
        max_turnover=math.nan,
        t_stat_ic=math.nan,
    )
    with pytest.raises(FrozenInstanceError):
        result.n_folds = 99


# ---------------------------------------------------------------------------
# Test 3: generate_factor_folds produces OOS-only contiguous folds
# ---------------------------------------------------------------------------


def test_generate_factor_folds_produces_oos_only(stub_execute_backtest: MagicMock) -> None:
    """With train_days=0, folds are OOS-only and contiguous (D-09, D-10)."""
    folds = generate_factor_folds(date(2024, 1, 1), date(2024, 4, 1), test_days=30)
    assert len(folds) == 3

    for f in folds:
        assert (f.test_end - f.test_start).days == 30
        # OOS-only: train_start == train_end == test_start (train_days=0).
        assert f.train_start == f.train_end == f.test_start

    # Contiguous test windows: each test_start picks up where the previous test_end left off.
    for prev, curr in zip(folds, folds[1:], strict=False):
        assert curr.test_start == prev.test_end


# ---------------------------------------------------------------------------
# Test 4: run_walk_forward calls _execute_backtest once per fold (WFV-01)
# ---------------------------------------------------------------------------


def test_run_walk_forward_calls_execute_backtest_once_per_fold(
    stub_execute_backtest: MagicMock,
    synthetic_data: pd.DataFrame,
    simple_factor_record: FactorRecord,
) -> None:
    """WFV-01: N folds → N _execute_backtest calls; all succeed."""
    folds = _make_folds(stub_execute_backtest, n=3, test_days=30)
    result = run_walk_forward(simple_factor_record, folds, synthetic_data)

    assert stub_execute_backtest.call_count == 3
    assert result.n_folds == 3
    assert result.n_folds_successful == 3
    assert len(result.folds) == 3
    assert all(f.error is None for f in result.folds)


# ---------------------------------------------------------------------------
# Test 5: Signal is evaluated once globally regardless of fold count (D-11)
# ---------------------------------------------------------------------------


def test_run_walk_forward_evaluates_signal_once(
    stub_execute_backtest: MagicMock,
    synthetic_data: pd.DataFrame,
    simple_factor_record: FactorRecord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-11: ``evaluate`` is called exactly once for any number of folds."""
    spy = MagicMock(wraps=wf_module.evaluate)
    monkeypatch.setattr(wf_module, "evaluate", spy)

    folds = _make_folds(stub_execute_backtest, n=4, test_days=30)
    run_walk_forward(simple_factor_record, folds, synthetic_data)

    assert spy.call_count == 1


# ---------------------------------------------------------------------------
# Test 6: Per-fold error is captured in FoldResult.error; siblings survive (D-14)
# ---------------------------------------------------------------------------


def test_per_fold_error_is_captured(
    stub_execute_backtest: MagicMock,
    synthetic_data: pd.DataFrame,
    simple_factor_record: FactorRecord,
) -> None:
    """D-14: a fold raising does not abort the run; error is recorded in FoldResult."""
    normal_return = {
        "sharpe_ratio": 1.0,
        "realized_pnl": 10.0,
        "pnl_pct": 0.01,
        "total_trades": 1,
    }
    stub_execute_backtest.side_effect = [
        normal_return,
        RuntimeError("kaboom"),
        normal_return,
    ]

    folds = _make_folds(stub_execute_backtest, n=3, test_days=30)
    result = run_walk_forward(simple_factor_record, folds, synthetic_data)

    assert result.n_folds == 3  # harness did NOT abort
    assert result.n_folds_successful == 2
    assert result.folds[1].error == "kaboom"
    assert math.isnan(result.folds[1].ic)
    assert math.isnan(result.folds[1].rank_ic)
    assert math.isnan(result.folds[1].turnover)
    # Surviving folds are intact.
    assert result.folds[0].error is None
    assert result.folds[2].error is None


# ---------------------------------------------------------------------------
# Test 7: Aggregate fields are populated and internally consistent (D-07, WFV-03)
# ---------------------------------------------------------------------------


def test_aggregate_fields_populated(
    stub_execute_backtest: MagicMock,
    synthetic_data: pd.DataFrame,
    simple_factor_record: FactorRecord,
) -> None:
    """WFV-03: aggregate fields exist, are finite, and obey min <= mean <= max + [0,1] hit_rate."""
    folds = _make_folds(stub_execute_backtest, n=3, test_days=30)
    result = run_walk_forward(simple_factor_record, folds, synthetic_data)

    for attr in (
        "mean_ic",
        "std_ic",
        "mean_rank_ic",
        "std_rank_ic",
        "mean_turnover",
        "std_turnover",
        "hit_rate",
        "min_ic",
        "max_ic",
        "min_rank_ic",
        "max_rank_ic",
        "min_turnover",
        "max_turnover",
        "t_stat_ic",
    ):
        val = getattr(result, attr)
        assert isinstance(val, float), f"{attr} should be float, got {type(val).__name__}"
        assert math.isfinite(val), f"{attr} should be finite, got {val}"

    # min <= mean <= max, with a small float-rounding tolerance for N=3 pandas reductions.
    eps = 1e-12
    assert result.min_ic - eps <= result.mean_ic <= result.max_ic + eps
    assert result.min_rank_ic - eps <= result.mean_rank_ic <= result.max_rank_ic + eps
    assert result.min_turnover - eps <= result.mean_turnover <= result.max_turnover + eps
    assert 0.0 <= result.hit_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 8: Aggregate when every fold fails — all NaN, no raise (D-14)
# ---------------------------------------------------------------------------


def test_aggregate_when_all_folds_fail(
    stub_execute_backtest: MagicMock,
    synthetic_data: pd.DataFrame,
    simple_factor_record: FactorRecord,
) -> None:
    """D-14: if every fold raises, harness returns NaN aggregates without exploding."""
    stub_execute_backtest.side_effect = RuntimeError("all broken")

    folds = _make_folds(stub_execute_backtest, n=2, test_days=30)
    result = run_walk_forward(simple_factor_record, folds, synthetic_data)

    assert result.n_folds == 2
    assert result.n_folds_successful == 0
    assert math.isnan(result.mean_ic)
    assert math.isnan(result.std_ic)
    assert math.isnan(result.hit_rate)
    assert math.isnan(result.t_stat_ic)


# ---------------------------------------------------------------------------
# Test 9: t_stat_ic matches mean / (std / sqrt(N)) on known inputs
# ---------------------------------------------------------------------------


def test_t_stat_matches_formula() -> None:
    """_aggregate's t_stat_ic equals mean_ic / (std_ic / sqrt(N)) with sample std."""
    folds = tuple(
        FoldResult(
            fold_index=i,
            test_start=date(2024, 1, 1),
            test_end=date(2024, 2, 1),
            n_bars=30,
            ic=ic,
            rank_ic=0.0,
            turnover=0.0,
            signal_mean=0.0,
            signal_std=0.0,
            signal_nan_count=0,
            sharpe=0.0,
            pnl=0.0,
            error=None,
        )
        for i, ic in enumerate([0.10, 0.20, 0.30])
    )
    agg = _aggregate(folds, ic_floor=0.02)

    ics = pd.Series([0.10, 0.20, 0.30], dtype=float)
    expected_mean = float(ics.mean())
    expected_std = float(ics.std(ddof=1))
    n = 3
    expected_t = expected_mean / (expected_std / math.sqrt(n))

    assert agg["mean_ic"] == pytest.approx(expected_mean)
    assert agg["std_ic"] == pytest.approx(expected_std)
    assert agg["t_stat_ic"] == pytest.approx(expected_t)


# ---------------------------------------------------------------------------
# Test 10: WalkForwardResult carries factor identity
# ---------------------------------------------------------------------------


def test_walkforward_result_contains_factor_identity(
    stub_execute_backtest: MagicMock,
    synthetic_data: pd.DataFrame,
    simple_factor_record: FactorRecord,
) -> None:
    """factor_id and factor_name map to FactorRecord.factor_id / .human_name."""
    folds = _make_folds(stub_execute_backtest, n=2, test_days=30)
    result = run_walk_forward(simple_factor_record, folds, synthetic_data)

    assert result.factor_id == str(simple_factor_record.factor_id)
    assert result.factor_name == simple_factor_record.human_name


# ---------------------------------------------------------------------------
# Test 11: No Flask import at module top (WFV-02 — source inspection)
# ---------------------------------------------------------------------------


def test_no_flask_import_at_module_top() -> None:
    """WFV-02: walkforward.py never imports flask at module top level."""
    src = inspect.getsource(wf_module)
    for line in src.splitlines():
        stripped = line.lstrip()
        # Module-top import lines are flush-left (no leading whitespace); indented
        # lines are inside function bodies (permitted per D-04 if it were relevant).
        if line == stripped and stripped.startswith(("from ", "import ")):
            assert "flask" not in stripped.lower(), (
                f"Flask must not be imported at module top: {line!r}"
            )


def test_no_cos_bte_import_at_module_top() -> None:
    """WFV-02 / D-04: cos_bte is lazy-imported inside function bodies only."""
    src = inspect.getsource(wf_module)
    for line in src.splitlines():
        stripped = line.lstrip()
        if line == stripped and stripped.startswith(("from ", "import ")):
            # Module-top imports must not reference cos_bte runtime modules. A
            # TYPE_CHECKING-gated block for WindowSpec is allowed (it's inside
            # ``if TYPE_CHECKING:`` so the line WILL be indented).
            assert not stripped.startswith(("from cos_bte", "import cos_bte")), (
                f"cos_bte must be lazy-imported, not module-top: {line!r}"
            )


# ---------------------------------------------------------------------------
# Test 12: No governance-writeback symbols in module source (D-13)
# ---------------------------------------------------------------------------


def test_no_governance_writeback_in_source() -> None:
    """D-13: harness never imports or calls compute_monitoring_update or save_factor."""
    src = inspect.getsource(wf_module)
    assert "compute_monitoring_update" not in src, (
        "D-13 forbids governance write-back from the harness"
    )
    assert "save_factor" not in src, "D-13 forbids registry writes from the harness"
