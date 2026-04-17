"""BRIDGE-02 + GUARD-03: assert signal_bridge never imports flask/fastapi/uvicorn.

Guards both import-time and runtime execution of run_walk_forward.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

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

# Forbidden HTTP-framework modules that MUST NOT appear in sys.modules after
# importing signal_bridge at top level or running run_walk_forward on a stubbed
# fixture. BRIDGE-02 pins the Flask-free invariant; GUARD-03 extends it to
# FastAPI + Uvicorn (and starlette, FastAPI's ASGI backbone) at zero marginal
# cost.
FORBIDDEN_MODULES = ("flask", "fastapi", "uvicorn", "starlette")


def _assert_no_forbidden_modules() -> None:
    """Fail loud when any forbidden HTTP framework is loaded in sys.modules."""
    leaked = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    assert not leaked, f"Forbidden modules loaded: {leaked}"


@pytest.fixture(autouse=True)
def _reset_forbidden_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pop any pre-existing forbidden-module entries from sys.modules pre-test.

    Uses monkeypatch.delitem so cleanup is automatic — the sys.modules entries
    are restored to their pre-test state after each test, which matters because
    other tests in the suite (notably test_walkforward.test_module_imports_
    without_cos_bte) rely on importlib.reload() against modules we remove here.
    """
    for m in FORBIDDEN_MODULES:
        monkeypatch.delitem(sys.modules, m, raising=False)
    yield


# ---------------------------------------------------------------------------
# Import-time Flask-free invariants (BRIDGE-02 + GUARD-03)
# ---------------------------------------------------------------------------


def test_import_signal_bridge_does_not_load_flask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import signal_bridge` MUST NOT pull flask/fastapi/uvicorn/starlette."""
    for m in (
        "signal_bridge",
        "signal_bridge.walkforward",
        "signal_bridge.evaluator",
        "signal_bridge.operator_signatures",
        "signal_bridge.ephemeral",
    ):
        monkeypatch.delitem(sys.modules, m, raising=False)
    import signal_bridge  # noqa: F401

    _assert_no_forbidden_modules()


def test_import_walkforward_does_not_load_flask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import signal_bridge.walkforward` MUST NOT pull HTTP frameworks."""
    monkeypatch.delitem(sys.modules, "signal_bridge.walkforward", raising=False)
    import signal_bridge.walkforward  # noqa: F401

    _assert_no_forbidden_modules()


def test_import_evaluator_does_not_load_flask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import signal_bridge.evaluator` MUST NOT pull HTTP frameworks."""
    monkeypatch.delitem(sys.modules, "signal_bridge.evaluator", raising=False)
    import signal_bridge.evaluator  # noqa: F401

    _assert_no_forbidden_modules()


def test_import_operator_signatures_does_not_load_flask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import signal_bridge.operator_signatures` MUST NOT pull HTTP frameworks."""
    monkeypatch.delitem(sys.modules, "signal_bridge.operator_signatures", raising=False)
    import signal_bridge.operator_signatures  # noqa: F401

    _assert_no_forbidden_modules()


def test_import_ephemeral_does_not_load_flask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import signal_bridge.ephemeral` MUST NOT pull HTTP frameworks."""
    monkeypatch.delitem(sys.modules, "signal_bridge.ephemeral", raising=False)
    import signal_bridge.ephemeral  # noqa: F401

    _assert_no_forbidden_modules()


def test_cos_bte_api_not_loaded_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cos_bte.api` (the Flask-bearing module per Pitfall #4) MUST NOT load eagerly.

    A future change to cos_bte.__init__.py (currently empty) that re-exports
    from cos_bte.api would poison the import closure. This test catches that
    at the library layer even before running any code.
    """
    monkeypatch.delitem(sys.modules, "cos_bte.api", raising=False)
    monkeypatch.delitem(sys.modules, "signal_bridge", raising=False)
    import signal_bridge  # noqa: F401

    assert (
        "cos_bte.api" not in sys.modules
    ), "cos_bte.api (flask-bearing) must not be eagerly loaded by signal_bridge"


# ---------------------------------------------------------------------------
# Runtime Flask-free invariant (BRIDGE-02)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_cos_bte(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub cos_bte.* modules — same shape as test_walkforward.stub_execute_backtest.

    The stub of the LOWER-LEVEL cos_bte.walkforward.windows.generate_windows
    accepts (start, end, train_days, test_days) to mirror that layer's real
    signature. The BRIDGE-level generate_factor_folds wraps it and exposes
    only (start, end, test_days) — bridge-layer call sites must NEVER pass
    train_days.
    """
    mock = MagicMock(
        return_value={
            "sharpe_ratio": 1.0,
            "realized_pnl": 0.0,
            "pnl_pct": 0.0,
            "total_trades": 0,
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
        # Mirrors cos_bte.walkforward.windows.generate_windows (the LOWER layer).
        # The bridge-level generate_factor_folds wraps this and exposes only
        # (start, end, test_days).
        tr_end = start + timedelta(days=train_days)
        te_end = tr_end + timedelta(days=test_days)
        return [StubWindowSpec(start, tr_end, tr_end, te_end, 0)]

    stub_root = types.ModuleType("cos_bte")
    stub_runners = types.ModuleType("cos_bte.runners")
    stub_factor_bt = types.ModuleType("cos_bte.runners.factor_backtest")
    stub_factor_bt._execute_backtest = mock  # type: ignore[attr-defined]
    stub_wf = types.ModuleType("cos_bte.walkforward")
    stub_windows = types.ModuleType("cos_bte.walkforward.windows")
    stub_windows.WindowSpec = StubWindowSpec  # type: ignore[attr-defined]
    stub_windows.generate_windows = stub_generate_windows  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cos_bte", stub_root)
    monkeypatch.setitem(sys.modules, "cos_bte.runners", stub_runners)
    monkeypatch.setitem(sys.modules, "cos_bte.runners.factor_backtest", stub_factor_bt)
    monkeypatch.setitem(sys.modules, "cos_bte.walkforward", stub_wf)
    monkeypatch.setitem(sys.modules, "cos_bte.walkforward.windows", stub_windows)
    return mock


@pytest.fixture
def minimal_factor() -> FactorRecord:
    """Minimal FactorRecord wrapping lag(close, n=1) for Flask-free smoke runs."""
    leaf = ExpressionNode(op=OperatorTag.close, inferred_type=TypeTag.Series)
    lag = ExpressionNode(
        op=OperatorTag.lag,
        children=[leaf],
        params={"n": 1},
        inferred_type=TypeTag.Series,
    )
    rec = FactorRecord(
        canonical_expr="lag(close, n=1)",
        expr_ir=lag,
        source_expr="lag(close, n=1)",
        description="Flask-free smoke factor.",
        output_type=TypeTag.Series,
        input_primitives=["close"],
        data_sources=[DATA_SOURCE_BTC_FORGE],
        lookback_bars=1,
        complexity_score=lag.complexity,
        discovery_method=DiscoveryMethod.hand_crafted,
        discovery_ts=datetime.now(UTC),
        author="test-suite",
    )
    rec.sgl_integration = SglIntegration(
        signal_name="lag_1",
        signal_version="v1",
        normalization=NormalizationConfig(
            method=NormMethod.passthrough,
            window=0,
            clip_lo=-999.0,
            clip_hi=999.0,
        ),
        invert=False,
    )
    return rec


def test_run_walk_forward_does_not_load_flask(
    stub_cos_bte: MagicMock, minimal_factor: FactorRecord
) -> None:
    """Calling run_walk_forward end-to-end on a stubbed fixture MUST NOT load flask.

    This exercises the real bridge-layer generate_factor_folds signature
    `(start, end, test_days)` — NOT the lower-level cos_bte signature that
    takes train_days.
    """
    from signal_bridge.walkforward import generate_factor_folds, run_walk_forward

    data = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    folds = generate_factor_folds(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        test_days=2,
    )
    run_walk_forward(minimal_factor, folds, data)

    _assert_no_forbidden_modules()
