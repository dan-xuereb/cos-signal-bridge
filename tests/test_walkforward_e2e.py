"""End-to-end test: walkforward harness against real ``_execute_backtest``.

Requires the 3.12 COS-BTE venv with NautilusTrader, ``xuer_sgl``, ``cos_bte``,
and a populated ``BTC_FORGE_ROOT`` OHLCV fixture. The entire module is
skipped otherwise — collection is safe under the 3.11 bridge venv thanks to
module-level ``pytest.importorskip`` gates.

Proves Phase 11 requirements end-to-end against the real ``cos_bte`` primitive
with NO stubbing, NO Flask server, NO governance write-back:

* WFV-01 — ``run_walk_forward`` drives N folds against the real backtest.
* WFV-02 — Harness imports ``cos_bte.runners.factor_backtest._execute_backtest``
  directly (no Flask ``api`` import).
* WFV-03 — Returned ``WalkForwardResult`` carries finite per-fold IC /
  turnover and finite aggregate fields over surviving folds.

Run instructions
----------------
Under the 3.12 COS-BTE venv with the BTC-Forge OHLCV fixture populated::

    export BTC_FORGE_ROOT=/data/ohlcv   # or /tmp/cos-test/ohlcv for a local fixture
    cd /home/btc/github/cos-signal-bridge
    PYTHONPATH=src:../COS-BTE/src \\
        /home/btc/github/COS-BTE/.venv/bin/python -m pytest \\
        tests/test_walkforward_e2e.py -v

Under the 3.11 bridge venv, the module is skipped cleanly — no collection
errors, no false failures::

    cd /home/btc/github/cos-signal-bridge
    .venv/bin/python3.11 -m pytest tests/test_walkforward_e2e.py -v  # SKIPPED
"""

from __future__ import annotations

import math
import os
import socket
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

# Hard-skip at collection time if heavy deps are missing (3.11 bridge venv path).
pytest.importorskip("nautilus_trader")
pytest.importorskip("cos_bte")
pytest.importorskip("xuer_sgl")

# Skip if the OHLCV fixture root isn't present (BTE venv with no data mount).
_OHLCV_ROOT = Path(os.environ.get("BTC_FORGE_ROOT", "/data/ohlcv"))
pytestmark = pytest.mark.skipif(
    not _OHLCV_ROOT.exists(),
    reason=f"BTC_FORGE_ROOT={_OHLCV_ROOT} missing — skip walkforward E2E",
)

from sdl.models.config import NormalizationConfig, SglIntegration  # noqa: E402
from sdl.models.factor import FactorRecord  # noqa: E402
from sdl.models.ir import ExpressionNode  # noqa: E402
from sdl.types import (  # noqa: E402
    DATA_SOURCE_BTC_FORGE,
    DiscoveryMethod,
    NormMethod,
    OperatorTag,
    TypeTag,
)
from xuer_sgl.loaders.btc_forge import BTCForgeLoader  # noqa: E402

from signal_bridge.walkforward import (  # noqa: E402
    WalkForwardResult,
    generate_factor_folds,
    run_walk_forward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ohlcv_dataframe() -> pd.DataFrame:
    """Load a real BTC/USD daily slice from BTC_FORGE_ROOT for end-to-end runs.

    Uses the ``xuer_sgl`` ``BTCForgeLoader`` — the same code path ``_execute_backtest``
    itself hits internally, so "the data fits the backtest" is guaranteed by
    construction.
    """
    loader = BTCForgeLoader(root=_OHLCV_ROOT, exchange="coinbase", granularity="1d")
    sf = loader.load("2023-01-01", "2024-04-01")
    df = sf.data.copy()
    if df.empty:
        pytest.skip(f"BTC_FORGE_ROOT={_OHLCV_ROOT} returned empty DataFrame")
    return df


@pytest.fixture
def close_factor_record() -> FactorRecord:
    """Minimal ``FactorRecord`` whose expression is the primitive ``close`` leaf.

    Keeps evaluation semantics trivial (``evaluate(expr_ir, df) == df['close']``)
    so per-fold IC/turnover math reflects the backtest data, not the expression.
    """
    close_leaf = ExpressionNode(op=OperatorTag.close, inferred_type=TypeTag.Series)
    record = FactorRecord(
        canonical_expr="close",
        expr_ir=close_leaf,
        source_expr="close",
        description="Primitive close — E2E walkforward fixture.",
        output_type=TypeTag.Series,
        input_primitives=["close"],
        data_sources=[DATA_SOURCE_BTC_FORGE],
        lookback_bars=0,
        complexity_score=close_leaf.complexity,
        discovery_method=DiscoveryMethod.hand_crafted,
        discovery_ts=datetime.now(UTC),
        author="test-suite",
    )
    record.sgl_integration = SglIntegration(
        signal_name="close_e2e",
        signal_version="v1",
        normalization=NormalizationConfig(
            method=NormMethod.passthrough,
            window=0,
            clip_lo=-1e9,
            clip_hi=1e9,
        ),
        invert=False,
    )
    return record


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_walk_forward_end_to_end(
    tmp_path: Path,
    close_factor_record: FactorRecord,
    ohlcv_dataframe: pd.DataFrame,
) -> None:
    """Full round-trip: real ``_execute_backtest`` per fold, real OHLCV data.

    Asserts WFV-01 / WFV-03 at the E2E tier:

    * ``result.n_folds == len(folds)``.
    * At least one fold succeeded (``n_folds_successful >= 1``).
    * Every successful ``FoldResult`` carries finite IC / rank IC / turnover
      / sharpe / pnl and ``error is None``.
    * Aggregate mean IC / mean turnover are finite whenever at least one
      fold survived.
    * ``result.factor_id`` round-trips the SDL factor identity.
    """
    folds = generate_factor_folds(date(2023, 1, 1), date(2024, 1, 1), test_days=90)
    assert len(folds) >= 3, f"expected >=3 folds, got {len(folds)}"

    result = run_walk_forward(
        factor_record=close_factor_record,
        folds=folds,
        data=ohlcv_dataframe,
        results_root=tmp_path,
    )

    assert isinstance(result, WalkForwardResult)
    assert result.n_folds == len(folds)
    assert result.n_folds_successful >= 1, (
        "E2E harness produced zero successful folds — every fold hit an exception. "
        f"First fold error: {result.folds[0].error!r}"
    )
    assert result.factor_id == str(close_factor_record.factor_id)

    # Aggregate fields are finite over surviving folds (WFV-03).
    assert math.isfinite(result.mean_ic)
    assert math.isfinite(result.mean_turnover)
    assert math.isfinite(result.mean_rank_ic)
    assert 0.0 <= result.hit_rate <= 1.0

    # Every successful fold has finite primary metrics.
    for fold in result.folds:
        if fold.error is None:
            assert math.isfinite(fold.ic), f"fold {fold.fold_index} ic not finite: {fold.ic}"
            assert math.isfinite(
                fold.rank_ic
            ), f"fold {fold.fold_index} rank_ic not finite: {fold.rank_ic}"
            assert math.isfinite(
                fold.turnover
            ), f"fold {fold.fold_index} turnover not finite: {fold.turnover}"
            assert fold.sharpe is not None and math.isfinite(fold.sharpe)
            assert fold.pnl is not None and math.isfinite(fold.pnl)
            assert fold.n_bars > 0


def test_no_flask_server_started(
    tmp_path: Path,
    close_factor_record: FactorRecord,
    ohlcv_dataframe: pd.DataFrame,
) -> None:
    """WFV-02 — harness drives the pure primitive, never instantiates Flask.

    Two guardrails:

    1. After the run, no module whose dotted-path is ``cos_bte.api`` or
       rooted at ``cos_bte.api.`` is present in ``sys.modules``. The harness
       routes through ``cos_bte.runners.factor_backtest`` exclusively.
    2. The Flask default BTE port (from ``BTE_PORT`` env, or 5004) remains
       bindable after the run — nothing is listening on it.
    """
    # Snapshot sys.modules before the run.
    pre_keys = set(sys.modules.keys())

    # Pick a free port and pin BTE_PORT to it; if the harness ever booted a
    # Flask listener, it would bind this port (or its own default).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        chosen_port = s.getsockname()[1]
    os.environ["BTE_PORT"] = str(chosen_port)

    folds = generate_factor_folds(date(2023, 4, 1), date(2023, 8, 1), test_days=60)
    assert len(folds) >= 1

    result = run_walk_forward(
        factor_record=close_factor_record,
        folds=folds[:1],  # one fold is enough to prove absence of Flask
        data=ohlcv_dataframe,
        results_root=tmp_path,
    )
    assert result.n_folds == 1

    # Guardrail 1: cos_bte.api never got imported as a side effect of the run.
    post_keys = set(sys.modules.keys())
    flask_bearing = {
        k
        for k in post_keys
        if k == "cos_bte.api" or k.startswith("cos_bte.api.") or k == "flask"
    }
    # flask itself is allowed to exist in sys.modules (NautilusTrader / other deps
    # may have imported it), but it must not have been brought in by *this* run
    # when it wasn't already loaded.
    newly_imported = flask_bearing - pre_keys
    assert "cos_bte.api" not in newly_imported, (
        "cos_bte.api was imported during run_walk_forward — WFV-02 violated. "
        f"newly imported: {sorted(newly_imported)}"
    )
    assert not any(k.startswith("cos_bte.api.") for k in newly_imported), (
        "A cos_bte.api.* submodule was imported during run_walk_forward. "
        f"newly imported: {sorted(newly_imported)}"
    )

    # Guardrail 2: the chosen Flask-default port is still bindable after the
    # run, proving no listener is in flight.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", chosen_port))
        except OSError as exc:  # pragma: no cover - defensive
            pytest.fail(f"Port {chosen_port} is occupied after walkforward run: {exc}")


def test_per_fold_failure_does_not_abort(
    tmp_path: Path,
    close_factor_record: FactorRecord,
    ohlcv_dataframe: pd.DataFrame,
) -> None:
    """D-14 empirical check — one bad fold, one good fold, both land in result.

    The bad fold's test window (2099) sits well beyond any OHLCV data, so
    ``_execute_backtest`` raises ``ValueError("No bars produced ...")``. The
    good fold runs normally. The harness must:

    * NOT propagate the exception to the caller.
    * Produce ``result.n_folds == 2``, ``result.n_folds_successful == 1``.
    * Populate ``FoldResult.error`` on the failing fold and leave the
      successful fold untouched.
    """
    good_folds = generate_factor_folds(date(2023, 1, 1), date(2023, 4, 1), test_days=60)
    assert len(good_folds) >= 1
    good_fold = good_folds[0]

    # Import WindowSpec lazily — it lives in the 3.12 BTE venv.
    from cos_bte.walkforward.windows import WindowSpec  # type: ignore[import-not-found]

    bad_fold = WindowSpec(
        train_start=date(2099, 1, 1),
        train_end=date(2099, 1, 1),
        test_start=date(2099, 1, 1),
        test_end=date(2099, 2, 1),
        fold_index=99,
    )

    result = run_walk_forward(
        factor_record=close_factor_record,
        folds=[good_fold, bad_fold],
        data=ohlcv_dataframe,
        results_root=tmp_path,
    )

    assert result.n_folds == 2
    assert result.n_folds_successful == 1, (
        f"expected 1 successful fold, got {result.n_folds_successful}; "
        f"errors: {[f.error for f in result.folds]}"
    )

    # Match folds by fold_index — order should be preserved but be defensive.
    by_index = {f.fold_index: f for f in result.folds}
    good = by_index[good_fold.fold_index]
    bad = by_index[99]

    assert good.error is None
    assert math.isfinite(good.ic)
    assert bad.error is not None
    assert "No bars" in bad.error or "2099" in bad.error or bad.error != ""
    assert math.isnan(bad.ic)
    assert math.isnan(bad.turnover)
