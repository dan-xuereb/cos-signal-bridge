"""
End-to-end integration test: SDL -> SGL -> BTE -> SDL pipeline.

Exercises all 6 bridge modules in sequence:
  1. registry.py   — save/load FactorRecord to disk
  2. adapter.py    — factor_to_indicator_spec (ExpressionNode → IndicatorSpec)
  3. BTCForgeLoader — real BTC OHLCV data from disk
  4. provider.py   — extract_signal_dict (SignalFrame → {ts_ns: float})
  5. COS-BTE       — bars_from_signal_frame + run_backtest with SignalDrivenStrategy
  6. feedback.py   — compute_monitoring_update (BTE → SDL status transition)

Step 7 (look-ahead bias assertion): lag(close, 1) at bar T == close at bar T-1.

Requires:
  BTC_FORGE_ROOT — path to BTC-Forge OHLCV data (e.g. /data/ohlcv)

Hard fail if BTC_FORGE_ROOT is missing or data cannot be loaded. No pytest.skip.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from cos_bte.data.loaders import bars_from_signal_frame
from cos_bte.runners.backtest import RunConfig, VenueConfig, run_backtest
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import BTC, USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy
from sdl.models.config import OosMonitoringConfig
from sdl.models.search import SglFactorMonitoringUpdate
from sdl.types import FactorStatus
from xuer_sgl.loaders.btc_forge import BTCForgeLoader
from xuer_sgl.models import IndicatorSpec, NaNReport, SignalManifest, StalenessReport, TimeGrid
from xuer_sgl.signal_frame import SignalFrame
from xuer_sgl.types import BarAvailabilityState, GapMode

from signal_bridge.adapter import factor_to_indicator_spec
from signal_bridge.feedback import compute_monitoring_update
from signal_bridge.provider import extract_signal_dict
from signal_bridge.registry import load_factor, save_factor

# ---------------------------------------------------------------------------
# Strategy: SignalDrivenStrategy (per D-01, D-02, D-03)
# Long/flat only. signal > 0 → go long; signal <= 0 → close long.
# Signal dict injected at constructor time: {ts_ns: signal_value}.
# ---------------------------------------------------------------------------


class SignalDrivenStrategyConfig(StrategyConfig, frozen=True):
    """Config for SignalDrivenStrategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    signal_dict: dict[int, float]  # {ts_ns: signal_value} injected at construction


class SignalDrivenStrategy(Strategy):
    """
    Minimal signal-driven strategy for E2E backtest.

    Reads pre-computed SDL factor values from a dict injected at construction.
    Long/flat only (per D-02): signal > 0 → go long; signal <= 0 → close long.
    """

    def __init__(self, config: SignalDrivenStrategyConfig) -> None:
        super().__init__(config)
        self._signal_dict = config.signal_dict
        self._instrument_id = config.instrument_id
        self._bar_type = config.bar_type
        self._trade_size = config.trade_size

    def on_start(self) -> None:
        instrument = self.cache.instrument(self._instrument_id)
        self._instrument = instrument
        self.subscribe_bars(self._bar_type)

    def on_bar(self, bar: Bar) -> None:
        signal = self._signal_dict.get(bar.ts_event)
        if signal is None:
            return
        position = self.cache.positions(instrument_id=self._instrument_id)
        is_long = any(p.is_open and p.is_long for p in position)
        if signal > 0 and not is_long:
            # Go long
            qty = Quantity(float(self._trade_size), self._instrument.size_precision)
            order = self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=OrderSide.BUY,
                quantity=qty,
            )
            self.submit_order(order)
        elif signal <= 0 and is_long:
            # Close long position
            qty = Quantity(float(self._trade_size), self._instrument.size_precision)
            order = self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=OrderSide.SELL,
                quantity=qty,
            )
            self.submit_order(order)


# ---------------------------------------------------------------------------
# Helper: make_btcusd_instrument (verbatim from COS-BTE/examples/btc_fred_integration.py)
# ---------------------------------------------------------------------------

SIM = Venue("SIM")


def make_btcusd_instrument() -> CurrencyPair:
    """Build a BTC/USD CurrencyPair for the SIM venue."""
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol("BTC/USD"), SIM),
        raw_symbol=Symbol("BTC/USD"),
        base_currency=BTC,
        quote_currency=USD,
        price_precision=2,
        size_precision=6,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.000001"),
        lot_size=None,
        max_quantity=None,
        min_quantity=Quantity.from_str("0.000001"),
        max_notional=None,
        min_notional=Money(10.00, USD),
        max_price=None,
        min_price=Price.from_str("0.01"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.02"),
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.002"),
        ts_event=0,
        ts_init=0,
    )


# ---------------------------------------------------------------------------
# Helper: apply_indicator_spec_to_df (verbatim from tests/test_adapter.py)
# ---------------------------------------------------------------------------


def apply_indicator_spec_to_df(spec: IndicatorSpec, df: pd.DataFrame) -> SignalFrame:
    """Apply an SDL-derived IndicatorSpec to a DataFrame, producing a valid SignalFrame."""
    result_series = spec.func(df)
    col = spec.outputs[0]
    avail_values = [
        (
            BarAvailabilityState.INSUFFICIENT_DATA.value
            if i < spec.lookback
            else (
                BarAvailabilityState.MISSING_NATIVE.value
                if pd.isna(v)
                else BarAvailabilityState.VALID.value
            )
        )
        for i, v in enumerate(result_series)
    ]
    data = pd.DataFrame({col: result_series}, index=df.index, dtype=float)
    avail = pd.DataFrame({col: avail_values}, index=df.index)
    return SignalFrame(
        data=data,
        availability=avail,
        manifest=SignalManifest(columns=[col]),
        time_grid=TimeGrid(freq="1h", gap_mode=GapMode.GAPLESS, clock="UTC", index=df.index),
        nan_report=NaNReport.from_frame(data),
        staleness_report=StalenessReport(),
    )


# ---------------------------------------------------------------------------
# Main E2E test
# ---------------------------------------------------------------------------


def test_e2e_pipeline(lag1_factor_with_sgl) -> None:  # type: ignore[no-untyped-def]
    """
    Full 7-step SDL->SGL->BTE->SDL pipeline test.

    Steps:
      1. Write FactorRecord to temp registry
      2. Load and convert to IndicatorSpec
      3. Load BTC OHLCV via BTCForgeLoader (real data — hard fail if missing)
      4. Compute SignalFrame with sdl_lag_close_1_v1 column
      5. Extract signal dict and run BTE backtest with SignalDrivenStrategy
      6. Assert monitoring update written to registry with status transition
      7. Look-ahead bias assertion: lag(close,1) == close.shift(1) on common index
    """
    with tempfile.TemporaryDirectory() as _tmpdir:
        registry_dir = Path(_tmpdir) / "registry"

        # ------------------------------------------------------------------
        # Step 1: Write FactorRecord to temp registry
        # ------------------------------------------------------------------
        record = lag1_factor_with_sgl
        # Set status to active so status transition can fire in feedback
        record.status = FactorStatus.active
        record.activation_date = datetime.now(UTC)
        # High IC thresholds: lag(close,1) vs pct_change returns will have IC
        # well below 0.99, triggering a status transition
        record.oos_monitoring = OosMonitoringConfig(ic_floor=0.99, ic_invalidation=0.98)

        save_factor(record, registry_dir)

        assert (
            registry_dir / f"{record.factor_id}.json"
        ).exists(), f"Registry file not written: {record.factor_id}.json"

        # ------------------------------------------------------------------
        # Step 2: Load and convert to IndicatorSpec
        # ------------------------------------------------------------------
        loaded = load_factor(record.factor_id, registry_dir)
        spec = factor_to_indicator_spec(loaded)

        assert (
            spec.outputs[0] == "sdl_lag_close_1_v1"
        ), f"Expected column name 'sdl_lag_close_1_v1', got '{spec.outputs[0]}'"
        assert (
            spec.lookback == 1
        ), f"Expected lookback == 1 (lag=1, passthrough norm window=0), got {spec.lookback}"

        # ------------------------------------------------------------------
        # Step 3: Load BTC OHLCV via BTCForgeLoader
        # Hard fail per D-06: KeyError if BTC_FORGE_ROOT is missing
        # ------------------------------------------------------------------
        root = Path(os.environ["BTC_FORGE_ROOT"])
        loader = BTCForgeLoader(root=root, exchange="coinbase", granularity="1h")
        ohlcv_sf = loader.load("2024-01-01", "2024-03-31")
        df = ohlcv_sf.data  # the raw OHLCV DataFrame

        assert len(df) > 100, (
            f"Expected > 100 bars from BTCForgeLoader, got {len(df)}. "
            f"Check BTC_FORGE_ROOT={root}"
        )

        # ------------------------------------------------------------------
        # Step 4: Compute SignalFrame with SDL column
        # ------------------------------------------------------------------
        sf = apply_indicator_spec_to_df(spec, df)
        col = "sdl_lag_close_1_v1"

        assert col in sf.data.columns, f"Column '{col}' not found in SignalFrame"

        # First spec.lookback rows must be INSUFFICIENT_DATA
        assert (
            sf.availability[col].iloc[: spec.lookback]
            == BarAvailabilityState.INSUFFICIENT_DATA.value
        ).all(), "First spec.lookback bars must be INSUFFICIENT_DATA"

        # Remaining non-NaN rows must be VALID
        remaining_avail = sf.availability[col].iloc[spec.lookback :]
        remaining_data = sf.data[col].iloc[spec.lookback :]
        valid_rows = remaining_data.notna()
        assert (
            remaining_avail[valid_rows] == BarAvailabilityState.VALID.value
        ).all(), "Non-NaN rows after warmup must be VALID"

        # ------------------------------------------------------------------
        # Step 5: Extract signal dict and run BTE backtest
        # ------------------------------------------------------------------
        signal_dict = extract_signal_dict(sf, col)
        assert len(signal_dict) > 0, "signal_dict is empty — no VALID bars in SignalFrame"

        instrument = make_btcusd_instrument()
        bar_type = BarType.from_str("BTC/USD.SIM-1-HOUR-LAST-EXTERNAL")
        bars = bars_from_signal_frame(ohlcv_sf, instrument, bar_type)
        assert len(bars) > 0, "bars_from_signal_frame produced no bars"

        strategy = SignalDrivenStrategy(
            SignalDrivenStrategyConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=Decimal("0.001"),
                signal_dict=signal_dict,
            )
        )

        config = RunConfig(
            trader_id="E2E-TEST-001",
            log_level="WARNING",
            venue=VenueConfig(name="SIM", starting_balance=100_000.0),
        )
        engine = run_backtest(
            strategies=[strategy],
            instruments=[instrument],
            data=[bars],
            config=config,
            print_reports=False,
        )
        engine.reset()
        engine.dispose()

        # ------------------------------------------------------------------
        # Step 6: Assert monitoring update written to registry
        # compute_monitoring_update requires sf.data["close"] for IC computation.
        # Build a combined SignalFrame with both the SDL signal column and close.
        # ------------------------------------------------------------------

        # Close column availability from OHLCV sf (all VALID for BTC 24/7)
        close_data = ohlcv_sf.data[["close"]]
        close_avail = ohlcv_sf.availability[["close"]]

        # Concat signal + close columns
        combined_data = pd.concat([sf.data, close_data], axis=1)
        combined_avail = pd.concat([sf.availability, close_avail], axis=1)

        sf_with_close = SignalFrame(
            data=combined_data,
            availability=combined_avail,
            manifest=SignalManifest(columns=[col, "close"]),
            time_grid=sf.time_grid,
            nan_report=NaNReport.from_frame(combined_data),
            staleness_report=StalenessReport(),
        )

        # df.index is UTC-aware (BTCForgeLoader always returns UTC)
        window_start = df.index[0].to_pydatetime()
        window_end = df.index[-1].to_pydatetime()

        # Ensure timezone awareness
        if window_start.tzinfo is None:
            window_start = window_start.replace(tzinfo=UTC)
        if window_end.tzinfo is None:
            window_end = window_end.replace(tzinfo=UTC)

        update = compute_monitoring_update(
            sf_with_close, col, record.factor_id, window_start, window_end, registry_dir
        )

        assert isinstance(
            update, SglFactorMonitoringUpdate
        ), f"Expected SglFactorMonitoringUpdate, got {type(update)}"
        assert (
            update.signal_id == record.factor_id
        ), f"update.signal_id {update.signal_id} != record.factor_id {record.factor_id}"

        # Reload and confirm status transition fired
        # IC of lag(close,1) vs pct_change returns is near 0 — well below 0.99 floor
        updated_record = load_factor(record.factor_id, registry_dir)
        assert updated_record.status != FactorStatus.active, (
            f"Status should have transitioned from 'active' — IC floor=0.99 should trigger. "
            f"Realized IC was {update.realized_ic:.4f}"
        )

        # ------------------------------------------------------------------
        # Step 7: Look-ahead bias assertion (per D-09)
        # lag(close, 1) at bar T must equal close at bar T-1.
        #
        # Use the raw evaluator output (before normalization/clipping) to
        # compare against close.shift(1). The spec.func applies normalization
        # and clipping (clip_hi=999 in the passthrough config), which would
        # distort BTC prices at ~$42k+. The look-ahead bias is a property of
        # the ExpressionNode evaluator, not of normalization.
        # ------------------------------------------------------------------
        from signal_bridge.evaluator import evaluate

        raw_factor_vals = evaluate(record.expr_ir, df).dropna()
        expected = df["close"].shift(1).dropna()
        common_idx = raw_factor_vals.index.intersection(expected.index)

        assert (
            len(common_idx) > 0
        ), "No common index between raw_factor_vals and expected shifted close"

        pd.testing.assert_series_equal(
            raw_factor_vals.loc[common_idx],
            expected.loc[common_idx],
            check_names=False,
        )
