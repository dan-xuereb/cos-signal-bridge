"""Tests for signal_bridge.ephemeral — BRIDGE-02 ephemeral FactorRecord helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sdl.models.ir import ExpressionNode
from sdl.types import DiscoveryMethod, FactorStatus, OperatorTag

import signal_bridge.ephemeral as ephemeral_mod
from signal_bridge.ephemeral import (
    EphemeralFactorMetadata,
    _build_ephemeral_factor_record,
)
from tests.conftest import make_leaf, make_unary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metadata() -> EphemeralFactorMetadata:
    return EphemeralFactorMetadata(
        name="test factor",
        signal_name="test_signal",
        version="0.1.0",
        author="pytest-suite",
    )


@pytest.fixture
def lag5_expression() -> ExpressionNode:
    leaf = make_leaf(OperatorTag.close)
    return make_unary(OperatorTag.lag, leaf, n=5)


# ---------------------------------------------------------------------------
# Pending-status construction + identity-preserving expr_ir
# ---------------------------------------------------------------------------


def test_builds_factor_record_with_pending_status(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.status == FactorStatus.pending


def test_identity_preserving_expr_ir(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.expr_ir is lag5_expression


# ---------------------------------------------------------------------------
# Derived-field correctness
# ---------------------------------------------------------------------------


def test_lookback_bars_matches_compute_lookback(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.lookback_bars == 5


def test_populated_fields_from_metadata(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.human_name == metadata.name
    assert rec.version == metadata.version
    assert rec.author == metadata.author


def test_discovery_method_is_hand_crafted(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.discovery_method == DiscoveryMethod.hand_crafted


def test_data_sources_is_notebook_singleton(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.data_sources == ["notebook"]


# ---------------------------------------------------------------------------
# SglIntegration wiring — W4 fix
# ---------------------------------------------------------------------------


def test_sgl_integration_signal_name_wired_through(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.sgl_integration.signal_name == metadata.signal_name
    assert rec.sgl_integration.signal_name == "test_signal"


def test_sgl_integration_signal_version_wired_through(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert rec.sgl_integration.signal_version == metadata.version
    assert rec.sgl_integration.signal_version == "0.1.0"


# ---------------------------------------------------------------------------
# Optional-metadata fallbacks
# ---------------------------------------------------------------------------


def test_notes_populates_source_expr_when_present(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    metadata_with_notes = metadata.model_copy(update={"notes": "custom source"})
    rec = _build_ephemeral_factor_record(lag5_expression, metadata_with_notes)
    assert rec.source_expr == "custom source"


def test_rationale_populates_description_when_present(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    metadata_with_rationale = metadata.model_copy(update={"rationale": "custom desc"})
    rec = _build_ephemeral_factor_record(lag5_expression, metadata_with_rationale)
    assert rec.description == "custom desc"


def test_default_description_when_rationale_absent(
    lag5_expression: ExpressionNode, metadata: EphemeralFactorMetadata
) -> None:
    rec = _build_ephemeral_factor_record(lag5_expression, metadata)
    assert metadata.name in rec.description


# ---------------------------------------------------------------------------
# Extra-field rejection (Pydantic v2 extra="forbid")
# ---------------------------------------------------------------------------


def test_extra_metadata_field_rejected() -> None:
    with pytest.raises(ValidationError):
        EphemeralFactorMetadata(
            name="x",
            signal_name="y",
            version="0.1.0",
            author="me",
            unexpected="z",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# GUARD-01 / GUARD-02: no registry-write references at module-source level
# ---------------------------------------------------------------------------


FORBIDDEN_SYMBOLS = (
    "save_factor",
    "FactorRegistry",
    "load_factor",
    "list_factors",
    "from signal_bridge.registry",
)


def test_ephemeral_module_has_no_registry_imports() -> None:
    """GUARD-01/02 invariant: ephemeral.py must not reference any SDL registry write path."""
    module_path = Path(ephemeral_mod.__file__)
    source = module_path.read_text(encoding="utf-8")
    for symbol in FORBIDDEN_SYMBOLS:
        assert (
            symbol not in source
        ), f"Forbidden symbol {symbol!r} found in {module_path} — violates GUARD-01/02"


def test_builder_does_not_invoke_registry_save(
    lag5_expression: ExpressionNode,
    metadata: EphemeralFactorMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime sentinel: builder must not invoke registry.save_factor."""
    import signal_bridge.registry as registry_mod

    calls: list[object] = []

    def _boom(*a: object, **kw: object) -> None:
        calls.append((a, kw))
        raise AssertionError("save_factor was called by ephemeral builder")

    monkeypatch.setattr(registry_mod, "save_factor", _boom)
    _build_ephemeral_factor_record(lag5_expression, metadata)
    assert calls == []
