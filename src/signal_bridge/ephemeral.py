"""
Ephemeral FactorRecord helper — BRIDGE-02 public integration surface.

This module hosts the typed input shape (`EphemeralFactorMetadata`) and the
factory (`_build_ephemeral_factor_record`) that Phase 14/15 Marimo notebooks
use to wrap a composed `ExpressionNode` into a `FactorRecord(status=pending)`
for `run_walk_forward` consumption WITHOUT triggering SDL registry persistence.

GUARD-01 / GUARD-02 invariant: this module has no dependency on the bridge
registry module by design — the no-registry-write guarantee is enforced at the
module-source level by a CI source-grep test
(`tests/test_ephemeral.py::test_ephemeral_module_has_no_registry_imports`).
The guard scans this file for any reference to registry write/read symbols.
Do not introduce one here, even behind `TYPE_CHECKING` or as a commented
reference — the grep is strict.

The underscore prefix on `_build_ephemeral_factor_record` is intentional: it
signals "not a user-facing registry API" while remaining importable from
notebook cells.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict
from sdl.models.config import SglIntegration
from sdl.models.factor import FactorRecord
from sdl.models.ir import ExpressionNode
from sdl.types import DiscoveryMethod, FactorStatus

from signal_bridge.evaluator import compute_lookback


class EphemeralFactorMetadata(BaseModel):
    """Typed input metadata for building an ephemeral FactorRecord from a notebook cell.

    No ``dict[str, Any]`` — every field is explicitly typed per Phase 13 CONTEXT
    decision. Consumable from a Marimo ``mo.ui.form`` without additional shaping.
    ``extra="forbid"`` guarantees no silent ``dict[str, Any]`` leak.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    signal_name: str
    version: str
    author: str
    notes: str | None = None
    rationale: str | None = None


def _build_ephemeral_factor_record(
    expression: ExpressionNode,
    metadata: EphemeralFactorMetadata,
) -> FactorRecord:
    """Build an in-memory ``FactorRecord(status=pending)`` for notebook use.

    NEVER writes to the SDL registry — this module does not import registry at
    all. The resulting FactorRecord is safe to pass to ``run_walk_forward`` but
    must not be persisted through the bridge registry module in v3.1 (see
    GUARD-02).

    Field-population strategy:
    - ``expr_ir`` is identity-preserving (the exact ``ExpressionNode`` passed in
      is stored, not a copy).
    - ``canonical_expr`` is derived from ``expression.canonical_str``.
    - ``source_expr`` falls back to ``expression.canonical_str`` when
      ``metadata.notes`` is absent.
    - ``description`` falls back to a generated default naming the factor when
      ``metadata.rationale`` is absent.
    - ``data_sources = ["notebook"]`` and ``discovery_method = hand_crafted``
      document these as notebook-composed factors, not pipeline-discovered ones.
    - ``sgl_integration`` is populated with ``metadata.signal_name`` and
      ``metadata.version`` (wired through to ``signal_name`` / ``signal_version``);
      remaining SglIntegration fields keep their defaults.
    - ``status = FactorStatus.pending`` — the v3.0 Phase 9 lifecycle state for
      pre-admission records.

    Args:
        expression: The ExpressionNode DAG to wrap.
        metadata: Typed metadata supplied by the researcher.

    Returns:
        FactorRecord with ``status=FactorStatus.pending``, ``lookback_bars``
        derived from ``signal_bridge.evaluator.compute_lookback``,
        identity-preserving ``expr_ir``, and an ``sgl_integration`` carrying
        ``metadata.signal_name`` + ``metadata.version``.
    """
    return FactorRecord(
        version=metadata.version,
        human_name=metadata.name,
        canonical_expr=expression.canonical_str,
        expr_ir=expression,
        source_expr=metadata.notes or expression.canonical_str,
        description=(
            metadata.rationale or f"Ephemeral notebook factor (Phase 13): {metadata.name}"
        ),
        output_type=expression.inferred_type,
        input_primitives=[],
        data_sources=["notebook"],
        lookback_bars=compute_lookback(expression),
        complexity_score=expression.complexity,
        discovery_method=DiscoveryMethod.hand_crafted,
        discovery_ts=datetime.now(UTC),
        author=metadata.author,
        sgl_integration=SglIntegration(
            signal_name=metadata.signal_name,
            signal_version=metadata.version,
        ),
        status=FactorStatus.pending,
    )


__all__ = ["EphemeralFactorMetadata", "_build_ephemeral_factor_record"]
