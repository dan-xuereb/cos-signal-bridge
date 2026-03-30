"""
File-based FactorRecord registry.

Provides atomic read/write operations for SDL FactorRecord instances stored as
JSON files on disk. Each factor is stored as `{factor_id}.json` within a
configurable registry directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import UUID

from sdl.models.factor import FactorRecord
from sdl.models.ir import ExpressionNode
from sdl.types import FactorStatus


def save_factor(record: FactorRecord, registry_dir: Path) -> None:
    """
    Persist a FactorRecord to the registry directory atomically.

    Uses tempfile.mkstemp + os.replace to ensure that a reader can never
    observe a partially-written file. On any write error the temporary file
    is cleaned up and the exception is re-raised unchanged.

    Args:
        record: The FactorRecord to persist.
        registry_dir: Directory in which JSON files are stored. Created if absent.

    Raises:
        ValueError: If any node_id UUID appears more than once in the
            ExpressionNode tree (the only structural error detectable at write time).
    """
    _validate_unique_node_ids(record.expr_ir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{record.factor_id}.json"
    data = record.model_dump_json(indent=2).encode()
    fd, tmp = tempfile.mkstemp(dir=str(registry_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_factor(factor_id: UUID, registry_dir: Path) -> FactorRecord:
    """
    Load a FactorRecord by its UUID from the registry directory.

    Args:
        factor_id: UUID of the factor to load.
        registry_dir: Directory in which JSON files are stored.

    Returns:
        The deserialized FactorRecord.

    Raises:
        FileNotFoundError: If no file for the given factor_id exists.
    """
    path = registry_dir / f"{factor_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No factor {factor_id} in {registry_dir}")
    return FactorRecord.model_validate_json(path.read_bytes())


def list_factors(
    registry_dir: Path,
    status: FactorStatus | None = None,
) -> list[FactorRecord]:
    """
    Return all FactorRecords from the registry directory, optionally filtered.

    Args:
        registry_dir: Directory in which JSON files are stored.
        status: If provided, only records with this status are returned.

    Returns:
        List of FactorRecord instances. Empty list if the directory does not exist.
    """
    if not registry_dir.exists():
        return []
    records = [
        FactorRecord.model_validate_json(p.read_bytes())
        for p in registry_dir.glob("*.json")
    ]
    if status is not None:
        records = [r for r in records if r.status == status]
    return records


def _validate_unique_node_ids(node: ExpressionNode) -> None:
    """
    Validate that all node_id UUIDs in the ExpressionNode tree are unique.

    ExpressionNode trees cannot have true cycles after JSON deserialization
    (JSON is acyclic by construction). This validates that all node_id UUIDs
    are unique — catching the only meaningful structural error possible at
    write time.

    Args:
        node: Root of the ExpressionNode tree to validate.

    Raises:
        ValueError: If any node_id UUID appears more than once in the tree.
    """
    ids: list[UUID] = []
    _collect_node_ids(node, ids)
    seen: set[UUID] = set()
    for nid in ids:
        if nid in seen:
            raise ValueError(f"Duplicate node_id {nid} in ExpressionNode tree")
        seen.add(nid)


def _collect_node_ids(node: ExpressionNode, ids: list[UUID]) -> None:
    """Recursively collect all node_id values from the tree into ids."""
    ids.append(node.node_id)
    for child in node.children:
        _collect_node_ids(child, ids)
