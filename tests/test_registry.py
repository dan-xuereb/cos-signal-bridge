"""Unit tests for signal_bridge.registry."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sdl.models.factor import FactorRecord
from sdl.models.ir import ExpressionNode
from sdl.types import DATA_SOURCE_BTC_FORGE, DiscoveryMethod, FactorStatus, OperatorTag, TypeTag

from signal_bridge.registry import list_factors, load_factor, save_factor
from tests.conftest import make_leaf, make_unary


def _make_factor(**kwargs) -> FactorRecord:
    """Build a minimal valid FactorRecord for test use."""
    close_leaf = make_leaf(OperatorTag.close)
    lag_node = make_unary(OperatorTag.lag, close_leaf, n=1)
    defaults: dict = {
        "canonical_expr": "lag(close, n=1)",
        "expr_ir": lag_node,
        "source_expr": "lag(close, n=1)",
        "description": "Lagged close by 1 bar.",
        "output_type": TypeTag.Series,
        "input_primitives": ["close"],
        "data_sources": [DATA_SOURCE_BTC_FORGE],
        "lookback_bars": 1,
        "complexity_score": lag_node.complexity,
        "discovery_method": DiscoveryMethod.hand_crafted,
        "discovery_ts": datetime.now(UTC),
        "author": "test-suite",
    }
    defaults.update(kwargs)
    return FactorRecord(**defaults)


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        """save_factor then load_factor returns an identical FactorRecord."""
        record = _make_factor()
        save_factor(record, tmp_path)
        loaded = load_factor(record.factor_id, tmp_path)
        assert loaded.factor_id == record.factor_id
        assert loaded.canonical_expr == record.canonical_expr
        assert loaded.author == record.author
        assert loaded.status == record.status
        assert loaded.expr_ir.op == record.expr_ir.op
        assert loaded.expr_ir.node_id == record.expr_ir.node_id

    def test_save_atomic_write_no_tmp_left(self, tmp_path):
        """After save_factor, no .tmp files remain in registry_dir."""
        record = _make_factor()
        save_factor(record, tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_load_nonexistent_raises(self, tmp_path):
        """load_factor raises FileNotFoundError for an unknown factor_id."""
        unknown_id = uuid4()
        with pytest.raises(FileNotFoundError):
            load_factor(unknown_id, tmp_path)


class TestListFactors:
    def test_list_factors_empty_dir(self, tmp_path):
        """list_factors on a nonexistent directory returns empty list."""
        nonexistent = tmp_path / "no_such_dir"
        result = list_factors(nonexistent)
        assert result == []

    def test_list_factors_returns_all(self, tmp_path):
        """list_factors returns all saved factors."""
        r1 = _make_factor()
        r2 = _make_factor()
        save_factor(r1, tmp_path)
        save_factor(r2, tmp_path)
        result = list_factors(tmp_path)
        ids = {r.factor_id for r in result}
        assert r1.factor_id in ids
        assert r2.factor_id in ids
        assert len(result) == 2

    def test_list_factors_filter_by_status(self, tmp_path):
        """list_factors(status=candidate) returns only candidate records."""
        candidate = _make_factor()  # default status = candidate
        approved = _make_factor(status=FactorStatus.approved)
        save_factor(candidate, tmp_path)
        save_factor(approved, tmp_path)

        candidates = list_factors(tmp_path, status=FactorStatus.candidate)
        assert len(candidates) == 1
        assert candidates[0].factor_id == candidate.factor_id

        approved_list = list_factors(tmp_path, status=FactorStatus.approved)
        assert len(approved_list) == 1
        assert approved_list[0].factor_id == approved.factor_id


class TestValidation:
    def test_save_rejects_duplicate_node_ids(self, tmp_path):
        """save_factor raises ValueError when two nodes share the same node_id UUID."""
        shared_uuid = uuid4()
        leaf_a = ExpressionNode(
            node_id=shared_uuid,
            op=OperatorTag.close,
            inferred_type=TypeTag.Series,
        )
        leaf_b = ExpressionNode(
            node_id=shared_uuid,  # intentional duplicate
            op=OperatorTag.volume,
            inferred_type=TypeTag.Series,
        )
        parent = ExpressionNode(
            op=OperatorTag.add,
            children=[leaf_a, leaf_b],
            inferred_type=TypeTag.Series,
        )
        record = _make_factor(expr_ir=parent)
        with pytest.raises(ValueError, match="Duplicate node_id"):
            save_factor(record, tmp_path)
