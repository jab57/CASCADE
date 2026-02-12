"""Tests for tools/dorothea.py -- DoRothEA TF regulon queries."""

import pytest
import pandas as pd
from unittest.mock import patch

import tools.dorothea as dorothea_module
from tools.dorothea import (
    get_tf_targets,
    validate_tf_classification,
    get_dorothea_stats,
)


@pytest.fixture(autouse=True)
def reset_dorothea_cache():
    """Reset the module-level DoRothEA data cache before each test."""
    dorothea_module._dorothea_data = None
    yield
    dorothea_module._dorothea_data = None


@pytest.fixture
def patch_dorothea_data(mock_dorothea_df):
    """Patch load_dorothea_regulons to return mock data."""
    with patch.object(dorothea_module, "_dorothea_data", mock_dorothea_df):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=mock_dorothea_df):
            yield mock_dorothea_df


class TestGetTFTargets:
    def test_finds_targets(self, patch_dorothea_data):
        results = get_tf_targets("TP53")
        assert len(results) == 3
        target_genes = [r["target"] for r in results]
        assert "CDKN1A" in target_genes
        assert "BAX" in target_genes
        assert "MDM2" in target_genes

    def test_confidence_filtering(self, mock_dorothea_df):
        """When filtering for A-only, TP53 should have 2 targets (CDKN1A, BAX)."""
        dorothea_module._dorothea_data = mock_dorothea_df
        results = get_tf_targets("TP53", confidence_levels=["A"])
        assert len(results) == 2
        for r in results:
            assert r["confidence"] == "A"

    def test_case_insensitive(self, patch_dorothea_data):
        results = get_tf_targets("tp53")
        assert len(results) == 3

    def test_unknown_gene_returns_empty(self, patch_dorothea_data):
        results = get_tf_targets("NONEXISTENT_GENE")
        assert results == []

    def test_top_k_limits(self, patch_dorothea_data):
        results = get_tf_targets("TP53", top_k=1)
        assert len(results) == 1

    def test_result_structure(self, patch_dorothea_data):
        results = get_tf_targets("TP53")
        for r in results:
            assert "target" in r
            assert "mor" in r
            assert "confidence" in r

    def test_mor_values(self, patch_dorothea_data):
        results = get_tf_targets("TP53")
        mor_map = {r["target"]: r["mor"] for r in results}
        assert mor_map["MDM2"] == -1.0  # repression


class TestValidateTFClassification:
    def test_known_tf(self, patch_dorothea_data):
        # Use all levels for validation
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            result = validate_tf_classification("TP53")
        assert result["is_known_tf"] is True
        assert result["best_confidence"] == "A"
        assert result["total_targets"] == 3

    def test_unknown_gene(self, patch_dorothea_data):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            result = validate_tf_classification("FAKEGENE")
        assert result["is_known_tf"] is False
        assert result["best_confidence"] is None

    def test_case_insensitive(self, patch_dorothea_data):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            result = validate_tf_classification("myc")
        assert result["is_known_tf"] is True

    def test_targets_by_level(self, patch_dorothea_data):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            result = validate_tf_classification("TP53")
        levels = result["num_targets_by_level"]
        assert levels.get("A", 0) == 2
        assert levels.get("B", 0) == 1

    def test_evidence_summary_present(self, patch_dorothea_data):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            result = validate_tf_classification("TP53")
        assert "evidence_summary" in result
        assert "TP53" in result["evidence_summary"]


class TestDorotheaStats:
    def test_stats_structure(self, patch_dorothea_data):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            stats = get_dorothea_stats()
        assert "total_interactions" in stats
        assert "unique_tfs" in stats
        assert "unique_targets" in stats
        assert "interactions_by_level" in stats

    def test_stats_values(self, patch_dorothea_data):
        with patch("tools.dorothea.load_dorothea_regulons", return_value=patch_dorothea_data):
            stats = get_dorothea_stats()
        assert stats["total_interactions"] == 6
        assert stats["unique_tfs"] == 3  # TP53, MYC, STAT3


class TestDorotheaDataLoading:
    def test_import_error_handled(self):
        """If decoupler not installed, get_tf_targets returns error dict."""
        with patch.dict("sys.modules", {"decoupler": None}):
            # Force reimport failure
            dorothea_module._dorothea_data = None
            with patch("tools.dorothea.load_dorothea_regulons",
                        side_effect=ImportError("decoupler package not installed")):
                results = get_tf_targets("TP53")
                assert len(results) == 1
                assert "error" in results[0]

    def test_cache_reuse(self, mock_dorothea_df):
        """Once loaded, data is served from cache."""
        dorothea_module._dorothea_data = mock_dorothea_df
        # Should return from cache without calling decoupler
        df = dorothea_module.load_dorothea_regulons(levels=["A", "B", "C"])
        assert len(df) > 0
