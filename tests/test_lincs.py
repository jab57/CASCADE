"""Tests for tools/lincs.py — LINCS L1000 knockdown data queries."""

import pytest
import pandas as pd
from unittest.mock import patch

import tools.lincs as lincs_module
from tools.lincs import (
    find_expression_regulators,
    get_knockdown_effects,
    get_lincs_stats,
)


@pytest.fixture(autouse=True)
def reset_lincs_cache():
    """Reset the module-level LINCS data cache before each test."""
    lincs_module._lincs_data = None
    yield
    lincs_module._lincs_data = None


@pytest.fixture
def patch_lincs_data(mock_lincs_df):
    """Patch load_lincs_data to return mock data."""
    with patch.object(lincs_module, "_lincs_data", mock_lincs_df):
        with patch("tools.lincs.load_lincs_data", return_value=mock_lincs_df):
            yield mock_lincs_df


class TestFindExpressionRegulators:
    def test_finds_regulators(self, patch_lincs_data):
        results = find_expression_regulators("CDKN1A")
        assert len(results) == 2
        ko_genes = [r["gene_ko"] for r in results]
        assert "TP53" in ko_genes
        assert "RB1" in ko_genes

    def test_direction_filter_down(self, patch_lincs_data):
        results = find_expression_regulators("CDKN1A", direction="down")
        for r in results:
            assert r["direction"] == "down"

    def test_direction_filter_up(self, patch_lincs_data):
        results = find_expression_regulators("MYC", direction="up")
        for r in results:
            assert r["direction"] == "up"

    def test_case_insensitive(self, patch_lincs_data):
        results = find_expression_regulators("cdkn1a")
        assert len(results) == 2

    def test_unknown_gene_returns_empty(self, patch_lincs_data):
        results = find_expression_regulators("NONEXISTENT_GENE")
        assert results == []

    def test_top_k_limits(self, patch_lincs_data):
        results = find_expression_regulators("CDKN1A", top_k=1)
        assert len(results) == 1

    def test_result_has_interpretation(self, patch_lincs_data):
        results = find_expression_regulators("CDKN1A")
        for r in results:
            assert "interpretation" in r
            assert "gene_ko" in r
            assert "effect" in r


class TestGetKnockdownEffects:
    def test_finds_effects(self, patch_lincs_data):
        results = get_knockdown_effects("TP53")
        assert len(results) == 1
        assert results[0]["gene"] == "CDKN1A"

    def test_unknown_ko_returns_empty(self, patch_lincs_data):
        results = get_knockdown_effects("FAKE_GENE")
        assert results == []

    def test_direction_filter(self, patch_lincs_data):
        results = get_knockdown_effects("BRD4", direction="down")
        assert len(results) == 1
        assert results[0]["direction"] == "down"


class TestLincsStats:
    def test_stats_structure(self, patch_lincs_data):
        stats = get_lincs_stats()
        assert "total_associations" in stats
        assert "unique_genes_measured" in stats
        assert "unique_knockdowns" in stats
        assert "upregulated_associations" in stats
        assert "downregulated_associations" in stats


class TestLincsDataLoading:
    def test_missing_file_raises(self):
        with patch.object(lincs_module, "LINCS_DATA_PATH", lincs_module.Path("/nonexistent/path.gz")):
            with pytest.raises(FileNotFoundError):
                lincs_module.load_lincs_data()
