"""Tests for tools/super_enhancers.py — super-enhancer lookup and batch screening."""

import pytest
import pandas as pd
from unittest.mock import patch

import tools.super_enhancers as se_module
from tools.super_enhancers import (
    has_super_enhancer,
    get_super_enhancer_info,
    check_genes_for_super_enhancers,
    get_super_enhancer_stats,
)


@pytest.fixture(autouse=True)
def reset_se_cache():
    """Reset the module-level SE data cache before each test."""
    se_module._se_data = None
    se_module._se_genes = None
    yield
    se_module._se_data = None
    se_module._se_genes = None


@pytest.fixture
def patch_se_data(mock_se_df):
    """Patch super-enhancer data to use mock data."""
    with patch.object(se_module, "_se_data", mock_se_df):
        with patch.object(se_module, "_se_genes", None):
            with patch("tools.super_enhancers.load_super_enhancer_data", return_value=mock_se_df):
                yield mock_se_df


class TestHasSuperEnhancer:
    def test_gene_with_se(self, patch_se_data):
        assert has_super_enhancer("MYC") is True

    def test_gene_without_se(self, patch_se_data):
        assert has_super_enhancer("TP53") is False

    def test_case_insensitive(self, patch_se_data):
        assert has_super_enhancer("myc") is True


class TestGetSuperEnhancerInfo:
    def test_gene_with_se(self, patch_se_data):
        info = get_super_enhancer_info("MYC")
        assert info["has_super_enhancer"] is True
        assert info["brd4_sensitive"] is True
        assert info["cell_type_count"] == 2
        assert "K562" in info["cell_types"]
        assert "HepG2" in info["cell_types"]
        assert "therapeutic_implication" in info

    def test_gene_without_se(self, patch_se_data):
        info = get_super_enhancer_info("TP53")
        assert info["has_super_enhancer"] is False
        assert info["brd4_sensitive"] is False
        assert info["cell_types"] == []


class TestBatchScreening:
    def test_check_multiple_genes(self, patch_se_data):
        results = check_genes_for_super_enhancers(["MYC", "BCL2", "TP53"])
        assert len(results) == 3
        myc_result = next(r for r in results if r["gene"] == "MYC")
        assert myc_result["has_super_enhancer"] is True
        tp53_result = next(r for r in results if r["gene"] == "TP53")
        assert tp53_result["has_super_enhancer"] is False


class TestStats:
    def test_stats_structure(self, patch_se_data):
        stats = get_super_enhancer_stats()
        assert "total_associations" in stats
        assert "unique_genes" in stats
        assert "unique_cell_types" in stats
        assert "data_source" in stats


class TestDataLoading:
    def test_missing_file_raises(self):
        with patch.object(se_module, "SE_DATA_PATH", se_module.Path("/nonexistent/path.tsv")):
            with pytest.raises(FileNotFoundError):
                se_module.load_super_enhancer_data()
