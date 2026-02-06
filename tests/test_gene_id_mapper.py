"""Tests for tools/gene_id_mapper.py — symbol/Ensembl conversion and caching."""

import pytest
import os
import pickle
from unittest.mock import patch, MagicMock
from pathlib import Path

from tools.gene_id_mapper import GeneIDMapper


@pytest.fixture
def mapper_with_cache(tmp_path):
    """Create a GeneIDMapper with a pre-populated cache file."""
    cache_file = str(tmp_path / "gene_cache.pkl")
    cache_data = {
        "symbol_to_ensembl": {
            "MYC": "ENSG00000136997",
            "TP53": "ENSG00000141510",
        },
        "ensembl_to_symbol": {
            "ENSG00000136997": "MYC",
            "ENSG00000141510": "TP53",
        },
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    return GeneIDMapper(cache_file=cache_file)


@pytest.fixture
def mapper_empty(tmp_path):
    """Create a GeneIDMapper with an empty cache."""
    cache_file = str(tmp_path / "empty_cache.pkl")
    return GeneIDMapper(cache_file=cache_file)


class TestSymbolToEnsembl:
    def test_cached_lookup(self, mapper_with_cache):
        assert mapper_with_cache.symbol_to_ensembl("MYC") == "ENSG00000136997"

    def test_case_insensitive(self, mapper_with_cache):
        assert mapper_with_cache.symbol_to_ensembl("myc") == "ENSG00000136997"

    def test_ensembl_id_passthrough(self, mapper_with_cache):
        """If input already looks like an Ensembl ID, return as-is."""
        result = mapper_with_cache.symbol_to_ensembl("ENSG00000136997")
        assert result == "ENSG00000136997"

    @patch("tools.gene_id_mapper.requests.get")
    def test_api_fallback(self, mock_get, mapper_empty):
        """When not in cache, query Ensembl REST API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ENSG00000012048"}
        mock_get.return_value = mock_response

        result = mapper_empty.symbol_to_ensembl("BRCA1")
        assert result == "ENSG00000012048"
        mock_get.assert_called_once()

    @patch("tools.gene_id_mapper.requests.get")
    def test_unknown_gene_returns_none(self, mock_get, mapper_empty):
        """Unknown gene should return None."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = mapper_empty.symbol_to_ensembl("FAKEGENE123")
        assert result is None

    @patch("tools.gene_id_mapper.requests.get")
    def test_api_error_returns_none(self, mock_get, mapper_empty):
        """Network error should return None gracefully."""
        mock_get.side_effect = Exception("Connection timeout")
        result = mapper_empty.symbol_to_ensembl("TP53")
        assert result is None


class TestEnsemblToSymbol:
    def test_cached_lookup(self, mapper_with_cache):
        assert mapper_with_cache.ensembl_to_symbol("ENSG00000136997") == "MYC"

    @patch("tools.gene_id_mapper.requests.get")
    def test_api_fallback(self, mock_get, mapper_empty):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"display_name": "APC"}
        mock_get.return_value = mock_response

        result = mapper_empty.ensembl_to_symbol("ENSG00000134982")
        assert result == "APC"

    @patch("tools.gene_id_mapper.requests.get")
    def test_unknown_id_returns_none(self, mock_get, mapper_empty):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = mapper_empty.ensembl_to_symbol("ENSG99999999999")
        assert result is None


class TestCachePersistence:
    @patch("tools.gene_id_mapper.requests.get")
    def test_cache_saved_after_api_lookup(self, mock_get, mapper_empty):
        """After an API lookup, the result should be cached to disk."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ENSG00000012048"}
        mock_get.return_value = mock_response

        mapper_empty.symbol_to_ensembl("BRCA1")

        # Verify cache file was written
        assert os.path.exists(mapper_empty.cache_file)
        with open(mapper_empty.cache_file, "rb") as f:
            data = pickle.load(f)
        assert "BRCA1" in data["symbol_to_ensembl"]

    def test_cache_stats(self, mapper_with_cache):
        stats = mapper_with_cache.get_cache_stats()
        assert stats["cached_symbols"] == 2
        assert stats["cached_ensembls"] == 2


class TestBatchConversion:
    def test_batch_symbol_to_ensembl(self, mapper_with_cache):
        result = mapper_with_cache.batch_symbol_to_ensembl(["MYC", "TP53"])
        assert result["MYC"] == "ENSG00000136997"
        assert result["TP53"] == "ENSG00000141510"
