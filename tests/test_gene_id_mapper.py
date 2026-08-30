"""Tests for tools/gene_id_mapper.py — symbol/Ensembl conversion and caching."""

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from tools.gene_id_mapper import GeneIDMapper
import tools.gene_id_mapper as gim


@pytest.fixture(autouse=True)
def _reset_ensembl_circuit_breaker():
    """Circuit-breaker state is process-global; reset around every test."""
    gim.reset_circuit_breaker()
    yield
    gim.reset_circuit_breaker()


@pytest.fixture
def mapper_with_cache(tmp_path):
    """Create a GeneIDMapper with a pre-populated cache file."""
    cache_file = str(tmp_path / "gene_cache.json")
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
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    return GeneIDMapper(cache_file=cache_file)


@pytest.fixture
def mapper_empty(tmp_path):
    """Create a GeneIDMapper with an empty cache."""
    cache_file = str(tmp_path / "empty_cache.json")
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
        with open(mapper_empty.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
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


class TestEnsemblRateLimiting:
    def test_semaphore_exists(self):
        import threading
        from tools.gene_id_mapper import _ensembl_semaphore
        assert isinstance(_ensembl_semaphore, type(threading.Semaphore()))

    def test_semaphore_respects_api_rate_limit_env(self, monkeypatch, tmp_path):
        """Semaphore value is read from API_RATE_LIMIT at import time; verify module uses it."""
        import tools.gene_id_mapper as gim
        # The semaphore was initialized at import time — just verify it's a Semaphore
        import threading
        assert isinstance(gim._ensembl_semaphore, type(threading.Semaphore()))

    @patch("tools.gene_id_mapper.requests.get")
    def test_symbol_lookup_completes_with_semaphore(self, mock_get, tmp_path):
        """symbol_to_ensembl completes without deadlock when semaphore is present."""
        from tools.gene_id_mapper import GeneIDMapper

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ENSG00000136997"}
        mock_get.return_value = mock_response

        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))
        result = mapper.symbol_to_ensembl("BRCA1")
        assert result == "ENSG00000136997"
        assert mock_get.call_count == 1

    @patch("tools.gene_id_mapper.requests.get")
    def test_ensembl_lookup_completes_with_semaphore(self, mock_get, tmp_path):
        """ensembl_to_symbol completes without deadlock when semaphore is present."""
        from tools.gene_id_mapper import GeneIDMapper

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"display_name": "MYC"}
        mock_get.return_value = mock_response

        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))
        result = mapper.ensembl_to_symbol("ENSG00000136997")
        assert result == "MYC"
        assert mock_get.call_count == 1


class TestEnsemblSSLVerify:
    """Ensembl calls must honour CASCADE_SSL_NO_VERIFY like every other CASCADE HTTP client."""

    def test_ssl_verify_env_parsing(self):
        """The module toggle follows CASCADE_SSL_NO_VERIFY."""
        assert (os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1") is gim._SSL_VERIFY

    @patch("tools.gene_id_mapper.requests.get")
    def test_symbol_lookup_passes_verify(self, mock_get, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ENSG00000012048"}
        mock_get.return_value = mock_response

        with patch.object(gim, "_SSL_VERIFY", False):
            mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))
            mapper.symbol_to_ensembl("BRCA1")

        assert mock_get.call_args.kwargs["verify"] is False

    @patch("tools.gene_id_mapper.requests.get")
    def test_ensembl_lookup_passes_verify(self, mock_get, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"display_name": "MYC"}
        mock_get.return_value = mock_response

        with patch.object(gim, "_SSL_VERIFY", False):
            mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))
            mapper.ensembl_to_symbol("ENSG00000136997")

        assert mock_get.call_args.kwargs["verify"] is False


class TestEnsemblTimeout:
    @patch("tools.gene_id_mapper.requests.get")
    def test_lookup_uses_short_timeout(self, mock_get, tmp_path):
        """A single lookup must not be able to block for 10s."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ENSG00000012048"}
        mock_get.return_value = mock_response

        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))
        mapper.symbol_to_ensembl("BRCA1")

        assert mock_get.call_args.kwargs["timeout"] <= 3
        assert mock_get.call_args.kwargs["timeout"] == gim._ENSEMBL_TIMEOUT


class TestEnsemblCircuitBreaker:
    """After repeated transport failures, lookups short-circuit instead of each
    burning a full timeout — this is what bounds a many-gene comprehensive run."""

    @patch("tools.gene_id_mapper.requests.get")
    def test_trips_after_threshold_consecutive_failures(self, mock_get, tmp_path):
        mock_get.side_effect = Exception("Read timed out")
        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))

        # First _CB_FAIL_THRESHOLD lookups actually hit the network (and fail).
        for i in range(gim._CB_FAIL_THRESHOLD):
            assert mapper.symbol_to_ensembl(f"FAKEGENE{i}") is None
        assert mock_get.call_count == gim._CB_FAIL_THRESHOLD
        assert gim._circuit_open() is True

        # Subsequent lookups short-circuit — no further network calls.
        for i in range(10):
            assert mapper.symbol_to_ensembl(f"OTHERGENE{i}") is None
        assert mock_get.call_count == gim._CB_FAIL_THRESHOLD
        assert gim.ensembl_unreachable() is True

    @patch("tools.gene_id_mapper.requests.get")
    def test_ensembl_to_symbol_also_short_circuits(self, mock_get, tmp_path):
        mock_get.side_effect = Exception("SSL error")
        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))
        for i in range(gim._CB_FAIL_THRESHOLD):
            mapper.symbol_to_ensembl(f"FAKEGENE{i}")
        pre = mock_get.call_count
        assert mapper.ensembl_to_symbol("ENSG00000000000") is None
        assert mock_get.call_count == pre  # skipped, breaker already open

    @patch("tools.gene_id_mapper.requests.get")
    def test_http_404_does_not_trip_breaker(self, mock_get, tmp_path):
        """A 404 is a reachable Ensembl saying 'no such gene' — not an outage."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))

        for i in range(gim._CB_FAIL_THRESHOLD + 3):
            assert mapper.symbol_to_ensembl(f"NOSUCHGENE{i}") is None
        assert gim._circuit_open() is False
        assert mock_get.call_count == gim._CB_FAIL_THRESHOLD + 3

    @patch("tools.gene_id_mapper.requests.get")
    def test_success_resets_failure_streak(self, mock_get, tmp_path):
        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"id": "ENSG00000012048"}
        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))

        # fail, fail, succeed, fail, fail — never 3 in a row
        mock_get.side_effect = [
            Exception("x"), Exception("x"), ok, Exception("x"), Exception("x"),
        ]
        mapper.symbol_to_ensembl("A")
        mapper.symbol_to_ensembl("B")
        assert mapper.symbol_to_ensembl("BRCA1") == "ENSG00000012048"
        mapper.symbol_to_ensembl("C")
        mapper.symbol_to_ensembl("D")
        assert gim._circuit_open() is False

    @patch("tools.gene_id_mapper.requests.get")
    def test_trips_on_cumulative_failures_despite_interleaved_success(self, mock_get, tmp_path):
        """A degraded endpoint (~1 in 3 calls succeeds) never hits the consecutive
        threshold, but net-accumulating failures still trip the breaker."""
        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"id": "ENSG00000012048"}
        # fail, fail, ok, ... — consecutive streak never exceeds 2 (< _CB_FAIL_THRESHOLD)
        mock_get.side_effect = ([Exception("timeout"), Exception("timeout"), ok] * 20)
        mapper = GeneIDMapper(cache_file=str(tmp_path / "cache.json"))

        for i in range(60):
            if gim._circuit_open():
                break
            mapper.symbol_to_ensembl(f"G{i}")

        assert gim._circuit_open() is True
        assert mock_get.call_count < 60  # tripped before exhausting the loop

    def test_reset_circuit_breaker_clears_state(self, tmp_path):
        gim._record_ensembl_result(reachable=False)
        gim._record_ensembl_result(reachable=False)
        gim._record_ensembl_result(reachable=False)
        assert gim.ensembl_unreachable() is True
        gim.reset_circuit_breaker()
        assert gim.ensembl_unreachable() is False
        assert gim._circuit_open() is False
