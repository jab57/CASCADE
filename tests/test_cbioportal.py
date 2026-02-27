"""
Unit tests for tools/cbioportal.py — cBioPortal TCGA primary tumor data module.

All HTTP calls are mocked; no live API calls are made during testing.
"""

import pytest
import requests
from unittest.mock import patch, MagicMock

from tools.cbioportal import (
    _get_entrez_id,
    _fetch_expression_for_study,
    _fetch_alterations_for_study,
    get_gene_tumor_expression,
    get_gene_alteration_frequency,
    get_cbioportal_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_gene_resp(entrez_id=4609):
    """Return a mock requests.Response for the /genes/{symbol} endpoint."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"hugoGeneSymbol": "MYC", "entrezGeneId": entrez_id}
    mock.raise_for_status.return_value = None
    return mock


def _mock_expression_resp(values):
    """Return a mock requests.Response for the molecular-data endpoint."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = [
        {"value": v, "sampleId": f"TCGA-{i:02d}", "molecularProfileId": "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna_median_Zscores"}
        for i, v in enumerate(values)
    ]
    mock.raise_for_status.return_value = None
    return mock


def _mock_404():
    mock = MagicMock()
    mock.status_code = 404
    return mock


# ---------------------------------------------------------------------------
# _get_entrez_id
# ---------------------------------------------------------------------------

class TestGetEntrezId:
    @patch("tools.cbioportal.requests.get")
    def test_returns_entrez_id(self, mock_get):
        mock_get.return_value = _mock_gene_resp(4609)
        result = _get_entrez_id("MYC")
        assert result == 4609
        mock_get.assert_called_once()

    @patch("tools.cbioportal.requests.get")
    def test_returns_none_on_404(self, mock_get):
        mock_get.return_value = _mock_404()
        result = _get_entrez_id("NOTAREAL")
        assert result is None

    @patch("tools.cbioportal.requests.get")
    def test_returns_none_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("unreachable")
        result = _get_entrez_id("MYC")
        assert result is None


# ---------------------------------------------------------------------------
# _fetch_expression_for_study
# ---------------------------------------------------------------------------

class TestFetchExpressionForStudy:
    @patch("tools.cbioportal.requests.get")
    def test_returns_z_scores(self, mock_get):
        mock_get.return_value = _mock_expression_resp([1.5, 2.0, -0.3])
        ct, values = _fetch_expression_for_study("brca", 4609)
        assert ct == "brca"
        assert len(values) == 3
        assert 1.5 in values

    @patch("tools.cbioportal.requests.get")
    def test_returns_empty_on_404(self, mock_get):
        mock_get.return_value = _mock_404()
        ct, values = _fetch_expression_for_study("brca", 4609)
        assert ct == "brca"
        assert values == []

    @patch("tools.cbioportal.requests.get")
    def test_returns_empty_on_timeout(self, mock_get):
        mock_get.side_effect = requests.Timeout("timed out")
        ct, values = _fetch_expression_for_study("brca", 4609)
        assert values == []


# ---------------------------------------------------------------------------
# get_gene_tumor_expression
# ---------------------------------------------------------------------------

class TestGetGeneTumorExpression:
    @patch("tools.cbioportal._get_entrez_id")
    @patch("tools.cbioportal._fetch_expression_for_study")
    def test_returns_sorted_cancer_stats(self, mock_fetch, mock_entrez):
        mock_entrez.return_value = 4609
        # Simulate 3 cancer types returning z-scores
        def fake_fetch(ct, entrez_id):
            data = {
                "brca": [1.5, 2.0, 0.8],
                "luad": [0.1, -0.2, 0.0],
                "gbm":  [3.0, 2.5, 2.8],
            }
            return ct, data.get(ct, [])
        mock_fetch.side_effect = fake_fetch

        result = get_gene_tumor_expression("MYC", top_n=2)

        assert result["gene"] == "MYC"
        assert "error" not in result
        assert "top_overexpressed" in result
        assert "pan_cancer_mean_z" in result
        # GBM has the highest mean — should appear first after sorting
        top = result["top_overexpressed"]
        assert len(top) <= 2
        assert top[0]["mean_z_score"] >= top[-1]["mean_z_score"]

    @patch("tools.cbioportal._get_entrez_id")
    def test_returns_error_for_unknown_gene(self, mock_entrez):
        mock_entrez.return_value = None
        result = get_gene_tumor_expression("NOTAREAL")
        assert "error" in result

    @patch("tools.cbioportal._get_entrez_id")
    @patch("tools.cbioportal._fetch_expression_for_study")
    def test_overexpressed_flag_correct(self, mock_fetch, mock_entrez):
        """Pan-cancer mean z > 1.0 should be reflected in the result."""
        mock_entrez.return_value = 4609
        mock_fetch.side_effect = lambda ct, _: (ct, [2.0, 2.5] if ct == "brca" else [])

        result = get_gene_tumor_expression("MYC")
        assert result["pan_cancer_mean_z"] > 1.0

    @patch("tools.cbioportal._get_entrez_id")
    @patch("tools.cbioportal._fetch_expression_for_study")
    def test_returns_error_when_no_data(self, mock_fetch, mock_entrez):
        mock_entrez.return_value = 4609
        mock_fetch.side_effect = lambda ct, _: (ct, [])

        result = get_gene_tumor_expression("MYC")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_gene_alteration_frequency
# ---------------------------------------------------------------------------

class TestGetGeneAlterationFrequency:
    @patch("tools.cbioportal._get_entrez_id")
    @patch("tools.cbioportal._fetch_alterations_for_study")
    def test_computes_pan_cancer_percentages(self, mock_fetch, mock_entrez):
        mock_entrez.return_value = 7157  # TP53
        # Only brca returns data; all others return zeros
        def fake_fetch(ct, entrez_id):
            if ct == "brca":
                return {
                    "cancer_prefix": ct,
                    "mut_count": 40, "mut_total": 100,
                    "amp_count": 5,  "del_count": 1, "cna_total": 100,
                }
            return {"cancer_prefix": ct, "mut_count": 0, "mut_total": 0,
                    "amp_count": 0, "del_count": 0, "cna_total": 0}
        mock_fetch.side_effect = fake_fetch

        result = get_gene_alteration_frequency("TP53")

        assert result["gene"] == "TP53"
        assert "error" not in result
        assert result["mutation_frequency_pct"] == pytest.approx(40.0)
        assert result["amplification_frequency_pct"] == pytest.approx(5.0)
        assert result["most_altered_cancer_type"] == "Breast Invasive Carcinoma"

    @patch("tools.cbioportal._get_entrez_id")
    def test_returns_error_for_unknown_gene(self, mock_entrez):
        mock_entrez.return_value = None
        result = get_gene_alteration_frequency("NOTAREAL")
        assert "error" in result

    @patch("tools.cbioportal._get_entrez_id")
    @patch("tools.cbioportal._fetch_alterations_for_study")
    def test_alteration_by_cancer_sorted_descending(self, mock_fetch, mock_entrez):
        mock_entrez.return_value = 4609
        def fake_fetch(ct, entrez_id):
            data = {
                "brca": {"cancer_prefix": "brca", "mut_count": 10, "mut_total": 100,
                         "amp_count": 30, "del_count": 0, "cna_total": 100},
                "luad": {"cancer_prefix": "luad", "mut_count": 5,  "mut_total": 100,
                         "amp_count": 5,  "del_count": 0, "cna_total": 100},
            }
            return data.get(ct, {"cancer_prefix": ct, "mut_count": 0, "mut_total": 0,
                                  "amp_count": 0, "del_count": 0, "cna_total": 0})
        mock_fetch.side_effect = fake_fetch

        result = get_gene_alteration_frequency("MYC")
        by_cancer = result["alteration_by_cancer"]
        totals = [e["total_altered_pct"] for e in by_cancer]
        assert totals == sorted(totals, reverse=True)


# ---------------------------------------------------------------------------
# get_cbioportal_stats
# ---------------------------------------------------------------------------

class TestGetCbioportalStats:
    @patch("tools.cbioportal.requests.get")
    def test_api_reachable(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = get_cbioportal_stats()
        assert result["api_reachable"] is True
        assert result["num_tcga_studies"] == 32

    @patch("tools.cbioportal.requests.get")
    def test_api_unreachable(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("unreachable")
        result = get_cbioportal_stats()
        assert result["api_reachable"] is False
        assert result["num_tcga_studies"] == 32
