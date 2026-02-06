"""Tests for tools/ppi/string_client.py — STRING protein-protein interaction client."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from tools.ppi.string_client import STRINGClient, get_string_client


class TestResolveToStringId:
    @patch("tools.ppi.string_client.requests.get")
    def test_resolve_success(self, mock_get):
        mock_get.return_value.json.return_value = [
            {"stringId": "9606.ENSP00000256078"}
        ]
        mock_get.return_value.raise_for_status = MagicMock()

        client = STRINGClient()
        result = client._resolve_to_string_id("TP53")
        assert result == "9606.ENSP00000256078"

    @patch("tools.ppi.string_client.requests.get")
    def test_resolve_empty_response(self, mock_get):
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status = MagicMock()

        client = STRINGClient()
        result = client._resolve_to_string_id("FAKEGENE")
        assert result is None

    @patch("tools.ppi.string_client.requests.get")
    def test_resolve_network_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection timeout")

        client = STRINGClient()
        result = client._resolve_to_string_id("TP53")
        assert result is None


class TestGetInteractions:
    @patch.object(STRINGClient, "_resolve_to_string_id")
    @patch("tools.ppi.string_client.requests.get")
    def test_basic_interactions(self, mock_get, mock_resolve):
        mock_resolve.return_value = "9606.ENSP00000256078"
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = [
            {
                "preferredName_A": "TP53",
                "preferredName_B": "MDM2",
                "stringId_A": "9606.ENSP00000256078",
                "stringId_B": "9606.ENSP00000270066",
                "score": 999,
                "escore": 0.9,
                "dscore": 0.8,
                "tscore": 0.95,
                "ascore": 0.3,
                "nscore": 0.0,
            },
            {
                "preferredName_A": "TP53",
                "preferredName_B": "CDKN1A",
                "stringId_A": "9606.ENSP00000256078",
                "stringId_B": "9606.ENSP00000244741",
                "score": 850,
                "escore": 0.7,
                "dscore": 0.6,
                "tscore": 0.8,
                "ascore": 0.2,
                "nscore": 0.0,
            },
        ]

        client = STRINGClient()
        result = client.get_interactions("TP53")

        assert result["query_gene"] == "TP53"
        assert result["count"] == 2
        assert len(result["interactions"]) == 2
        # Sorted by score descending
        assert result["interactions"][0]["partner"] == "MDM2"
        assert result["interactions"][0]["combined_score"] == 999
        assert result["interactions"][1]["partner"] == "CDKN1A"

    @patch.object(STRINGClient, "_resolve_to_string_id")
    def test_gene_not_found(self, mock_resolve):
        mock_resolve.return_value = None

        client = STRINGClient()
        result = client.get_interactions("NOTAREALGENE")
        assert "error" in result
        assert "Could not find protein" in result["error"]

    @patch.object(STRINGClient, "_resolve_to_string_id")
    @patch("tools.ppi.string_client.requests.get")
    def test_empty_interactions(self, mock_get, mock_resolve):
        mock_resolve.return_value = "9606.ENSP00000256078"
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = []

        client = STRINGClient()
        result = client.get_interactions("TP53", min_score=900)
        assert result["count"] == 0
        assert result["interactions"] == []
        assert "No interactions found" in result["note"]

    @patch.object(STRINGClient, "_resolve_to_string_id")
    @patch("tools.ppi.string_client.requests.get")
    def test_network_error_during_interactions(self, mock_get, mock_resolve):
        mock_resolve.return_value = "9606.ENSP00000256078"
        mock_get.side_effect = requests.RequestException("API down")

        client = STRINGClient()
        result = client.get_interactions("TP53")
        assert "error" in result
        assert "STRING API request failed" in result["error"]

    @patch.object(STRINGClient, "_resolve_to_string_id")
    @patch("tools.ppi.string_client.requests.get")
    def test_skips_self_interactions(self, mock_get, mock_resolve):
        mock_resolve.return_value = "9606.ENSP00000256078"
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = [
            {
                "preferredName_A": "TP53",
                "preferredName_B": "TP53",
                "stringId_A": "9606.ENSP00000256078",
                "stringId_B": "9606.ENSP00000256078",
                "score": 999,
                "escore": 0, "dscore": 0, "tscore": 0, "ascore": 0, "nscore": 0,
            },
        ]

        client = STRINGClient()
        result = client.get_interactions("TP53")
        assert result["count"] == 0

    @patch.object(STRINGClient, "_resolve_to_string_id")
    @patch("tools.ppi.string_client.requests.get")
    def test_skips_duplicate_partners(self, mock_get, mock_resolve):
        mock_resolve.return_value = "9606.ENSP00000256078"
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = [
            {
                "preferredName_A": "TP53",
                "preferredName_B": "MDM2",
                "stringId_A": "9606.ENSP00000256078",
                "stringId_B": "9606.ENSP00000270066",
                "score": 999,
                "escore": 0.9, "dscore": 0.8, "tscore": 0.95, "ascore": 0.3, "nscore": 0.0,
            },
            {
                "preferredName_A": "TP53",
                "preferredName_B": "MDM2",
                "stringId_A": "9606.ENSP00000256078",
                "stringId_B": "9606.ENSP00000270066",
                "score": 850,
                "escore": 0.7, "dscore": 0.6, "tscore": 0.8, "ascore": 0.2, "nscore": 0.0,
            },
        ]

        client = STRINGClient()
        result = client.get_interactions("TP53")
        assert result["count"] == 1

    @patch.object(STRINGClient, "_resolve_to_string_id")
    @patch("tools.ppi.string_client.requests.get")
    def test_partner_on_a_side(self, mock_get, mock_resolve):
        """When the query gene appears as preferredName_B, the partner is on the A side."""
        mock_resolve.return_value = "9606.ENSP00000256078"
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = [
            {
                "preferredName_A": "MDM2",
                "preferredName_B": "TP53",
                "stringId_A": "9606.ENSP00000270066",
                "stringId_B": "9606.ENSP00000256078",
                "score": 999,
                "escore": 0.9, "dscore": 0.8, "tscore": 0.95, "ascore": 0.3, "nscore": 0.0,
            },
        ]

        client = STRINGClient()
        result = client.get_interactions("TP53")
        assert result["count"] == 1
        assert result["interactions"][0]["partner"] == "MDM2"


class TestEvidenceSummary:
    def test_evidence_fields(self):
        client = STRINGClient()
        item = {
            "escore": 0.9,
            "dscore": 0.8,
            "tscore": 0.7,
            "ascore": 0.6,
            "nscore": 0.5,
        }
        evidence = client._get_evidence_summary(item)
        assert evidence["experimental"] == 0.9
        assert evidence["database"] == 0.8
        assert evidence["textmining"] == 0.7
        assert evidence["coexpression"] == 0.6
        assert evidence["neighborhood"] == 0.5

    def test_missing_evidence_defaults_to_zero(self):
        client = STRINGClient()
        evidence = client._get_evidence_summary({})
        assert evidence["experimental"] == 0
        assert evidence["database"] == 0


class TestScoreInterpretation:
    def test_highest_confidence(self):
        client = STRINGClient()
        assert client._score_interpretation(900) == "highest confidence only"
        assert client._score_interpretation(950) == "highest confidence only"

    def test_high_confidence(self):
        client = STRINGClient()
        assert client._score_interpretation(700) == "high confidence"
        assert client._score_interpretation(899) == "high confidence"

    def test_medium_confidence(self):
        client = STRINGClient()
        assert client._score_interpretation(400) == "medium confidence"

    def test_low_confidence(self):
        client = STRINGClient()
        assert client._score_interpretation(150) == "low confidence"

    def test_all_interactions(self):
        client = STRINGClient()
        assert client._score_interpretation(0) == "all interactions"
        assert client._score_interpretation(149) == "all interactions"


class TestSingleton:
    def test_get_string_client_returns_instance(self):
        # Reset singleton
        import tools.ppi.string_client as mod
        mod._client = None
        client = get_string_client()
        assert isinstance(client, STRINGClient)

    def test_get_string_client_returns_same_instance(self):
        import tools.ppi.string_client as mod
        mod._client = None
        c1 = get_string_client()
        c2 = get_string_client()
        assert c1 is c2

    def test_custom_timeout(self):
        client = STRINGClient(timeout=60)
        assert client.timeout == 60
