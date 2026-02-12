"""Tests for cascade_langgraph_workflow.py — LangGraph integration tests.

These tests verify workflow routing logic and state management
without requiring real data files, model checkpoints, or external APIs.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestGeneRoleClassification:
    """Test the gene classification logic used for routing."""

    def _classify_gene(self, num_targets, num_regulators):
        """Replicate the classification logic from the workflow."""
        if num_targets > 50:
            return "master_regulator"
        elif num_targets > 10:
            return "transcription_factor"
        elif num_targets > 0:
            return "minor_regulator"
        elif num_regulators > 0:
            return "effector"
        else:
            return "isolated"

    def test_master_regulator(self):
        assert self._classify_gene(num_targets=100, num_regulators=5) == "master_regulator"

    def test_transcription_factor(self):
        assert self._classify_gene(num_targets=25, num_regulators=3) == "transcription_factor"

    def test_minor_regulator(self):
        assert self._classify_gene(num_targets=5, num_regulators=2) == "minor_regulator"

    def test_effector(self):
        assert self._classify_gene(num_targets=0, num_regulators=10) == "effector"

    def test_isolated(self):
        assert self._classify_gene(num_targets=0, num_regulators=0) == "isolated"

    def test_boundary_master_regulator(self):
        assert self._classify_gene(num_targets=51, num_regulators=0) == "master_regulator"

    def test_boundary_transcription_factor(self):
        assert self._classify_gene(num_targets=11, num_regulators=0) == "transcription_factor"

    def test_boundary_exactly_50(self):
        """50 targets is TF, not master regulator (threshold is >50)."""
        assert self._classify_gene(num_targets=50, num_regulators=0) == "transcription_factor"


class TestRoutingDecisions:
    """Test which analysis batches are selected based on gene role and depth."""

    def _decide_batches(self, gene_role, depth):
        """Replicate simplified routing logic."""
        batches = set()

        # Core analysis is always included
        batches.add("batch_core")

        if depth in ("comprehensive", "focused"):
            batches.add("batch_external")

        if depth == "comprehensive":
            batches.add("batch_insights")

        # Effectors always get external (PPI is important for them)
        if gene_role == "effector":
            batches.add("batch_external")

        return batches

    def test_basic_depth_only_core(self):
        batches = self._decide_batches("transcription_factor", "basic")
        assert batches == {"batch_core"}

    def test_focused_includes_external(self):
        batches = self._decide_batches("transcription_factor", "focused")
        assert "batch_external" in batches

    def test_comprehensive_includes_all(self):
        batches = self._decide_batches("master_regulator", "comprehensive")
        assert "batch_core" in batches
        assert "batch_external" in batches
        assert "batch_insights" in batches

    def test_effector_always_gets_external(self):
        batches = self._decide_batches("effector", "basic")
        assert "batch_external" in batches


class TestStateSchema:
    """Test that state schema fields are properly typed."""

    def test_state_dict_structure(self):
        """Verify the expected state keys exist as a TypedDict."""
        try:
            from cascade_langgraph_workflow import PerturbationAnalysisState
            # Check it has the expected annotation keys
            annotations = PerturbationAnalysisState.__annotations__
            required_keys = [
                "gene", "cell_type", "perturbation_type",
                "gene_role", "comprehensive_report",
                "dorothea_regulons",
            ]
            for key in required_keys:
                assert key in annotations, f"Missing state key: {key}"
        except ImportError:
            pytest.skip("cascade_langgraph_workflow not importable (missing dependencies)")


class TestVulnerabilityScoring:
    """Test the vulnerability score formula: V = h + 0.3*c + 10*w_mean + 5/(r+1)."""

    def _compute_vulnerability(self, hub_score, cascade_reach, mean_weight, regulator_count):
        return hub_score + 0.3 * cascade_reach + 10 * mean_weight + 5 / (regulator_count + 1)

    def test_high_hub_high_vulnerability(self):
        v = self._compute_vulnerability(hub_score=50, cascade_reach=100, mean_weight=0.5, regulator_count=2)
        assert v > 50  # High hub score dominates

    def test_few_regulators_increases_vulnerability(self):
        v0 = self._compute_vulnerability(10, 10, 0.5, regulator_count=0)
        v10 = self._compute_vulnerability(10, 10, 0.5, regulator_count=10)
        assert v0 > v10  # Fewer regulators = higher vulnerability

    def test_zero_regulators(self):
        v = self._compute_vulnerability(0, 0, 0, regulator_count=0)
        assert v == 5.0  # Only the 5/(0+1) = 5 term remains


class TestBuildRoleContext:
    """Test the _build_role_context helper function."""

    def _get_helpers(self):
        from cascade_langgraph_workflow import _build_role_context, _build_key_findings
        return _build_role_context, _build_key_findings

    def test_tf_with_agreements(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("transcription_factor", total_affected=20, ppi_count=5, agreement_count=2)
        assert "network_propagation" in result["primary"]
        assert "lincs_experimental" in result["primary"]
        assert "2 prediction(s)" in result["context"]

    def test_tf_without_agreements(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("transcription_factor", total_affected=20, ppi_count=5, agreement_count=0)
        assert result["primary"] == "network_propagation"
        assert "No overlapping LINCS" in result["context"]

    def test_tf_with_dorothea_validation(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("transcription_factor", total_affected=20, ppi_count=5,
                                    agreement_count=1, dorothea_validated=True)
        assert "dorothea" in result["primary"]
        assert "DoRothEA" in result["context"]

    def test_effector_with_ppi(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("effector", total_affected=0, ppi_count=5, agreement_count=0)
        assert result["primary"] == "string_ppi"
        assert "no transcriptional targets" in result["context"]

    def test_effector_without_ppi(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("effector", total_affected=0, ppi_count=0, agreement_count=0)
        assert result["primary"] == "embedding_similarity"

    def test_isolated(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("isolated", total_affected=0, ppi_count=0, agreement_count=0)
        assert "not present in the cell-type regulatory network" in result["context"]

    def test_unknown_role(self):
        build_role_context, _ = self._get_helpers()
        result = build_role_context("unknown", total_affected=0, ppi_count=0, agreement_count=0)
        assert result["primary"] == "unknown"


class TestBuildKeyFindings:
    """Test the _build_key_findings helper function."""

    def _get_helper(self):
        from cascade_langgraph_workflow import _build_key_findings
        return _build_key_findings

    def test_multi_source_finding(self):
        build = self._get_helper()
        multi_source = [{"symbol": "CDKN1A", "sources": ["network_propagation", "string_ppi"], "source_count": 2}]
        findings = build("transcription_factor", multi_source, [], [], 20, 5, 10)
        assert any("1 gene(s) supported by multiple" in f for f in findings)

    def test_agreement_finding(self):
        build = self._get_helper()
        agreements = ["CDKN1A: network down + LINCS down"]
        findings = build("transcription_factor", [], agreements, [], 20, 5, 10)
        assert any("1 gene(s) confirmed" in f for f in findings)
        assert any("No directional disagreements" in f for f in findings)

    def test_disagreement_finding(self):
        build = self._get_helper()
        disagreements = ["CDKN1A: network down vs LINCS up"]
        findings = build("transcription_factor", [], [], disagreements, 20, 5, 10)
        assert any("1 gene(s) show directional disagreement" in f for f in findings)

    def test_ppi_only_genes(self):
        build = self._get_helper()
        multi_source = [
            {"symbol": "BRCA1", "sources": ["embedding_similarity", "string_ppi"], "source_count": 2}
        ]
        findings = build("transcription_factor", multi_source, [], [], 20, 5, 0)
        assert any("STRING identifies 1 protein interaction partner" in f for f in findings)

    def test_effector_no_targets_with_ppi(self):
        build = self._get_helper()
        findings = build("effector", [], [], [], 0, 5, 0)
        assert any("no transcriptional targets but 5 STRING" in f for f in findings)

    def test_effector_no_targets_no_ppi(self):
        build = self._get_helper()
        findings = build("effector", [], [], [], 0, 0, 0)
        assert any("Embedding similarity is the only" in f for f in findings)

    def test_dorothea_validated_finding(self):
        build = self._get_helper()
        findings = build("transcription_factor", [], [], [], 20, 5, 10, dorothea_validated=True)
        assert any("DoRothEA" in f for f in findings)

    def test_empty_inputs(self):
        build = self._get_helper()
        findings = build("transcription_factor", [], [], [], 20, 5, 10)
        assert isinstance(findings, list)


class TestSynthesizeEvidence:
    """Test the _synthesize_evidence method with mock state dicts."""

    def _get_helpers(self):
        try:
            from cascade_langgraph_workflow import _build_role_context, _build_key_findings
            return _build_role_context, _build_key_findings
        except ImportError:
            pytest.skip("cascade_langgraph_workflow not importable")

    def test_multi_source_detection(self):
        """Gene appearing in both network propagation and STRING should be multi-source."""
        from cascade_langgraph_workflow import _build_role_context
        from collections import defaultdict

        # Simulate what _synthesize_evidence does
        evidence = defaultdict(lambda: {"sources": [], "evidence": {}})

        # CDKN1A in network propagation
        evidence["CDKN1A"]["sources"].append("network_propagation")
        evidence["CDKN1A"]["evidence"]["network"] = {"effect": -0.8, "direction": "down"}

        # CDKN1A in STRING
        evidence["CDKN1A"]["sources"].append("string_ppi")
        evidence["CDKN1A"]["evidence"]["string"] = {"combined_score": 900}

        multi_source = [
            {"symbol": sym, "sources": sorted(data["sources"]),
             "source_count": len(data["sources"]), "evidence": data["evidence"]}
            for sym, data in evidence.items() if len(data["sources"]) >= 2
        ]

        assert len(multi_source) == 1
        assert multi_source[0]["symbol"] == "CDKN1A"
        assert multi_source[0]["source_count"] == 2

    def test_directional_agreement(self):
        """Network and LINCS both saying 'down' should be an agreement."""
        multi_source = [{
            "symbol": "CDKN1A",
            "sources": ["lincs_experimental", "network_propagation"],
            "source_count": 2,
            "evidence": {
                "network": {"direction": "down"},
                "lincs": {"direction": "down"}
            }
        }]

        agreements = []
        for gene_data in multi_source:
            ev = gene_data["evidence"]
            if "network" in ev and "lincs" in ev:
                if ev["network"]["direction"] == ev["lincs"]["direction"]:
                    agreements.append(f"{gene_data['symbol']}: network {ev['network']['direction']} + LINCS {ev['lincs']['direction']}")

        assert len(agreements) == 1
        assert "CDKN1A: network down + LINCS down" in agreements[0]

    def test_directional_disagreement(self):
        """Network 'down' vs LINCS 'up' should be a disagreement."""
        multi_source = [{
            "symbol": "CDKN1A",
            "sources": ["lincs_experimental", "network_propagation"],
            "source_count": 2,
            "evidence": {
                "network": {"direction": "down"},
                "lincs": {"direction": "up"}
            }
        }]

        disagreements = []
        for gene_data in multi_source:
            ev = gene_data["evidence"]
            if "network" in ev and "lincs" in ev:
                if ev["network"]["direction"] != ev["lincs"]["direction"]:
                    disagreements.append(f"{gene_data['symbol']}: network {ev['network']['direction']} vs LINCS {ev['lincs']['direction']}")

        assert len(disagreements) == 1
        assert "CDKN1A: network down vs LINCS up" in disagreements[0]

    def test_lincs_type_hazard(self):
        """LINCS as dict (error) should not crash — should produce empty list."""
        lincs = {"error": "API down"}
        lincs_list = lincs if isinstance(lincs, list) else []
        assert lincs_list == []

    def test_all_empty_state(self):
        """All None state fields should produce valid empty synthesis."""
        from cascade_langgraph_workflow import _build_role_context, _build_key_findings

        role_context = _build_role_context("unknown", 0, 0, 0)
        key_findings = _build_key_findings("unknown", [], [], [], 0, 0, 0)

        assert isinstance(role_context, dict)
        assert "context" in role_context
        assert "primary" in role_context
        assert isinstance(key_findings, list)
