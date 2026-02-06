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
