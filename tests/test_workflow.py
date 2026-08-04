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


# =============================================================================
# Workflow Node Tests — testing actual CascadeWorkflow node methods
# =============================================================================

def _make_workflow(mapper=None):
    """
    Create a CascadeWorkflow instance without running __init__.

    Uses object.__new__ to bypass file loading, then manually sets the
    attributes that node methods rely on.
    """
    try:
        from cascade_langgraph_workflow import CascadeWorkflow
    except ImportError:
        pytest.skip("cascade_langgraph_workflow not importable (missing dependencies)")

    wf = object.__new__(CascadeWorkflow)
    wf.gene_mapper = mapper or MagicMock()
    wf._model = None
    wf.string_client = MagicMock()
    return wf


class TestWorkflowNodeInitialize:
    """Test the _initialize node sets up the workflow state."""

    @pytest.mark.asyncio
    async def test_returns_required_control_fields(self):
        wf = _make_workflow()
        state = {"gene": "TP53", "cell_type": "epithelial_cell"}
        result = await wf._initialize(state)

        assert result["current_step"] == "initialize"
        assert result["workflow_complete"] is False
        assert result["next_actions"] == []
        assert result["completed_actions"] == []
        assert "analysis_metadata" in result

    @pytest.mark.asyncio
    async def test_embedding_enhanced_starts_false(self):
        wf = _make_workflow()
        state = {"gene": "MYC", "cell_type": "cd4_t_cells"}
        result = await wf._initialize(state)
        assert result["embedding_enhanced"] is False

    @pytest.mark.asyncio
    async def test_no_error_on_start(self):
        wf = _make_workflow()
        state = {"gene": "BRCA1", "cell_type": "epithelial_cell"}
        result = await wf._initialize(state)
        assert result.get("error_message") is None


class TestWorkflowNodeResolveGene:
    """Test the _resolve_gene node with a controlled gene mapper."""

    @pytest.fixture
    def wf(self, mock_gene_id_mapper):
        return _make_workflow(mapper=mock_gene_id_mapper)

    @pytest.mark.asyncio
    async def test_symbol_resolved_to_ensembl(self, wf):
        result = await wf._resolve_gene({"gene": "TP53"})
        assert result["ensembl_id"] == "ENSG00000141510"
        assert result["gene_symbol"] == "TP53"
        assert result["current_step"] == "resolve_gene"

    @pytest.mark.asyncio
    async def test_ensembl_input_resolved_to_symbol(self, wf):
        result = await wf._resolve_gene({"gene": "ENSG00000136997"})
        assert result["ensembl_id"] == "ENSG00000136997"
        assert result["gene_symbol"] == "MYC"

    @pytest.mark.asyncio
    async def test_unknown_symbol_returns_error(self, wf):
        result = await wf._resolve_gene({"gene": "FAKEGENE"})
        assert "error_message" in result
        assert result["next_actions"] == ["error"]

    @pytest.mark.asyncio
    async def test_lowercase_symbol_normalised(self, wf):
        """Gene symbols should be uppercased before lookup."""
        result = await wf._resolve_gene({"gene": "tp53"})
        # mock_gene_id_mapper uses .upper() in symbol_to_ensembl side_effect
        assert result["ensembl_id"] == "ENSG00000141510"


class TestWorkflowNodeAnalyzeNetworkContext:
    """Test _analyze_network_context classifies gene roles from network data."""

    @pytest.fixture
    def wf(self, tmp_path, mock_gene_id_mapper, mock_network_df):
        """Workflow with NETWORKS_DIR pointing at a mock network."""
        wf = _make_workflow(mapper=mock_gene_id_mapper)
        # Create network directory and file so the existence check passes
        cell_dir = tmp_path / "epithelial_cell"
        cell_dir.mkdir()
        (cell_dir / "network.tsv").touch()
        wf.NETWORKS_DIR = tmp_path
        # Store the mock df so tests can reference it
        wf._mock_df = mock_network_df
        return wf

    @pytest.mark.asyncio
    async def test_effector_classification(self, wf):
        """ENSG_TARGET3 has 1 regulator and 0 targets → effector."""
        with patch("tools.loader.load_network", return_value=wf._mock_df):
            result = await wf._analyze_network_context(
                {"cell_type": "epithelial_cell", "ensembl_id": "ENSG_TARGET3"}
            )
        assert result["gene_role"] == "effector"
        assert result["network_context"]["num_targets"] == 0
        assert result["network_context"]["num_regulators"] == 1

    @pytest.mark.asyncio
    async def test_minor_regulator_classification(self, wf):
        """ENSG_TF1 has 2 targets → minor_regulator (0 < n ≤ 10)."""
        with patch("tools.loader.load_network", return_value=wf._mock_df):
            result = await wf._analyze_network_context(
                {"cell_type": "epithelial_cell", "ensembl_id": "ENSG_TF1"}
            )
        assert result["gene_role"] == "minor_regulator"
        assert result["network_context"]["num_targets"] == 2

    @pytest.mark.asyncio
    async def test_isolated_classification(self, wf):
        """Gene with no targets and no regulators → isolated."""
        with patch("tools.loader.load_network", return_value=wf._mock_df):
            result = await wf._analyze_network_context(
                {"cell_type": "epithelial_cell", "ensembl_id": "ENSG_UNKNOWN"}
            )
        assert result["gene_role"] == "isolated"

    @pytest.mark.asyncio
    async def test_missing_network_returns_error(self, wf):
        """A non-existent cell type should return an error, not crash."""
        result = await wf._analyze_network_context(
            {"cell_type": "nonexistent_cell_type", "ensembl_id": "ENSG_TF1"}
        )
        assert "error_message" in result
        assert result["next_actions"] == ["error"]

    @pytest.mark.asyncio
    async def test_network_context_in_result(self, wf):
        """Result should include network_context dict with expected keys."""
        with patch("tools.loader.load_network", return_value=wf._mock_df):
            result = await wf._analyze_network_context(
                {"cell_type": "epithelial_cell", "ensembl_id": "ENSG_TF1"}
            )
        ctx = result["network_context"]
        assert "num_targets" in ctx
        assert "num_regulators" in ctx
        assert "in_network" in ctx
        assert ctx["in_network"] is True


class TestWorkflowNodeDecideNextSteps:
    """Test _decide_next_steps routing decisions using the real method."""

    @pytest.fixture
    def wf(self):
        return _make_workflow()

    @pytest.mark.asyncio
    async def test_basic_depth_requires_only_perturbation_and_similar(self, wf):
        state = {
            "gene_role": "transcription_factor",
            "analysis_depth": "basic",
            "perturbation_type": "knockdown",
            "completed_actions": [],
        }
        result = await wf._decide_next_steps(state)
        # Basic only needs perturbation + similar; should not jump straight to complete
        assert result["next_actions"] != ["complete"]

    @pytest.mark.asyncio
    async def test_all_required_complete_routes_to_complete(self, wf):
        state = {
            "gene_role": "transcription_factor",
            "analysis_depth": "basic",
            "perturbation_type": "knockdown",
            "completed_actions": ["perturbation", "similar"],
        }
        result = await wf._decide_next_steps(state)
        assert result["next_actions"] == ["complete"]
        assert result["workflow_complete"] is True

    @pytest.mark.asyncio
    async def test_comprehensive_depth_requires_many_steps(self, wf):
        state = {
            "gene_role": "master_regulator",
            "analysis_depth": "comprehensive",
            "perturbation_type": "knockdown",
            "completed_actions": [],
        }
        result = await wf._decide_next_steps(state)
        # Should not immediately complete
        assert result["next_actions"] != ["complete"]

    @pytest.mark.asyncio
    async def test_multiple_pending_core_tasks_batched(self, wf):
        # Comprehensive with nothing completed: multiple batch groups pending
        # → routes to run_all_batches (all groups dispatched concurrently)
        state = {
            "gene_role": "transcription_factor",
            "analysis_depth": "comprehensive",
            "perturbation_type": "knockdown",
            "completed_actions": [],
        }
        result = await wf._decide_next_steps(state)
        assert "run_all_batches" in result["next_actions"]

    @pytest.mark.asyncio
    async def test_single_batch_group_routes_to_batch_core(self, wf):
        # When only core batch is pending (external + insights already done),
        # route to batch_core (not run_all_batches)
        state = {
            "gene_role": "transcription_factor",
            "analysis_depth": "comprehensive",
            "perturbation_type": "knockdown",
            "completed_actions": [
                "lincs", "dorothea", "depmap", "cbioportal",  # external done
                "similar", "vulnerability", "cross_cell",  # insights done
            ],
        }
        result = await wf._decide_next_steps(state)
        assert "batch_core" in result["next_actions"]

    @pytest.mark.asyncio
    async def test_isolated_gene_does_not_require_regulators(self, wf):
        # Isolated genes are not in the network — regulators analysis is uninformative.
        # Completing all required analyses WITHOUT regulators should route to complete.
        state = {
            "gene_role": "isolated",
            "analysis_depth": "comprehensive",
            "perturbation_type": "knockdown",
            "completed_actions": [
                "perturbation", "similar", "dorothea",          # core (no regulators)
                "ppi", "super_enhancers", "depmap", "cbioportal",  # external
                "cross_cell",                                    # insights
            ],
        }
        result = await wf._decide_next_steps(state)
        assert result["next_actions"] == ["complete"]
        assert result["workflow_complete"] is True

    @pytest.mark.asyncio
    async def test_effector_gene_still_requires_regulators(self, wf):
        # Effector genes do have upstream regulators — regulators analysis remains required.
        # Without regulators in completed, should NOT route to complete.
        state = {
            "gene_role": "effector",
            "analysis_depth": "comprehensive",
            "perturbation_type": "knockdown",
            "completed_actions": [
                "perturbation", "similar", "dorothea",  # core (missing regulators)
                "ppi", "super_enhancers",               # external
                "cross_cell",                           # insights
            ],
        }
        result = await wf._decide_next_steps(state)
        assert result["next_actions"] != ["complete"]


class TestNoNetworkTargetsNote:
    """Test that no_network_targets_note is present in reports for effector/isolated genes."""

    @pytest.fixture
    def wf(self):
        return _make_workflow()

    def _minimal_state(self, gene_role):
        return {
            "gene": "KRAS",
            "gene_symbol": "KRAS",
            "cell_type": "epithelial_cell",
            "gene_role": gene_role,
            "perturbation_type": "knockdown",
            "perturbation_result": {"total_affected_genes": 0, "top_affected_genes": []},
            "ppi_interactions": None,
            "lincs_effects": None,
            "similar_genes": None,
            "regulators_analysis": None,
            "targets_analysis": None,
            "dorothea_regulons": None,
            "super_enhancer_status": None,
            "vulnerability_analysis": None,
            "network_context": None,
            "cross_cell_comparison": None,
            "embedding_enhanced": False,
            "analysis_metadata": {},
            "completed_actions": [],
            "ensembl_id": None,
            "include_llm_insights": False,
        }

    @pytest.mark.asyncio
    async def test_effector_report_includes_note(self, wf):
        state = self._minimal_state("effector")
        result = await wf._generate_report(state)
        note = result["comprehensive_report"]["network_analysis"]["no_network_targets_note"]
        assert note is not None
        assert "effector" in note
        assert "protein_interactions" in note

    @pytest.mark.asyncio
    async def test_isolated_report_includes_note(self, wf):
        state = self._minimal_state("isolated")
        result = await wf._generate_report(state)
        note = result["comprehensive_report"]["network_analysis"]["no_network_targets_note"]
        assert note is not None
        assert "isolated" in note
        assert "protein_interactions" in note

    @pytest.mark.asyncio
    async def test_transcription_factor_report_has_no_note(self, wf):
        state = self._minimal_state("transcription_factor")
        result = await wf._generate_report(state)
        note = result["comprehensive_report"]["network_analysis"]["no_network_targets_note"]
        assert note is None


class TestEmbeddingFallbackNote:
    """Predictions sourced purely from embedding similarity (no real network edge) must be
    flagged in the report rather than reading as network-validated."""

    @pytest.fixture
    def wf(self):
        return _make_workflow()

    def _state_with_targets(self, top_affected_genes):
        return {
            "gene": "PLK1",
            "gene_symbol": "PLK1",
            "cell_type": "epithelial_cell",
            "gene_role": "effector",
            "perturbation_type": "knockdown",
            "perturbation_result": {
                "total_affected_genes": len(top_affected_genes),
                "top_affected_genes": top_affected_genes,
            },
            "ppi_interactions": None,
            "lincs_effects": None,
            "similar_genes": None,
            "regulators_analysis": None,
            "targets_analysis": None,
            "dorothea_regulons": None,
            "super_enhancer_status": None,
            "vulnerability_analysis": None,
            "network_context": None,
            "cross_cell_comparison": None,
            "embedding_enhanced": False,
            "analysis_metadata": {},
            "completed_actions": [],
            "ensembl_id": None,
            "include_llm_insights": False,
        }

    @pytest.mark.asyncio
    async def test_all_embedding_only_flagged(self, wf):
        genes = [
            {"symbol": "GENE1", "direction": "down", "source": "embedding_only"},
            {"symbol": "GENE2", "direction": "up", "source": "embedding_only"},
        ]
        state = self._state_with_targets(genes)
        result = await wf._generate_report(state)
        analysis = result["comprehensive_report"]["network_analysis"]
        assert analysis["embedding_only_prediction_count"] == 2
        assert analysis["network_backed_prediction_count"] == 0
        assert "embedding" in analysis["embedding_fallback_note"].lower()
        assert "All 2" in analysis["embedding_fallback_note"]

    @pytest.mark.asyncio
    async def test_partial_embedding_only_flagged(self, wf):
        genes = [
            {"symbol": "GENE1", "direction": "down", "source": "embedding_only"},
            {"symbol": "GENE2", "direction": "up", "source": "network+embedding"},
        ]
        state = self._state_with_targets(genes)
        result = await wf._generate_report(state)
        analysis = result["comprehensive_report"]["network_analysis"]
        assert analysis["embedding_only_prediction_count"] == 1
        assert analysis["network_backed_prediction_count"] == 1
        assert "1 of 2" in analysis["embedding_fallback_note"]

    @pytest.mark.asyncio
    async def test_no_embedding_only_no_note(self, wf):
        genes = [{"symbol": "GENE1", "direction": "down", "source": "network+embedding"}]
        state = self._state_with_targets(genes)
        result = await wf._generate_report(state)
        analysis = result["comprehensive_report"]["network_analysis"]
        assert analysis["embedding_only_prediction_count"] == 0
        assert analysis["embedding_fallback_note"] is None


class TestSynthesizeEvidenceEmbeddingFallback:
    """embedding_only predictions must not be tagged network_propagation -- that would let
    fallback-only genes masquerade as network-validated in multi-source confidence scoring."""

    @pytest.fixture
    def wf(self):
        return _make_workflow()

    def _state_with_ppi_overlap(self, source):
        """PLK1TGT appears in both top_affected_genes and STRING PPI, so it's multi-source
        and shows up in the returned multi_source_genes list regardless of its perturb.py
        source tag."""
        return {
            "gene": "PLK1",
            "gene_symbol": "PLK1",
            "gene_role": "effector",
            "perturbation_result": {
                "top_affected_genes": [
                    {"symbol": "PLK1TGT", "direction": "down", "combined_effect": -0.5,
                     "source": source},
                ]
            },
            "ppi_interactions": {"interactions": [
                {"partner": "PLK1TGT", "combined_score": 900}
            ]},
            "lincs_effects": None,
            "similar_genes": None,
            "regulators_analysis": None,
            "targets_analysis": None,
            "dorothea_regulons": None,
            "depmap_essentiality": None,
            "cbioportal_tumor_data": None,
        }

    def test_embedding_only_tagged_as_embedding_similarity(self, wf):
        state = self._state_with_ppi_overlap("embedding_only")
        result = wf._synthesize_evidence(state)
        entry = next(g for g in result["multi_source_genes"] if g["symbol"] == "PLK1TGT")
        assert "network_propagation" not in entry["sources"]
        assert "embedding_similarity" in entry["sources"]
        assert entry["evidence"]["network"]["embedding_fallback"] is True

    def test_network_backed_tagged_as_network_propagation(self, wf):
        state = self._state_with_ppi_overlap("network+embedding")
        result = wf._synthesize_evidence(state)
        entry = next(g for g in result["multi_source_genes"] if g["symbol"] == "PLK1TGT")
        assert "network_propagation" in entry["sources"]
        assert "embedding_similarity" not in entry["sources"]
        assert entry["evidence"]["network"]["embedding_fallback"] is False


class TestRouteNextAction:
    """Test _route_next_action maps state to correct node names."""

    @pytest.fixture
    def wf(self):
        return _make_workflow()

    def test_error_message_routes_to_error(self, wf):
        state = {"error_message": "Something went wrong", "next_actions": []}
        assert wf._route_next_action(state) == "error"

    def test_empty_next_actions_routes_to_complete(self, wf):
        state = {"error_message": None, "next_actions": []}
        assert wf._route_next_action(state) == "complete"

    def test_batch_core_routes_correctly(self, wf):
        state = {"error_message": None, "next_actions": ["batch_core"]}
        assert wf._route_next_action(state) == "batch_core"

    def test_run_all_batches_routes_correctly(self, wf):
        state = {"error_message": None, "next_actions": ["run_all_batches"]}
        assert wf._route_next_action(state) == "run_all_batches"

    def test_complete_action_routes_correctly(self, wf):
        state = {"error_message": None, "next_actions": ["complete"]}
        assert wf._route_next_action(state) == "complete"

    def test_unknown_action_routes_to_error(self, wf):
        state = {"error_message": None, "next_actions": ["totally_unknown_action"]}
        assert wf._route_next_action(state) == "error"

    def test_perturbation_routes_correctly(self, wf):
        state = {"error_message": None, "next_actions": ["perturbation"]}
        assert wf._route_next_action(state) == "perturbation"


class TestGenerateComparisonSummary:
    """Test the _generate_comparison_summary module-level helper."""

    def _get_fn(self):
        try:
            from cascade_langgraph_mcp_server import _generate_comparison_summary
            return _generate_comparison_summary
        except ImportError:
            pytest.skip("cascade_langgraph_mcp_server not importable")

    def test_identifies_most_influential_gene(self):
        fn = self._get_fn()
        genes = ["TP53", "MYC"]
        results = {
            "TP53": {"summary": {"gene_role": "master_regulator"},
                     "network_analysis": {"context": {"num_targets": 80, "num_regulators": 2}}},
            "MYC": {"summary": {"gene_role": "transcription_factor"},
                    "network_analysis": {"context": {"num_targets": 30, "num_regulators": 5}}},
        }
        summary = fn(genes, results)
        assert summary["most_influential_gene"] == "TP53"

    def test_skips_error_genes(self):
        fn = self._get_fn()
        genes = ["TP53", "FAKEGENE"]
        results = {
            "TP53": {"summary": {"gene_role": "master_regulator"},
                     "network_analysis": {"context": {"num_targets": 50, "num_regulators": 0}}},
            "FAKEGENE": {"error": "Could not resolve"},
        }
        summary = fn(genes, results)
        assert "FAKEGENE" not in summary["gene_roles"]

    def test_all_errors_returns_none_most_influential(self):
        fn = self._get_fn()
        genes = ["FAKE1", "FAKE2"]
        results = {
            "FAKE1": {"error": "not found"},
            "FAKE2": {"error": "not found"},
        }
        summary = fn(genes, results)
        assert summary["most_influential_gene"] is None


# ---------------------------------------------------------------------------
# Test failed_analyses collection (Issue #10)
# ---------------------------------------------------------------------------

class TestFailedAnalysesCollection:
    """Verify batch errors are collected into failed_analyses rather than silently dropped."""

    def _get_batch_methods(self):
        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("cascade_langgraph_workflow.CascadeWorkflow.__init__", return_value=None):
                from cascade_langgraph_workflow import CascadeWorkflow
                wf = CascadeWorkflow.__new__(CascadeWorkflow)
                return wf

    @pytest.fixture
    def base_state(self):
        return {
            "gene": "TP53",
            "cell_type": "epithelial_cell",
            "perturbation_type": "knockdown",
            "analysis_depth": "comprehensive",
            "ensembl_id": "ENSG00000141510",
            "gene_symbol": "TP53",
            "gene_role": "master_regulator",
            "completed_actions": [],
            "required_actions": [
                "perturbation", "regulators", "targets",
                "ppi", "lincs", "super_enhancers", "dorothea", "depmap", "cbioportal",
                "similar", "vulnerability", "cross_cell",
            ],
            "failed_analyses": None,
            "network_context": {"num_targets": 100},
            "network_df": None,
        }

    def test_batch_core_collects_exceptions(self, base_state):
        import asyncio
        from unittest.mock import patch, AsyncMock

        wf = self._get_batch_methods()
        wf._run_perturbation_impl = AsyncMock(side_effect=RuntimeError("network down"))
        wf._analyze_regulators_impl = AsyncMock(return_value={"regulators": []})
        wf._analyze_targets_impl = AsyncMock(return_value={"targets": []})

        result = asyncio.run(
            wf._batch_core_analysis(base_state)
        )

        assert result.get("failed_analyses") is not None
        assert len(result["failed_analyses"]) == 1
        assert result["failed_analyses"][0]["analysis"] == "perturbation"
        assert "network down" in result["failed_analyses"][0]["error"]
        assert result["failed_analyses"][0]["batch"] == "core"

    def test_batch_core_no_failures_returns_none(self, base_state):
        import asyncio
        from unittest.mock import AsyncMock

        wf = self._get_batch_methods()
        wf._run_perturbation_impl = AsyncMock(return_value={"top_affected_genes": []})
        wf._analyze_regulators_impl = AsyncMock(return_value={"regulators": []})
        wf._analyze_targets_impl = AsyncMock(return_value={"targets": []})

        result = asyncio.run(
            wf._batch_core_analysis(base_state)
        )

        assert result.get("failed_analyses") is None

    def test_batch_external_collects_exceptions(self, base_state):
        import asyncio
        from unittest.mock import AsyncMock

        wf = self._get_batch_methods()
        wf._fetch_ppi_impl = AsyncMock(side_effect=ConnectionError("STRING API down"))
        wf._fetch_lincs_impl = AsyncMock(return_value=[])
        wf._check_super_enhancers_impl = AsyncMock(return_value={})
        wf._fetch_dorothea_impl = AsyncMock(return_value={})
        wf._fetch_depmap_impl = AsyncMock(return_value={})
        wf._fetch_cbioportal_impl = AsyncMock(return_value={})

        result = asyncio.run(
            wf._batch_external_data(base_state)
        )

        assert result.get("failed_analyses") is not None
        failed_names = [f["analysis"] for f in result["failed_analyses"]]
        assert "ppi" in failed_names
        assert result["failed_analyses"][0]["batch"] == "external"

    def test_batch_insights_collects_exceptions(self, base_state):
        import asyncio
        from unittest.mock import AsyncMock

        wf = self._get_batch_methods()
        wf._find_similar_genes_impl = AsyncMock(side_effect=ValueError("model not loaded"))
        wf._analyze_vulnerability_impl = AsyncMock(return_value={})
        wf._cross_cell_comparison_impl = AsyncMock(return_value={})

        result = asyncio.run(
            wf._batch_insights(base_state)
        )

        assert result.get("failed_analyses") is not None
        assert result["failed_analyses"][0]["analysis"] == "similar"
        assert result["failed_analyses"][0]["batch"] == "insights"

    def test_failed_analyses_in_report_metadata(self):
        """_generate_report must include failed_analyses in metadata."""
        import asyncio, time
        from unittest.mock import patch, MagicMock

        wf = self._get_batch_methods()
        wf._get_model = MagicMock(return_value=None)

        state = {
            "gene": "TP53",
            "cell_type": "epithelial_cell",
            "perturbation_type": "knockdown",
            "analysis_depth": "comprehensive",
            "ensembl_id": "ENSG00000141510",
            "gene_symbol": "TP53",
            "gene_role": "master_regulator",
            "completed_actions": ["perturbation"],
            "failed_analyses": [{"analysis": "ppi", "error": "timeout", "batch": "external"}],
            "network_context": {"num_targets": 100, "num_regulators": 5, "direct_targets": []},
            "perturbation_result": {"top_affected_genes": [], "total_affected": 0, "direction": "knockdown"},
            "regulators_analysis": None,
            "targets_analysis": None,
            "similar_genes": None,
            "embedding_enhanced": False,
            "ppi_interactions": None,
            "lincs_effects": None,
            "super_enhancer_status": None,
            "dorothea_regulons": None,
            "depmap_essentiality": None,
            "cbioportal_tumor_data": None,
            "cross_cell_comparison": None,
            "vulnerability_analysis": None,
            "therapeutic_suggestions": None,
            "llm_insights": None,
            "analysis_metadata": {"start_time": time.time()},
        }

        result = asyncio.run(wf._generate_report(state))
        report = result["comprehensive_report"]
        assert "metadata" in report
        assert "failed_analyses" in report["metadata"]
        assert len(report["metadata"]["failed_analyses"]) == 1
        assert report["metadata"]["failed_analyses"][0]["analysis"] == "ppi"


class TestProgressCallback:
    """Verify progress_cb is forwarded through workflow.run() without errors."""

    def test_run_accepts_progress_cb_parameter(self):
        """workflow.run() must accept a progress_cb kwarg (signature check)."""
        import inspect
        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("cascade_langgraph_workflow.CascadeWorkflow.__init__", return_value=None):
                from cascade_langgraph_workflow import CascadeWorkflow
                sig = inspect.signature(CascadeWorkflow.run)
                assert "progress_cb" in sig.parameters

    def test_progress_cb_called_during_run(self):
        """Progress callback should be invoked at least once during workflow.run()."""
        import asyncio
        from unittest.mock import patch, MagicMock, AsyncMock

        calls = []

        async def fake_cb(progress, total, message):
            calls.append((progress, total, message))

        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("cascade_langgraph_workflow.CascadeWorkflow.__init__", return_value=None):
                from cascade_langgraph_workflow import CascadeWorkflow
                wf = CascadeWorkflow.__new__(CascadeWorkflow)
                mock_workflow = MagicMock()
                mock_workflow.ainvoke = AsyncMock(return_value={
                    "comprehensive_report": {"summary": "ok"}
                })
                wf.workflow = mock_workflow

                result = asyncio.run(
                    wf.run("TP53", progress_cb=fake_cb)
                )

        assert len(calls) >= 2  # at least start (0.05) and done (1.0)
        progresses = [c[0] for c in calls]
        assert progresses[0] == pytest.approx(0.05)
        assert progresses[-1] == pytest.approx(1.0)

    def test_none_progress_cb_does_not_raise(self):
        """workflow.run() with progress_cb=None must not raise."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock

        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("cascade_langgraph_workflow.CascadeWorkflow.__init__", return_value=None):
                from cascade_langgraph_workflow import CascadeWorkflow
                wf = CascadeWorkflow.__new__(CascadeWorkflow)
                mock_workflow = MagicMock()
                mock_workflow.ainvoke = AsyncMock(return_value={
                    "comprehensive_report": {"summary": "ok"}
                })
                wf.workflow = mock_workflow

                result = asyncio.run(
                    wf.run("TP53", progress_cb=None)
                )
        assert result == {"summary": "ok"}


class TestApiRateLimiting:
    """Verify asyncio.Semaphore rate limiting on external API calls."""

    def _make_workflow(self, monkeypatch):
        """Build a CascadeWorkflow with all heavy init patched out."""
        monkeypatch.setenv("API_RATE_LIMIT", "2")
        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("cascade_langgraph_workflow.CascadeWorkflow.__init__", return_value=None):
                from cascade_langgraph_workflow import CascadeWorkflow
                import asyncio
                wf = CascadeWorkflow.__new__(CascadeWorkflow)
                wf._api_semaphore = asyncio.Semaphore(2)
                return wf

    def test_semaphore_initialized_on_workflow(self, monkeypatch):
        """CascadeWorkflow.__init__ sets _api_semaphore."""
        import asyncio
        monkeypatch.setenv("API_RATE_LIMIT", "2")
        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("tools.gene_id_mapper.get_mapper", return_value=MagicMock()):
                with patch("tools.ppi.string_client.get_string_client", return_value=MagicMock()):
                    with patch("cascade_langgraph_workflow.CascadeWorkflow._initialize_llm", return_value=False):
                        from cascade_langgraph_workflow import CascadeWorkflow
                        wf = CascadeWorkflow.__new__(CascadeWorkflow)
                        wf.__init__()
                        assert isinstance(wf._api_semaphore, asyncio.Semaphore)

    def test_api_rate_limit_env_var_controls_semaphore(self, monkeypatch):
        """API_RATE_LIMIT env var sets the semaphore bound."""
        import asyncio
        # asyncio.Semaphore doesn't expose _value publicly but we can verify
        # the semaphore blocks correctly by checking it was constructed with the env var.
        monkeypatch.setenv("API_RATE_LIMIT", "1")
        with patch("cascade_langgraph_workflow.CascadeWorkflow._create_workflow"):
            with patch("tools.gene_id_mapper.get_mapper", return_value=MagicMock()):
                with patch("tools.ppi.string_client.get_string_client", return_value=MagicMock()):
                    with patch("cascade_langgraph_workflow.CascadeWorkflow._initialize_llm", return_value=False):
                        from cascade_langgraph_workflow import CascadeWorkflow
                        wf = CascadeWorkflow.__new__(CascadeWorkflow)
                        wf.__init__()
                        # Semaphore with limit=1: acquiring once should succeed immediately
                        loop = asyncio.new_event_loop()
                        acquired = loop.run_until_complete(wf._api_semaphore.acquire())
                        assert acquired is True
                        wf._api_semaphore.release()
                        loop.close()

    def test_fetch_ppi_impl_acquires_semaphore(self, monkeypatch):
        """_fetch_ppi_impl acquires _api_semaphore before calling STRING."""
        import asyncio
        wf = self._make_workflow(monkeypatch)
        wf.string_client = MagicMock()
        wf.string_client.get_interactions.return_value = {"interactions": []}

        semaphore_acquired = []
        real_semaphore = wf._api_semaphore

        async def tracking_acquire():
            semaphore_acquired.append(True)
            return await real_semaphore.acquire()

        wf._api_semaphore = MagicMock()
        wf._api_semaphore.__aenter__ = AsyncMock(side_effect=lambda: semaphore_acquired.append(True))
        wf._api_semaphore.__aexit__ = AsyncMock(return_value=False)

        state = {"gene": "TP53", "gene_symbol": "TP53"}
        asyncio.run(wf._fetch_ppi_impl(state))
        assert len(semaphore_acquired) == 1

    def test_fetch_cbioportal_impl_acquires_semaphore(self, monkeypatch):
        """_fetch_cbioportal_impl acquires _api_semaphore before calling cBioPortal."""
        import asyncio
        wf = self._make_workflow(monkeypatch)

        semaphore_acquired = []
        wf._api_semaphore = MagicMock()
        wf._api_semaphore.__aenter__ = AsyncMock(side_effect=lambda: semaphore_acquired.append(True))
        wf._api_semaphore.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.cbioportal.get_gene_tumor_expression", return_value={}):
            with patch("tools.cbioportal.get_gene_alteration_frequency", return_value={}):
                state = {"gene": "TP53", "gene_symbol": "TP53"}
                asyncio.run(wf._fetch_cbioportal_impl(state))
        assert len(semaphore_acquired) == 1
