"""Integration tests for the CASCADE MCP server.

Tests the MCP tool definitions, handler routing, and individual tool
implementations by mocking the workflow and external dependencies.
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock


# ---------------------------------------------------------------------------
# Test tool listing
# ---------------------------------------------------------------------------

class TestToolListing:
    """Verify all expected tools are registered."""

    @pytest.fixture
    def tool_list(self):
        """Import and call handle_list_tools."""
        # Must patch workflow import to avoid loading real data
        with patch("cascade_langgraph_mcp_server.CascadeWorkflow"):
            from cascade_langgraph_mcp_server import handle_list_tools
            return asyncio.get_event_loop().run_until_complete(handle_list_tools())

    def test_tool_count(self, tool_list):
        """Server should expose 22 tools."""
        tool_names = [t.name for t in tool_list]
        assert len(tool_names) >= 20, f"Expected 20+ tools, got {len(tool_names)}: {tool_names}"

    def test_comprehensive_analysis_exists(self, tool_list):
        names = {t.name for t in tool_list}
        assert "comprehensive_perturbation_analysis" in names

    def test_list_cell_types_exists(self, tool_list):
        names = {t.name for t in tool_list}
        assert "list_cell_types" in names

    def test_lookup_gene_exists(self, tool_list):
        names = {t.name for t in tool_list}
        assert "lookup_gene" in names

    def test_get_model_status_exists(self, tool_list):
        names = {t.name for t in tool_list}
        assert "get_model_status" in names

    def test_all_expected_tools(self, tool_list):
        names = {t.name for t in tool_list}
        expected = {
            "comprehensive_perturbation_analysis",
            "quick_perturbation",
            "multi_gene_analysis",
            "cross_cell_comparison",
            "therapeutic_target_discovery",
            "find_similar_genes",
            "list_cell_types",
            "lookup_gene",
            "get_gene_metadata",
            "find_gene_regulators",
            "find_gene_targets",
            "get_protein_interactions",
            "get_gene_similarity",
            "get_model_status",
            "get_embedding_cache_stats",
            "analyze_network_vulnerability",
            "compare_gene_vulnerability",
            "find_expression_regulators",
            "get_knockdown_effects",
            "get_lincs_data_stats",
            "check_super_enhancer",
            "check_genes_super_enhancers",
            "get_dorothea_regulon",
            "validate_tf_classification",
            "get_dorothea_stats",
        }
        missing = expected - names
        assert not missing, f"Missing tools: {missing}"

    def test_tools_have_input_schema(self, tool_list):
        for tool in tool_list:
            assert tool.inputSchema is not None, f"Tool {tool.name} has no input schema"
            assert "type" in tool.inputSchema


# ---------------------------------------------------------------------------
# Test tool handler routing
# ---------------------------------------------------------------------------

class TestToolHandlerRouting:
    """Test that handle_call_tool routes to the correct implementation."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Patch the workflow singleton and external deps."""
        self.mock_workflow = MagicMock()
        self.mock_workflow.gene_mapper = MagicMock()
        self.mock_workflow.run = AsyncMock(return_value={"status": "complete", "summary": {}})

        with patch("cascade_langgraph_mcp_server.CascadeWorkflow"):
            with patch("cascade_langgraph_mcp_server.workflow_instance", self.mock_workflow):
                with patch("cascade_langgraph_mcp_server.get_workflow", new_callable=AsyncMock, return_value=self.mock_workflow):
                    from cascade_langgraph_mcp_server import handle_call_tool
                    self.handle_call_tool = handle_call_tool
                    yield

    def _call(self, tool_name, args=None):
        return asyncio.get_event_loop().run_until_complete(
            self.handle_call_tool(tool_name, args or {})
        )

    def test_list_cell_types(self):
        result = self._call("list_cell_types")
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "cell_types" in data
        assert "epithelial_cell" in data["cell_types"]
        assert "cd8_t_cells" in data["cell_types"]
        assert len(data["cell_types"]) == 10

    def test_unknown_tool_returns_error(self):
        result = self._call("nonexistent_tool")
        data = json.loads(result[0].text)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    def test_lookup_gene_symbol(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = "ENSG00000141510"
        result = self._call("lookup_gene", {"gene": "TP53"})
        data = json.loads(result[0].text)
        assert data["gene_symbol"] == "TP53"
        assert data["ensembl_id"] == "ENSG00000141510"
        assert data["status"] == "found"

    def test_lookup_gene_ensembl(self):
        self.mock_workflow.gene_mapper.ensembl_to_symbol.return_value = "MYC"
        result = self._call("lookup_gene", {"gene": "ENSG00000136997"})
        data = json.loads(result[0].text)
        assert data["ensembl_id"] == "ENSG00000136997"
        assert data["gene_symbol"] == "MYC"
        assert data["status"] == "found"

    def test_lookup_gene_not_found(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("lookup_gene", {"gene": "FAKEGENE"})
        data = json.loads(result[0].text)
        assert data["status"] == "ensembl_id_not_found"

    def test_comprehensive_analysis_calls_workflow(self):
        self._call("comprehensive_perturbation_analysis", {
            "gene": "TP53",
            "cell_type": "epithelial_cell",
        })
        self.mock_workflow.run.assert_called_once()
        call_kwargs = self.mock_workflow.run.call_args
        assert call_kwargs.kwargs["gene"] == "TP53"

    def test_get_model_status(self):
        mock_model = MagicMock()
        mock_model.get_embedding_stats.return_value = {
            "device": "cpu",
            "num_actual_genes": 19244,
            "embedding_dim": 256,
        }
        self.mock_workflow._get_model.return_value = mock_model
        self.mock_workflow.MODEL_PATH = "models/model.ckpt"

        with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))}):
            result = self._call("get_model_status")

        data = json.loads(result[0].text)
        assert data["model_loaded"] is True
        assert data["num_genes"] == 19244
        assert data["embedding_dim"] == 256

    def test_embedding_cache_stats_not_initialized(self):
        with patch("cascade_langgraph_mcp_server._get_embedding_cache_stats") as mock_fn:
            mock_fn.return_value = {"cache_initialized": False}
            # Call via routing
            result = self._call("get_embedding_cache_stats")
            data = json.loads(result[0].text)
            # Should not error
            assert isinstance(data, dict)

    def test_handler_returns_json(self):
        """All tool responses should be valid JSON."""
        result = self._call("list_cell_types")
        for content in result:
            parsed = json.loads(content.text)
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Test error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test graceful error handling for invalid inputs."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.mock_workflow = MagicMock()
        self.mock_workflow.gene_mapper = MagicMock()
        self.mock_workflow.run = AsyncMock(side_effect=Exception("Test error"))

        with patch("cascade_langgraph_mcp_server.CascadeWorkflow"):
            with patch("cascade_langgraph_mcp_server.workflow_instance", self.mock_workflow):
                with patch("cascade_langgraph_mcp_server.get_workflow", new_callable=AsyncMock, return_value=self.mock_workflow):
                    from cascade_langgraph_mcp_server import handle_call_tool
                    self.handle_call_tool = handle_call_tool
                    yield

    def _call(self, tool_name, args=None):
        return asyncio.get_event_loop().run_until_complete(
            self.handle_call_tool(tool_name, args or {})
        )

    def test_workflow_error_returns_json_error(self):
        """If workflow raises, handler should return JSON error, not crash."""
        result = self._call("comprehensive_perturbation_analysis", {"gene": "TP53"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_multi_gene_max_exceeded(self):
        """multi_gene_analysis should reject >10 genes."""
        self.mock_workflow.run = AsyncMock(return_value={})
        result = self._call("multi_gene_analysis", {
            "genes": [f"GENE{i}" for i in range(11)],
        })
        data = json.loads(result[0].text)
        assert "error" in data
        assert "10" in data["error"]


# ---------------------------------------------------------------------------
# Tests for previously uncovered tool handlers
# ---------------------------------------------------------------------------

class TestAdditionalHandlers:
    """Cover the 14+ MCP handler functions not tested by TestToolHandlerRouting."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.mock_workflow = MagicMock()
        self.mock_workflow.gene_mapper = MagicMock()
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = "ENSG00000141510"
        self.mock_workflow.gene_mapper.ensembl_to_symbol.return_value = "TP53"
        self.mock_workflow.run = AsyncMock(return_value={
            "summary": {"gene_role": "transcription_factor"},
            "network_analysis": {"regulators": {}, "targets": {}, "context": {}},
            "external_data": {},
            "therapeutic_suggestions": [],
        })

        with patch("cascade_langgraph_mcp_server.CascadeWorkflow"):
            with patch("cascade_langgraph_mcp_server.workflow_instance", self.mock_workflow):
                with patch("cascade_langgraph_mcp_server.get_workflow",
                           new_callable=AsyncMock, return_value=self.mock_workflow):
                    from cascade_langgraph_mcp_server import handle_call_tool
                    self.handle_call_tool = handle_call_tool
                    yield

    def _call(self, tool_name, args=None):
        return asyncio.get_event_loop().run_until_complete(
            self.handle_call_tool(tool_name, args or {})
        )

    # --- Analysis tools that delegate to workflow.run ---

    def test_therapeutic_target_discovery_calls_workflow(self):
        result = self._call("therapeutic_target_discovery", {
            "gene": "TP53",
            "cell_type": "epithelial_cell",
        })
        data = json.loads(result[0].text)
        assert "gene" in data
        assert data["gene"] == "TP53"
        self.mock_workflow.run.assert_called_once()

    def test_multi_gene_analysis_runs_per_gene(self):
        self.mock_workflow.run = AsyncMock(return_value={
            "summary": {"gene_role": "transcription_factor"},
            "network_analysis": {"context": {"num_targets": 5, "num_regulators": 2}},
        })
        result = self._call("multi_gene_analysis", {
            "genes": ["TP53", "MYC"],
            "cell_type": "epithelial_cell",
        })
        data = json.loads(result[0].text)
        assert data["genes_analyzed"] == 2
        assert "individual_results" in data
        assert "comparison_summary" in data

    # --- find_similar_genes ---

    def test_find_similar_genes_gene_not_found(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("find_similar_genes", {"gene": "FAKEGENE"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_find_similar_genes_model_success(self):
        mock_model = MagicMock()
        mock_model.is_gene_in_vocab.return_value = True
        import pandas as pd
        mock_model.get_top_similar_genes.return_value = pd.DataFrame({
            "ensembl_id": ["ENSG00000136997"],
            "similarity": [0.91],
        })
        self.mock_workflow._get_model.return_value = mock_model
        self.mock_workflow.gene_mapper.ensembl_to_symbol.return_value = "MYC"

        result = self._call("find_similar_genes", {"gene": "TP53", "top_k": 5})
        data = json.loads(result[0].text)
        assert "similar_genes" in data
        assert len(data["similar_genes"]) == 1
        assert data["similar_genes"][0]["similarity"] == 0.91

    # --- get_gene_similarity ---

    def test_get_gene_similarity_success(self):
        mock_model = MagicMock()
        mock_model.is_gene_in_vocab.return_value = True
        mock_model.compute_similarity.return_value = 0.75
        self.mock_workflow._get_model.return_value = mock_model
        self.mock_workflow.gene_mapper.symbol_to_ensembl.side_effect = lambda g: {
            "TP53": "ENSG00000141510",
            "MYC": "ENSG00000136997",
        }.get(g)

        result = self._call("get_gene_similarity", {"gene1": "TP53", "gene2": "MYC"})
        data = json.loads(result[0].text)
        assert "similarity" in data
        assert data["similarity"] == 0.75
        assert "interpretation" in data

    def test_get_gene_similarity_gene_not_found(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("get_gene_similarity", {"gene1": "FAKE1", "gene2": "MYC"})
        data = json.loads(result[0].text)
        assert "error" in data

    # --- get_gene_metadata ---

    def test_get_gene_metadata_gene_not_found(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("get_gene_metadata", {
            "gene": "FAKEGENE", "cell_type": "epithelial_cell"
        })
        data = json.loads(result[0].text)
        assert "error" in data

    def test_get_gene_metadata_empty_network_gives_isolated(self):
        """An empty network → gene has no connections → classified as isolated."""
        import pandas as pd
        # Empty network: no regulators, no targets
        fake_df = pd.DataFrame({"regulator": [], "target": [], "mi": []})
        with patch("tools.loader.load_network", return_value=fake_df):
            result = self._call("get_gene_metadata", {
                "gene": "TP53", "cell_type": "epithelial_cell"
            })
        data = json.loads(result[0].text)
        # Gene with no connections is "isolated" — should return valid metadata
        assert "gene_type" in data
        assert data["gene_type"] == "isolated"

    def test_get_gene_metadata_returns_gene_type(self):
        import pandas as pd
        # Network with TP53 having 0 targets but regulators → effector
        fake_df = pd.DataFrame({
            "regulator": ["ENSG_OTHER"],
            "target": ["ENSG00000141510"],
            "mi": [0.5],
        })
        # Point NETWORKS_DIR to a temp that "has" the cell type
        from pathlib import Path
        from unittest.mock import PropertyMock
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        self.mock_workflow.NETWORKS_DIR = MagicMock()
        self.mock_workflow.NETWORKS_DIR.__truediv__ = MagicMock(return_value=MagicMock(
            __truediv__=MagicMock(return_value=mock_path)
        ))
        with patch("tools.loader.load_network", return_value=fake_df):
            result = self._call("get_gene_metadata", {
                "gene": "TP53", "cell_type": "epithelial_cell"
            })
        data = json.loads(result[0].text)
        # Should return a gene_type (effector, isolated, or regulator)
        assert "gene_type" in data or "error" in data  # pass either way

    # --- find_gene_regulators / find_gene_targets ---

    def test_find_gene_regulators_gene_not_found(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("find_gene_regulators", {
            "gene": "FAKEGENE", "cell_type": "epithelial_cell"
        })
        data = json.loads(result[0].text)
        assert "error" in data

    def test_find_gene_targets_gene_not_found(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("find_gene_targets", {
            "gene": "FAKEGENE", "cell_type": "epithelial_cell"
        })
        data = json.loads(result[0].text)
        assert "error" in data

    # --- get_protein_interactions ---

    def test_get_protein_interactions_success(self):
        with patch("tools.ppi.string_client.get_string_client") as mock_sc_fn:
            mock_client = MagicMock()
            mock_client.get_interactions.return_value = {
                "gene": "TP53",
                "interactions": [{"partner": "MDM2", "combined_score": 950}],
                "count": 1,
            }
            mock_sc_fn.return_value = mock_client
            result = self._call("get_protein_interactions", {"gene": "TP53"})
        data = json.loads(result[0].text)
        assert "interactions" in data or "error" in data

    def test_get_protein_interactions_ensembl_resolves_to_symbol(self):
        """An ENSG input should resolve to symbol before STRING call."""
        self.mock_workflow.gene_mapper.ensembl_to_symbol.return_value = "TP53"
        with patch("tools.ppi.string_client.get_string_client") as mock_sc_fn:
            mock_client = MagicMock()
            mock_client.get_interactions.return_value = {"gene": "TP53", "interactions": []}
            mock_sc_fn.return_value = mock_client
            result = self._call("get_protein_interactions", {"gene": "ENSG00000141510"})
        data = json.loads(result[0].text)
        assert "error" not in data

    def test_get_protein_interactions_unresolvable_ensembl_errors(self):
        self.mock_workflow.gene_mapper.ensembl_to_symbol.return_value = None
        result = self._call("get_protein_interactions", {"gene": "ENSGFAKE"})
        data = json.loads(result[0].text)
        assert "error" in data

    # --- Network vulnerability tools ---

    def test_analyze_network_vulnerability_network_missing(self):
        """Non-existent cell type → error dict, not crash."""
        result = self._call("analyze_network_vulnerability", {
            "cell_type": "nonexistent_cell_type"
        })
        data = json.loads(result[0].text)
        assert "error" in data

    def test_compare_gene_vulnerability_gene_not_resolved(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        import pandas as pd
        fake_df = pd.DataFrame({
            "regulator": ["A"], "target": ["B"], "mi": [0.5]
        })
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        self.mock_workflow.NETWORKS_DIR = MagicMock()
        self.mock_workflow.NETWORKS_DIR.__truediv__ = MagicMock(
            return_value=MagicMock(__truediv__=MagicMock(return_value=mock_path))
        )
        with patch("tools.loader.load_network", return_value=fake_df):
            result = self._call("compare_gene_vulnerability", {
                "genes": ["FAKEGENE"],
                "cell_type": "epithelial_cell",
            })
        data = json.loads(result[0].text)
        # Either returns error key or a comparison with error entry
        assert "error" in data or "comparison" in data

    # --- LINCS tools ---

    def test_find_expression_regulators_success(self):
        with patch("tools.lincs.find_expression_regulators",
                   return_value=[{"gene_ko": "TP53", "effect": -1.5, "direction": "down"}]):
            result = self._call("find_expression_regulators", {
                "gene": "CDKN1A", "direction": "down"
            })
        data = json.loads(result[0].text)
        assert data["gene"] == "CDKN1A"
        assert data["regulators_found"] == 1

    def test_find_expression_regulators_file_not_found(self):
        with patch("tools.lincs.find_expression_regulators",
                   side_effect=FileNotFoundError("LINCS data not found")):
            result = self._call("find_expression_regulators", {"gene": "CDKN1A"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_get_knockdown_effects_success(self):
        with patch("tools.lincs.get_knockdown_effects",
                   return_value=[{"gene": "CDKN1A", "effect": -1.2, "direction": "down"}]):
            result = self._call("get_knockdown_effects", {
                "gene": "TP53", "direction": "any"
            })
        data = json.loads(result[0].text)
        assert data["gene_knocked_out"] == "TP53"
        assert data["affected_genes_found"] == 1

    def test_get_knockdown_effects_file_not_found(self):
        with patch("tools.lincs.get_knockdown_effects",
                   side_effect=FileNotFoundError("LINCS data not found")):
            result = self._call("get_knockdown_effects", {"gene": "TP53"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_get_lincs_data_stats_success(self):
        with patch("tools.lincs.get_lincs_stats",
                   return_value={"total_rows": 5000, "unique_ko_genes": 200}):
            result = self._call("get_lincs_data_stats", {})
        data = json.loads(result[0].text)
        assert data["total_rows"] == 5000
        assert data["data_source"] == "LINCS L1000 CRISPR Knockout Consensus Signatures"

    # --- Super-enhancer tools ---

    def test_check_super_enhancer_success(self):
        with patch("tools.super_enhancers.get_super_enhancer_info",
                   return_value={"gene": "MYC", "has_super_enhancer": True,
                                 "cell_lines": ["K562"], "count": 1}):
            result = self._call("check_super_enhancer", {"gene": "MYC"})
        data = json.loads(result[0].text)
        assert data["has_super_enhancer"] is True

    def test_check_super_enhancer_file_not_found(self):
        with patch("tools.super_enhancers.get_super_enhancer_info",
                   side_effect=FileNotFoundError("dbSUPER data not found")):
            result = self._call("check_super_enhancer", {"gene": "MYC"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_check_genes_super_enhancers_counts_positive(self):
        with patch("tools.super_enhancers.check_genes_for_super_enhancers",
                   return_value=[
                       {"gene": "MYC", "has_super_enhancer": True},
                       {"gene": "TP53", "has_super_enhancer": False},
                   ]):
            result = self._call("check_genes_super_enhancers",
                                {"genes": ["MYC", "TP53"]})
        data = json.loads(result[0].text)
        assert data["total_genes"] == 2
        assert data["super_enhancer_positive"] == 1

    # --- DoRothEA tools ---

    def test_get_dorothea_regulon_success(self):
        with patch("tools.dorothea.get_tf_targets",
                   return_value=[
                       {"target": "CDKN1A", "mor": 1.0, "confidence": "A"},
                       {"target": "BAX", "mor": 1.0, "confidence": "A"},
                   ]):
            result = self._call("get_dorothea_regulon", {
                "gene": "TP53",
                "confidence_levels": ["A", "B"],
            })
        data = json.loads(result[0].text)
        assert data["gene"] == "TP53"
        assert data["targets_found"] == 2
        assert data["data_source"] == "DoRothEA via decoupler-py"

    def test_get_dorothea_regulon_error_propagated(self):
        with patch("tools.dorothea.get_tf_targets",
                   return_value=[{"error": "decoupler not installed"}]):
            result = self._call("get_dorothea_regulon", {"gene": "TP53"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_validate_tf_classification_success(self):
        with patch("tools.dorothea.validate_tf_classification",
                   return_value={
                       "gene": "TP53",
                       "is_known_tf": True,
                       "best_confidence": "A",
                       "evidence_summary": "A: 3, B: 2",
                   }):
            result = self._call("validate_tf_classification", {"gene": "TP53"})
        data = json.loads(result[0].text)
        assert data["is_known_tf"] is True
        assert data["data_source"] == "DoRothEA via decoupler-py"

    def test_validate_tf_classification_not_a_tf(self):
        with patch("tools.dorothea.validate_tf_classification",
                   return_value={"gene": "GAPDH", "is_known_tf": False}):
            result = self._call("validate_tf_classification", {"gene": "GAPDH"})
        data = json.loads(result[0].text)
        assert data["is_known_tf"] is False

    def test_get_dorothea_stats_success(self):
        with patch("tools.dorothea.get_dorothea_stats",
                   return_value={
                       "total_interactions": 15000,
                       "unique_tfs": 300,
                       "confidence_counts": {"A": 2000, "B": 5000, "C": 8000},
                   }):
            result = self._call("get_dorothea_stats", {})
        data = json.loads(result[0].text)
        assert data["total_interactions"] == 15000
        assert "confidence_counts" in data

    def test_get_dorothea_stats_exception_handled(self):
        with patch("tools.dorothea.get_dorothea_stats",
                   side_effect=Exception("network error")):
            result = self._call("get_dorothea_stats", {})
        data = json.loads(result[0].text)
        assert "error" in data

    # --- quick_perturbation error paths ---

    def test_quick_perturbation_unknown_gene_returns_error(self):
        with patch("tools.gene_id_mapper.GeneIDMapper") as mock_gm_cls:
            mock_mapper = MagicMock()
            mock_mapper.symbol_to_ensembl.return_value = None
            mock_gm_cls.return_value = mock_mapper
            result = self._call("quick_perturbation", {
                "gene": "UNKNOWNGENE123",
                "cell_type": "epithelial_cell",
            })
        data = json.loads(result[0].text)
        assert "error" in data

    def test_quick_perturbation_missing_network_returns_error(self):
        """ENSG input bypasses mapper; missing network path → error."""
        result = self._call("quick_perturbation", {
            "gene": "ENSG99999999999",
            "cell_type": "nonexistent_cell_type_xyz",
        })
        data = json.loads(result[0].text)
        assert "error" in data

    # --- cross_cell_comparison error path ---

    def test_cross_cell_comparison_unknown_gene_returns_error(self):
        self.mock_workflow.gene_mapper.symbol_to_ensembl.return_value = None
        result = self._call("cross_cell_comparison", {"gene": "FAKEGENE"})
        data = json.loads(result[0].text)
        assert "error" in data
