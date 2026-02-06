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
