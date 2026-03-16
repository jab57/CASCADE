#!/usr/bin/env python3
"""
CASCADE LangGraph-Powered MCP Server - SKETCH/DRAFT
====================================================

Model Context Protocol (MCP) server that exposes CASCADE perturbation analysis
through LangGraph workflow orchestration.

This server provides high-level analysis tools that automatically orchestrate
multiple analysis steps, run independent analyses in parallel, and generate
comprehensive reports.

Available Tools:
    1. comprehensive_perturbation_analysis - Full workflow-driven analysis (recommended)
    2. quick_perturbation - Fast knockdown/overexpression with minimal context
    3. multi_gene_analysis - Parallel analysis of multiple genes
    4. cross_cell_comparison - Compare gene across all cell types
    5. therapeutic_target_discovery - Find drug targets for a pathway/gene set
    6. workflow_status - Check analysis progress

vs Original MCP Server:
    - Original: 20+ individual tools, user/LLM orchestrates
    - LangGraph: 6 high-level tools, automatic orchestration

Author: [Your name]
License: MIT
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Sequence

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio as stdio
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    ResourceTemplate,
    TextResourceContents,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    Prompt,
    PromptArgument,
    PromptMessage,
    GetPromptResult,
)
from pydantic import AnyUrl
import mcp.types as types
from mcp.server.lowlevel.server import request_ctx

# Import our LangGraph workflow
from cascade_langgraph_workflow import CascadeWorkflow, CellType, PerturbationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cell type validation helper
_VALID_CELL_TYPES = [ct.value for ct in CellType]

def _validate_cell_type(cell_type: str) -> dict | None:
    """Return an error dict if cell_type is invalid, else None."""
    if cell_type not in _VALID_CELL_TYPES:
        return {
            "error": f"Invalid cell type: '{cell_type}'. "
                     f"Valid options are: {', '.join(_VALID_CELL_TYPES)}"
        }
    return None


# Initialize the server
server = Server("cascade-langgraph-server")

# Global workflow instance (singleton)
workflow_instance = None


async def get_workflow():
    """Get or create the global workflow instance."""
    global workflow_instance
    if workflow_instance is None:
        logger.info("Initializing CASCADE LangGraph workflow...")
        workflow_instance = CascadeWorkflow()
        logger.info("Workflow ready")
    return workflow_instance


# =============================================================================
# RESOURCE DEFINITIONS
# =============================================================================

@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List static discoverable resources."""
    return [
        Resource(
            uri=AnyUrl("cascade://cell-types"),
            name="Available Cell Types",
            description="All 10 supported cell types with their regulatory networks",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("cascade://lincs/summary"),
            name="LINCS Dataset Summary",
            description="Coverage stats for the LINCS L1000 experimental knockdown dataset",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("cascade://model/status"),
            name="Model Status",
            description="GREmLN embedding model checkpoint status and GPU availability",
            mimeType="application/json",
        ),
    ]


@server.list_resource_templates()
async def handle_list_resource_templates() -> list[ResourceTemplate]:
    """List URI templates for dynamic resources."""
    return [
        ResourceTemplate(
            uriTemplate="cascade://network/{cell_type}/summary",
            name="Network Summary",
            description="Edge count, gene count, and top hub regulators for a cell type network",
            mimeType="application/json",
        ),
        ResourceTemplate(
            uriTemplate="cascade://gene/{symbol}/{cell_type}",
            name="Gene Metadata",
            description="Gene role, target count, and regulator count in a given cell type network",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> list[TextResourceContents]:
    """Dispatch resource reads by URI pattern."""
    uri_str = str(uri)

    if uri_str == "cascade://cell-types":
        data = await _resource_cell_types()
    elif uri_str == "cascade://lincs/summary":
        data = await _resource_lincs_summary()
    elif uri_str == "cascade://model/status":
        data = await _resource_model_status()
    elif uri_str.startswith("cascade://network/") and uri_str.endswith("/summary"):
        cell_type = uri_str[len("cascade://network/"):-len("/summary")]
        data = await _resource_network_summary(cell_type)
    elif uri_str.startswith("cascade://gene/"):
        parts = uri_str[len("cascade://gene/"):].split("/")
        if len(parts) == 2:
            symbol, cell_type = parts
            data = await _resource_gene_metadata(symbol, cell_type)
        else:
            data = {"error": "Invalid URI. Expected cascade://gene/{symbol}/{cell_type}"}
    else:
        data = {"error": f"Unknown resource URI: {uri_str}"}

    return [TextResourceContents(
        uri=uri,
        text=json.dumps(data, indent=2, default=str),
        mimeType="application/json",
    )]


# =============================================================================
# PROMPT DEFINITIONS
# =============================================================================

@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """List available MCP prompt templates for common analysis workflows."""
    return [
        Prompt(
            name="quick_knockdown",
            description=(
                "What happens if we knock down a gene? Runs a fast perturbation "
                "simulation and returns predicted downstream effects."
            ),
            arguments=[
                PromptArgument(
                    name="gene",
                    description="Gene symbol or Ensembl ID (e.g. TP53, MYC, ENSG00000141510)",
                    required=True,
                ),
                PromptArgument(
                    name="cell_type",
                    description=(
                        "Cell type context. One of: epithelial_cell, cd4_t_cells, "
                        "cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, "
                        "nk_cells, nkt_cells, erythrocytes, "
                        "monocyte-derived_dendritic_cells. Defaults to epithelial_cell."
                    ),
                    required=False,
                ),
            ],
        ),
        Prompt(
            name="comprehensive_gene_analysis",
            description=(
                "Comprehensively analyze a gene: network propagation, protein "
                "interactions, experimental knockdown data, super-enhancer status, "
                "DepMap essentiality, and cross-source evidence synthesis."
            ),
            arguments=[
                PromptArgument(
                    name="gene",
                    description="Gene symbol or Ensembl ID (e.g. TP53, MYC, ENSG00000141510)",
                    required=True,
                ),
                PromptArgument(
                    name="cell_type",
                    description=(
                        "Cell type context. One of: epithelial_cell, cd4_t_cells, "
                        "cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, "
                        "nk_cells, nkt_cells, erythrocytes, "
                        "monocyte-derived_dendritic_cells."
                    ),
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="drug_target_discovery",
            description=(
                "A gene is mutated or dysregulated in disease. Find upstream "
                "regulators and protein partners that could serve as drug targets, "
                "combining network analysis with STRING PPI and LINCS data."
            ),
            arguments=[
                PromptArgument(
                    name="gene",
                    description="Disease-relevant gene symbol or Ensembl ID (e.g. KRAS, BRAF)",
                    required=True,
                ),
                PromptArgument(
                    name="cell_type",
                    description=(
                        "Cell type context. One of: epithelial_cell, cd4_t_cells, "
                        "cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, "
                        "nk_cells, nkt_cells, erythrocytes, "
                        "monocyte-derived_dendritic_cells."
                    ),
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="cross_cell_comparison",
            description=(
                "How does a gene behave across different cell types? Compares "
                "perturbation effects and network role across all 10 supported "
                "cell type regulatory networks."
            ),
            arguments=[
                PromptArgument(
                    name="gene",
                    description="Gene symbol or Ensembl ID to compare across cell types",
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="gene_set_analysis",
            description=(
                "Analyze and compare a set of genes in parallel. Useful for "
                "pathway members, co-expressed genes, or a hit list from a screen."
            ),
            arguments=[
                PromptArgument(
                    name="genes",
                    description="Comma-separated gene symbols or Ensembl IDs (e.g. TP53,MYC,CDKN1A)",
                    required=True,
                ),
                PromptArgument(
                    name="cell_type",
                    description=(
                        "Cell type context. One of: epithelial_cell, cd4_t_cells, "
                        "cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, "
                        "nk_cells, nkt_cells, erythrocytes, "
                        "monocyte-derived_dendritic_cells."
                    ),
                    required=True,
                ),
            ],
        ),
    ]


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Return a filled prompt message for the requested template."""
    args = arguments or {}

    if name == "quick_knockdown":
        gene = args.get("gene", "{gene}")
        cell_type = args.get("cell_type", "epithelial_cell")
        return GetPromptResult(
            description=f"Quick knockdown simulation for {gene}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Use the quick_perturbation tool to simulate knocking down {gene} "
                            f"in {cell_type} cells. Report the top predicted downstream effects, "
                            f"noting any genes with strong up- or down-regulation."
                        ),
                    ),
                )
            ],
        )

    elif name == "comprehensive_gene_analysis":
        gene = args.get("gene", "{gene}")
        cell_type = args.get("cell_type", "{cell_type}")
        return GetPromptResult(
            description=f"Comprehensive analysis of {gene} in {cell_type}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Run a comprehensive perturbation analysis of {gene} in "
                            f"{cell_type} cells using the comprehensive_perturbation_analysis "
                            f"tool. Summarize: (1) network propagation effects, (2) protein "
                            f"interaction partners from STRING, (3) experimental knockdown "
                            f"data from LINCS, (4) super-enhancer and DepMap essentiality "
                            f"status, and (5) the cross-source evidence synthesis highlighting "
                            f"genes corroborated by multiple data sources."
                        ),
                    ),
                )
            ],
        )

    elif name == "drug_target_discovery":
        gene = args.get("gene", "{gene}")
        cell_type = args.get("cell_type", "{cell_type}")
        return GetPromptResult(
            description=f"Drug target discovery for {gene} in {cell_type}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"{gene} is mutated or dysregulated in disease. Use the "
                            f"therapeutic_target_discovery tool in {cell_type} cells to "
                            f"identify upstream regulators and protein interaction partners "
                            f"that could serve as drug targets. Highlight candidates "
                            f"supported by multiple evidence sources and flag any with "
                            f"known BET/BRD4 super-enhancer associations."
                        ),
                    ),
                )
            ],
        )

    elif name == "cross_cell_comparison":
        gene = args.get("gene", "{gene}")
        return GetPromptResult(
            description=f"Cross-cell-type comparison for {gene}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Use the cross_cell_comparison tool to compare how {gene} "
                            f"behaves across all supported cell types. Identify which cell "
                            f"types show the strongest perturbation effects and where the "
                            f"gene plays distinct regulatory roles."
                        ),
                    ),
                )
            ],
        )

    elif name == "gene_set_analysis":
        genes = args.get("genes", "{genes}")
        cell_type = args.get("cell_type", "{cell_type}")
        gene_list = [g.strip() for g in genes.split(",") if g.strip()]
        return GetPromptResult(
            description=f"Gene set analysis for {len(gene_list)} genes in {cell_type}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Use the multi_gene_analysis tool to analyze the following "
                            f"genes in {cell_type} cells: {genes}. Compare their network "
                            f"roles, predicted perturbation effects, and identify any "
                            f"functional relationships between them."
                        ),
                    ),
                )
            ],
        )

    else:
        return GetPromptResult(
            description="Unknown prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Unknown prompt template: {name}",
                    ),
                )
            ],
        )


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available LangGraph-powered tools."""

    cell_type_enum = [ct.value for ct in CellType]
    tcga_network_enum = ["brca", "coad", "hnsc", "luad", "lusc", "ov", "prad", "ucec"]

    return [
        Tool(
            name="comprehensive_perturbation_analysis",
            description="""
            Analyze what happens when a gene is knocked down or overexpressed.

            This is the RECOMMENDED tool for most analyses. It automatically:
            - Determines if the gene is a transcription factor or effector
            - Runs the appropriate analyses based on gene type
            - Fetches protein interactions for non-TF genes
            - Checks experimental data (LINCS knockdown effects)
            - Validates essentiality with DepMap CRISPR screens (Chronos scores across 1000+ cancer cell lines)
            - Finds similar genes using AI embeddings
            - Generates therapeutic targeting suggestions

            Perfect for questions like:
            - "What happens if we knock down TP53?"
            - "What are the effects of MYC overexpression?"
            - "Is APC a good drug target?"

            Returns a comprehensive report with all findings including DepMap essentiality validation.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol (e.g., TP53, MYC, APC) or Ensembl ID"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "description": "Cell type for network analysis",
                        "default": "epithelial_cell"
                    },
                    "perturbation_type": {
                        "type": "string",
                        "enum": ["knockdown", "overexpression"],
                        "description": "Type of perturbation to simulate",
                        "default": "knockdown"
                    },
                    "analysis_depth": {
                        "type": "string",
                        "enum": ["basic", "comprehensive", "focused"],
                        "description": "How deep to analyze (basic=fast, comprehensive=full, focused=gene-type-specific)",
                        "default": "comprehensive"
                    },
                    "include_llm_insights": {
                        "type": "boolean",
                        "description": "Generate LLM-powered biological interpretation (requires Ollama, adds latency)",
                        "default": False
                    },
                    "network_source": {
                        "type": "string",
                        "enum": ["cell_type", "tcga"],
                        "description": "Use 'tcga' for tumor-state regulatory network; requires tcga_network param",
                        "default": "cell_type"
                    },
                    "tcga_network": {
                        "type": "string",
                        "enum": tcga_network_enum,
                        "description": "TCGA cancer type network (required when network_source=tcga). Epithelial-origin only."
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="quick_perturbation",
            description="""
            Fast perturbation analysis - just the core knockdown/overexpression effects.

            Use this when you only need the direct effects without all the extras.
            Much faster than comprehensive analysis (~1-2 seconds vs ~5-10 seconds).

            Returns:
            - Affected genes and their predicted expression changes
            - Whether embeddings enhanced the prediction

            Use comprehensive_perturbation_analysis if you also want:
            - Protein interactions
            - LINCS experimental data
            - Similar genes
            - Therapeutic suggestions
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol or Ensembl ID"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "description": "Cell type for network analysis",
                        "default": "epithelial_cell"
                    },
                    "perturbation_type": {
                        "type": "string",
                        "enum": ["knockdown", "overexpression"],
                        "default": "knockdown"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Network propagation depth (1=direct, 2=indirect)",
                        "default": 2
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top affected genes to return",
                        "default": 25
                    },
                    "network_source": {
                        "type": "string",
                        "enum": ["cell_type", "tcga"],
                        "description": "Use 'tcga' for tumor-state regulatory network; requires tcga_network param",
                        "default": "cell_type"
                    },
                    "tcga_network": {
                        "type": "string",
                        "enum": tcga_network_enum,
                        "description": "TCGA cancer type network (required when network_source=tcga). Epithelial-origin only."
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="multi_gene_analysis",
            description="""
            Analyze multiple genes at once - MUCH faster than analyzing one at a time.

            All genes are analyzed in PARALLEL, so analyzing 5 genes takes about
            the same time as analyzing 1 gene.

            Perfect for:
            - Comparing related genes (TP53, MDM2, CDKN1A)
            - Analyzing a gene panel or signature
            - Batch processing research gene lists

            Example: Analyze [TP53, BRCA1, APC, MYC] to compare their network roles

            Returns individual reports for each gene plus a comparison summary.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene symbols or Ensembl IDs (max 10)",
                        "maxItems": 10
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    },
                    "analysis_depth": {
                        "type": "string",
                        "enum": ["basic", "comprehensive"],
                        "description": "Use 'basic' for fastest results",
                        "default": "basic"
                    }
                },
                "required": ["genes"]
            }
        ),

        Tool(
            name="cross_cell_comparison",
            description="""
            Compare how a gene behaves across ALL available cell types.

            Shows whether the gene is:
            - A regulator in some cells but a target in others
            - More/less connected in specific cell types
            - Active in immune cells vs epithelial cells

            Great for understanding tissue-specific gene regulation.

            Available cell types: epithelial cells, CD4/CD8 T cells, B cells,
            NK cells, monocytes, dendritic cells, erythrocytes.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol or Ensembl ID"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="therapeutic_target_discovery",
            description="""
            Find the best drug targets to modulate a gene of interest.

            If you can't drug a gene directly, this finds:
            - Upstream regulators that control it (knock them down to reduce target)
            - Protein interaction partners (disrupt the complex)
            - Super-enhancer status (use BET inhibitors)
            - Network vulnerability scores (prioritize high-impact targets)

            Example: "APC is mutated in colon cancer but hard to drug directly.
            What upstream regulators could we target instead?"

            Returns ranked list of therapeutic targets with rationale.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene you want to modulate"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    },
                    "goal": {
                        "type": "string",
                        "enum": ["reduce_expression", "increase_expression", "find_alternatives"],
                        "description": "What you're trying to achieve",
                        "default": "reduce_expression"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="find_similar_genes",
            description="""
            Find genes that behave similarly based on AI-learned embeddings.

            The GREmLN model learned gene representations from 11 million cells.
            Similar genes often:
            - Participate in the same pathways
            - Have related functions
            - Respond similarly to perturbations

            Useful for:
            - Finding backup targets if your primary target fails
            - Discovering new pathway members
            - Understanding gene function through guilt-by-association
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol or Ensembl ID"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar genes to return",
                        "default": 20
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="list_cell_types",
            description="""
            List all available cell types with pre-computed regulatory networks.

            Use this to see what cell types are available for analysis.
            """,
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # =====================================================================
        # UTILITY TOOLS
        # =====================================================================

        Tool(
            name="lookup_gene",
            description="""
            Convert between gene symbol and Ensembl ID.

            Use this to:
            - Convert a gene symbol (like TP53) to Ensembl ID (ENSG00000141510)
            - Convert an Ensembl ID back to gene symbol
            - Verify a gene exists in our database
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol (e.g., TP53) or Ensembl ID (e.g., ENSG00000141510)"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_gene_metadata",
            description="""
            Get detailed information about a gene's role in the regulatory network.

            Returns:
            - Gene type (master_regulator, transcription_factor, effector, isolated)
            - Number of targets it regulates
            - Number of regulators controlling it
            - Recommended analysis tools based on gene type
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol or Ensembl ID"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="find_gene_regulators",
            description="""
            Find all upstream regulators (transcription factors) that control a gene.

            Use this to understand what controls a gene's expression.
            Returns regulators ranked by regulatory strength (mutual information).
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Target gene to find regulators for"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    },
                    "max_regulators": {
                        "type": "integer",
                        "description": "Maximum regulators to return",
                        "default": 50
                    },
                    "network_source": {
                        "type": "string",
                        "enum": ["cell_type", "tcga"],
                        "description": "Use 'tcga' for tumor-state regulatory network; requires tcga_network param",
                        "default": "cell_type"
                    },
                    "tcga_network": {
                        "type": "string",
                        "enum": tcga_network_enum,
                        "description": "TCGA cancer type network (required when network_source=tcga). Epithelial-origin only."
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="find_gene_targets",
            description="""
            Find all downstream targets that a gene regulates.

            Use this to see what genes are controlled by a transcription factor.
            Returns targets ranked by regulatory strength.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Regulator gene to find targets for"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    },
                    "max_targets": {
                        "type": "integer",
                        "description": "Maximum targets to return",
                        "default": 50
                    },
                    "network_source": {
                        "type": "string",
                        "enum": ["cell_type", "tcga"],
                        "description": "Use 'tcga' for tumor-state regulatory network; requires tcga_network param",
                        "default": "cell_type"
                    },
                    "tcga_network": {
                        "type": "string",
                        "enum": tcga_network_enum,
                        "description": "TCGA cancer type network (required when network_source=tcga). Epithelial-origin only."
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_protein_interactions",
            description="""
            Get protein-protein interactions from STRING database.

            Use this for genes that aren't transcription factors - they often
            function through protein interactions rather than transcriptional regulation.

            Returns interaction partners with confidence scores and evidence types.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol"
                    },
                    "min_score": {
                        "type": "integer",
                        "description": "Minimum confidence (150=low, 400=medium, 700=high, 900=highest)",
                        "default": 400
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum interactions to return",
                        "default": 25
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_gene_similarity",
            description="""
            Get embedding similarity between two specific genes.

            Returns a similarity score from -1 to 1 where:
            - 1 = identical behavior
            - 0 = unrelated
            - negative = opposite behavior
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene1": {
                        "type": "string",
                        "description": "First gene"
                    },
                    "gene2": {
                        "type": "string",
                        "description": "Second gene"
                    }
                },
                "required": ["gene1", "gene2"]
            }
        ),

        Tool(
            name="get_model_status",
            description="""
            Check CASCADE model and GPU status.

            Returns:
            - Whether model is loaded
            - GPU availability
            - Number of genes in vocabulary
            - Embedding dimensions
            """,
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        Tool(
            name="get_embedding_cache_stats",
            description="""
            Get statistics about the embedding similarity cache.

            Shows cache hit rate and size - useful for performance monitoring.
            """,
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # =====================================================================
        # NETWORK VULNERABILITY TOOLS
        # =====================================================================

        Tool(
            name="analyze_network_vulnerability",
            description="""
            Find critical hub genes in the network - potential drug targets.

            Analyzes network topology to identify genes whose disruption would
            cause maximum downstream impact. High vulnerability = good drug target.

            Returns genes ranked by:
            - Hub score (direct targets)
            - Cascade reach (indirect effects)
            - Vulnerability score (combined metric)
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top vulnerable genes",
                        "default": 20
                    }
                },
                "required": []
            }
        ),

        Tool(
            name="compare_gene_vulnerability",
            description="""
            Compare vulnerability scores for specific genes of interest.

            Use this to evaluate candidate drug targets or compare disease genes.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of genes to compare"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": cell_type_enum,
                        "default": "epithelial_cell"
                    }
                },
                "required": ["genes"]
            }
        ),

        # =====================================================================
        # LINCS EXPERIMENTAL DATA TOOLS
        # =====================================================================

        Tool(
            name="find_expression_regulators",
            description="""
            Find genes whose knockdown affects a target gene's expression (LINCS data).

            This uses EXPERIMENTAL data from CRISPR knockdowns, not just network predictions.
            Captures effects that transcriptional networks may miss (epigenetic, post-translational).

            Example: find_expression_regulators("CDKN1A") returns TP53 because
            TP53 knockdown reduces CDKN1A expression.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Target gene to find regulators for"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["down", "up", "any"],
                        "description": "Filter by effect direction",
                        "default": "any"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 20
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_knockdown_effects",
            description="""
            Find genes affected when a specific gene is knocked out (LINCS data).

            The inverse of find_expression_regulators - shows what changes when
            you knock down a gene.

            Example: get_knockdown_effects("TP53") shows CDKN1A goes down.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene that was knocked out"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["down", "up", "any"],
                        "default": "any"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 20
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_lincs_data_stats",
            description="""
            Get statistics about the LINCS L1000 perturbation dataset.

            Shows number of genes, knockdowns, and associations available.
            """,
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # =====================================================================
        # SUPER-ENHANCER TOOLS
        # =====================================================================

        Tool(
            name="check_super_enhancer",
            description="""
            Check if a gene has super-enhancers (BRD4/BET inhibitor sensitivity).

            Super-enhancers drive high expression of key genes. If a gene can't
            be targeted directly, BET inhibitors (JQ1, OTX015) may reduce its
            expression if it has super-enhancers.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol to check"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="check_genes_super_enhancers",
            description="""
            Check multiple genes for super-enhancer associations.

            Batch version of check_super_enhancer - screen a gene list for
            BET inhibitor sensitivity.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene symbols"
                    }
                },
                "required": ["genes"]
            }
        ),

        # =====================================================================
        # DOROTHEA TF REGULON TOOLS
        # =====================================================================

        Tool(
            name="get_dorothea_regulon",
            description="""
            Get curated TF regulon targets from DoRothEA.

            DoRothEA provides multi-evidence transcription factor regulons with
            confidence levels A-E (A=highest: literature + ChIP-seq + motifs).

            Use this to:
            - Find known targets of a transcription factor
            - Validate network-predicted TF-target relationships
            - Get confidence-weighted regulatory evidence

            Returns targets with mode of regulation (activation/repression)
            and confidence level.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Transcription factor gene symbol (e.g., TP53, MYC)"
                    },
                    "confidence_levels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                        "description": "Confidence levels to include (default: A, B, C)",
                        "default": ["A", "B", "C"]
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="validate_tf_classification",
            description="""
            Validate whether a gene is a known transcription factor using DoRothEA.

            CASCADE classifies TFs using network topology (target count thresholds).
            This tool cross-references against DoRothEA's curated multi-evidence
            TF regulons to confirm or refute the classification.

            Returns:
            - Whether the gene is a known TF in DoRothEA
            - Best confidence level (A-E)
            - Number of targets per confidence level
            - Evidence summary
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol to validate as TF"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_dorothea_stats",
            description="""
            Get statistics about the DoRothEA TF regulon dataset.

            Shows total interactions, unique TFs, unique targets, and
            interaction counts per confidence level.
            """,
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # =====================================================================
        # DEPMAP ESSENTIALITY TOOLS
        # =====================================================================

        Tool(
            name="get_depmap_essentiality",
            description="""
            Get DepMap CRISPR essentiality profile for a gene across cancer cell lines.

            Uses Chronos gene effect scores from DepMap CRISPR screens across 1,000+
            cancer cell lines. Negative scores indicate essentiality:
              score < -0.5  → essential in that cell line
              score < -1.0  → strongly essential

            Returns:
            - Mean/median/std Chronos score across all tested lines
            - Fraction of lines where the gene is essential (< -0.5)
            - Fraction where strongly essential (< -1.0)
            - pan_cancer_essential flag (essential in >50% of lines)
            - common_essential flag (essential in >90%, housekeeping-like)
            - Top 5 cancer lineages most dependent on this gene

            Use this to:
            - Assess cancer dependency / therapeutic vulnerability
            - Identify lineage-specific essentiality
            - Distinguish oncogene addiction from housekeeping essentiality
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol (e.g., MYC, TP53, KRAS)"
                    }
                },
                "required": ["gene"]
            }
        ),

        # =====================================================================
        # CBIOPORTAL TCGA PRIMARY TUMOR TOOLS
        # =====================================================================

        Tool(
            name="get_tumor_expression_profile",
            description="""
            Get mRNA expression z-scores for a gene across TCGA PanCancer Atlas cancer types.

            Queries primary tumor RNA-seq data from the TCGA PanCancer Atlas 2018
            (~10,000 samples, 32 cancer types) via cBioPortal. Returns mean z-scores
            per cancer type, identifying where a gene is overexpressed or underexpressed
            in primary patient tumors — complementing cell-line data from DepMap and LINCS.

            Returns:
            - top_overexpressed: cancer types where gene is most overexpressed (z-score ranked)
            - top_underexpressed: cancer types where gene is most underexpressed
            - pan_cancer_mean_z: mean z-score across all cancer types (>1.0 = overexpressed)
            - num_cancer_types_queried: number of TCGA cancer type studies with data

            Use this to:
            - Confirm cell-line findings in primary patient tissue
            - Identify cancer lineages with highest expression (therapeutic targeting)
            - Distinguish housekeeping expression from cancer-specific overexpression
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol (e.g., MYC, TP53, BRCA1)"
                    }
                },
                "required": ["gene"]
            }
        ),

        Tool(
            name="get_tumor_alteration_frequency",
            description="""
            Get somatic mutation and copy-number alteration (CNA) frequency for a gene
            across TCGA PanCancer Atlas cancer types.

            Queries mutation and GISTIC discrete CNA data from the TCGA PanCancer Atlas 2018
            via cBioPortal. Returns pan-cancer and per-cancer rates for somatic mutations,
            high-level amplification (GISTIC=2), and homozygous deletion (GISTIC=-2).

            Returns:
            - mutation_frequency_pct: pan-cancer % of sequenced samples with any somatic mutation
            - amplification_frequency_pct: pan-cancer % of samples with high-level amplification
            - deletion_frequency_pct: pan-cancer % of samples with homozygous deletion
            - most_altered_cancer_type: cancer type with highest combined alteration rate
            - alteration_by_cancer: per-cancer breakdown of mutation, amplification, deletion rates

            Use this to:
            - Identify cancer lineages with highest mutation/amplification burden
            - Assess whether a gene is a recurrent driver (high somatic alteration rate)
            - Guide cancer-type-specific therapeutic targeting
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol (e.g., MYC, TP53, KRAS)"
                    }
                },
                "required": ["gene"]
            }
        )
    ]


# =============================================================================
# TOOL HANDLERS
# =============================================================================

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool execution requests."""

    async def _maybe_send_progress(progress: float, total: float, message: str) -> None:
        try:
            ctx = request_ctx.get()
            token = ctx.meta.progressToken if ctx.meta else None
            if token is not None:
                await ctx.session.send_progress_notification(
                    progress_token=token,
                    progress=progress,
                    total=total,
                    message=message,
                )
        except LookupError:
            pass
        except Exception as e:
            logger.debug(f"Progress notification failed (non-fatal): {e}")

    try:
        if name == "comprehensive_perturbation_analysis":
            result = await _comprehensive_analysis(arguments, progress_cb=_maybe_send_progress)

        elif name == "quick_perturbation":
            result = await _quick_perturbation(arguments)

        elif name == "multi_gene_analysis":
            result = await _multi_gene_analysis(arguments)

        elif name == "cross_cell_comparison":
            result = await _cross_cell_comparison(arguments)

        elif name == "therapeutic_target_discovery":
            result = await _therapeutic_discovery(arguments)

        elif name == "find_similar_genes":
            result = await _find_similar(arguments)

        elif name == "list_cell_types":
            result = {"cell_types": [ct.value for ct in CellType]}

        # Utility tools
        elif name == "lookup_gene":
            result = await _lookup_gene(arguments)

        elif name == "get_gene_metadata":
            result = await _get_gene_metadata(arguments)

        elif name == "find_gene_regulators":
            result = await _find_gene_regulators(arguments)

        elif name == "find_gene_targets":
            result = await _find_gene_targets(arguments)

        elif name == "get_protein_interactions":
            result = await _get_protein_interactions(arguments)

        elif name == "get_gene_similarity":
            result = await _get_gene_similarity(arguments)

        elif name == "get_model_status":
            result = await _get_model_status(arguments)

        elif name == "get_embedding_cache_stats":
            result = await _get_embedding_cache_stats(arguments)

        # Network vulnerability tools
        elif name == "analyze_network_vulnerability":
            result = await _analyze_network_vulnerability(arguments)

        elif name == "compare_gene_vulnerability":
            result = await _compare_gene_vulnerability(arguments)

        # LINCS tools
        elif name == "find_expression_regulators":
            result = await _find_expression_regulators(arguments)

        elif name == "get_knockdown_effects":
            result = await _get_knockdown_effects(arguments)

        elif name == "get_lincs_data_stats":
            result = await _get_lincs_data_stats(arguments)

        # Super-enhancer tools
        elif name == "check_super_enhancer":
            result = await _check_super_enhancer(arguments)

        elif name == "check_genes_super_enhancers":
            result = await _check_genes_super_enhancers(arguments)

        # DoRothEA tools
        elif name == "get_dorothea_regulon":
            result = await _get_dorothea_regulon(arguments)

        elif name == "validate_tf_classification":
            result = await _validate_tf_classification(arguments)

        elif name == "get_dorothea_stats":
            result = await _get_dorothea_stats(arguments)

        # DepMap tools
        elif name == "get_depmap_essentiality":
            result = await _get_depmap_essentiality(arguments)

        # cBioPortal tools
        elif name == "get_tumor_expression_profile":
            result = await _get_tumor_expression_profile(arguments)

        elif name == "get_tumor_alteration_frequency":
            result = await _get_tumor_alteration_frequency(arguments)

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]

    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]


async def _comprehensive_analysis(args: dict, progress_cb=None) -> dict:
    """Run comprehensive perturbation analysis via workflow."""
    network_source = args.get("network_source", "cell_type")
    tcga_network = args.get("tcga_network")

    if network_source == "tcga":
        if not tcga_network:
            return {"error": "tcga_network is required when network_source='tcga'. "
                             "Valid options: brca, coad, hnsc, luad, lusc, ov, prad, ucec"}
    else:
        cell_type = args.get("cell_type", "epithelial_cell")
        if err := _validate_cell_type(cell_type):
            return err

    workflow = await get_workflow()

    return await workflow.run(
        gene=args["gene"],
        cell_type=args.get("cell_type", "epithelial_cell"),
        perturbation_type=args.get("perturbation_type", "knockdown"),
        analysis_depth=args.get("analysis_depth", "comprehensive"),
        include_llm_insights=args.get("include_llm_insights", False),
        network_source=network_source,
        tcga_network=tcga_network,
        progress_cb=progress_cb,
    )


async def _quick_perturbation(args: dict) -> dict:
    """Run quick perturbation without full workflow."""
    from tools.loader import load_network, load_tcga_network
    from tools.perturb import (
        simulate_knockdown_with_embeddings,
        simulate_overexpression_with_embeddings,
        simulate_knockdown,
        simulate_overexpression
    )

    workflow = await get_workflow()

    gene = args["gene"]
    perturbation_type = args.get("perturbation_type", "knockdown")
    depth = args.get("depth", 2)
    top_k = args.get("top_k", 25)
    network_source = args.get("network_source", "cell_type")
    tcga_network = args.get("tcga_network")

    if network_source == "tcga":
        if not tcga_network:
            return {"error": "tcga_network is required when network_source='tcga'. "
                             "Valid options: brca, coad, hnsc, luad, lusc, ov, prad, ucec"}
        # TCGA networks use gene symbols — resolve symbol if Ensembl ID given
        if gene.upper().startswith("ENSG"):
            symbol = workflow.gene_mapper.ensembl_to_symbol(gene)
            if symbol is None:
                return {"error": f"Could not resolve Ensembl ID '{gene}'"}
            gene_id = symbol
        else:
            gene_id = gene.upper()
        network_df = load_tcga_network(tcga_network)
        if isinstance(network_df, dict) and "error" in network_df:
            return network_df
        cell_type = tcga_network  # for result annotation
    else:
        cell_type = args.get("cell_type", "epithelial_cell")
        if err := _validate_cell_type(cell_type):
            return err
        # Resolve gene using the workflow's shared mapper
        if gene.upper().startswith("ENSG"):
            gene_id = gene.upper()
        else:
            gene_id = workflow.gene_mapper.symbol_to_ensembl(gene)
            if gene_id is None:
                return {"error": f"Could not resolve gene '{gene}'"}
        # Load network (cached after first access)
        network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            return {"error": f"Network not found for {cell_type}"}
        network_df = load_network(network_path)

    # Reuse the workflow's pre-loaded model singleton instead of reloading 120MB checkpoint
    try:
        model = workflow._get_model()

        if perturbation_type == "knockdown":
            result = simulate_knockdown_with_embeddings(
                network_df, gene_id, model, depth=depth, top_k=top_k
            )
        else:
            result = simulate_overexpression_with_embeddings(
                network_df, gene_id, model, depth=depth, top_k=top_k
            )
        result["embedding_enhanced"] = True
    except Exception as e:
        if perturbation_type == "knockdown":
            result = simulate_knockdown(network_df, gene_id, depth=depth, top_k=top_k)
        else:
            result = simulate_overexpression(network_df, gene_id, depth=depth, top_k=top_k)
        result["embedding_enhanced"] = False
        result["note"] = f"Network-only (model unavailable: {str(e)[:50]})"

    result["gene"] = gene
    result["cell_type"] = cell_type
    result["perturbation_type"] = perturbation_type
    if network_source == "tcga":
        result["network_source"] = "tcga"
        result["tcga_network"] = tcga_network

    if result.get("total_affected_genes", 0) == 0:
        result["suggestions"] = [
            "This gene has no transcriptional targets in the network. Try these alternatives:",
            "- get_protein_interactions: find protein-level interaction partners",
            "- find_gene_regulators: find upstream transcription factors controlling this gene",
            "- find_similar_genes: find functionally similar genes by embedding similarity",
        ]

    return result


async def _multi_gene_analysis(args: dict) -> dict:
    """Analyze multiple genes in parallel."""
    workflow = await get_workflow()

    genes = args["genes"]
    cell_type = args.get("cell_type", "epithelial_cell")
    if err := _validate_cell_type(cell_type):
        return err
    analysis_depth = args.get("analysis_depth", "basic")

    if len(genes) > 10:
        return {"error": "Maximum 10 genes allowed"}

    # Run all analyses in parallel
    tasks = [
        workflow.run(
            gene=gene,
            cell_type=cell_type,
            perturbation_type="knockdown",
            analysis_depth=analysis_depth
        )
        for gene in genes
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Compile results
    gene_results = {}
    for gene, result in zip(genes, results):
        if isinstance(result, Exception):
            gene_results[gene] = {"error": str(result)}
        else:
            gene_results[gene] = result

    # Generate comparison summary
    summary = _generate_comparison_summary(genes, gene_results)

    return {
        "genes_analyzed": len(genes),
        "cell_type": cell_type,
        "individual_results": gene_results,
        "comparison_summary": summary
    }


def _generate_comparison_summary(genes: list, results: dict) -> dict:
    """Generate a comparison summary across genes."""
    roles = {}
    target_counts = {}
    regulator_counts = {}

    for gene in genes:
        result = results.get(gene, {})
        if "error" in result:
            continue

        summary = result.get("summary", {})
        roles[gene] = summary.get("gene_role", "unknown")

        # Get network data - prefer context (always available), then detailed analyses
        network = result.get("network_analysis") or {}
        context = network.get("context") or {}
        targets = network.get("targets") or {}
        regulators = network.get("regulators") or {}

        # Use context data (from initial network analysis) as primary source
        target_count = context.get("num_targets", 0)
        regulator_count = context.get("num_regulators", 0)

        # Override with detailed analysis if available
        if targets.get("total_targets"):
            target_count = targets["total_targets"]
        if regulators.get("total_regulators"):
            regulator_count = regulators["total_regulators"]

        target_counts[gene] = target_count
        regulator_counts[gene] = regulator_count

    # Find most influential (most targets) - handle empty dicts
    most_influential = None
    if target_counts and any(v > 0 for v in target_counts.values()):
        most_influential = max(target_counts, key=lambda k: target_counts.get(k, 0))

    # Find most regulated
    most_regulated = None
    if regulator_counts and any(v > 0 for v in regulator_counts.values()):
        most_regulated = max(regulator_counts, key=lambda k: regulator_counts.get(k, 0))

    return {
        "gene_roles": roles,
        "target_counts": target_counts,
        "regulator_counts": regulator_counts,
        "most_influential_gene": most_influential,
        "most_regulated_gene": most_regulated
    }


async def _cross_cell_comparison(args: dict) -> dict:
    """Compare gene across all cell types."""
    workflow = await get_workflow()

    gene = args["gene"]

    # Use the workflow's cross-cell implementation
    state = {
        "gene": gene,
        "ensembl_id": None,
        "gene_symbol": None
    }

    # Resolve gene first
    if gene.upper().startswith("ENSG"):
        state["ensembl_id"] = gene.upper()
        state["gene_symbol"] = workflow.gene_mapper.ensembl_to_symbol(gene)
    else:
        state["gene_symbol"] = gene.upper()
        state["ensembl_id"] = workflow.gene_mapper.symbol_to_ensembl(gene)

    if state["ensembl_id"] is None:
        return {"error": f"Could not resolve gene '{gene}'"}

    result = await workflow._cross_cell_comparison_impl(state)

    # Add interpretation
    comparison = result.get("cell_type_comparison", {})

    # Find where gene is most active
    max_targets = 0
    max_targets_cell = None
    max_regulators = 0
    max_regulators_cell = None

    for cell_type, data in comparison.items():
        if data.get("num_targets", 0) > max_targets:
            max_targets = data["num_targets"]
            max_targets_cell = cell_type
        if data.get("num_regulators", 0) > max_regulators:
            max_regulators = data["num_regulators"]
            max_regulators_cell = cell_type

    result["interpretation"] = {
        "most_influential_in": max_targets_cell,
        "most_regulated_in": max_regulators_cell,
        "max_targets": max_targets,
        "max_regulators": max_regulators
    }

    return result


async def _therapeutic_discovery(args: dict) -> dict:
    """Discover therapeutic targets for a gene."""
    workflow = await get_workflow()

    gene = args["gene"]
    cell_type = args.get("cell_type", "epithelial_cell")
    if err := _validate_cell_type(cell_type):
        return err
    goal = args.get("goal", "reduce_expression")

    # Run comprehensive analysis to get all data
    result = await workflow.run(
        gene=gene,
        cell_type=cell_type,
        perturbation_type="knockdown",
        analysis_depth="comprehensive"
    )

    # Extract therapeutic suggestions
    suggestions = result.get("therapeutic_suggestions", [])

    # Add regulator-based targets
    regulators = result.get("network_analysis", {}).get("regulators", {})
    if regulators and goal in ["reduce_expression", "increase_expression"]:
        top_regulators = regulators.get("regulators", [])[:5]
        for reg in top_regulators:
            action = "inhibit" if goal == "reduce_expression" else "activate"
            suggestions.append({
                "target": reg.get("gene_symbol", reg.get("ensembl_id")),
                "action": action,
                "reason": f"Upstream regulator of {gene}",
                "priority": "high"
            })

    # Add PPI-based targets
    ppi = result.get("external_data", {}).get("protein_interactions", {})
    if ppi and ppi.get("interactions"):
        top_partners = ppi["interactions"][:3]
        for partner in top_partners:
            suggestions.append({
                "target": partner.get("preferredName", "unknown"),
                "action": "disrupt_interaction",
                "reason": f"Protein interaction partner of {gene}",
                "score": partner.get("score", 0),
                "priority": "medium"
            })

    return {
        "gene": gene,
        "goal": goal,
        "cell_type": cell_type,
        "gene_role": result.get("summary", {}).get("gene_role"),
        "therapeutic_targets": suggestions,
        "super_enhancer_status": result.get("external_data", {}).get("super_enhancers"),
        "vulnerability": result.get("network_analysis", {}).get("vulnerability")
    }


async def _find_similar(args: dict) -> dict:
    """Find similar genes using embeddings."""
    workflow = await get_workflow()

    gene = args["gene"]
    top_k = args.get("top_k", 20)

    # Resolve gene
    if gene.upper().startswith("ENSG"):
        ensembl_id = gene.upper()
        symbol = workflow.gene_mapper.ensembl_to_symbol(gene)
    else:
        symbol = gene.upper()
        ensembl_id = workflow.gene_mapper.symbol_to_ensembl(gene)

    if ensembl_id is None:
        return {"error": f"Could not resolve gene '{gene}'"}

    try:
        model = workflow._get_model()

        if not model.is_gene_in_vocab(ensembl_id):
            return {"error": f"Gene {gene} ({ensembl_id}) not in model vocabulary"}

        from tools.cache import get_embedding_cache
        cache = get_embedding_cache(model)
        similar_df = cache.get_top_similar(ensembl_id, top_k=top_k)

        if similar_df is None:
            return {"error": "Could not compute similarities"}

        similar_genes = []
        for _, row in similar_df.iterrows():
            target_ensembl = row["ensembl_id"]
            target_symbol = workflow.gene_mapper.ensembl_to_symbol(target_ensembl) or target_ensembl
            similar_genes.append({
                "gene_symbol": target_symbol,
                "ensembl_id": target_ensembl,
                "similarity": round(row["similarity"], 4)
            })

        return {
            "query_gene": symbol or gene,
            "query_ensembl": ensembl_id,
            "similar_genes": similar_genes,
            "note": "Similarity based on GREmLN embeddings learned from 11M cells"
        }

    except Exception as e:
        return {"error": f"Model unavailable: {str(e)}"}


# =============================================================================
# UTILITY TOOL IMPLEMENTATIONS
# =============================================================================

async def _lookup_gene(args: dict) -> dict:
    """Look up gene information and convert between symbol and Ensembl ID."""
    workflow = await get_workflow()
    gene = args["gene"]

    if gene.upper().startswith("ENSG"):
        ensembl_id = gene.upper()
        symbol = workflow.gene_mapper.ensembl_to_symbol(gene)
        return {
            "input": gene,
            "ensembl_id": ensembl_id,
            "gene_symbol": symbol,
            "status": "found" if symbol else "symbol_not_found"
        }
    else:
        ensembl_id = workflow.gene_mapper.symbol_to_ensembl(gene)
        return {
            "input": gene,
            "gene_symbol": gene.upper(),
            "ensembl_id": ensembl_id,
            "status": "found" if ensembl_id else "ensembl_id_not_found"
        }


async def _get_gene_metadata(args: dict) -> dict:
    """Get gene classification and network role metadata."""
    from pathlib import Path
    from tools.loader import load_network

    workflow = await get_workflow()
    gene = args["gene"]
    cell_type = args.get("cell_type", "epithelial_cell")
    if err := _validate_cell_type(cell_type):
        return err

    # Resolve gene
    if gene.upper().startswith("ENSG"):
        ensembl_id = gene.upper()
    else:
        ensembl_id = workflow.gene_mapper.symbol_to_ensembl(gene)
    if ensembl_id is None:
        return {"error": f"Could not resolve gene '{gene}'"}

    # Load network
    network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for {cell_type}"}

    network_df = load_network(network_path)

    # Count targets and regulators
    targets = network_df[network_df["regulator"] == ensembl_id]
    regulators = network_df[network_df["target"] == ensembl_id]

    num_targets = len(targets)
    num_regulators = len(regulators)

    # Determine gene role
    if num_targets > 50:
        gene_type = "master_regulator"
        description = "Hub transcription factor with many downstream targets"
    elif num_targets > 10:
        gene_type = "transcription_factor"
        description = "Regulates multiple downstream genes"
    elif num_targets > 0:
        gene_type = "minor_regulator"
        description = "Regulates a small number of genes"
    elif num_regulators > 0:
        gene_type = "effector"
        description = "Regulated by network but does not regulate others"
    else:
        gene_type = "isolated"
        description = "Not connected in this cell type's regulatory network"

    # Recommendations
    recommendations = []
    if num_targets > 0:
        recommendations.append("Use analyze_gene_knockdown to see downstream effects")
        recommendations.append("Use find_gene_targets to see all regulated genes")
    else:
        recommendations.append("Use get_protein_interactions - gene functions through protein interactions")
        recommendations.append("Use find_gene_regulators to see upstream controllers")

    return {
        "gene": gene,
        "ensembl_id": ensembl_id,
        "cell_type": cell_type,
        "gene_type": gene_type,
        "description": description,
        "num_targets": num_targets,
        "num_regulators": num_regulators,
        "is_transcription_factor": num_targets > 0,
        "recommendations": recommendations
    }


async def _find_gene_regulators(args: dict) -> dict:
    """Find upstream regulators of a gene."""
    from tools.loader import load_network, load_tcga_network
    from tools.perturb import get_regulators

    workflow = await get_workflow()
    gene = args["gene"]
    max_regulators = args.get("max_regulators", 50)
    network_source = args.get("network_source", "cell_type")
    tcga_network = args.get("tcga_network")

    if network_source == "tcga":
        if not tcga_network:
            return {"error": "tcga_network is required when network_source='tcga'. "
                             "Valid options: brca, coad, hnsc, luad, lusc, ov, prad, ucec"}
        gene_id = gene if not gene.upper().startswith("ENSG") else (
            workflow.gene_mapper.ensembl_to_symbol(gene) or gene
        )
        network_df = load_tcga_network(tcga_network)
        if isinstance(network_df, dict) and "error" in network_df:
            return network_df
        cell_type = tcga_network
    else:
        cell_type = args.get("cell_type", "epithelial_cell")
        if err := _validate_cell_type(cell_type):
            return err
        gene_id = workflow.gene_mapper.symbol_to_ensembl(gene)
        if gene_id is None:
            return {"error": f"Could not resolve gene '{gene}'"}
        network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            return {"error": f"Network not found for {cell_type}"}
        network_df = load_network(network_path)

    result = get_regulators(network_df, gene_id, max_regulators=max_regulators)
    result["input_gene"] = gene
    result["cell_type"] = cell_type
    if network_source == "tcga":
        result["network_source"] = "tcga"
        result["tcga_network"] = tcga_network
    return result


async def _find_gene_targets(args: dict) -> dict:
    """Find downstream targets of a gene."""
    from tools.loader import load_network, load_tcga_network
    from tools.perturb import get_targets

    workflow = await get_workflow()
    gene = args["gene"]
    max_targets = args.get("max_targets", 50)
    network_source = args.get("network_source", "cell_type")
    tcga_network = args.get("tcga_network")

    if network_source == "tcga":
        if not tcga_network:
            return {"error": "tcga_network is required when network_source='tcga'. "
                             "Valid options: brca, coad, hnsc, luad, lusc, ov, prad, ucec"}
        gene_id = gene if not gene.upper().startswith("ENSG") else (
            workflow.gene_mapper.ensembl_to_symbol(gene) or gene
        )
        network_df = load_tcga_network(tcga_network)
        if isinstance(network_df, dict) and "error" in network_df:
            return network_df
        cell_type = tcga_network
    else:
        cell_type = args.get("cell_type", "epithelial_cell")
        if err := _validate_cell_type(cell_type):
            return err
        gene_id = workflow.gene_mapper.symbol_to_ensembl(gene)
        if gene_id is None:
            return {"error": f"Could not resolve gene '{gene}'"}
        network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            return {"error": f"Network not found for {cell_type}"}
        network_df = load_network(network_path)

    result = get_targets(network_df, gene_id, max_targets=max_targets)
    result["input_gene"] = gene
    result["cell_type"] = cell_type
    if network_source == "tcga":
        result["network_source"] = "tcga"
        result["tcga_network"] = tcga_network
    return result


async def _get_protein_interactions(args: dict) -> dict:
    """Get protein-protein interactions from STRING."""
    from tools.ppi.string_client import get_string_client

    workflow = await get_workflow()
    gene = args["gene"]
    min_score = args.get("min_score", 400)
    limit = args.get("limit", 25)

    # Resolve Ensembl to symbol if needed
    if gene.upper().startswith("ENSG"):
        symbol = workflow.gene_mapper.ensembl_to_symbol(gene)
        if symbol is None:
            return {"error": f"Could not resolve Ensembl ID '{gene}'"}
        gene = symbol

    try:
        client = get_string_client()
        result = client.get_interactions(gene, min_score=min_score, limit=limit)
        return result
    except Exception as e:
        return {"error": f"STRING API error: {str(e)}"}


async def _get_gene_similarity(args: dict) -> dict:
    """Get embedding similarity between two genes."""
    workflow = await get_workflow()
    gene1 = args["gene1"]
    gene2 = args["gene2"]

    # Resolve genes
    ensembl1 = workflow.gene_mapper.symbol_to_ensembl(gene1)
    ensembl2 = workflow.gene_mapper.symbol_to_ensembl(gene2)

    if ensembl1 is None:
        return {"error": f"Could not resolve gene '{gene1}'"}
    if ensembl2 is None:
        return {"error": f"Could not resolve gene '{gene2}'"}

    try:
        model = workflow._get_model()

        if not model.is_gene_in_vocab(ensembl1):
            return {"error": f"Gene {gene1} ({ensembl1}) not in model vocabulary"}
        if not model.is_gene_in_vocab(ensembl2):
            return {"error": f"Gene {gene2} ({ensembl2}) not in model vocabulary"}

        similarity = model.compute_similarity(ensembl1, ensembl2)

        # Interpret
        if similarity >= 0.8:
            interpretation = "Very high similarity - genes likely have similar functions"
        elif similarity >= 0.5:
            interpretation = "High similarity - genes may be functionally related"
        elif similarity >= 0.3:
            interpretation = "Moderate similarity - some functional overlap possible"
        elif similarity >= 0.0:
            interpretation = "Low similarity - likely unrelated functions"
        else:
            interpretation = "Negative similarity - potentially opposite functions"

        return {
            "gene1": gene1,
            "gene1_ensembl": ensembl1,
            "gene2": gene2,
            "gene2_ensembl": ensembl2,
            "similarity": round(similarity, 4) if similarity is not None else None,
            "interpretation": interpretation
        }

    except Exception as e:
        return {"error": f"Model error: {str(e)}"}


async def _get_model_status(args: dict) -> dict:
    """Check CASCADE model and GPU status."""
    import torch

    workflow = await get_workflow()

    try:
        model = await asyncio.to_thread(workflow._get_model)
        stats = model.get_embedding_stats()
        return {
            "model_loaded": True,
            "gpu_available": torch.cuda.is_available(),
            "device": stats["device"],
            "num_genes": stats["num_actual_genes"],
            "embedding_dim": stats["embedding_dim"],
            "checkpoint_path": str(workflow.MODEL_PATH)
        }
    except Exception as e:
        return {
            "model_loaded": False,
            "gpu_available": torch.cuda.is_available(),
            "error": str(e)
        }


async def _get_embedding_cache_stats(args: dict) -> dict:
    """Get embedding similarity cache statistics."""
    try:
        from tools.cache import _embedding_cache
        if _embedding_cache is None:
            return {
                "cache_initialized": False,
                "note": "Cache initializes on first model-based query"
            }
        return {
            "cache_initialized": True,
            **_embedding_cache.get_stats()
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# NETWORK VULNERABILITY IMPLEMENTATIONS
# =============================================================================

async def _analyze_network_vulnerability(args: dict) -> dict:
    """Find critical hub genes in the network."""
    from tools.loader import load_network
    from collections import defaultdict

    workflow = await get_workflow()
    cell_type = args.get("cell_type", "epithelial_cell")
    if err := _validate_cell_type(cell_type):
        return err
    top_k = args.get("top_k", 20)

    network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for {cell_type}"}

    network_df = load_network(network_path)

    # Build adjacency (fast)
    forward_adj = defaultdict(list)
    reverse_adj = defaultdict(list)

    for _, row in network_df.iterrows():
        reg = row["regulator"]
        tgt = row["target"]
        weight = row.get("mi", 1.0)
        if weight > 0:
            forward_adj[reg].append((tgt, float(weight)))
            reverse_adj[tgt].append((reg, float(weight)))

    # First pass: quick scoring based on hub_score only (fast)
    quick_scores = []
    for gene in set(network_df["regulator"].unique()):
        targets = forward_adj.get(gene, [])
        hub_score = len(targets)
        quick_scores.append((gene, hub_score))

    # Sort by hub score and take top candidates for detailed analysis
    quick_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = quick_scores[:top_k * 2]  # Analyze 2x more than needed

    # Second pass: detailed scoring for top candidates only
    scores = []
    for gene, hub_score in top_candidates:
        targets = forward_adj.get(gene, [])
        regulators = reverse_adj.get(gene, [])

        reg_count = len(regulators)
        avg_weight = sum(w for _, w in targets) / len(targets) if targets else 0

        # Cascade calculation only for top candidates
        cascade = set()
        for tgt, _ in targets[:50]:
            second = forward_adj.get(tgt, [])
            cascade.update(t for t, _ in second[:20])

        vulnerability = (
            hub_score * 1.0 +
            len(cascade) * 0.3 +
            avg_weight * 10 +
            (1 / (reg_count + 1)) * 5
        )

        symbol = workflow.gene_mapper.ensembl_to_symbol(gene) or gene
        scores.append({
            "symbol": symbol,
            "ensembl_id": gene,
            "hub_score": hub_score,
            "regulator_count": reg_count,
            "cascade_reach": len(cascade),
            "vulnerability_score": round(vulnerability, 2)
        })

    scores.sort(key=lambda x: x["vulnerability_score"], reverse=True)

    return {
        "cell_type": cell_type,
        "total_regulators": len(quick_scores),
        "top_vulnerable_genes": scores[:top_k],
        "interpretation": "High vulnerability = good drug target (maximum network impact)"
    }


async def _compare_gene_vulnerability(args: dict) -> dict:
    """Compare vulnerability scores for specific genes."""
    from tools.loader import load_network
    from collections import defaultdict

    workflow = await get_workflow()
    genes = args["genes"]
    cell_type = args.get("cell_type", "epithelial_cell")
    if err := _validate_cell_type(cell_type):
        return err

    network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for {cell_type}"}

    network_df = load_network(network_path)

    # Build adjacency
    forward_adj = defaultdict(list)
    reverse_adj = defaultdict(list)

    for _, row in network_df.iterrows():
        reg = row["regulator"]
        tgt = row["target"]
        weight = row.get("mi", 1.0)
        if weight > 0:
            forward_adj[reg].append((tgt, float(weight)))
            reverse_adj[tgt].append((reg, float(weight)))

    results = []
    for gene in genes:
        ensembl_id = workflow.gene_mapper.symbol_to_ensembl(gene)
        if ensembl_id is None:
            results.append({"gene": gene, "error": "Could not resolve"})
            continue

        targets = forward_adj.get(ensembl_id, [])
        regulators = reverse_adj.get(ensembl_id, [])

        hub_score = len(targets)
        reg_count = len(regulators)
        avg_weight = sum(w for _, w in targets) / len(targets) if targets else 0

        cascade = set()
        for tgt, _ in targets[:50]:
            second = forward_adj.get(tgt, [])
            cascade.update(t for t, _ in second[:20])

        vulnerability = (
            hub_score * 1.0 +
            len(cascade) * 0.3 +
            avg_weight * 10 +
            (1 / (reg_count + 1)) * 5
        )

        results.append({
            "gene": gene,
            "ensembl_id": ensembl_id,
            "hub_score": hub_score,
            "regulator_count": reg_count,
            "cascade_reach": len(cascade),
            "vulnerability_score": round(vulnerability, 2)
        })

    results.sort(key=lambda x: x.get("vulnerability_score", 0), reverse=True)

    return {
        "cell_type": cell_type,
        "comparison": results,
        "best_target": results[0]["gene"] if results and "vulnerability_score" in results[0] else None
    }


# =============================================================================
# LINCS IMPLEMENTATIONS
# =============================================================================

async def _find_expression_regulators(args: dict) -> dict:
    """Find genes whose knockdown affects target expression."""
    from tools.lincs import find_expression_regulators

    gene = args["gene"]
    direction = args.get("direction", "any")
    top_k = args.get("top_k", 20)

    try:
        results = find_expression_regulators(gene, direction=direction, top_k=top_k)
        return {
            "gene": gene,
            "direction_filter": direction,
            "regulators_found": len(results) if results else 0,
            "regulators": results or [],
            "data_source": "LINCS L1000 CRISPR Knockout"
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"LINCS query failed: {str(e)}"}


async def _get_knockdown_effects(args: dict) -> dict:
    """Find genes affected by a knockdown."""
    from tools.lincs import get_knockdown_effects

    gene = args["gene"]
    direction = args.get("direction", "any")
    top_k = args.get("top_k", 20)

    try:
        results = get_knockdown_effects(gene, direction=direction, top_k=top_k)
        return {
            "gene_knocked_out": gene,
            "direction_filter": direction,
            "affected_genes_found": len(results) if results else 0,
            "affected_genes": results or [],
            "data_source": "LINCS L1000 CRISPR Knockout"
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"LINCS query failed: {str(e)}"}


async def _get_lincs_data_stats(args: dict) -> dict:
    """Get LINCS dataset statistics."""
    from tools.lincs import get_lincs_stats

    try:
        stats = get_lincs_stats()
        return {
            **stats,
            "data_source": "LINCS L1000 CRISPR Knockout Consensus Signatures"
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"LINCS load failed: {str(e)}"}


# =============================================================================
# SUPER-ENHANCER IMPLEMENTATIONS
# =============================================================================

async def _check_super_enhancer(args: dict) -> dict:
    """Check super-enhancer status for a gene."""
    from tools.super_enhancers import get_super_enhancer_info

    gene = args["gene"]

    try:
        return get_super_enhancer_info(gene)
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Super-enhancer check failed: {str(e)}"}


async def _check_genes_super_enhancers(args: dict) -> dict:
    """Check multiple genes for super-enhancers."""
    from tools.super_enhancers import check_genes_for_super_enhancers

    genes = args["genes"]

    try:
        results = check_genes_for_super_enhancers(genes)
        se_positive = [r for r in results if r.get("has_super_enhancer")]

        return {
            "total_genes": len(genes),
            "super_enhancer_positive": len(se_positive),
            "results": results,
            "interpretation": f"{len(se_positive)}/{len(genes)} genes may respond to BET inhibitors",
            "data_source": "dbSUPER"
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Super-enhancer check failed: {str(e)}"}


# =============================================================================
# DOROTHEA IMPLEMENTATIONS
# =============================================================================

async def _get_dorothea_regulon(args: dict) -> dict:
    """Get DoRothEA TF regulon targets."""
    from tools.dorothea import get_tf_targets

    gene = args["gene"]
    confidence_levels = args.get("confidence_levels", ["A", "B", "C"])

    try:
        results = get_tf_targets(gene, confidence_levels=confidence_levels, top_k=50)
        # Check if results contain an error dict
        if results and isinstance(results[0], dict) and "error" in results[0]:
            return {"error": results[0]["error"]}
        return {
            "gene": gene,
            "confidence_levels": confidence_levels,
            "targets_found": len(results),
            "targets": results,
            "data_source": "DoRothEA via decoupler-py"
        }
    except Exception as e:
        return {"error": f"DoRothEA query failed: {str(e)}"}


async def _validate_tf_classification(args: dict) -> dict:
    """Validate TF classification against DoRothEA."""
    from tools.dorothea import validate_tf_classification

    gene = args["gene"]

    try:
        result = validate_tf_classification(gene)
        result["data_source"] = "DoRothEA via decoupler-py"
        return result
    except Exception as e:
        return {"error": f"DoRothEA validation failed: {str(e)}"}


async def _get_dorothea_stats(args: dict) -> dict:
    """Get DoRothEA dataset statistics."""
    from tools.dorothea import get_dorothea_stats

    try:
        return get_dorothea_stats()
    except Exception as e:
        return {"error": f"DoRothEA stats failed: {str(e)}"}


# =============================================================================
# DEPMAP IMPLEMENTATIONS
# =============================================================================

async def _get_depmap_essentiality(args: dict) -> dict:
    """Get DepMap CRISPR essentiality profile for a gene."""
    from tools.depmap import get_gene_essentiality

    gene = args["gene"]

    try:
        return get_gene_essentiality(gene)
    except FileNotFoundError as e:
        return {"error": str(e), "not_found": True}
    except Exception as e:
        return {"error": f"DepMap query failed: {str(e)}", "not_found": True}


# =============================================================================
# CBIOPORTAL IMPLEMENTATIONS
# =============================================================================

async def _get_tumor_expression_profile(args: dict) -> dict:
    """Get mRNA expression z-scores across TCGA PanCancer Atlas cancer types."""
    from tools.cbioportal import get_gene_tumor_expression

    gene = args["gene"]

    try:
        return await asyncio.to_thread(get_gene_tumor_expression, gene)
    except Exception as e:
        return {"error": f"cBioPortal expression query failed: {str(e)}", "gene": gene}


async def _get_tumor_alteration_frequency(args: dict) -> dict:
    """Get somatic mutation and CNA frequency across TCGA PanCancer Atlas cancer types."""
    from tools.cbioportal import get_gene_alteration_frequency

    gene = args["gene"]

    try:
        return await asyncio.to_thread(get_gene_alteration_frequency, gene)
    except Exception as e:
        return {"error": f"cBioPortal alteration query failed: {str(e)}", "gene": gene}


# =============================================================================
# RESOURCE IMPLEMENTATIONS
# =============================================================================

async def _resource_cell_types() -> dict:
    """List all cell types with availability."""
    workflow = await get_workflow()
    cell_types = []
    for ct in CellType:
        network_path = workflow.NETWORKS_DIR / ct.value / "network.tsv"
        cell_types.append({
            "name": ct.value,
            "available": network_path.exists(),
        })
    return {
        "cell_types": cell_types,
        "total": len(cell_types),
        "note": "Use cascade://network/{cell_type}/summary for edge/gene counts",
    }


async def _resource_network_summary(cell_type: str) -> dict:
    """Return edge count, gene count, and top hub regulators for a cell type."""
    from tools.loader import load_network

    workflow = await get_workflow()
    valid = [ct.value for ct in CellType]
    if cell_type not in valid:
        return {"error": f"Unknown cell type '{cell_type}'. Valid: {valid}"}

    network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network file not found for '{cell_type}'"}

    network_df = await asyncio.to_thread(load_network, network_path)
    edge_count = len(network_df)
    all_genes = set(network_df["regulator"].unique()) | set(network_df["target"].unique())
    gene_count = len(all_genes)

    # Top 10 hub regulators by out-degree
    hub_series = network_df.groupby("regulator").size().nlargest(10)
    top_hubs = []
    for ensembl_id, target_count in hub_series.items():
        symbol = workflow.gene_mapper.ensembl_to_symbol(ensembl_id) or ensembl_id
        top_hubs.append({"symbol": symbol, "ensembl_id": ensembl_id, "target_count": int(target_count)})

    return {
        "cell_type": cell_type,
        "edge_count": edge_count,
        "gene_count": gene_count,
        "top_regulators": top_hubs,
    }


async def _resource_gene_metadata(symbol: str, cell_type: str) -> dict:
    """Return gene role and network position for a gene in a given cell type."""
    from tools.loader import load_network

    workflow = await get_workflow()
    valid = [ct.value for ct in CellType]
    if cell_type not in valid:
        return {"error": f"Unknown cell type '{cell_type}'. Valid: {valid}"}

    ensembl_id = workflow.gene_mapper.symbol_to_ensembl(symbol)
    if ensembl_id is None:
        return {"error": f"Could not resolve gene symbol '{symbol}'"}

    network_path = workflow.NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network file not found for '{cell_type}'"}

    network_df = await asyncio.to_thread(load_network, network_path)
    num_targets = int((network_df["regulator"] == ensembl_id).sum())
    num_regulators = int((network_df["target"] == ensembl_id).sum())

    if num_targets > 50:
        gene_role = "master_regulator"
    elif num_targets > 10:
        gene_role = "transcription_factor"
    elif num_targets > 0:
        gene_role = "minor_regulator"
    elif num_regulators > 0:
        gene_role = "effector"
    else:
        gene_role = "isolated"

    return {
        "gene": symbol.upper(),
        "ensembl_id": ensembl_id,
        "cell_type": cell_type,
        "gene_role": gene_role,
        "num_targets": num_targets,
        "num_regulators": num_regulators,
        "in_network": num_targets > 0 or num_regulators > 0,
    }


async def _resource_lincs_summary() -> dict:
    """Return LINCS dataset coverage statistics."""
    from tools.lincs import get_lincs_stats
    try:
        return await asyncio.to_thread(get_lincs_stats)
    except Exception as e:
        return {"error": f"LINCS stats unavailable: {e}"}


async def _resource_model_status() -> dict:
    """Return GREmLN model checkpoint status and GPU info."""
    import torch
    workflow = await get_workflow()
    model_path = workflow.MODEL_PATH
    loaded = workflow._model is not None
    gene_count = None
    if loaded:
        try:
            gene_count = len(workflow._model.gene_ids)
        except Exception:
            pass
    return {
        "model_path": str(model_path),
        "checkpoint_exists": model_path.exists(),
        "loaded": loaded,
        "gene_count": gene_count,
        "gpu_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# =============================================================================
# SERVER STARTUP
# =============================================================================

async def main():
    """Run the MCP server."""
    logger.info("Starting CASCADE LangGraph MCP Server...")

    # Initialize workflow eagerly to avoid lazy-load hanging in MCP context
    logger.info("Pre-initializing workflow...")
    await get_workflow()
    logger.info("Workflow ready")

    logger.info("Starting MCP server...")

    async with stdio_server() as (read_stream, write_stream):
        # Pre-warm DoRothEA regulons as a background task AFTER the server
        # is ready to handle the initialize handshake. Running it before
        # stdio_server() blocked the handshake for >60 s, causing Claude
        # Desktop to time out and disconnect.
        async def _prewarm_dorothea():
            try:
                logger.info("Pre-warming DoRothEA regulons in background...")
                from tools.dorothea import load_dorothea_regulons
                await asyncio.to_thread(load_dorothea_regulons)
                logger.info("DoRothEA regulons ready")
            except Exception as e:
                logger.warning(f"DoRothEA pre-warm failed (will retry on first use): {e}")

        async def _prewarm_model():
            try:
                logger.info("Pre-warming GREmLN model in background...")
                wf = await get_workflow()
                await asyncio.to_thread(wf._get_model)
                logger.info("GREmLN model ready")
            except Exception as e:
                logger.warning(f"GREmLN pre-warm failed (will retry on first use): {e}")

        async def _prewarm_embedding_cache():
            try:
                logger.info("Pre-warming embedding similarity cache in background...")
                wf = await get_workflow()
                def _init_cache():
                    model = wf._get_model()
                    from tools.cache import get_embedding_cache
                    get_embedding_cache(model)
                await asyncio.to_thread(_init_cache)
                logger.info("Embedding cache ready")
            except Exception as e:
                logger.warning(f"Embedding cache pre-warm failed (will retry on first use): {e}")

        async def _prewarm_depmap():
            try:
                logger.info("Pre-warming DepMap CRISPR data in background...")
                from tools.depmap import load_depmap_data
                await asyncio.to_thread(load_depmap_data)
                logger.info("DepMap data ready")
            except Exception as e:
                logger.warning(f"DepMap pre-warm failed (will retry on first use): {e}")

        async def _prewarm_lincs():
            try:
                logger.info("Pre-warming LINCS data in background...")
                from tools.lincs import load_lincs_data
                await asyncio.to_thread(load_lincs_data)
                logger.info("LINCS data ready")
            except Exception as e:
                logger.warning(f"LINCS pre-warm failed (will retry on first use): {e}")

        async def _prewarm_super_enhancers():
            try:
                logger.info("Pre-warming super-enhancer data in background...")
                from tools.super_enhancers import load_super_enhancer_data
                await asyncio.to_thread(load_super_enhancer_data)
                logger.info("Super-enhancer data ready")
            except Exception as e:
                logger.warning(f"Super-enhancer pre-warm failed (will retry on first use): {e}")

        async def _prewarm_networks():
            try:
                logger.info("Pre-warming cell type networks in background...")
                from tools.loader import load_network, NETWORKS_DIR
                from cascade_langgraph_workflow import CellType
                for ct in CellType:
                    network_path = NETWORKS_DIR / ct.value / "network.tsv"
                    if network_path.exists():
                        await asyncio.to_thread(load_network, network_path)
                logger.info("All cell type networks ready")
            except Exception as e:
                logger.warning(f"Network pre-warm failed (will retry on first use): {e}")

        asyncio.create_task(_prewarm_dorothea())
        asyncio.create_task(_prewarm_model())
        asyncio.create_task(_prewarm_embedding_cache())
        asyncio.create_task(_prewarm_depmap())
        asyncio.create_task(_prewarm_lincs())
        asyncio.create_task(_prewarm_super_enhancers())
        asyncio.create_task(_prewarm_networks())

        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cascade-langgraph-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
