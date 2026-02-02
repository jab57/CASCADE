#!/usr/bin/env python3
"""
GREmLN LangGraph Workflow - SKETCH/DRAFT
=========================================

Multi-agent orchestration for in silico gene perturbation analysis.

This module wraps the existing GREmLN tools into a LangGraph workflow that:
- Automatically determines the best analysis path based on gene characteristics
- Runs independent analyses in parallel (network + embeddings + LINCS + PPI)
- Provides intelligent suggestions and follow-up recommendations
- Generates comprehensive perturbation reports

Architecture:
    - State-based workflow orchestration (LangGraph StateGraph)
    - Parallel batch processing of independent analyses
    - Conditional routing based on gene network position (TF vs effector)
    - Graceful fallback (embedding model → network-only)

Author: [Your name]
License: MIT
"""

from typing import Dict, List, TypedDict, Optional, Any, Literal
from langgraph.graph import StateGraph, END
from enum import Enum
import asyncio
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CellType(Enum):
    """Available cell types with pre-computed regulatory networks."""
    EPITHELIAL_CELL = "epithelial_cell"
    CD4_T_CELLS = "cd4_t_cells"
    CD8_T_CELLS = "cd8_t_cells"
    CD14_MONOCYTES = "cd14_monocytes"
    CD16_MONOCYTES = "cd16_monocytes"
    CD20_B_CELLS = "cd20_b_cells"
    NK_CELLS = "nk_cells"
    NKT_CELLS = "nkt_cells"
    ERYTHROCYTES = "erythrocytes"
    MONOCYTE_DERIVED_DENDRITIC_CELLS = "monocyte-derived_dendritic_cells"


class PerturbationType(Enum):
    """Types of perturbation analysis."""
    KNOCKDOWN = "knockdown"
    OVEREXPRESSION = "overexpression"
    SIMILARITY = "similarity"  # Find similar genes only


class GeneRole(Enum):
    """Gene's role in the regulatory network."""
    MASTER_REGULATOR = "master_regulator"      # >50 targets
    TRANSCRIPTION_FACTOR = "transcription_factor"  # 10-50 targets
    MINOR_REGULATOR = "minor_regulator"        # 1-10 targets
    EFFECTOR = "effector"                      # No targets, but regulated
    ISOLATED = "isolated"                      # Not in network


# =============================================================================
# STATE SCHEMA
# =============================================================================

class PerturbationAnalysisState(TypedDict):
    """
    State object tracking the entire perturbation analysis workflow.

    This is the central data structure that flows through all workflow nodes.
    Each node reads from and writes to this state.
    """
    # === Input Parameters ===
    gene: str                                   # Gene symbol or Ensembl ID
    cell_type: str                              # Cell type for network context
    perturbation_type: str                      # knockdown, overexpression, similarity
    analysis_depth: str                         # basic, comprehensive, focused

    # === Resolved Gene Info ===
    ensembl_id: Optional[str]                   # Resolved Ensembl ID
    gene_symbol: Optional[str]                  # Resolved gene symbol
    gene_role: Optional[str]                    # master_regulator, tf, effector, isolated

    # === Workflow Control ===
    current_step: str                           # Current workflow node
    workflow_complete: bool                     # Whether analysis is done
    error_message: Optional[str]                # Error if any
    next_actions: List[str]                     # Pending analysis steps
    completed_actions: List[str]                # Completed analysis steps

    # === Core Network Analysis ===
    network_context: Optional[Dict]             # Gene's position in network
    perturbation_result: Optional[Dict]         # Knockdown/overexpression result
    regulators_analysis: Optional[Dict]         # Upstream regulators
    targets_analysis: Optional[Dict]            # Downstream targets

    # === Embedding-Based Analysis ===
    similar_genes: Optional[Dict]               # Embedding-based similar genes
    embedding_enhanced: bool                    # Whether embeddings were used

    # === External Data Integration ===
    ppi_interactions: Optional[Dict]            # STRING protein interactions
    lincs_effects: Optional[Dict]               # LINCS knockdown effects
    super_enhancer_status: Optional[Dict]       # Super-enhancer info (BET sensitivity)

    # === Cross-Cell Analysis ===
    cross_cell_comparison: Optional[Dict]       # Same gene across cell types

    # === Therapeutic Insights ===
    vulnerability_analysis: Optional[Dict]      # Network vulnerability scores
    therapeutic_suggestions: Optional[List]     # Drug target recommendations

    # === Final Output ===
    comprehensive_report: Optional[Dict]        # Final compiled report
    analysis_metadata: Dict                     # Timing, versions, etc.

    # === LLM Insights (Optional) ===
    include_llm_insights: bool                  # Whether to generate LLM synthesis
    llm_insights: Optional[Dict]                # LLM-generated biological interpretation


# =============================================================================
# WORKFLOW CLASS
# =============================================================================

class GREmLNWorkflow:
    """
    LangGraph workflow for comprehensive gene perturbation analysis.

    Orchestrates multiple analysis tools into a coherent workflow that:
    1. Resolves gene identity and determines network role
    2. Routes to appropriate analyses based on gene type
    3. Runs independent analyses in parallel
    4. Integrates results and generates recommendations

    Example:
        >>> workflow = GREmLNWorkflow()
        >>> result = await workflow.run(
        ...     gene="TP53",
        ...     cell_type="epithelial_cell",
        ...     perturbation_type="knockdown"
        ... )
    """

    def __init__(self):
        """Initialize workflow with GREmLN components."""
        # Import existing GREmLN components
        from pathlib import Path
        from tools.loader import load_network, get_available_cell_types, MODEL_PATH
        from tools.gene_id_mapper import GeneIDMapper
        from tools.ppi.string_client import get_string_client

        self.BASE_DIR = Path(__file__).parent
        self.NETWORKS_DIR = self.BASE_DIR / "data" / "networks"
        self.MODEL_PATH = MODEL_PATH

        # Initialize components
        self.gene_mapper = GeneIDMapper()
        self.string_client = get_string_client()
        self._model = None  # Lazy loaded

        # LLM configuration (Ollama)
        self.use_llm = os.getenv('USE_LLM_INSIGHTS', 'false').lower() == 'true'
        self.ollama_client = None
        self.ollama_available = self._initialize_ollama() if self.use_llm else False
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        self.ollama_temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        self.ollama_max_tokens = int(os.getenv('OLLAMA_MAX_TOKENS', '2000'))

        # Build the workflow graph
        self.workflow = self._create_workflow()
        logger.info("GREmLN LangGraph workflow initialized")

    def _get_model(self):
        """Lazy load the GREmLN model."""
        if self._model is None:
            from tools.model_inference import GREmLNModel
            self._model = GREmLNModel(self.MODEL_PATH)
            self._model.load()
        return self._model

    def _initialize_ollama(self) -> bool:
        """Initialize Ollama client (auto-detects local vs cloud)."""
        try:
            import ollama
        except ImportError:
            logger.warning("ollama package not installed, LLM insights disabled")
            return False

        api_key = os.getenv('OLLAMA_API_KEY')

        if api_key:
            # Cloud mode
            logger.info("Using Ollama Cloud (API key detected)")
            self.ollama_client = ollama.Client(
                host='https://ollama.com',
                headers={'Authorization': f'Bearer {api_key}'}
            )
        else:
            # Local mode
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            logger.info(f"Using local Ollama at {host}")
            self.ollama_client = ollama.Client(host=host)

        # Test connection
        try:
            self.ollama_client.list()
            logger.info("Ollama connection successful")
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    # =========================================================================
    # WORKFLOW GRAPH CONSTRUCTION
    # =========================================================================

    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow structure.

        Workflow stages:
        1. Initialize → Resolve gene, load network
        2. Analyze Context → Determine gene role in network
        3. Route → Decide which analyses to run
        4. Batch Core → Parallel: perturbation + regulators + targets
        5. Batch External → Parallel: PPI + LINCS + super-enhancers
        6. Batch Insights → Parallel: similar genes + vulnerability
        7. Generate Report → Compile final results

        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(PerturbationAnalysisState)

        # === Stage 1: Initialization ===
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("resolve_gene", self._resolve_gene)
        workflow.add_node("analyze_network_context", self._analyze_network_context)

        # === Stage 2: Routing ===
        workflow.add_node("decide_next_steps", self._decide_next_steps)

        # === Stage 3: Core Analysis (can run in parallel) ===
        workflow.add_node("batch_core_analysis", self._batch_core_analysis)
        workflow.add_node("run_perturbation", self._run_perturbation)
        workflow.add_node("analyze_regulators", self._analyze_regulators)
        workflow.add_node("analyze_targets", self._analyze_targets)

        # === Stage 4: External Data (can run in parallel) ===
        workflow.add_node("batch_external_data", self._batch_external_data)
        workflow.add_node("fetch_ppi", self._fetch_ppi)
        workflow.add_node("fetch_lincs", self._fetch_lincs)
        workflow.add_node("check_super_enhancers", self._check_super_enhancers)

        # === Stage 5: Advanced Analysis ===
        workflow.add_node("batch_insights", self._batch_insights)
        workflow.add_node("find_similar_genes", self._find_similar_genes)
        workflow.add_node("analyze_vulnerability", self._analyze_vulnerability)
        workflow.add_node("cross_cell_comparison", self._cross_cell_comparison)

        # === Stage 6: Report Generation ===
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("synthesize_insights", self._synthesize_insights)
        workflow.add_node("handle_error", self._handle_error)

        # === Define Edges ===

        # Sequential initialization
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "resolve_gene")
        workflow.add_edge("resolve_gene", "analyze_network_context")
        workflow.add_edge("analyze_network_context", "decide_next_steps")

        # Conditional routing based on gene role and analysis depth
        workflow.add_conditional_edges(
            "decide_next_steps",
            self._route_next_action,
            {
                # Batch processing routes
                "batch_core": "batch_core_analysis",
                "batch_external": "batch_external_data",
                "batch_insights": "batch_insights",

                # Individual analysis routes (for focused mode)
                "perturbation": "run_perturbation",
                "regulators": "analyze_regulators",
                "targets": "analyze_targets",
                "ppi": "fetch_ppi",
                "lincs": "fetch_lincs",
                "super_enhancers": "check_super_enhancers",
                "similar": "find_similar_genes",
                "vulnerability": "analyze_vulnerability",
                "cross_cell": "cross_cell_comparison",

                # Terminal routes
                "complete": "generate_report",
                "error": "handle_error"
            }
        )

        # Batch nodes flow back to routing
        workflow.add_edge("batch_core_analysis", "decide_next_steps")
        workflow.add_edge("batch_external_data", "decide_next_steps")
        workflow.add_edge("batch_insights", "decide_next_steps")

        # Individual nodes flow back to routing
        workflow.add_edge("run_perturbation", "decide_next_steps")
        workflow.add_edge("analyze_regulators", "decide_next_steps")
        workflow.add_edge("analyze_targets", "decide_next_steps")
        workflow.add_edge("fetch_ppi", "decide_next_steps")
        workflow.add_edge("fetch_lincs", "decide_next_steps")
        workflow.add_edge("check_super_enhancers", "decide_next_steps")
        workflow.add_edge("find_similar_genes", "decide_next_steps")
        workflow.add_edge("analyze_vulnerability", "decide_next_steps")
        workflow.add_edge("cross_cell_comparison", "decide_next_steps")

        # Terminal edges
        workflow.add_edge("generate_report", "synthesize_insights")
        workflow.add_edge("synthesize_insights", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    # =========================================================================
    # WORKFLOW NODE IMPLEMENTATIONS
    # =========================================================================

    async def _initialize(self, state: PerturbationAnalysisState) -> Dict:
        """Initialize the workflow state."""
        import time

        logger.info(f"Initializing analysis for gene: {state['gene']}")

        return {
            "current_step": "initialize",
            "workflow_complete": False,
            "error_message": None,
            "next_actions": [],
            "completed_actions": [],
            "embedding_enhanced": False,
            "analysis_metadata": {
                "start_time": time.time(),
                "workflow_version": "1.0.0"
            }
        }

    async def _resolve_gene(self, state: PerturbationAnalysisState) -> Dict:
        """Resolve gene symbol to Ensembl ID and vice versa."""
        gene = state["gene"]

        if gene.upper().startswith("ENSG"):
            ensembl_id = gene.upper()
            symbol = self.gene_mapper.ensembl_to_symbol(gene)
        else:
            symbol = gene.upper()
            ensembl_id = self.gene_mapper.symbol_to_ensembl(gene)

        if ensembl_id is None:
            return {
                "current_step": "resolve_gene",
                "error_message": f"Could not resolve gene '{gene}' to Ensembl ID",
                "next_actions": ["error"]
            }

        logger.info(f"Resolved gene: {symbol} ({ensembl_id})")

        return {
            "current_step": "resolve_gene",
            "ensembl_id": ensembl_id,
            "gene_symbol": symbol or ensembl_id
        }

    async def _analyze_network_context(self, state: PerturbationAnalysisState) -> Dict:
        """Analyze gene's position and role in the regulatory network."""
        from tools.loader import load_network

        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]

        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            return {
                "current_step": "analyze_network_context",
                "error_message": f"Network not found for cell type: {cell_type}",
                "next_actions": ["error"]
            }

        network_df = load_network(network_path)

        # Count targets and regulators
        targets = network_df[network_df["regulator"] == ensembl_id]
        regulators = network_df[network_df["target"] == ensembl_id]

        num_targets = len(targets)
        num_regulators = len(regulators)

        # Determine gene role
        if num_targets > 50:
            gene_role = GeneRole.MASTER_REGULATOR.value
        elif num_targets > 10:
            gene_role = GeneRole.TRANSCRIPTION_FACTOR.value
        elif num_targets > 0:
            gene_role = GeneRole.MINOR_REGULATOR.value
        elif num_regulators > 0:
            gene_role = GeneRole.EFFECTOR.value
        else:
            gene_role = GeneRole.ISOLATED.value

        network_context = {
            "num_targets": num_targets,
            "num_regulators": num_regulators,
            "gene_role": gene_role,
            "is_transcription_factor": num_targets > 0,
            "is_regulated": num_regulators > 0,
            "in_network": num_targets > 0 or num_regulators > 0
        }

        logger.info(f"Gene role: {gene_role} (targets={num_targets}, regulators={num_regulators})")

        return {
            "current_step": "analyze_network_context",
            "gene_role": gene_role,
            "network_context": network_context
        }

    async def _decide_next_steps(self, state: PerturbationAnalysisState) -> Dict:
        """
        Decide what analyses to run next based on current state.

        This is the central routing logic that determines the analysis path.
        """
        completed = set(state.get("completed_actions", []))
        gene_role = state.get("gene_role", GeneRole.ISOLATED.value)
        analysis_depth = state.get("analysis_depth", "comprehensive")
        perturbation_type = state.get("perturbation_type", "knockdown")

        next_actions = []

        # === Determine required analyses based on depth ===

        if analysis_depth == "basic":
            # Basic: Just perturbation + similar genes
            required = {"perturbation", "similar"}

        elif analysis_depth == "focused":
            # Focused: Perturbation + relevant follow-ups based on gene role
            required = {"perturbation"}
            if gene_role in [GeneRole.MASTER_REGULATOR.value, GeneRole.TRANSCRIPTION_FACTOR.value]:
                required.add("targets")
            else:
                required.add("ppi")  # Effectors need PPI analysis

        else:  # comprehensive
            # Comprehensive: Everything relevant to gene role
            required = {"perturbation", "regulators", "similar"}

            if gene_role in [GeneRole.MASTER_REGULATOR.value, GeneRole.TRANSCRIPTION_FACTOR.value]:
                required.update({"targets", "vulnerability", "lincs"})
            else:
                # Effectors/isolated genes need protein-level analysis
                required.update({"ppi", "super_enhancers"})

            # Always include cross-cell for comprehensive
            required.add("cross_cell")

        # === Determine what's still pending ===
        pending = required - completed

        if not pending:
            # All required analyses complete
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["complete"],
                "workflow_complete": True
            }

        # === Group into batches for parallel execution ===

        # Core batch: perturbation + regulators + targets
        core_pending = pending & {"perturbation", "regulators", "targets"}
        if len(core_pending) > 1:
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["batch_core"]
            }

        # External batch: ppi + lincs + super_enhancers
        external_pending = pending & {"ppi", "lincs", "super_enhancers"}
        if len(external_pending) > 1:
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["batch_external"]
            }

        # Insights batch: similar + vulnerability + cross_cell
        insights_pending = pending & {"similar", "vulnerability", "cross_cell"}
        if len(insights_pending) > 1:
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["batch_insights"]
            }

        # Single pending action
        next_action = list(pending)[0]
        return {
            "current_step": "decide_next_steps",
            "next_actions": [next_action]
        }

    def _route_next_action(self, state: PerturbationAnalysisState) -> str:
        """Route to the next workflow node based on state."""
        if state.get("error_message"):
            return "error"

        next_actions = state.get("next_actions", [])
        if not next_actions:
            return "complete"

        action = next_actions[0]

        # Map action names to node names
        action_map = {
            "batch_core": "batch_core",
            "batch_external": "batch_external",
            "batch_insights": "batch_insights",
            "perturbation": "perturbation",
            "regulators": "regulators",
            "targets": "targets",
            "ppi": "ppi",
            "lincs": "lincs",
            "super_enhancers": "super_enhancers",
            "similar": "similar",
            "vulnerability": "vulnerability",
            "cross_cell": "cross_cell",
            "complete": "complete",
            "error": "error"
        }

        return action_map.get(action, "error")

    # =========================================================================
    # BATCH PROCESSING NODES (Parallel Execution)
    # =========================================================================

    async def _batch_core_analysis(self, state: PerturbationAnalysisState) -> Dict:
        """Run core analyses in parallel: perturbation + regulators + targets."""
        logger.info("Running batch core analysis (parallel)")

        completed = set(state.get("completed_actions", []))
        tasks = []
        task_names = []

        if "perturbation" not in completed:
            tasks.append(self._run_perturbation_impl(state))
            task_names.append("perturbation")

        if "regulators" not in completed:
            tasks.append(self._analyze_regulators_impl(state))
            task_names.append("regulators")

        if "targets" not in completed:
            tasks.append(self._analyze_targets_impl(state))
            task_names.append("targets")

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        updates = {
            "current_step": "batch_core_analysis",
            "completed_actions": list(completed | set(task_names))
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {name}: {result}")
                continue
            if name == "perturbation":
                updates["perturbation_result"] = result
            elif name == "regulators":
                updates["regulators_analysis"] = result
            elif name == "targets":
                updates["targets_analysis"] = result

        return updates

    async def _batch_external_data(self, state: PerturbationAnalysisState) -> Dict:
        """Run external data fetches in parallel: PPI + LINCS + super-enhancers."""
        logger.info("Running batch external data (parallel)")

        completed = set(state.get("completed_actions", []))
        tasks = []
        task_names = []

        if "ppi" not in completed:
            tasks.append(self._fetch_ppi_impl(state))
            task_names.append("ppi")

        if "lincs" not in completed:
            tasks.append(self._fetch_lincs_impl(state))
            task_names.append("lincs")

        if "super_enhancers" not in completed:
            tasks.append(self._check_super_enhancers_impl(state))
            task_names.append("super_enhancers")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        updates = {
            "current_step": "batch_external_data",
            "completed_actions": list(completed | set(task_names))
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {name}: {result}")
                continue
            if name == "ppi":
                updates["ppi_interactions"] = result
            elif name == "lincs":
                updates["lincs_effects"] = result
            elif name == "super_enhancers":
                updates["super_enhancer_status"] = result

        return updates

    async def _batch_insights(self, state: PerturbationAnalysisState) -> Dict:
        """Run insight analyses in parallel: similar genes + vulnerability + cross-cell."""
        logger.info("Running batch insights (parallel)")

        completed = set(state.get("completed_actions", []))
        tasks = []
        task_names = []

        if "similar" not in completed:
            tasks.append(self._find_similar_genes_impl(state))
            task_names.append("similar")

        if "vulnerability" not in completed:
            tasks.append(self._analyze_vulnerability_impl(state))
            task_names.append("vulnerability")

        if "cross_cell" not in completed:
            tasks.append(self._cross_cell_comparison_impl(state))
            task_names.append("cross_cell")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        updates = {
            "current_step": "batch_insights",
            "completed_actions": list(completed | set(task_names))
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {name}: {result}")
                continue
            if name == "similar":
                updates["similar_genes"] = result
            elif name == "vulnerability":
                updates["vulnerability_analysis"] = result
            elif name == "cross_cell":
                updates["cross_cell_comparison"] = result

        return updates

    # =========================================================================
    # INDIVIDUAL ANALYSIS NODE WRAPPERS
    # =========================================================================

    async def _run_perturbation(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for perturbation analysis."""
        result = await self._run_perturbation_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "run_perturbation",
            "perturbation_result": result,
            "completed_actions": list(completed | {"perturbation"})
        }

    async def _analyze_regulators(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for regulators analysis."""
        result = await self._analyze_regulators_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "analyze_regulators",
            "regulators_analysis": result,
            "completed_actions": list(completed | {"regulators"})
        }

    async def _analyze_targets(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for targets analysis."""
        result = await self._analyze_targets_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "analyze_targets",
            "targets_analysis": result,
            "completed_actions": list(completed | {"targets"})
        }

    async def _fetch_ppi(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for PPI fetch."""
        result = await self._fetch_ppi_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "fetch_ppi",
            "ppi_interactions": result,
            "completed_actions": list(completed | {"ppi"})
        }

    async def _fetch_lincs(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for LINCS fetch."""
        result = await self._fetch_lincs_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "fetch_lincs",
            "lincs_effects": result,
            "completed_actions": list(completed | {"lincs"})
        }

    async def _check_super_enhancers(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for super-enhancer check."""
        result = await self._check_super_enhancers_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "check_super_enhancers",
            "super_enhancer_status": result,
            "completed_actions": list(completed | {"super_enhancers"})
        }

    async def _find_similar_genes(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for similar genes."""
        result = await self._find_similar_genes_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "find_similar_genes",
            "similar_genes": result,
            "completed_actions": list(completed | {"similar"})
        }

    async def _analyze_vulnerability(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for vulnerability analysis."""
        result = await self._analyze_vulnerability_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "analyze_vulnerability",
            "vulnerability_analysis": result,
            "completed_actions": list(completed | {"vulnerability"})
        }

    async def _cross_cell_comparison(self, state: PerturbationAnalysisState) -> Dict:
        """Node wrapper for cross-cell comparison."""
        result = await self._cross_cell_comparison_impl(state)
        completed = set(state.get("completed_actions", []))
        return {
            "current_step": "cross_cell_comparison",
            "cross_cell_comparison": result,
            "completed_actions": list(completed | {"cross_cell"})
        }

    # =========================================================================
    # ANALYSIS IMPLEMENTATIONS (Wrap existing GREmLN tools)
    # =========================================================================

    async def _run_perturbation_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Run perturbation analysis using existing GREmLN tools."""
        from tools.loader import load_network
        from tools.perturb import (
            simulate_knockdown_with_embeddings,
            simulate_overexpression_with_embeddings,
            simulate_knockdown,
            simulate_overexpression
        )

        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]
        perturbation_type = state.get("perturbation_type", "knockdown")

        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"
        network_df = load_network(network_path)

        # Try with embeddings, fall back to network-only
        try:
            model = self._get_model()
            if perturbation_type == "knockdown":
                result = simulate_knockdown_with_embeddings(
                    network_df, ensembl_id, model,
                    depth=2, top_k=25, alpha=0.7
                )
            else:
                result = simulate_overexpression_with_embeddings(
                    network_df, ensembl_id, model,
                    fold_change=2.0, depth=2, top_k=25, alpha=0.7
                )
            result["embedding_enhanced"] = True
        except Exception as e:
            logger.warning(f"Model unavailable, using network-only: {e}")
            if perturbation_type == "knockdown":
                result = simulate_knockdown(network_df, ensembl_id, depth=2, top_k=25)
            else:
                result = simulate_overexpression(network_df, ensembl_id, fold_change=2.0, depth=2, top_k=25)
            result["embedding_enhanced"] = False

        return result

    async def _analyze_regulators_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Get upstream regulators using existing GREmLN tools."""
        from tools.loader import load_network
        from tools.perturb import get_regulators

        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]

        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"
        network_df = load_network(network_path)

        return get_regulators(network_df, ensembl_id, max_regulators=50)

    async def _analyze_targets_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Get downstream targets using existing GREmLN tools."""
        from tools.loader import load_network
        from tools.perturb import get_targets

        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]

        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"
        network_df = load_network(network_path)

        return get_targets(network_df, ensembl_id, max_targets=50)

    async def _fetch_ppi_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch protein-protein interactions from STRING."""
        gene_symbol = state.get("gene_symbol", state["gene"])

        try:
            return self.string_client.get_interactions(gene_symbol, min_score=400, limit=25)
        except Exception as e:
            logger.error(f"STRING API error: {e}")
            return {"error": str(e), "interactions": []}

    async def _fetch_lincs_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch LINCS knockdown effects."""
        from tools.lincs import get_knockdown_effects

        gene_symbol = state.get("gene_symbol", state["gene"])

        try:
            return get_knockdown_effects(gene_symbol, direction="any", top_k=20)
        except Exception as e:
            logger.error(f"LINCS error: {e}")
            return {"error": str(e), "effects": []}

    async def _check_super_enhancers_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Check super-enhancer status."""
        from tools.super_enhancers import get_super_enhancer_info

        gene_symbol = state.get("gene_symbol", state["gene"])

        try:
            return get_super_enhancer_info(gene_symbol)
        except Exception as e:
            logger.error(f"Super-enhancer error: {e}")
            return {"error": str(e), "has_super_enhancer": False}

    async def _find_similar_genes_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Find similar genes using embeddings."""
        ensembl_id = state["ensembl_id"]

        try:
            model = self._get_model()
            if not model.is_gene_in_vocab(ensembl_id):
                return {"error": f"Gene {ensembl_id} not in model vocabulary"}

            similar_df = model.get_top_similar_genes(ensembl_id, top_k=20)
            if similar_df is None:
                return {"error": "Could not compute similarities"}

            similar_genes = []
            for _, row in similar_df.iterrows():
                target_ensembl = row["ensembl_id"]
                symbol = self.gene_mapper.ensembl_to_symbol(target_ensembl) or target_ensembl
                similar_genes.append({
                    "gene_symbol": symbol,
                    "ensembl_id": target_ensembl,
                    "similarity": round(row["similarity"], 4)
                })

            return {"similar_genes": similar_genes}
        except Exception as e:
            logger.error(f"Similarity error: {e}")
            return {"error": str(e), "similar_genes": []}

    async def _analyze_vulnerability_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Analyze network vulnerability for therapeutic targeting."""
        # Simplified vulnerability based on network position
        network_context = state.get("network_context", {})
        gene_symbol = state.get("gene_symbol", state["gene"])

        num_targets = network_context.get("num_targets", 0)
        num_regulators = network_context.get("num_regulators", 0)

        # Calculate vulnerability score
        vulnerability_score = (
            num_targets * 1.0 +
            (1 / (num_regulators + 1)) * 5  # Less regulated = harder to compensate
        )

        return {
            "gene": gene_symbol,
            "vulnerability_score": round(vulnerability_score, 2),
            "hub_score": num_targets,
            "regulator_count": num_regulators,
            "therapeutic_potential": "high" if vulnerability_score > 50 else "moderate" if vulnerability_score > 10 else "low"
        }

    async def _cross_cell_comparison_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Compare gene across all cell types."""
        from tools.loader import load_network

        ensembl_id = state["ensembl_id"]
        gene_symbol = state.get("gene_symbol", state["gene"])

        results = {}
        for cell_type in CellType:
            network_path = self.NETWORKS_DIR / cell_type.value / "network.tsv"
            if not network_path.exists():
                continue

            network_df = load_network(network_path)

            targets = network_df[network_df["regulator"] == ensembl_id]
            regulators = network_df[network_df["target"] == ensembl_id]

            results[cell_type.value] = {
                "num_targets": len(targets),
                "num_regulators": len(regulators),
                "in_network": len(targets) > 0 or len(regulators) > 0
            }

        return {
            "gene": gene_symbol,
            "cell_type_comparison": results
        }

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    async def _generate_report(self, state: PerturbationAnalysisState) -> Dict:
        """Generate comprehensive analysis report."""
        import time

        gene_symbol = state.get("gene_symbol", state["gene"])
        gene_role = state.get("gene_role", "unknown")

        # Compile therapeutic suggestions based on results
        suggestions = []

        # Based on gene role
        if gene_role == GeneRole.EFFECTOR.value:
            ppi = state.get("ppi_interactions") or {}
            if ppi.get("interactions"):
                suggestions.append({
                    "action": "Target upstream regulators or protein partners",
                    "reason": f"{gene_symbol} is an effector - consider targeting its regulators or PPI partners",
                    "priority": "high"
                })

        # Based on super-enhancer status
        se_status = state.get("super_enhancer_status") or {}
        if se_status.get("has_super_enhancer"):
            suggestions.append({
                "action": "Consider BRD4/BET inhibitors",
                "reason": f"{gene_symbol} has super-enhancers and may respond to epigenetic drugs",
                "priority": "medium"
            })

        # Based on vulnerability
        vuln = state.get("vulnerability_analysis") or {}
        if vuln.get("therapeutic_potential") == "high":
            suggestions.append({
                "action": "Prioritize as drug target",
                "reason": f"{gene_symbol} has high network vulnerability score",
                "priority": "high"
            })

        # Calculate execution time
        metadata = state.get("analysis_metadata", {})
        start_time = metadata.get("start_time", time.time())
        execution_time = time.time() - start_time

        report = {
            "summary": {
                "gene": gene_symbol,
                "ensembl_id": state.get("ensembl_id"),
                "cell_type": state.get("cell_type"),
                "gene_role": gene_role,
                "perturbation_type": state.get("perturbation_type")
            },
            "perturbation_effects": state.get("perturbation_result"),
            "network_analysis": {
                "context": state.get("network_context"),  # Always available (num_targets, num_regulators)
                "regulators": state.get("regulators_analysis"),
                "targets": state.get("targets_analysis"),
                "vulnerability": state.get("vulnerability_analysis")
            },
            "external_data": {
                "protein_interactions": state.get("ppi_interactions"),
                "lincs_knockdown": state.get("lincs_effects"),
                "super_enhancers": state.get("super_enhancer_status")
            },
            "embedding_analysis": {
                "similar_genes": state.get("similar_genes"),
                "embedding_enhanced": state.get("embedding_enhanced", False)
            },
            "cross_cell_comparison": state.get("cross_cell_comparison"),
            "therapeutic_suggestions": suggestions,
            "metadata": {
                "execution_time_seconds": round(execution_time, 2),
                "completed_analyses": state.get("completed_actions", []),
                "workflow_version": metadata.get("workflow_version", "1.0.0")
            }
        }

        logger.info(f"Report generated in {execution_time:.2f}s")

        return {
            "current_step": "generate_report",
            "comprehensive_report": report,
            "workflow_complete": True,
            "therapeutic_suggestions": suggestions
        }

    # =========================================================================
    # LLM SYNTHESIS (Optional)
    # =========================================================================

    async def _synthesize_insights(self, state: PerturbationAnalysisState) -> Dict:
        """Generate LLM-powered biological interpretation of results."""

        if not state.get("include_llm_insights", False):
            return {"llm_insights": None}

        if not self.ollama_available:
            logger.warning("LLM insights requested but Ollama not available")
            return {"llm_insights": {"error": "Ollama not available", "llm_powered": False}}

        try:
            insights = await self._call_llm_synthesis(state)
            # Add llm_insights to the comprehensive_report
            report = state.get("comprehensive_report", {})
            if report:
                report["llm_insights"] = insights
            return {"llm_insights": insights, "comprehensive_report": report}
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return {"llm_insights": {"error": str(e), "llm_powered": False}}

    async def _call_llm_synthesis(self, state: PerturbationAnalysisState) -> Dict:
        """Call Ollama to synthesize biological insights."""

        gene = state.get("gene_symbol", state["gene"])
        cell_type = state.get("cell_type", "unknown")
        gene_role = state.get("gene_role", "unknown")
        perturbation_type = state.get("perturbation_type", "knockdown")

        # Build context from analysis results
        perturbation = state.get("perturbation_result") or {}
        ppi = state.get("ppi_interactions") or {}
        lincs = state.get("lincs_effects") or {}
        se = state.get("super_enhancer_status") or {}
        similar = state.get("similar_genes") or {}

        prompt = f"""Analyze the biological implications of {gene} {perturbation_type} in {cell_type} cells.

## Gene Context
- Gene Role: {gene_role}
- Network Position: {state.get('network_context', {})}

## Perturbation Analysis Results
- Total Affected Genes: {perturbation.get('total_affected_genes', 0)}
- Top Targets: {perturbation.get('top_affected', [])[:5]}

## Protein Interactions (STRING)
- Interaction Partners: {len(ppi.get('interactions', []))} found
- Key Partners: {[p.get('preferredName') for p in ppi.get('interactions', [])[:5]]}

## LINCS Knockdown Data
- Experimental Effects: {lincs.get('total_effects', 0)} genes affected in LINCS

## Super-Enhancer Status
- Has Super-Enhancer: {se.get('has_super_enhancer', False)}
- BRD4 Sensitive: {se.get('has_super_enhancer', False)}

## Similar Genes (Embedding-based)
- Top Similar: {[g.get('symbol') for g in similar.get('similar_genes', [])[:5]]}

Provide a biological interpretation in this EXACT JSON format:
{{
  "mechanism_summary": "2-3 sentence explanation of what this perturbation does mechanistically",
  "therapeutic_implications": "1-2 sentences on drug development relevance",
  "key_pathways_affected": ["pathway1", "pathway2", "pathway3"],
  "confidence_level": "high|medium|low",
  "confidence_rationale": "why this confidence level",
  "follow_up_suggestions": ["suggestion1", "suggestion2"],
  "biological_interpretation": "3-4 sentence narrative synthesis suitable for a research report"
}}

Base your analysis on the data provided. Be scientifically accurate. If data is limited, acknowledge uncertainty.
Provide only the JSON, no additional text."""

        system_prompt = """You are an expert molecular biologist specializing in gene regulatory networks and perturbation biology.
Provide scientifically accurate, evidence-based analysis. When data is limited, acknowledge uncertainty rather than speculating."""

        timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))

        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.ollama_client.chat,
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.ollama_temperature,
                    "num_predict": self.ollama_max_tokens
                }
            ),
            timeout=timeout
        )

        content = response['message']['content']
        parsed = self._parse_llm_json(content)
        parsed["llm_powered"] = True
        parsed["model"] = self.ollama_model

        return parsed

    def _parse_llm_json(self, response_text: str) -> dict:
        """Extract and validate JSON from LLM response."""

        # Handle markdown-wrapped JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        # Find JSON object
        if not response_text.strip().startswith('{'):
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1:
                response_text = response_text[json_start:json_end+1]

        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return {
                "parse_error": str(e),
                "raw_response": response_text[:500]
            }

    async def _handle_error(self, state: PerturbationAnalysisState) -> Dict:
        """Handle workflow errors gracefully."""
        error_msg = state.get("error_message", "Unknown error")
        logger.error(f"Workflow error: {error_msg}")

        return {
            "current_step": "handle_error",
            "workflow_complete": True,
            "comprehensive_report": {
                "error": error_msg,
                "partial_results": {
                    "gene": state.get("gene"),
                    "cell_type": state.get("cell_type"),
                    "completed_actions": state.get("completed_actions", [])
                }
            }
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def run(
        self,
        gene: str,
        cell_type: str = "epithelial_cell",
        perturbation_type: str = "knockdown",
        analysis_depth: str = "comprehensive",
        include_llm_insights: bool = False
    ) -> Dict:
        """
        Run the complete perturbation analysis workflow.

        Args:
            gene: Gene symbol or Ensembl ID
            cell_type: Cell type for network context
            perturbation_type: knockdown, overexpression, or similarity
            analysis_depth: basic, comprehensive, or focused
            include_llm_insights: Whether to generate LLM-powered biological interpretation

        Returns:
            Comprehensive analysis report
        """
        initial_state: PerturbationAnalysisState = {
            "gene": gene,
            "cell_type": cell_type,
            "perturbation_type": perturbation_type,
            "analysis_depth": analysis_depth,
            "ensembl_id": None,
            "gene_symbol": None,
            "gene_role": None,
            "current_step": "start",
            "workflow_complete": False,
            "error_message": None,
            "next_actions": [],
            "completed_actions": [],
            "network_context": None,
            "perturbation_result": None,
            "regulators_analysis": None,
            "targets_analysis": None,
            "similar_genes": None,
            "embedding_enhanced": False,
            "ppi_interactions": None,
            "lincs_effects": None,
            "super_enhancer_status": None,
            "cross_cell_comparison": None,
            "vulnerability_analysis": None,
            "therapeutic_suggestions": None,
            "comprehensive_report": None,
            "analysis_metadata": {},
            "include_llm_insights": include_llm_insights,
            "llm_insights": None
        }

        logger.info(f"Starting workflow for {gene} ({perturbation_type}, {analysis_depth})")

        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)

        return final_state.get("comprehensive_report", {"error": "No report generated"})


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the GREmLN workflow."""
    workflow = GREmLNWorkflow()

    # Run comprehensive analysis
    result = await workflow.run(
        gene="TP53",
        cell_type="epithelial_cell",
        perturbation_type="knockdown",
        analysis_depth="comprehensive"
    )

    import json
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
