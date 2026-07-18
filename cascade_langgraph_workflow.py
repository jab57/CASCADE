#!/usr/bin/env python3
"""
CASCADE LangGraph Workflow - SKETCH/DRAFT
=========================================

Multi-agent orchestration for in silico gene perturbation analysis.

This module wraps the existing CASCADE analysis tools into a LangGraph workflow that:
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
import threading

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
    network_source: str                         # "cell_type" (default) or "tcga"
    tcga_network: Optional[str]                 # TCGA cancer type (brca, coad, ...) when network_source="tcga"
    top_k: int                                  # Number of top-affected genes returned by perturbation propagation (default 25)

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
    dorothea_regulons: Optional[Dict]           # DoRothEA TF regulon validation
    depmap_essentiality: Optional[Dict]         # DepMap CRISPR essentiality scores
    cbioportal_tumor_data: Optional[Dict]       # cBioPortal TCGA primary tumor expression + alteration

    # === Cross-Cell Analysis ===
    cross_cell_comparison: Optional[Dict]       # Same gene across cell types

    # === Therapeutic Insights ===
    vulnerability_analysis: Optional[Dict]      # Network vulnerability scores
    therapeutic_suggestions: Optional[List]     # Drug target recommendations

    # === Final Output ===
    comprehensive_report: Optional[Dict]        # Final compiled report
    failed_analyses: Optional[List[Dict]]       # Errors from batch tasks, if any
    analysis_metadata: Dict                     # Timing, versions, etc.

    # === LLM Insights (Optional) ===
    include_llm_insights: bool                  # Whether to generate LLM synthesis
    llm_insights: Optional[Dict]                # LLM-generated biological interpretation


# =============================================================================
# EVIDENCE SYNTHESIS HELPERS
# =============================================================================

def _build_role_context(gene_role: str, total_affected: int, ppi_count: int,
                        agreement_count: int, dorothea_validated: bool = False) -> Dict[str, str]:
    """Return context string and primary evidence source based on gene role."""
    if gene_role in ("master_regulator", "transcription_factor", "minor_regulator"):
        primary = "network_propagation"
        if agreement_count > 0:
            context = (f"This {gene_role.replace('_', ' ')}'s transcriptional effects "
                      f"are captured by network propagation, with {agreement_count} "
                      f"prediction(s) experimentally confirmed by LINCS knockdown data.")
            primary = "network_propagation + lincs_experimental"
        else:
            context = (f"This {gene_role.replace('_', ' ')}'s effects are predicted by "
                      f"network propagation ({total_affected} affected genes). "
                      f"No overlapping LINCS experimental data available for validation.")
        if dorothea_validated:
            context += " TF classification confirmed by DoRothEA curated regulons."
            primary += " + dorothea"
    elif gene_role == "effector":
        primary = "string_ppi" if ppi_count > 0 else "embedding_similarity"
        context = (f"This gene has no transcriptional targets in the network (effector role). "
                  f"Network propagation is uninformative. "
                  f"{'STRING protein interactions' if ppi_count > 0 else 'Embedding similarity'} "
                  f"provides the most relevant evidence for this gene type.")
    elif gene_role == "isolated":
        primary = "string_ppi" if ppi_count > 0 else "embedding_similarity"
        context = (f"This gene is not present in the cell-type regulatory network. "
                  f"Network-based analyses (propagation, regulators, targets) are unavailable. "
                  f"{'STRING protein interactions' if ppi_count > 0 else 'Embedding similarity'} "
                  f"provides the most relevant evidence.")
    else:
        primary = "unknown"
        context = "Gene role could not be determined."
    return {"context": context, "primary": primary}


def _build_key_findings(gene_role: str, multi_source: list, agreements: list,
                        disagreements: list, total_affected: int, ppi_count: int,
                        lincs_count: int, dorothea_validated: bool = False) -> List[str]:
    """Generate human-readable key findings from synthesis results."""
    findings = []

    if len(multi_source) > 0:
        findings.append(
            f"{len(multi_source)} gene(s) supported by multiple independent evidence sources."
        )

    if len(agreements) > 0:
        findings.append(
            f"{len(agreements)} gene(s) confirmed by both network propagation and "
            f"LINCS experimental knockdown data (directional agreement)."
        )

    if len(disagreements) > 0:
        findings.append(
            f"{len(disagreements)} gene(s) show directional disagreement between "
            f"network prediction and LINCS experimental data — requires investigation."
        )
    elif len(agreements) > 0:
        findings.append("No directional disagreements between network and experimental evidence.")

    # PPI-only genes (in STRING but not network propagation)
    ppi_only = [g for g in multi_source
                if "string_ppi" in g["sources"] and "network_propagation" not in g["sources"]]
    if ppi_only:
        findings.append(
            f"STRING identifies {len(ppi_only)} protein interaction partner(s) "
            f"not detected at the mRNA level by network propagation."
        )

    if gene_role in ("effector", "isolated") and total_affected == 0:
        if ppi_count > 0:
            findings.append(
                f"Gene has no transcriptional targets but {ppi_count} STRING protein "
                f"interaction partners — protein-level evidence is primary."
            )
        else:
            findings.append(
                "Gene has no transcriptional targets and no STRING interactions. "
                "Embedding similarity is the only available evidence source."
            )

    if dorothea_validated:
        findings.append(
            "TF classification validated by DoRothEA curated regulons (multi-evidence: "
            "literature, ChIP-seq, motifs)."
        )

    return findings


# =============================================================================
# WORKFLOW CLASS
# =============================================================================

class CascadeWorkflow:
    """
    LangGraph workflow for comprehensive gene perturbation analysis.

    Orchestrates multiple analysis tools into a coherent workflow that:
    1. Resolves gene identity and determines network role
    2. Routes to appropriate analyses based on gene type
    3. Runs independent analyses in parallel
    4. Integrates results and generates recommendations

    Example:
        >>> workflow = CascadeWorkflow()
        >>> result = await workflow.run(
        ...     gene="TP53",
        ...     cell_type="epithelial_cell",
        ...     perturbation_type="knockdown"
        ... )
    """

    def __init__(self):
        """Initialize workflow with CASCADE components."""
        # Import existing CASCADE components
        from pathlib import Path
        from tools.loader import load_network, get_available_cell_types, MODEL_PATH
        from tools.gene_id_mapper import get_mapper
        from tools.ppi.string_client import get_string_client

        self.BASE_DIR = Path(__file__).parent
        self.NETWORKS_DIR = self.BASE_DIR / "data" / "networks"
        self.MODEL_PATH = MODEL_PATH

        # Initialize components
        self.gene_mapper = get_mapper()
        self.string_client = get_string_client()
        self._model = None  # Lazy loaded
        self._model_lock = threading.Lock()  # Thread-safe lazy init

        # Rate limiting: max concurrent external API calls (STRING, cBioPortal)
        _api_limit = int(os.getenv('API_RATE_LIMIT', '3'))
        self._api_semaphore = asyncio.Semaphore(_api_limit)

        # LLM configuration (multi-provider)
        self.use_llm = os.getenv('USE_LLM_INSIGHTS', 'false').lower() == 'true'
        self.llm_client = None
        self.ollama_client = None  # backward-compat alias
        self.ollama_available = self._initialize_llm() if self.use_llm else False
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        self.ollama_temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.3'))
        self.ollama_max_tokens = int(os.getenv('OLLAMA_MAX_TOKENS', '2000'))

        # Build the workflow graph
        self.workflow = self._create_workflow()
        logger.info("CASCADE LangGraph workflow initialized")

    def _get_model(self):
        """Lazy load the GREmLN model (thread-safe via lock)."""
        with self._model_lock:
            if self._model is None:
                from tools.model_inference import CascadeModel
                self._model = CascadeModel(self.MODEL_PATH)
                self._model.load()
        return self._model

    def _initialize_llm(self) -> bool:
        """Initialize LLM client based on LLM_PROVIDER env var (default: ollama)."""
        provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
        if provider == 'ollama':
            return self._initialize_ollama_provider()
        elif provider in ('openai', 'openai_compatible'):
            return self._initialize_openai_provider()
        elif provider == 'anthropic':
            return self._initialize_anthropic_provider()
        else:
            logger.error(f"Unknown LLM_PROVIDER: {provider}. Use: ollama | openai | openai_compatible | anthropic")
            return False

    # backward-compat alias
    def _initialize_ollama(self) -> bool:
        return self._initialize_llm()

    def _initialize_ollama_provider(self) -> bool:
        """Initialize Ollama client (auto-detects local vs cloud)."""
        try:
            import ollama
        except ImportError:
            logger.warning("ollama package not installed. Run: pip install ollama")
            return False

        api_key = os.getenv('OLLAMA_API_KEY')
        if api_key:
            logger.info("Using Ollama Cloud (API key detected)")
            self.llm_client = ollama.Client(
                host='https://ollama.com',
                headers={'Authorization': f'Bearer {api_key}'}
            )
        else:
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            logger.info(f"Using local Ollama at {host}")
            self.llm_client = ollama.Client(host=host)

        try:
            models_response = self.llm_client.list()
            available_models = []
            models_list = (models_response.models if hasattr(models_response, 'models')
                           else models_response.get('models', []) if isinstance(models_response, dict)
                           else models_response)
            for m in models_list:
                name = (m.get('name') or m.get('model') if isinstance(m, dict)
                        else getattr(m, 'model', None) or getattr(m, 'name', None))
                if name:
                    available_models.append(name)

            model_name = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
            if model_name not in available_models:
                logger.error(f"Ollama model '{model_name}' not found. Available: {available_models}")
                logger.error(f"Run: ollama pull {model_name}")
                return False

            logger.info(f"Ollama available, model: {model_name}")
            self.ollama_client = self.llm_client  # backward-compat
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            logger.warning(f"To use Ollama: 1) Install from https://ollama.com  2) Run: ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}")
            return False

    def _initialize_openai_provider(self) -> bool:
        """Initialize OpenAI or OpenAI-compatible provider."""
        try:
            import openai
            api_key = os.getenv('LLM_API_KEY')
            if not api_key:
                logger.error("LLM_API_KEY required for openai/openai_compatible provider")
                return False
            kwargs = {"api_key": api_key}
            api_base = os.getenv('LLM_API_BASE')
            if api_base:
                kwargs["base_url"] = api_base
            self.llm_client = openai.AsyncOpenAI(**kwargs)
            model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
            logger.info(f"OpenAI provider initialized, model: {model}")
            return True
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"OpenAI provider initialization failed: {e}")
            return False

    def _initialize_anthropic_provider(self) -> bool:
        """Initialize Anthropic provider."""
        try:
            import anthropic
            api_key = os.getenv('LLM_API_KEY')
            if not api_key:
                logger.error("LLM_API_KEY required for anthropic provider")
                return False
            self.llm_client = anthropic.AsyncAnthropic(api_key=api_key)
            model = os.getenv('LLM_MODEL', 'claude-haiku-4-5-20251001')
            logger.info(f"Anthropic provider initialized, model: {model}")
            return True
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Anthropic provider initialization failed: {e}")
            return False

    async def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Dispatch LLM call to the configured provider."""
        provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
        if provider == 'ollama':
            return await self._call_ollama_provider(prompt, system_prompt)
        elif provider in ('openai', 'openai_compatible'):
            return await self._call_openai_provider(prompt, system_prompt)
        elif provider == 'anthropic':
            return await self._call_anthropic_provider(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    async def _call_ollama_provider(self, prompt: str, system_prompt: str = None) -> str:
        """Call Ollama with retry logic."""
        timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm_client.chat,
                        model=self.ollama_model,
                        messages=messages,
                        options={"temperature": self.ollama_temperature,
                                 "num_predict": self.ollama_max_tokens}
                    ),
                    timeout=timeout
                )
                content = response['message']['content']
                if not content or len(content.strip()) < 10:
                    raise ValueError("Empty response from Ollama")
                return content
            except Exception as e:
                if attempt == 1:
                    raise
                logger.warning(f"Ollama call failed (attempt 1): {e}, retrying...")
                await asyncio.sleep(1)

    async def _call_openai_provider(self, prompt: str, system_prompt: str = None) -> str:
        """Call OpenAI or OpenAI-compatible provider with retry logic."""
        model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.ollama_temperature,
                        max_tokens=self.ollama_max_tokens
                    ),
                    timeout=timeout
                )
                content = response.choices[0].message.content
                if not content or len(content.strip()) < 10:
                    raise ValueError("Empty response from OpenAI provider")
                return content
            except Exception as e:
                if attempt == 1:
                    raise
                logger.warning(f"OpenAI call failed (attempt 1): {e}, retrying...")
                await asyncio.sleep(1)

    async def _call_anthropic_provider(self, prompt: str, system_prompt: str = None) -> str:
        """Call Anthropic provider with retry logic."""
        model = os.getenv('LLM_MODEL', 'claude-haiku-4-5-20251001')
        timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))

        for attempt in range(2):
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": self.ollama_max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                response = await asyncio.wait_for(
                    self.llm_client.messages.create(**kwargs),
                    timeout=timeout
                )
                content = response.content[0].text
                if not content or len(content.strip()) < 10:
                    raise ValueError("Empty response from Anthropic provider")
                return content
            except Exception as e:
                if attempt == 1:
                    raise
                logger.warning(f"Anthropic call failed (attempt 1): {e}, retrying...")
                await asyncio.sleep(1)

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
        workflow.add_node("run_all_batches", self._run_all_batches)

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
                # Concurrent all-batch route (comprehensive mode)
                "run_all_batches": "run_all_batches",

                # Batch processing routes (focused/basic or leftover single batches)
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

        # All-batch node flows back to routing (handles any remaining single items)
        workflow.add_edge("run_all_batches", "decide_next_steps")

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
        from tools.loader import load_network, load_tcga_network

        network_source = state.get("network_source", "cell_type")

        if network_source == "tcga":
            tcga_network = state.get("tcga_network")
            network_df = load_tcga_network(tcga_network)
            if isinstance(network_df, dict) and "error" in network_df:
                return {
                    "current_step": "analyze_network_context",
                    "error_message": network_df["error"],
                    "next_actions": ["error"]
                }
            # TCGA networks use gene symbols; match on symbol instead of Ensembl ID
            gene_id = state.get("gene_symbol") or state["ensembl_id"]
        else:
            cell_type = state.get("cell_type", "epithelial_cell")
            network_path = self.NETWORKS_DIR / cell_type / "network.tsv"
            if not network_path.exists():
                return {
                    "current_step": "analyze_network_context",
                    "error_message": f"Network not found for cell type: {cell_type}",
                    "next_actions": ["error"]
                }
            network_df = load_network(network_path)
            gene_id = state["ensembl_id"]

        # Count targets and regulators
        # TCGA networks use gene symbols; cell-type networks use Ensembl IDs
        targets = network_df[network_df["regulator"] == gene_id]
        regulators = network_df[network_df["target"] == gene_id]

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
            required = {"perturbation", "regulators", "similar", "dorothea", "depmap", "cbioportal"}

            if gene_role in [GeneRole.MASTER_REGULATOR.value, GeneRole.TRANSCRIPTION_FACTOR.value]:
                required.update({"targets", "vulnerability", "lincs"})
            else:
                # Effectors/isolated genes need protein-level analysis
                required.update({"ppi", "super_enhancers"})

            # Isolated genes are not in the network at all — regulators analysis is uninformative
            if gene_role == GeneRole.ISOLATED.value:
                required.discard("regulators")

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

        core_pending = pending & {"perturbation", "regulators", "targets"}
        external_pending = pending & {"ppi", "lincs", "super_enhancers", "dorothea", "depmap", "cbioportal"}
        insights_pending = pending & {"similar", "vulnerability", "cross_cell"}

        # Count how many batch groups have work to do
        batch_groups_with_work = sum([
            len(core_pending) > 0,
            len(external_pending) > 0,
            len(insights_pending) > 0,
        ])

        # If multiple batch groups are pending, run them all concurrently
        if batch_groups_with_work > 1:
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["run_all_batches"]
            }

        # Single batch group remaining — run its individual batch node
        if len(core_pending) > 1:
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["batch_core"]
            }

        if len(external_pending) > 1:
            return {
                "current_step": "decide_next_steps",
                "next_actions": ["batch_external"]
            }

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
            "run_all_batches": "run_all_batches",
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
        failed_analyses = list(state.get("failed_analyses") or [])
        updates = {
            "current_step": "batch_core_analysis",
            "completed_actions": list(completed | set(task_names))
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {name}: {result}")
                failed_analyses.append({"analysis": name, "error": str(result), "batch": "core"})
                continue
            if name == "perturbation":
                updates["perturbation_result"] = result
            elif name == "regulators":
                updates["regulators_analysis"] = result
            elif name == "targets":
                updates["targets_analysis"] = result

        updates["failed_analyses"] = failed_analyses or None
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

        if "dorothea" not in completed:
            tasks.append(self._fetch_dorothea_impl(state))
            task_names.append("dorothea")

        if "depmap" not in completed:
            tasks.append(self._fetch_depmap_impl(state))
            task_names.append("depmap")

        if "cbioportal" not in completed:
            tasks.append(self._fetch_cbioportal_impl(state))
            task_names.append("cbioportal")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_analyses = list(state.get("failed_analyses") or [])
        updates = {
            "current_step": "batch_external_data",
            "completed_actions": list(completed | set(task_names))
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {name}: {result}")
                failed_analyses.append({"analysis": name, "error": str(result), "batch": "external"})
                continue
            if name == "ppi":
                updates["ppi_interactions"] = result
            elif name == "lincs":
                updates["lincs_effects"] = result
            elif name == "super_enhancers":
                updates["super_enhancer_status"] = result
            elif name == "dorothea":
                updates["dorothea_regulons"] = result
            elif name == "depmap":
                updates["depmap_essentiality"] = result
            elif name == "cbioportal":
                updates["cbioportal_tumor_data"] = result

        updates["failed_analyses"] = failed_analyses or None
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

        failed_analyses = list(state.get("failed_analyses") or [])
        updates = {
            "current_step": "batch_insights",
            "completed_actions": list(completed | set(task_names))
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {name}: {result}")
                failed_analyses.append({"analysis": name, "error": str(result), "batch": "insights"})
                continue
            if name == "similar":
                updates["similar_genes"] = result
            elif name == "vulnerability":
                updates["vulnerability_analysis"] = result
            elif name == "cross_cell":
                updates["cross_cell_comparison"] = result

        updates["failed_analyses"] = failed_analyses or None
        return updates

    async def _run_all_batches(self, state: PerturbationAnalysisState) -> Dict:
        """Run all three batch groups concurrently via asyncio.gather.

        Called when multiple batch groups are pending (typical for comprehensive
        depth). Dispatches core, external, and insights analyses simultaneously
        instead of routing through decide_next_steps three times.
        """
        logger.info("Running all batch groups concurrently")

        results = await asyncio.gather(
            self._batch_core_analysis(state),
            self._batch_external_data(state),
            self._batch_insights(state),
            return_exceptions=True
        )

        # Merge updates from all three batches
        merged: Dict = {"current_step": "run_all_batches"}
        all_failed: list = list(state.get("failed_analyses") or [])
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch error in run_all_batches: {result}")
                all_failed.append({"analysis": "batch_group", "error": str(result), "batch": "run_all"})
                continue
            merged.update(result)
            if isinstance(result, dict) and result.get("failed_analyses"):
                all_failed.extend(result["failed_analyses"])

        # Union of all completed_actions from the three batches
        all_completed: set = set(state.get("completed_actions", []))
        for result in results:
            if isinstance(result, dict):
                all_completed.update(result.get("completed_actions", []))
        merged["completed_actions"] = list(all_completed)
        merged["failed_analyses"] = all_failed or None

        return merged

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
    # ANALYSIS IMPLEMENTATIONS (Wrap existing CASCADE tools)
    # =========================================================================

    async def _run_perturbation_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Run perturbation analysis using existing CASCADE tools."""
        from tools.loader import load_network, load_tcga_network
        from tools.perturb import (
            simulate_knockdown_with_embeddings,
            simulate_overexpression_with_embeddings,
            simulate_knockdown,
            simulate_overexpression
        )

        network_source = state.get("network_source", "cell_type")
        perturbation_type = state.get("perturbation_type", "knockdown")
        top_k = state.get("top_k") or 25

        if network_source == "tcga":
            tcga_network = state.get("tcga_network", "")
            # TCGA networks use gene symbols as node IDs
            gene_id = state.get("gene_symbol") or state["ensembl_id"]
            ensembl_id = state["ensembl_id"]

            def _sync_tcga():
                network_df = load_tcga_network(tcga_network)
                if isinstance(network_df, dict) and "error" in network_df:
                    return network_df
                # Use embedding_gene to pass Ensembl ID for model lookup while
                # gene_id (symbol) is used for network propagation.
                try:
                    model = self._get_model()
                    if perturbation_type == "knockdown":
                        result = simulate_knockdown_with_embeddings(
                            network_df, gene_id, model,
                            depth=2, top_k=top_k, alpha=0.7,
                            embedding_gene=ensembl_id,
                            embedding_threshold=0.1,
                        )
                    else:
                        result = simulate_overexpression_with_embeddings(
                            network_df, gene_id, model,
                            fold_change=2.0, depth=2, top_k=top_k, alpha=0.7,
                            embedding_gene=ensembl_id,
                            embedding_threshold=0.1,
                        )
                    result["embedding_enhanced"] = True
                except Exception as e:
                    logger.warning(f"Model unavailable for TCGA, using network-only: {e}")
                    if perturbation_type == "knockdown":
                        result = simulate_knockdown(network_df, gene_id, depth=2, top_k=top_k)
                    else:
                        result = simulate_overexpression(network_df, gene_id, fold_change=2.0, depth=2, top_k=top_k)
                    result["embedding_enhanced"] = False
                return result

            return await asyncio.to_thread(_sync_tcga)

        # Default: cell-type network (Ensembl IDs)
        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]
        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"

        def _sync():
            network_df = load_network(network_path)
            try:
                model = self._get_model()
                if perturbation_type == "knockdown":
                    result = simulate_knockdown_with_embeddings(
                        network_df, ensembl_id, model,
                        depth=2, top_k=top_k, alpha=0.7
                    )
                else:
                    result = simulate_overexpression_with_embeddings(
                        network_df, ensembl_id, model,
                        fold_change=2.0, depth=2, top_k=top_k, alpha=0.7
                    )
                result["embedding_enhanced"] = True
            except Exception as e:
                logger.warning(f"Model unavailable, using network-only: {e}")
                if perturbation_type == "knockdown":
                    result = simulate_knockdown(network_df, ensembl_id, depth=2, top_k=top_k)
                else:
                    result = simulate_overexpression(network_df, ensembl_id, fold_change=2.0, depth=2, top_k=top_k)
                result["embedding_enhanced"] = False
            return result

        return await asyncio.to_thread(_sync)

    async def _analyze_regulators_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Get upstream regulators using existing CASCADE tools."""
        from tools.loader import load_network, load_tcga_network
        from tools.perturb import get_regulators

        network_source = state.get("network_source", "cell_type")

        if network_source == "tcga":
            tcga_network = state.get("tcga_network", "")
            gene_id = state.get("gene_symbol") or state["ensembl_id"]

            def _sync_tcga():
                network_df = load_tcga_network(tcga_network)
                if isinstance(network_df, dict) and "error" in network_df:
                    return network_df
                return get_regulators(network_df, gene_id, max_regulators=50)

            return await asyncio.to_thread(_sync_tcga)

        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]
        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"

        def _sync():
            network_df = load_network(network_path)
            return get_regulators(network_df, ensembl_id, max_regulators=50)

        return await asyncio.to_thread(_sync)

    async def _analyze_targets_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Get downstream targets using existing CASCADE tools."""
        from tools.loader import load_network, load_tcga_network
        from tools.perturb import get_targets

        network_source = state.get("network_source", "cell_type")

        if network_source == "tcga":
            tcga_network = state.get("tcga_network", "")
            gene_id = state.get("gene_symbol") or state["ensembl_id"]

            def _sync_tcga():
                network_df = load_tcga_network(tcga_network)
                if isinstance(network_df, dict) and "error" in network_df:
                    return network_df
                return get_targets(network_df, gene_id, max_targets=50)

            return await asyncio.to_thread(_sync_tcga)

        cell_type = state.get("cell_type", "epithelial_cell")
        ensembl_id = state["ensembl_id"]
        network_path = self.NETWORKS_DIR / cell_type / "network.tsv"

        def _sync():
            network_df = load_network(network_path)
            return get_targets(network_df, ensembl_id, max_targets=50)

        return await asyncio.to_thread(_sync)

    async def _fetch_ppi_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch protein-protein interactions from STRING."""
        gene_symbol = state.get("gene_symbol", state["gene"])
        string_client = self.string_client

        def _sync():
            try:
                return string_client.get_interactions(gene_symbol, min_score=400, limit=25)
            except Exception as e:
                logger.error(f"STRING API error: {e}")
                return {"error": str(e), "interactions": []}

        async with self._api_semaphore:
            return await asyncio.to_thread(_sync)

    async def _fetch_lincs_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch LINCS knockdown effects."""
        from tools.lincs import get_knockdown_effects

        gene_symbol = state.get("gene_symbol", state["gene"])

        def _sync():
            try:
                return get_knockdown_effects(gene_symbol, direction="any", top_k=20)
            except Exception as e:
                logger.error(f"LINCS error: {e}")
                return {"error": str(e), "effects": []}

        return await asyncio.to_thread(_sync)

    async def _check_super_enhancers_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Check super-enhancer status."""
        from tools.super_enhancers import get_super_enhancer_info

        gene_symbol = state.get("gene_symbol", state["gene"])

        def _sync():
            try:
                return get_super_enhancer_info(gene_symbol)
            except Exception as e:
                logger.error(f"Super-enhancer error: {e}")
                return {"error": str(e), "has_super_enhancer": False}

        return await asyncio.to_thread(_sync)

    async def _fetch_dorothea_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch DoRothEA TF regulon data."""
        from tools.dorothea import get_tf_targets, validate_tf_classification

        gene_symbol = state.get("gene_symbol", state["gene"])

        def _sync():
            try:
                targets = get_tf_targets(gene_symbol, confidence_levels=["A", "B", "C"], top_k=50)
                validation = validate_tf_classification(gene_symbol)
                return {"targets": targets, "validation": validation}
            except Exception as e:
                logger.error(f"DoRothEA error: {e}")
                return {"error": str(e), "targets": [], "validation": {}}

        return await asyncio.to_thread(_sync)

    async def _fetch_depmap_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch DepMap CRISPR essentiality data for the query gene."""
        from tools.depmap import get_gene_essentiality

        gene_symbol = state.get("gene_symbol", state["gene"])

        def _sync():
            try:
                return get_gene_essentiality(gene_symbol)
            except Exception as e:
                logger.error(f"DepMap error: {e}")
                return {"error": str(e), "not_found": True}

        return await asyncio.to_thread(_sync)

    async def _fetch_cbioportal_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Fetch TCGA primary tumor expression and alteration data from cBioPortal."""
        from tools.cbioportal import get_gene_tumor_expression, get_gene_alteration_frequency

        gene_symbol = state.get("gene_symbol", state["gene"])

        def _sync():
            try:
                expression = get_gene_tumor_expression(gene_symbol)
                alteration = get_gene_alteration_frequency(gene_symbol)
                return {"expression": expression, "alteration": alteration}
            except Exception as e:
                logger.error(f"cBioPortal error: {e}")
                return {"error": str(e)}

        async with self._api_semaphore:
            return await asyncio.to_thread(_sync)

    async def _find_similar_genes_impl(self, state: PerturbationAnalysisState) -> Dict:
        """Find similar genes using embeddings."""
        ensembl_id = state["ensembl_id"]
        gene_mapper = self.gene_mapper

        def _sync():
            try:
                model = self._get_model()
                if not model.is_gene_in_vocab(ensembl_id):
                    return {"error": f"Gene {ensembl_id} not in model vocabulary"}

                from tools.cache import get_embedding_cache
                cache = get_embedding_cache(model)
                similar_df = cache.get_top_similar(ensembl_id, top_k=20)
                if similar_df is None:
                    return {"error": "Could not compute similarities"}

                similar_genes = []
                for _, row in similar_df.iterrows():
                    target_ensembl = row["ensembl_id"]
                    symbol = gene_mapper.ensembl_to_symbol(target_ensembl) or target_ensembl
                    similar_genes.append({
                        "gene_symbol": symbol,
                        "ensembl_id": target_ensembl,
                        "similarity": round(row["similarity"], 4)
                    })
                return {"similar_genes": similar_genes}
            except Exception as e:
                logger.error(f"Similarity error: {e}")
                return {"error": str(e), "similar_genes": []}

        return await asyncio.to_thread(_sync)

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
        networks_dir = self.NETWORKS_DIR

        def _sync():
            results = {}
            for cell_type in CellType:
                network_path = networks_dir / cell_type.value / "network.tsv"
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
            return {"gene": gene_symbol, "cell_type_comparison": results}

        return await asyncio.to_thread(_sync)

    # =========================================================================
    # EVIDENCE SYNTHESIS
    # =========================================================================

    def _synthesize_evidence(self, state: PerturbationAnalysisState) -> Dict:
        """Cross-reference genes across all evidence sources and produce synthesis."""
        from collections import defaultdict

        evidence = defaultdict(lambda: {"sources": [], "evidence": {}})

        # --- 1. Collect gene appearances across sources ---

        # Network propagation (perturbation_result -> top_affected_genes)
        perturbation = state.get("perturbation_result") or {}
        for gene in perturbation.get("top_affected_genes", []):
            sym = gene["symbol"].upper()
            evidence[sym]["sources"].append("network_propagation")
            evidence[sym]["evidence"]["network"] = {
                "effect": gene.get("predicted_effect") or gene.get("combined_effect"),
                "direction": gene["direction"],
                "magnitude": gene.get("magnitude")
            }

        # STRING PPI (ppi_interactions -> interactions)
        ppi = state.get("ppi_interactions") or {}
        for interaction in ppi.get("interactions", []):
            sym = interaction["partner"].upper()
            evidence[sym]["sources"].append("string_ppi")
            evidence[sym]["evidence"]["string"] = {
                "combined_score": interaction["combined_score"]
            }

        # LINCS experimental (lincs_effects — LIST on success, DICT on error)
        lincs = state.get("lincs_effects")
        lincs_list = lincs if isinstance(lincs, list) else []
        for entry in lincs_list:
            sym = entry["gene"].upper()
            evidence[sym]["sources"].append("lincs_experimental")
            evidence[sym]["evidence"]["lincs"] = {
                "effect": entry["effect"],
                "direction": entry["direction"]
            }

        # Embedding similarity (similar_genes -> similar_genes)
        similar = state.get("similar_genes") or {}
        for gene in similar.get("similar_genes", []):
            sym = gene["gene_symbol"].upper()
            evidence[sym]["sources"].append("embedding_similarity")
            evidence[sym]["evidence"]["embedding"] = {
                "similarity": gene["similarity"]
            }

        # Regulators (regulators_analysis -> regulators)
        regs = state.get("regulators_analysis") or {}
        for reg in regs.get("regulators", []):
            sym = reg["symbol"].upper()
            evidence[sym]["sources"].append("network_regulators")
            evidence[sym]["evidence"]["regulator"] = {
                "edge_weight": reg["edge_weight"]
            }

        # Targets (targets_analysis -> targets)
        tgts = state.get("targets_analysis") or {}
        for tgt in tgts.get("targets", []):
            sym = tgt["symbol"].upper()
            evidence[sym]["sources"].append("network_targets")
            evidence[sym]["evidence"]["target"] = {
                "edge_weight": tgt["edge_weight"]
            }

        # DoRothEA regulons (dorothea_regulons -> targets)
        dorothea = state.get("dorothea_regulons") or {}
        dorothea_targets = dorothea.get("targets") or []
        for dt in dorothea_targets:
            if isinstance(dt, dict) and "target" in dt:
                sym = dt["target"].upper()
                evidence[sym]["sources"].append("dorothea_regulon")
                evidence[sym]["evidence"]["dorothea"] = {
                    "mor": dt.get("mor"),
                    "confidence": dt.get("confidence")
                }

        # DepMap essentiality (validates the query gene itself, not partner genes)
        depmap = state.get("depmap_essentiality") or {}
        if not depmap.get("not_found") and "mean_chronos_score" in depmap:
            query_gene = state.get("gene_symbol", state["gene"]).upper()
            if query_gene in evidence:
                evidence[query_gene]["sources"].append("depmap_essentiality")
                evidence[query_gene]["evidence"]["depmap"] = {
                    "mean_chronos_score": depmap["mean_chronos_score"],
                    "pan_cancer_essential": depmap["pan_cancer_essential"]
                }

        # cBioPortal primary tumor data (validates the query gene in primary tissue)
        cbio = state.get("cbioportal_tumor_data") or {}
        cbio_expr = cbio.get("expression") or {}
        if isinstance(cbio_expr, dict) and "error" not in cbio_expr:
            pan_z = cbio_expr.get("pan_cancer_mean_z", 0)
            if pan_z is not None:
                query_gene = state.get("gene_symbol", state["gene"]).upper()
                if query_gene in evidence:
                    evidence[query_gene]["sources"].append("cbioportal_tumor")
                    evidence[query_gene]["evidence"]["cbioportal"] = {
                        "pan_cancer_mean_z": pan_z,
                        "tumor_overexpressed": pan_z > 1.0
                    }

        # --- 2. Filter to multi-source genes ---
        multi_source = []
        for sym, data in evidence.items():
            if len(data["sources"]) >= 2:
                entry = {
                    "symbol": sym,
                    "sources": sorted(data["sources"]),
                    "source_count": len(data["sources"]),
                    "evidence": data["evidence"]
                }
                multi_source.append(entry)

        multi_source.sort(key=lambda x: x["source_count"], reverse=True)

        # --- 3. Check directional agreement (network vs LINCS) ---
        agreements = []
        disagreements = []
        for gene_data in multi_source:
            ev = gene_data["evidence"]
            if "network" in ev and "lincs" in ev:
                net_dir = ev["network"]["direction"]
                lincs_dir = ev["lincs"]["direction"]
                sym = gene_data["symbol"]
                if net_dir == lincs_dir:
                    agreements.append(f"{sym}: network {net_dir} + LINCS {lincs_dir}")
                else:
                    disagreements.append(f"{sym}: network {net_dir} vs LINCS {lincs_dir}")

        # --- 4. Gene-role context ---
        gene_role = state.get("gene_role", "unknown")
        total_affected = perturbation.get("total_affected_genes", 0)
        ppi_count = ppi.get("count", len(ppi.get("interactions", [])))

        dorothea_validation = dorothea.get("validation") or {}
        dorothea_validated = dorothea_validation.get("is_known_tf", False)

        role_context = _build_role_context(
            gene_role, total_affected, ppi_count, len(agreements), dorothea_validated
        )

        # --- 5. cBioPortal convergent evidence flag ---
        cbio = state.get("cbioportal_tumor_data") or {}
        cbio_expr = cbio.get("expression") or {}
        tumor_overexpressed = False
        pan_z = None
        if isinstance(cbio_expr, dict) and "error" not in cbio_expr:
            pan_z = cbio_expr.get("pan_cancer_mean_z")
            tumor_overexpressed = pan_z is not None and pan_z > 1.0

        # --- 6. Build key findings ---
        key_findings = _build_key_findings(
            gene_role, multi_source, agreements, disagreements,
            total_affected, ppi_count, len(lincs_list), dorothea_validated
        )

        # Convergent cell-line + primary tumor evidence
        if depmap.get("pan_cancer_essential") and tumor_overexpressed:
            key_findings.append(
                "Convergent cell-line (DepMap CRISPR) and primary tumor (TCGA cBioPortal) evidence: "
                f"gene is pan-cancer essential in cell lines (Chronos {depmap.get('mean_chronos_score', 'n/a')}) "
                f"and overexpressed in primary tumors (pan-cancer z={pan_z:.2f}) — "
                "highest confidence multi-layer therapeutic target signal."
            )
        elif tumor_overexpressed and pan_z is not None:
            key_findings.append(
                f"Gene is overexpressed in primary TCGA tumors (pan-cancer mean z-score={pan_z:.2f}), "
                "providing primary tissue context beyond cell-line data."
            )

        return {
            "gene_role_context": role_context["context"],
            "primary_evidence_source": role_context["primary"],
            "multi_source_genes": multi_source,
            "multi_source_gene_count": len(multi_source),
            "source_agreements": agreements,
            "source_disagreements": disagreements,
            "key_findings": key_findings,
            "depmap_essentiality": depmap if not depmap.get("not_found") else None,
            "cbioportal_summary": {
                "pan_cancer_mean_z": pan_z,
                "tumor_overexpressed": tumor_overexpressed,
            } if pan_z is not None else None
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

        # Based on DoRothEA validation
        dorothea = state.get("dorothea_regulons") or {}
        dorothea_validation = dorothea.get("validation") or {}
        if dorothea_validation.get("is_known_tf") and dorothea_validation.get("best_confidence") in ("A", "B"):
            suggestions.append({
                "action": "Target known regulon members",
                "reason": (
                    f"{gene_symbol} is a DoRothEA-validated TF (confidence {dorothea_validation['best_confidence']}) "
                    f"with {dorothea_validation.get('total_targets', 0)} curated targets — "
                    f"consider targeting downstream regulon members"
                ),
                "priority": "medium"
            })

        # Based on DepMap CRISPR essentiality
        depmap = state.get("depmap_essentiality") or {}
        if not depmap.get("not_found") and not depmap.get("error"):
            if depmap.get("common_essential"):
                suggestions.append({
                    "action": "Exercise caution — common essential gene",
                    "reason": (
                        f"{gene_symbol} is essential in >90% of cancer cell lines "
                        f"(Chronos mean: {depmap['mean_chronos_score']:.2f}); broad toxicity risk"
                    ),
                    "priority": "high"
                })
            elif depmap.get("pan_cancer_essential"):
                top_lin = depmap.get("top_lineages", [])
                lineage_str = top_lin[0]["lineage"] if top_lin else "multiple lineages"
                suggestions.append({
                    "action": "Prioritize as cancer therapeutic target",
                    "reason": (
                        f"{gene_symbol} is essential in >50% of cancer cell lines; "
                        f"most essential in {lineage_str} "
                        f"(Chronos mean: {depmap['mean_chronos_score']:.2f})"
                    ),
                    "priority": "high"
                })
            elif depmap.get("top_lineages"):
                top = depmap["top_lineages"][0]
                suggestions.append({
                    "action": f"Consider as lineage-specific target in {top['lineage']}",
                    "reason": (
                        f"{gene_symbol} is lineage-selectively essential in {top['lineage']} "
                        f"(mean Chronos: {top['mean_score']:.2f}, {top['n_cell_lines']} cell lines)"
                    ),
                    "priority": "medium"
                })

        # Based on cBioPortal primary tumor data
        cbio = state.get("cbioportal_tumor_data") or {}
        cbio_expr = cbio.get("expression") or {}
        if isinstance(cbio_expr, dict) and "error" not in cbio_expr:
            pan_z = cbio_expr.get("pan_cancer_mean_z", 0) or 0
            top_over = cbio_expr.get("top_overexpressed", [])
            top_cancer = top_over[0]["cancer_type"] if top_over else None
            if pan_z > 1.0:
                suggestions.append({
                    "action": f"Prioritize in primary tumor context{f' — highest expression in {top_cancer}' if top_cancer else ''}",
                    "reason": (
                        f"{gene_symbol} is overexpressed in primary TCGA tumors "
                        f"(pan-cancer mean z={pan_z:.2f}); "
                        "this primary tissue evidence complements cell-line-based sources"
                    ),
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

        # Synthesize cross-source evidence
        evidence_synthesis = self._synthesize_evidence(state)

        # Calculate execution time
        metadata = state.get("analysis_metadata", {})
        start_time = metadata.get("start_time", time.time())
        execution_time = time.time() - start_time

        # Build no_network_targets_note for effector/isolated genes
        no_network_targets_note = None
        if gene_role in (GeneRole.EFFECTOR.value, GeneRole.ISOLATED.value):
            perturbation = state.get("perturbation_result") or {}
            total_affected = perturbation.get("total_affected_genes", 0)
            cell_type = state.get("cell_type", "this cell type")
            if gene_role == GeneRole.EFFECTOR.value:
                no_network_targets_note = (
                    f"{gene_symbol} is an effector gene with no transcriptional targets in the "
                    f"{cell_type} regulatory network (network propagation: {total_affected} affected genes). "
                    f"This is expected — effector genes are regulated by transcription factors but do not "
                    f"regulate other genes transcriptionally. See 'external_data.protein_interactions' "
                    f"and 'embedding_analysis.similar_genes' for the most informative evidence."
                )
            else:  # isolated
                no_network_targets_note = (
                    f"{gene_symbol} is not present in the {cell_type} regulatory network "
                    f"(isolated gene: no transcriptional targets or regulators). "
                    f"Network propagation, regulators, and targets analyses are uninformative. "
                    f"See 'external_data.protein_interactions' and 'embedding_analysis.similar_genes' "
                    f"for the most informative evidence."
                )

        # Build summary with optional DoRothEA validation
        summary = {
            "gene": gene_symbol,
            "ensembl_id": state.get("ensembl_id"),
            "cell_type": state.get("cell_type"),
            "gene_role": gene_role,
            "perturbation_type": state.get("perturbation_type")
        }
        if dorothea_validation.get("is_known_tf"):
            summary["dorothea_validated"] = True
            summary["dorothea_confidence"] = dorothea_validation.get("best_confidence")

        report = {
            "summary": summary,
            "evidence_synthesis": evidence_synthesis,
            "perturbation_effects": state.get("perturbation_result"),
            "network_analysis": {
                "context": state.get("network_context"),  # Always available (num_targets, num_regulators)
                "regulators": state.get("regulators_analysis"),
                "targets": state.get("targets_analysis"),
                "vulnerability": state.get("vulnerability_analysis"),
                "no_network_targets_note": no_network_targets_note
            },
            "external_data": {
                "protein_interactions": state.get("ppi_interactions"),
                "lincs_knockdown": state.get("lincs_effects"),
                "super_enhancers": state.get("super_enhancer_status"),
                "dorothea_regulons": state.get("dorothea_regulons"),
                "depmap_essentiality": state.get("depmap_essentiality"),
                "cbioportal_tumor_data": state.get("cbioportal_tumor_data")
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
                "failed_analyses": state.get("failed_analyses") or [],
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

        # Build context from analysis results.
        # Be defensive about shapes: some integrations return lists (e.g., LINCS),
        # while the LLM prompt expects summary fields.
        perturbation = state.get("perturbation_result")
        perturbation = perturbation if isinstance(perturbation, dict) else {}

        ppi = state.get("ppi_interactions")
        lincs = state.get("lincs_effects")
        se = state.get("super_enhancer_status")
        se = se if isinstance(se, dict) else {}
        similar = state.get("similar_genes")

        lincs_total_effects = 0
        lincs_up = 0
        lincs_down = 0
        lincs_top_genes = []
        if isinstance(lincs, list):
            lincs_total_effects = len(lincs)
            for e in lincs:
                if isinstance(e, dict):
                    direction = e.get("direction", e.get("effect_direction", ""))
                    if "up" in str(direction).lower():
                        lincs_up += 1
                    elif "down" in str(direction).lower():
                        lincs_down += 1
                    sym = e.get("gene_symbol") or e.get("gene") or e.get("symbol")
                    if sym and len(lincs_top_genes) < 5:
                        lincs_top_genes.append(sym)
        elif isinstance(lincs, dict):
            lincs_total_effects = int(lincs.get("total_effects", 0) or 0)

        ppi_interactions = []
        if isinstance(ppi, dict):
            ppi_interactions = ppi.get("interactions", []) or []
        elif isinstance(ppi, list):
            ppi_interactions = ppi
        ppi_names = [p.get("preferredName") or p.get("partner") or p.get("symbol")
                     for p in ppi_interactions[:8] if isinstance(p, dict)]
        ppi_names = [n for n in ppi_names if n]

        similar_genes_list = []
        if isinstance(similar, dict):
            similar_genes_list = similar.get("similar_genes", []) or []
        elif isinstance(similar, list):
            similar_genes_list = similar
        similar_names = [g.get("gene_symbol") or g.get("symbol")
                         for g in similar_genes_list[:5] if isinstance(g, dict)]
        similar_names = [n for n in similar_names if n]

        # Format top affected genes as readable list with direction and score
        top_affected_raw = perturbation.get("top_affected", []) or []
        top_affected_fmt = []
        for g in top_affected_raw[:15]:
            if not isinstance(g, dict):
                continue
            sym = (g.get("gene_symbol") or g.get("symbol") or
                   g.get("ensembl_id") or g.get("gene", ""))
            effect = g.get("combined_effect") or g.get("network_effect") or g.get("effect", 0)
            direction = "↓" if float(effect or 0) < 0 else "↑"
            top_affected_fmt.append(f"{sym} ({direction}{abs(float(effect or 0)):.2f})")

        # DoRothEA TF validation
        dorothea = state.get("dorothea_regulons") or {}
        dorothea_validated = isinstance(dorothea, dict) and dorothea.get("is_known_tf", False)
        dorothea_targets = (dorothea.get("total_targets", 0)
                            if isinstance(dorothea, dict) else 0)
        dorothea_conf = (dorothea.get("confidence_levels", [])
                         if isinstance(dorothea, dict) else [])

        # Network source context
        network_source = state.get("network_source", "cell_type")
        network_label = (f"TCGA {state.get('tcga_network', '').upper()} tumor-state network"
                         if network_source == "tcga"
                         else f"{cell_type} cell-type network (GREmLN)")

        # Regulators
        regulators = state.get("regulators_analysis") or {}
        top_regulators = []
        if isinstance(regulators, dict):
            for r in (regulators.get("top_regulators") or [])[:5]:
                sym = r.get("gene_symbol") or r.get("symbol") or r.get("ensembl_id", "")
                if sym:
                    top_regulators.append(sym)

        prompt = f"""You are analyzing results from CASCADE, an in silico gene perturbation tool.

Gene: {gene} | Perturbation: {perturbation_type} | Network: {network_label}
Gene Role: {gene_role}

## Network Propagation Results
- Total downstream genes affected: {perturbation.get('total_affected_genes', 0)}
- Direct network targets: {perturbation.get('direct_targets', perturbation.get('total_direct_targets', 'N/A'))}
- Top predicted {'downregulated' if perturbation_type == 'knockdown' else 'upregulated'} genes:
  {', '.join(top_affected_fmt[:10]) if top_affected_fmt else 'N/A'}

## Upstream Regulators
- Top regulators of {gene}: {', '.join(top_regulators) if top_regulators else 'N/A'}

## DoRothEA TF Validation
- Known TF: {dorothea_validated} | Curated targets: {dorothea_targets} | Confidence: {dorothea_conf}

## STRING Protein Interactions
- Partners ({len(ppi_names)} shown): {', '.join(ppi_names) if ppi_names else 'N/A'}

## LINCS Experimental Knockdown
- Genes affected: {lincs_total_effects} ({lincs_up} up / {lincs_down} down)
- Example affected genes: {', '.join(lincs_top_genes) if lincs_top_genes else 'N/A'}

## Super-Enhancer / BRD4 Status
- Has super-enhancer: {se.get('has_super_enhancer', False)}
- BRD4/BET inhibitor sensitive: {se.get('has_super_enhancer', False)}
- Cell types with SE: {se.get('cell_type_count', 'N/A')}

## Functionally Similar Genes
- {', '.join(similar_names) if similar_names else 'N/A'}

Based on the data above, provide a biological interpretation in this EXACT JSON format:
{{
  "mechanism_summary": "2-3 sentences explaining the mechanism, naming specific top-affected genes from the data",
  "therapeutic_implications": "1-2 sentences on drug development relevance, referencing BRD4/LINCS/PPI data where applicable",
  "key_pathways_affected": ["pathway1", "pathway2", "pathway3"],
  "confidence_level": "high|medium|low",
  "confidence_rationale": "cite specific data points that justify this confidence level",
  "follow_up_suggestions": ["specific actionable suggestion 1", "specific actionable suggestion 2"],
  "biological_interpretation": "3-4 sentence narrative suitable for a research report, naming key genes and their roles"
}}

Rules: Name specific genes from the data. Do not invent genes not listed above. Provide only the JSON, no other text."""

        system_prompt = """You are an expert molecular biologist specializing in gene regulatory networks, cancer biology, and perturbation analysis. Provide concise, data-driven interpretations that name specific genes. Never speculate beyond what the data supports."""

        content = await self._call_llm(prompt, system_prompt)
        parsed = self._parse_llm_json(content)
        parsed["llm_powered"] = True
        parsed["model"] = os.getenv('LLM_MODEL', self.ollama_model)

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
        include_llm_insights: bool = False,
        network_source: str = "cell_type",
        tcga_network: Optional[str] = None,
        top_k: int = 25,
        progress_cb=None,
    ) -> Dict:
        """
        Run the complete perturbation analysis workflow.

        Args:
            gene: Gene symbol or Ensembl ID
            cell_type: Cell type for network context
            perturbation_type: knockdown, overexpression, or similarity
            analysis_depth: basic, comprehensive, or focused
            include_llm_insights: Whether to generate LLM-powered biological interpretation
            top_k: Number of top-affected genes to return from perturbation propagation
                (default 25, matching CASCADE's historical default)

        Returns:
            Comprehensive analysis report
        """
        initial_state: PerturbationAnalysisState = {
            "gene": gene,
            "cell_type": cell_type,
            "perturbation_type": perturbation_type,
            "analysis_depth": analysis_depth,
            "network_source": network_source,
            "tcga_network": tcga_network,
            "top_k": top_k,
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
            "dorothea_regulons": None,
            "depmap_essentiality": None,
            "cross_cell_comparison": None,
            "vulnerability_analysis": None,
            "therapeutic_suggestions": None,
            "comprehensive_report": None,
            "failed_analyses": None,
            "analysis_metadata": {},
            "include_llm_insights": include_llm_insights,
            "llm_insights": None
        }

        logger.info(f"Starting workflow for {gene} ({perturbation_type}, {analysis_depth})")

        async def _notify(progress: float, message: str) -> None:
            if progress_cb is not None:
                try:
                    await progress_cb(progress, 1.0, message)
                except Exception:
                    pass

        # Heartbeat task: fires progress notifications while ainvoke runs
        heartbeat_schedule = [
            (5,  0.15, f"Resolving {gene} and analyzing regulatory network..."),
            (15, 0.40, "Running parallel analyses (network propagation, STRING PPI, LINCS, DepMap)..."),
            (30, 0.70, "Processing external data sources and cross-cell comparison..."),
            (45, 0.88, "Generating comprehensive report..."),
        ]

        async def _heartbeat() -> None:
            for delay, progress, message in heartbeat_schedule:
                await asyncio.sleep(delay)
                await _notify(progress, message)

        await _notify(0.05, f"Starting {perturbation_type} analysis of {gene} ({analysis_depth})...")
        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            final_state = await self.workflow.ainvoke(initial_state)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        await _notify(1.0, "Analysis complete.")

        return final_state.get("comprehensive_report", {"error": "No report generated"})


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the CASCADE workflow."""
    workflow = CascadeWorkflow()

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
