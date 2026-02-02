# GREmLN Future Roadmap: Unified Bio-Orchestrator

> **STATUS UPDATE (2025-02)**: LangGraph orchestration has been implemented for GREmLN. The Unified Bio-Orchestrator (cross-project integration with regnetagents) remains postponed.

## Architecture Options

| Aspect | Option 1: Two Separate Servers | Option 2: One Integrated Server |
|--------|--------------------------------|---------------------------------|
| **MCP Servers** | 2 (regnetagents + GREmLN) | 1 (Bio-Orchestrator) |
| **Orchestration** | Claude conversation-level | LangGraph StateGraph |
| **Tool Chaining** | Manual (user/Claude decides) | Automatic (gene-type routing) |
| **State Management** | None (stateless tools) | Unified state across analyses |
| **Complexity** | Simple, modular | More complex, integrated |
| **Maintenance** | Independent updates | Coordinated updates |
| **User Experience** | Multiple tool calls needed | Single query → full analysis |
| **Status** | **In Production** | Postponed |

---

## Option 1: Two Separate MCP Servers (Current)

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop / MCP Client              │
│              (orchestrates via conversation)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│   regnetagents      │   │      GREmLN         │
│    MCP Server       │   │    MCP Server       │
│                     │   │                     │
│ - Network analysis  │   │ - Perturbation sim  │
│ - Pathway enrich    │   │ - Gene embeddings   │
│ - Domain insights   │   │ - PPI (STRING)      │
│ - LangGraph workflow│   │ - Smart suggestions │
└─────────────────────┘   └─────────────────────┘

Location: c:\Dev\regnetagents   Location: c:\Dev\GREmLN
Status: Frozen (no changes)     Status: Active development
```

### Current Capabilities

**regnetagents MCP Server:**
- `comprehensive_gene_analysis` - Full LangGraph workflow
- `multi_gene_analysis` - Parallel gene processing
- `pathway_focused_analysis` - Reactome integration
- Domain-specific agents (cancer, drug, clinical, systems biology)

**GREmLN MCP Server:**
- `analyze_gene_knockdown` / `analyze_gene_overexpression` - Perturbation analysis (network + embeddings)
- `get_gene_metadata` - Gene type classification
- `get_protein_interactions` - STRING database
- `find_similar_genes` - Embedding-based similarity
- `analyze_network_vulnerability` - Drug target discovery
- Smart suggestions when results are empty

### Current Limitation

When analyzing genes like APC (scaffold proteins with no transcriptional targets), Claude must manually recognize the biological context and chain tools appropriately:

```
APC knockdown → empty results
    ↓ (Claude recognizes APC is effector)
APC PPI → shows CTNNB1 as key partner
    ↓ (Claude reasons about APC→CTNNB1 relationship)
CTNNB1 overexpression → rich cascade data
```

The new `gene_metadata` and `suggestions` fields help Claude make these decisions, but the reasoning still happens at the conversation level.

### Partial Solution: Claude Code Skill (Implemented 2025-01)

The GREmLN skill (`.claude/skills/gremln/SKILL.md`) provides workflow guidance to Claude Code, teaching it:
- When to use knockdown vs PPI analysis
- How to handle effector genes (scaffold proteins)
- Suggested tool chaining (e.g., empty knockdown → check metadata → use PPI)
- Trigger keywords to distinguish from other MCP servers (regnetagents)

This reduces manual intervention but doesn't fully automate the orchestration. Option 2 remains the long-term vision for seamless single-tool analysis.

---

## Option 2: One Integrated Server (Future)

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop / MCP Client              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Unified Bio-Orchestrator MCP Server            │
│                    (NEW PROJECT)                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LangGraph StateGraph                    │   │
│  │                                                      │   │
│  │  analyze_mutation_implications()                     │   │
│  │    → assess_gene_type                               │   │
│  │    → route_to_appropriate_analysis                  │   │
│  │    → synthesize_results                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│            ┌─────────────┴─────────────┐                   │
│            ▼                           ▼                    │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │  regnetagents   │         │     GREmLN      │          │
│   │  (as library)   │         │  (as library)   │          │
│   └─────────────────┘         └─────────────────┘          │
└─────────────────────────────────────────────────────────────┘

Total MCP Servers exposed to Claude: 1
```

### Key Benefits

1. **Intelligent Routing**: Agent automatically decides whether to use network analysis, PPI, or embeddings based on gene type
2. **Unified State**: Single state object tracks all findings across both systems
3. **Biological Reasoning**: LLM-powered decisions about analysis strategy
4. **Seamless UX**: User asks "Analyze APC mutation" → gets complete cascade analysis

### Proposed State Schema

```python
class UnifiedBioState(TypedDict):
    # Input
    gene: str
    cell_type: str
    analysis_type: str  # "mutation", "overexpression", "pathway"

    # Gene Classification
    gene_metadata: dict  # From GREmLN get_gene_metadata
    is_transcription_factor: bool
    known_complexes: list[str]

    # Analysis Results
    network_context: dict      # From regnetagents
    perturbation_results: dict # From GREmLN
    ppi_data: dict             # From GREmLN STRING
    pathway_enrichment: dict   # From regnetagents
    embedding_similar: list    # From GREmLN

    # Routing State
    analyses_completed: list[str]
    next_analysis: str
    suggestions: list[dict]

    # Output
    final_report: str
    confidence_score: float
```

### Proposed Workflow Graph

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           ▼
                ┌──────────────────────┐
                │  assess_gene_type    │
                │  (calls GREmLN       │
                │   get_gene_metadata) │
                └──────────┬───────────┘
                           ▼
              ┌────────────────────────────┐
              │     route_analysis         │
              │  (conditional edge)        │
              └─────┬──────────┬───────────┘
                    │          │
        ┌───────────┘          └───────────┐
        ▼                                  ▼
┌───────────────┐                 ┌───────────────┐
│ TF Analysis   │                 │ Effector      │
│ Path          │                 │ Analysis Path │
├───────────────┤                 ├───────────────┤
│ 1. knockdown  │                 │ 1. get_ppi    │
│ 2. pathways   │                 │ 2. find TF    │
│ 3. domain     │                 │    partners   │
│    insights   │                 │ 3. analyze TF │
└───────┬───────┘                 │    partners   │
        │                         └───────┬───────┘
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
              ┌───────────────┐
              │  synthesize   │
              │  results      │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │     END       │
              └───────────────┘
```

---

## Prerequisites Before Starting

### Technical Requirements

1. **Python Environment Compatibility**
   - Both projects use Python 3.10+
   - Verify no dependency conflicts between regnetagents and GREmLN

2. **Import Strategy**
   - regnetagents: Import `RegNetAgentsWorkflow` class directly
   - GREmLN: Import tool functions from `tools/` modules
   - Avoid MCP-over-MCP calls (use direct Python imports)

3. **State Management**
   - Design unified state schema that captures both systems' outputs
   - Handle async execution (both systems use asyncio)

### Architectural Decisions Needed (for Option 2)

1. **Where does orchestrator live?**
   - New standalone project (`c:\Dev\bio-orchestrator`) - *Recommended*
   - Inside GREmLN as optional module
   - Inside regnetagents as extension

2. **How to handle regnetagents being frozen?**
   - Import as-is without modification
   - Create wrapper classes if needed
   - Fork if changes become necessary

3. **Deployment model**
   - Single MCP server (orchestrator only)
   - Keep individual servers available for direct access?

### Testing Strategy

1. **Unit tests**: Each routing decision in isolation
2. **Integration tests**: Full workflows (APC → CTNNB1 cascade)
3. **Comparison tests**: Orchestrator results vs manual chaining

---

## Implementation Phases (When Ready)

### Phase 1: Foundation
- [ ] Create new project structure
- [ ] Set up shared state schema
- [ ] Import regnetagents and GREmLN as libraries
- [ ] Basic LangGraph workflow skeleton

### Phase 2: Gene Type Routing
- [ ] Implement `assess_gene_type` node
- [ ] Create conditional routing logic
- [ ] TF path: knockdown → pathways
- [ ] Effector path: PPI → partner analysis

### Phase 3: Intelligent Synthesis
- [ ] Result aggregation across systems
- [ ] LLM-powered biological interpretation
- [ ] Confidence scoring

### Phase 4: MCP Server Wrapper
- [ ] Expose orchestrator via MCP
- [ ] Single tool: `analyze_mutation_implications`
- [ ] Streaming progress updates

### Phase 5: Testing & Validation
- [ ] APC mutation test case (the original problem)
- [ ] TP53, MYC, BRCA1 test cases
- [ ] Performance benchmarks

---

## Reference: The Original Problem

This roadmap exists because of the APC mutation analysis scenario:

**What happened:**
1. User ran `analyze_gene_knockdown("APC")` → empty results
2. Tool returned no suggestions about why
3. User had to manually recognize APC is a scaffold protein
4. User manually switched to `get_protein_interactions("APC")`
5. User identified CTNNB1 as key partner
6. User reasoned that APC inhibits CTNNB1
7. User ran `analyze_gene_overexpression("CTNNB1")` → rich cascade

**What we want:**
1. User runs `analyze_mutation_implications("APC")`
2. Orchestrator detects APC is an effector (no transcriptional targets)
3. Orchestrator automatically queries PPI
4. Orchestrator identifies CTNNB1 as TF partner
5. Orchestrator reasons about APC→CTNNB1 relationship
6. Orchestrator runs CTNNB1 overexpression analysis
7. User receives integrated report with full cascade

---

## Near-Term Enhancements

| Feature | Status | Date | Description |
|---------|--------|------|-------------|
| LLM-Powered Insights | **COMPLETE** | 2025-02-02 | Ollama integration for biological interpretation |
| LangGraph Orchestration | **COMPLETE** | 2025-02-02 | Intelligent workflow routing with parallel execution |
| LINCS L1000 (Harmonizome) | **COMPLETE** | 2025-01-30 | Expression perturbation from CRISPR knockdowns |
| Super-Enhancer Annotations | **COMPLETE** | 2025-01-30 | BRD4 druggability from dbSUPER |
| Raw LINCS Integration | NOT STARTED | - | Full LINCS data for better coverage |
| Expression Data Fetching | NOT STARTED | - | CellxGene Census / HPA integration |

---

### Completed: LLM-Powered Biological Insights (2025-02-02)

Added optional Ollama integration for AI-generated interpretation of perturbation analysis results:

**New Feature:**
- `include_llm_insights=True` parameter in `comprehensive_perturbation_analysis`

**Key Capabilities:**
- **Mechanism Summary**: 2-3 sentence explanation of what the perturbation does mechanistically
- **Therapeutic Implications**: Drug development relevance assessment
- **Pathway Identification**: Automated identification of key affected pathways
- **Confidence Assessment**: High/medium/low confidence with rationale
- **Follow-up Suggestions**: Intelligent recommendations for further analysis
- **Biological Narrative**: 3-4 sentence synthesis suitable for research reports

**Configuration:**
- Auto-detects local vs cloud Ollama (set `OLLAMA_API_KEY` for cloud)
- Configurable model, temperature, and timeout via environment variables
- Graceful fallback: Returns structured data if LLM unavailable
- Default OFF to avoid latency for quick queries

**Files:**
- `gremln_langgraph_workflow.py` - Added `_synthesize_insights` node, `_call_llm_synthesis` method
- `gremln_langgraph_mcp_server.py` - Added `include_llm_insights` parameter to tool schema
- `.env.example` - LLM configuration template

---

### Completed: LangGraph Orchestration (2025-02-02)

Implemented LangGraph-based workflow orchestration within GREmLN MCP server:

**New Files:**
- `gremln_langgraph_workflow.py` - Core LangGraph StateGraph workflow
- `gremln_langgraph_mcp_server.py` - MCP server exposing 22 tools

**Key Features:**
- **Intelligent Routing**: Automatically selects analysis strategy based on gene type (master_regulator, transcription_factor, effector, isolated)
- **Parallel Batch Execution**: Independent analyses run concurrently (batch_core_analysis, batch_external_data, batch_insights)
- **Automatic Synthesis**: Generates comprehensive reports with actionable recommendations
- **Graceful Degradation**: Falls back to network-only if embeddings unavailable

**New Workflow Tools:**
- `comprehensive_perturbation_analysis` - Full automated analysis with intelligent routing
- `multi_gene_analysis` - Analyze multiple genes in parallel

**Performance:**
- Basic analysis: ~3s
- Comprehensive analysis: ~8-10s
- Multi-gene parallel (3 genes): ~10s

**Note:** This addresses the "GREmLN-only" orchestration need. The Unified Bio-Orchestrator (Option 2 - cross-project integration with regnetagents) remains postponed.

---

### Completed: LINCS L1000 Expression Perturbation (2025-01-30)

Added tools to find regulatory relationships from experimental CRISPR knockdown data:
- `find_expression_regulators(gene)` - what knockdowns affect this gene?
- `get_knockdown_effects(gene)` - what does this knockdown affect?

**Data source**: Harmonizome LINCS L1000 CRISPR Knockout Consensus Signatures

**Limitation**: Harmonizome pre-filters data, removing some validated relationships (e.g., BRD4 → MYC).

### Planned: Raw LINCS Integration

**Status**: NOT STARTED

**Problem**: BRD4 → MYC (validated BET inhibitor target) not in Harmonizome data

**Solution**: Integrate raw LINCS L1000 data from clue.io (GEO: GSE92742)

**Effort**: Medium (larger dataset, GCTX format parsing)

**Validation**: `find_expression_regulators("MYC")` should return BRD4

**Tasks**:
- [ ] Download raw LINCS L1000 data (level 5 - moderated z-scores)
- [ ] Implement GCTX file parser (HDF5 format)
- [ ] Create gene-knockdown → affected-genes mapping
- [ ] Add tool: `find_expression_regulators_raw(gene)`
- [ ] Validate: BRD4 knockdown should show MYC downregulation

### Completed: Super-Enhancer Annotations (2025-01-30)

Added tools to identify BRD4/BET inhibitor sensitive genes:
- `check_super_enhancer(gene)` - Check if gene has super-enhancers
- `check_genes_super_enhancers(genes)` - Screen multiple genes

**Data source**: dbSUPER (69K associations, 10K genes, 102 cell types)

**Validation**: MYC has SE in 32 cell types (BRD4-sensitive), TP53 has none

**Therapeutic value**: Enables drug discovery for "undruggable" targets like MYC - if a gene has super-enhancers, BRD4 inhibitors (JQ1, OTX015) may reduce its expression

### Planned: Expression Data Fetching

**Status**: NOT STARTED (stub in `tools/expression_fetcher.py`)

**Goal**: Automatically fetch baseline expression profiles for each cell type

**Data Sources** (in order of priority):
1. **Human Protein Atlas** - Simple, pre-computed averages, no complex dependencies
2. **CellxGene Census** - Most comprehensive, true single-cell data
3. **Tabula Sapiens** - Human cell atlas

**Use Cases**:
- Context-aware gene embeddings (cell-type-specific)
- Baseline for perturbation predictions
- Gene rank embeddings for model input

**Tasks**:
- [ ] Implement HPA data download and parsing
- [ ] Map cell type names to standardized ontology terms (mapping exists in stub)
- [ ] Create local cache for expression profiles
- [ ] Add tool: `get_baseline_expression(cell_type)`
- [ ] Integrate with model for context-aware embeddings

---

## Document History

| Date | Change |
|------|--------|
| 2025-02-02 | Added LLM-powered biological insights (Ollama integration) |
| 2025-02-02 | Added LangGraph orchestration (COMPLETE), updated architecture diagrams |
| 2025-02-01 | Added status summary table, Expression Data Fetching section, task checklists |
| 2025-01-30 | Added super-enhancer annotations (dbSUPER) for BRD4 druggability |
| 2025-01-30 | Added LINCS L1000 tools, documented limitations, planned raw LINCS integration |
| 2025-01-25 | Initial roadmap created (Jose A. Bird, PhD) |

**Current Status**: Active development - LLM insights and LangGraph orchestration complete, raw LINCS and expression fetching planned
