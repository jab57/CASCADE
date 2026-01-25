# GREmLN Future Roadmap: Unified Bio-Orchestrator

> **STATUS: POSTPONED** - This document outlines future integration work that is not currently scheduled. The current architecture (Option A) is in production.

## Current Architecture (Option A - In Production)

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
- `analyze_gene_knockdown` / `analyze_gene_overexpression` - Network perturbation
- `get_gene_metadata` - Gene type classification (NEW)
- `get_protein_interactions` - STRING database
- `find_similar_genes` - Embedding-based similarity
- `analyze_network_vulnerability` - Drug target discovery
- Smart suggestions when results are empty (NEW)

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

---

## Future Vision (Option C - Unified Orchestrator)

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

### Architectural Decisions Needed

1. **Where does orchestrator live?**
   - Option A: New standalone project (`c:\Dev\bio-orchestrator`)
   - Option B: Inside GREmLN as optional module
   - Option C: Inside regnetagents as extension

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

## Document History

- **2025-01-25**: Initial roadmap created (Jose A. Bird, PhD)
- **Status**: Postponed - current Option A architecture with smart suggestions is sufficient for now
