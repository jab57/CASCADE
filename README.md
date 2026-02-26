# CASCADE MCP Server

**Computational Analysis of Simulated Cell And Drug Effects**

A Model Context Protocol (MCP) server for **in silico gene perturbation analysis** using pre-computed gene regulatory networks and GREmLN model embeddings. Features **LangGraph-based workflow orchestration** for intelligent, automated analysis pipelines.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Desktop / MCP Client                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              CASCADE LangGraph MCP Server                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LangGraph StateGraph                        │   │
│  │                                                          │   │
│  │  comprehensive_perturbation_analysis()                   │   │
│  │    → Gene classification (TF, effector, master reg)      │   │
│  │    → Intelligent routing based on gene type              │   │
│  │    → Parallel batch execution                            │   │
│  │    → Automatic synthesis & recommendations               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│     ┌────────────────────────┼────────────────────────┐        │
│     ▼                        ▼                        ▼        │
│  ┌──────────┐         ┌──────────────┐         ┌──────────┐   │
│  │ Network  │         │  Embeddings  │         │ External │   │
│  │ Analysis │         │   (GREmLN)   │         │   APIs   │   │
│  │ (BFS)    │         │   256-dim    │         │ (STRING, │   │
│  └──────────┘         └──────────────┘         │  LINCS,  │   │
│                                                │ dbSUPER, │   │
│                                                │ DoRothEA,│   │
│                                                │  DepMap) │   │
│                                                └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Benefits of LangGraph Architecture

- **Intelligent Routing**: Automatically selects analysis strategy based on gene type
- **Parallel Execution**: Independent analyses run concurrently (3-5x faster)
- **Automatic Synthesis**: Generates comprehensive reports with actionable recommendations
- **LLM-Powered Insights**: Biological interpretation via configurable LLM (Ollama local/cloud by default; adaptable to other providers)
- **Graceful Degradation**: Falls back to network-only if embeddings unavailable
- **MCP Resources**: Browsable resource endpoints for metadata discovery without running full analyses

## Features

### Network-Based Analysis
- **Gene Knockdown Simulation**: Predict downstream effects of silencing a gene
- **Gene Overexpression Simulation**: Predict effects of increased gene expression
- **Regulator Discovery**: Find transcription factors controlling a target gene
- **Target Discovery**: Find genes controlled by a regulator
- **Gene ID Mapping**: Convert between gene symbols (MYC) and Ensembl IDs (ENSG...)
- **Gene Metadata & Classification**: Determine if a gene is a transcription factor, effector, or scaffold protein based on network position

### Intelligent Tool Guidance
- **Automatic Suggestions**: When perturbation tools return no targets (e.g., scaffold proteins like APC), the response includes actionable suggestions for alternative analyses
- **Known Complex Partners**: For well-characterized proteins, suggestions include known interaction partners (e.g., APC → CTNNB1, AXIN1, GSK3B)
- **Recommended Follow-ups**: Specific tool calls suggested based on biological context (e.g., "Run overexpression on CTNNB1 to see effects of APC loss")
- **Gene-Role-Aware Routing**: Effector and isolated genes skip uninformative network analyses and receive a `no_network_targets_note` in the report explaining why network propagation returned zero results and directing users to protein interaction and embedding evidence

### Model-Enhanced Analysis
- **Embedding-Enhanced Knockdown**: Combines network topology with learned gene representations
- **Embedding-Enhanced Overexpression**: More accurate predictions using 11M-cell-trained embeddings
- **Gene Similarity**: Compute functional similarity between genes using learned embeddings
- **Similar Gene Discovery**: Find functionally related genes even without direct network connections

### LLM-Powered Biological Insights
- **Narrative Synthesis**: LLM-powered interpretation of analysis results
- **Mechanism Explanation**: Automatic generation of biological mechanism summaries
- **Therapeutic Implications**: AI-generated drug development relevance
- **Pathway Identification**: Automated identification of key affected pathways
- **Follow-up Suggestions**: Intelligent recommendations for further analysis

### Network Vulnerability Analysis (Drug Target Discovery)
- **Hub Gene Identification**: Find genes with the most downstream targets
- **Vulnerability Scoring**: Rank genes by network criticality (cascade impact, connectivity)
- **Drug Target Comparison**: Compare candidate genes to identify best therapeutic targets
- **Master Regulator Detection**: Identify genes that control large portions of the network

### Experimental Perturbation Data (LINCS L1000)
- **Knockdown Effects**: Query experimental CRISPR knockdown signatures to validate network predictions
- **Expression Regulators**: Find genes whose knockdown affects a target gene's expression
- **Directional Filtering**: Filter by up- or down-regulation

### Super-Enhancer / BRD4 Druggability (dbSUPER)
- **Super-Enhancer Detection**: Check if a gene is driven by super-enhancers
- **BET Inhibitor Sensitivity**: Identify genes targetable by BRD4/BET inhibitors (JQ1, OTX015)
- **Batch Screening**: Screen multiple genes for super-enhancer associations

### TF Regulon Validation (DoRothEA)
- **Curated Regulons**: Query multi-evidence TF-target relationships with confidence levels (A-E)
- **TF Classification Validation**: Cross-reference network-derived TF classifications against curated regulons
- **Evidence Integration**: Combine literature, ChIP-seq, and motif evidence

### CRISPR Essentiality (DepMap)
- **Pan-Cancer Essentiality**: Chronos gene effect scores across 1,000+ cancer cell lines — negative scores indicate fitness dependency
- **Lineage Profiles**: Identify which cancer lineages show the strongest essentiality for a gene (e.g., MYC most essential in Pancreas and Myeloid)
- **Phenotypic Validation**: The only empirical phenotypic evidence layer in CASCADE — validates whether predicted network hubs are actually lethal to cancer cells
- **Therapeutic Triage**: Three-tier suggestions: common essential (>90% of lines, broad toxicity risk), pan-cancer essential (>50%), or lineage-selective target

### Protein-Protein Interactions (STRING Database)
- **Interaction Partners**: Query physical and functional protein interactions from STRING
- **Confidence Scoring**: Filter by experimental evidence, database annotations, or text mining
- **Mechanism Discovery**: Understand protein-level effects of gene perturbations (e.g., APC knockdown disrupts β-catenin binding)

## Use Cases

### Cancer Research & Immuno-Oncology
- **Tumor Microenvironment**: Analyze immune cell networks (CD8 T cells, NK cells, monocytes) for immunotherapy target discovery
- **Drug Target Prioritization**: Use vulnerability analysis to identify high-value therapeutic targets
- **Checkpoint Biology**: Explore PD-1, CTLA-4, LAG-3 regulatory networks
- **CAR-T Engineering**: Understand T cell exhaustion and persistence pathways

### General Applications
- Perturbation prediction for CRISPR experiments
- Transcription factor target mapping
- Functional gene annotation via embeddings
- Pathway exploration

## How It Works

The server provides analysis across several categories:

1. **Perturbation simulation** (`comprehensive_perturbation_analysis`, `quick_perturbation`): Combines BFS propagation through the regulatory network with gene embeddings learned from 11 million cells. Discovers both direct network effects and indirect functional relationships. Falls back to network-only if model is unavailable.

2. **Vulnerability analysis** (`analyze_network_vulnerability`, `compare_gene_vulnerability`): Identifies critical network nodes (hub genes, master regulators) for drug target discovery. Ranks genes by downstream impact if disrupted.

3. **Protein-protein interactions** (`get_protein_interactions`): Queries STRING database for physical and functional protein interactions. Explains what happens at the protein level after perturbation.

4. **Experimental corroboration** (`get_knockdown_effects`, `find_expression_regulators`): Queries LINCS L1000 CRISPR knockdown signatures to validate or complement network predictions with experimental data.

5. **Druggability assessment** (`check_super_enhancer`): Checks super-enhancer annotations from dbSUPER to identify BRD4/BET inhibitor sensitivity.

6. **TF regulon validation** (`get_dorothea_regulon`, `validate_tf_classification`): Cross-references network-derived TF classifications against DoRothEA curated multi-evidence regulons.

7. **Gene similarity** (`find_similar_genes`, `get_gene_similarity`): Computes functional similarity using GREmLN embeddings to discover pathway members and alternative targets.

8. **CRISPR essentiality** (`get_depmap_essentiality`, also integrated in `comprehensive_perturbation_analysis`): Queries pre-downloaded DepMap Chronos gene effect scores to provide empirical phenotypic validation — confirming whether a predicted network hub is actually lethal to cancer cells across 1,000+ cell lines and 30+ cancer lineages.

## Supported Cell Types

| Cell Type | Network File |
|-----------|--------------|
| Epithelial cells | `epithelial_cell/network.tsv` |
| CD4 T cells | `cd4_t_cells/network.tsv` |
| CD8 T cells | `cd8_t_cells/network.tsv` |
| CD14 Monocytes | `cd14_monocytes/network.tsv` |
| CD16 Monocytes | `cd16_monocytes/network.tsv` |
| CD20 B cells | `cd20_b_cells/network.tsv` |
| NK cells | `nk_cells/network.tsv` |
| NKT cells | `nkt_cells/network.tsv` |
| Erythrocytes | `erythrocytes/network.tsv` |
| Monocyte-derived DCs | `monocyte-derived_dendritic_cells/network.tsv` |

> **Note:** Networks represent population-averaged regulatory relationships and do not capture cell-state-specific dynamics. Analysis of state-dependent regulation (e.g., resting vs. activated T cells) requires cell-state-specific networks generated from appropriately resolved single-cell data and supplied in the same format as the bundled networks.

## Installation

```bash
# Create virtual environment
python -m venv env

# Activate (Windows)
.\env\Scripts\activate

# Activate (Linux/Mac)
# source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended), install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Install Ollama for LLM-powered biological insights
# Download from https://ollama.ai and run: ollama pull llama3.1:8b
```

### Verify Installation

After installing, run the verification script to confirm everything works:

```bash
python verify_installation.py
```

This checks core dependencies, loads all 10 cell-type networks, verifies the model checkpoint, tests gene ID mapping (via Ensembl API), runs a perturbation simulation, and validates embedding similarity search. To skip internet-dependent checks:

```bash
python verify_installation.py --offline
```

## Usage

### Run the MCP Server

```bash
# LangGraph-based server (recommended)
python cascade_langgraph_mcp_server.py

# With LLM insights enabled
USE_LLM_INSIGHTS=true python cascade_langgraph_mcp_server.py

# Original FastMCP server (deprecated, kept for reference)
python cascade_mcp_server_original.py
```

### LLM Insights Configuration

CASCADE uses Ollama (local or cloud) by default for LLM-powered biological insights. The LLM integration is modular and can be adapted to other providers. Copy `.env.example` to `.env` and configure:

```bash
# Enable LLM-powered biological insights
USE_LLM_INSIGHTS=true

# Local Ollama (default)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Or use Ollama Cloud
# OLLAMA_API_KEY=your-api-key
```

### Claude Desktop Configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac):

```json
{
  "mcpServers": {
    "CASCADE": {
      "command": "C:/Dev/CASCADE/env/Scripts/python.exe",
      "args": ["C:/Dev/CASCADE/cascade_langgraph_mcp_server.py"]
    }
  }
}
```

### Claude Code Skill

The repo includes a skill at `.claude/skills/cascade/SKILL.md` that teaches Claude Code when and how to use CASCADE tools. It triggers automatically on keywords like "knockdown", "knockout", "perturbation", "overexpress", "similar genes", etc.

### Example Prompts

**Perturbation Analysis:**
- "Simulate knocking down MYC in epithelial cells"
- "What genes does TP53 regulate in CD4 T cells?"
- "Find all regulators of BRCA1 in epithelial cells"
- "What happens if we overexpress HNF4A 3-fold?"

**Gene Similarity (Embeddings):**
- "How similar are MYC and TP53 based on the model embeddings?"
- "Find genes functionally similar to BRCA1"
- "Check the model status and GPU availability"

**Drug Target Discovery (Network Vulnerability):**
- "Find the top 20 most critical genes in epithelial cells"
- "Compare STAT3, MYC, and TP53 as potential drug targets"
- "What are the master regulators in CD8 T cells?"
- "Which gene would cause the most network disruption if knocked out?"

**Protein-Protein Interactions:**
- "What proteins does APC interact with?"
- "Get high-confidence interactions for TP53"
- "What protein interactions would be disrupted if I knock down BRCA1?"

## Available MCP Tools (26 total)

### Workflow Tools (LangGraph Orchestration)
| Tool | Description |
|------|-------------|
| `comprehensive_perturbation_analysis` | **Main entry point** - Full automated workflow with intelligent routing. Supports `include_llm_insights=true` for AI-powered biological interpretation |
| `multi_gene_analysis` | Analyze multiple genes in parallel |
| `cross_cell_comparison` | Compare how a gene behaves across all available cell types |
| `therapeutic_target_discovery` | Find upstream regulators, interaction partners, and druggability for a gene of interest |

### Perturbation Analysis Tools
| Tool | Description |
|------|-------------|
| `quick_perturbation` | Fast knockdown/overexpression without full workflow context |
| `list_cell_types` | List available cell types with networks |
| `get_gene_metadata` | Get gene classification (TF, effector, scaffold) and analysis recommendations |
| `find_gene_regulators` | Find upstream regulators of a gene |
| `find_gene_targets` | Find downstream targets of a regulator |
| `lookup_gene` | Convert between symbol and Ensembl ID |

### Gene Similarity Tools (Embeddings)
| Tool | Description |
|------|-------------|
| `get_model_status` | Check model loading status and GPU |
| `get_gene_similarity` | Cosine similarity between two genes |
| `find_similar_genes` | Find top-k functionally similar genes |
| `get_embedding_cache_stats` | Check embedding cache performance |

### Network Vulnerability Tools (Drug Target Discovery)
| Tool | Description |
|------|-------------|
| `analyze_network_vulnerability` | Find top hub genes and critical network nodes |
| `compare_gene_vulnerability` | Compare vulnerability scores for candidate genes |

### Protein-Protein Interaction Tools
| Tool | Description |
|------|-------------|
| `get_protein_interactions` | Get interaction partners from STRING database |

### Experimental Perturbation Data (LINCS L1000)
| Tool | Description |
|------|-------------|
| `find_expression_regulators` | Find genes whose knockdown affects target expression |
| `get_knockdown_effects` | Find genes affected when a specific gene is knocked out |
| `get_lincs_data_stats` | Check LINCS dataset statistics |

### Super-Enhancer / BRD4 Druggability
| Tool | Description |
|------|-------------|
| `check_super_enhancer` | Check if a gene has super-enhancers (BRD4/BET inhibitor sensitive) |
| `check_genes_super_enhancers` | Screen multiple genes for super-enhancer status |

### DoRothEA TF Regulon Validation
| Tool | Description |
|------|-------------|
| `get_dorothea_regulon` | Get curated TF regulon targets with confidence levels (A-E) |
| `validate_tf_classification` | Validate gene as known TF against DoRothEA curated regulons |
| `get_dorothea_stats` | Get DoRothEA dataset statistics |

### CRISPR Essentiality (DepMap)
| Tool | Description |
|------|-------------|
| `get_depmap_essentiality` | Chronos gene effect scores across 1,000+ cancer cell lines — pan-cancer essential flag, lineage breakdown, strongly essential fraction |

## MCP Resources

In addition to tools, CASCADE exposes **browsable MCP Resources** that allow clients to discover available data without triggering a full analysis. Resources are read-only and return JSON.

### Static Resources

| URI | Description |
|-----|-------------|
| `cascade://cell-types` | All 10 supported cell types with their regulatory networks |
| `cascade://lincs/summary` | Coverage stats for the LINCS L1000 experimental knockdown dataset |
| `cascade://model/status` | GREmLN embedding model checkpoint status and GPU availability |

### URI Template Resources

| URI Template | Description |
|--------------|-------------|
| `cascade://network/{cell_type}/summary` | Edge count, gene count, and top hub regulators for a cell type |
| `cascade://gene/{symbol}/{cell_type}` | Gene role, target count, and regulator count in a given cell type |

**Examples:**
```
cascade://cell-types                          → list all supported cell types
cascade://network/epithelial_cell/summary     → edge/gene counts for epithelial network
cascade://gene/MYC/cd8_t_cells               → MYC's role and connectivity in CD8 T cells
cascade://model/status                        → whether GREmLN checkpoint is loaded
```

Resources are useful for orientation queries (e.g., "what cell types are available?", "how many genes are in the NK cell network?") before running perturbation analyses.

## Project Structure

```
CASCADE/
├── cascade_langgraph_mcp_server.py  # LangGraph MCP server (main entry)
├── cascade_langgraph_workflow.py    # LangGraph StateGraph workflow
├── cascade_mcp_server_original.py   # Original FastMCP server (deprecated)
├── verify_installation.py           # Installation verification script
├── .claude/
│   └── skills/cascade/              # Claude Code skill for perturbation analysis
├── tools/
│   ├── loader.py                   # Network/model loading utilities
│   ├── perturb.py                  # Perturbation analysis (network + embeddings)
│   ├── model_inference.py          # GREmLN model wrapper for embeddings
│   ├── cache.py                    # Embedding similarity cache
│   ├── gene_id_mapper.py           # Gene symbol/Ensembl ID conversion
│   ├── lincs.py                    # LINCS L1000 expression perturbation data
│   ├── super_enhancers.py          # Super-enhancer annotations (BRD4 druggability)
│   ├── dorothea.py                # DoRothEA TF regulon validation (via decoupler)
│   ├── depmap.py                   # DepMap CRISPR essentiality (Chronos scores)
│   └── ppi/
│       └── string_client.py        # STRING database API client
├── data/
│   ├── networks/                   # Pre-computed regulatory networks (10 cell types)
│   ├── lincs/                      # LINCS L1000 knockdown expression data
│   ├── super_enhancers/            # dbSUPER super-enhancer annotations
│   └── depmap/                     # DepMap CRISPR gene effect data (download separately)
├── models/
│   └── model.ckpt                  # GREmLN model checkpoint (120MB)
└── cache/
    └── gene_id_cache.pkl           # Cached gene ID mappings
```

## Performance

### Individual Tools
| Operation | CPU | GPU |
|-----------|-----|-----|
| Model loading | ~2-5s | ~0.15s |
| Single gene similarity | <1ms | <1ms |
| All-gene similarity (19K genes) | ~500ms | ~30ms |
| Full knockdown analysis | ~3-5s | ~2s |

### LangGraph Workflows
| Workflow | Depth | Typical Time | Notes |
|----------|-------|--------------|-------|
| `comprehensive_perturbation_analysis` | basic | ~2-4s | Network propagation + embeddings only |
| `comprehensive_perturbation_analysis` | comprehensive | ~5-10s | 8 data sources in parallel; STRING API is main variable (10s cap) |
| `comprehensive_perturbation_analysis` | focused | ~3-5s | Role-dependent subset of analyses |
| `comprehensive_perturbation_analysis` | + LLM insights | +5-15s | Depends on Ollama model and hardware |
| `multi_gene_analysis` (3 genes) | basic | ~8s | Parallel workflow per gene |
| `cross_cell_comparison` | — | ~1s first call; <0.1s cached | All 10 networks pre-warmed and cached in-process |

### Startup Performance

| Event | Behavior |
|-------|----------|
| MCP initialize handshake | Completes in milliseconds |
| Background pre-warming | Server pre-warms all 6 data sources concurrently on startup: GREmLN model, LINCS, DepMap, super-enhancers, DoRothEA (disk cache), and all 10 cell type networks — completes in ~5s so first tool call finds everything ready |
| DoRothEA regulons | Cached to disk (`data/dorothea/`) after first download; subsequent server restarts load in ~0.1s instead of re-downloading |
| Network adjacency | Built once per cell type per session using vectorized numpy operations; shared across all analyses for that cell type |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- LangGraph (for workflow orchestration)
- MCP (Model Context Protocol SDK)
- pandas, numpy
- scGraphLLM (GREmLN package from CZI)
- requests (for Ensembl API gene ID lookups)
- Ollama (local or cloud) for LLM-powered biological insights; adaptable to other LLM providers

## Technical Details

### Regulatory Networks

CASCADE's cell-type-specific regulatory networks are inferred from single-cell RNA-seq data. Each directed edge (`regulator → target`) is a statistically inferred regulatory relationship with the following attributes:

- **mi** (mutual information): how strongly the regulator's mRNA level predicts the target's mRNA level across cells
- **scc** (Spearman correlation): direction and strength of the expression relationship
- **count**: reproducibility of the edge across bootstrap iterations
- **log_p**: statistical significance

These edges capture real co-expression-based regulatory relationships. However, because inference is based on mRNA co-variation, the networks may not include regulatory relationships where the transcription factor's activity is controlled post-translationally rather than at the mRNA level (e.g., TP53, whose protein is stabilized by phosphorylation rather than transcriptionally upregulated).

### Gene Embeddings

The GREmLN model contains a gene embedding table with 256-dimensional vectors for ~19,247 genes. These embeddings were learned during pre-training on 11 million single cells and capture functional relationships between genes.

### Combined Scoring

The model-enhanced tools use a weighted combination:
```
combined_effect = α × network_effect + (1-α) × embedding_similarity × network_effect
```

Where `α` (default 0.7) controls the balance between network and embedding signals.

### Embedding-Only Effects

Genes with high embedding similarity but no direct network connection are also reported as potential indirect effects, allowing discovery of relationships not captured in the static network.

### Protein-Protein Interaction Integration

The `get_protein_interactions` tool queries the STRING database to complement gene regulatory network analysis with protein-level mechanisms. This helps explain *why* perturbations have downstream effects.

**Example: Understanding APC Knockdown (with Intelligent Suggestions)**

APC is a scaffold protein with no transcriptional targets. The tools now guide you:

```
User: "What happens when APC is knocked down?"

Step 1: Check gene type
> get_gene_metadata("APC", cell_type="epithelial_cell")
Result: {
  "gene_type": "effector",
  "is_transcription_factor": false,
  "num_targets": 0,
  "analysis_recommendations": [
    {"tool": "get_protein_interactions", "reason": "Gene does not regulate transcription"}
  ]
}

Step 2: Simulate knockdown (returns suggestions since no targets)
> quick_perturbation("APC", cell_type="epithelial_cell")
Result: {
  "total_affected_genes": 0,
  "suggestions": [
    {"action": "get_protein_interactions", "priority": "high"},
    {"action": "analyze_functional_partners",
     "genes": ["CTNNB1", "AXIN1", "GSK3B", "CSNK1A1"],
     "recommended_followup": "Run overexpression on CTNNB1..."}
  ]
}

Step 3: Follow suggestions - analyze CTNNB1 (the key functional partner)
> quick_perturbation("CTNNB1", cell_type="epithelial_cell", perturbation_type="overexpression")
Result: 2,739 genes affected (MYC, CCND1, GLUT1, etc.)

Interpretation: APC normally degrades β-catenin via the destruction complex.
APC loss → β-catenin accumulates → activates oncogenic transcription.
```

**STRING Confidence Scores:**
- 900+: Highest confidence (experimentally validated)
- 700-899: High confidence
- 400-699: Medium confidence
- 150-399: Low confidence

### Network Vulnerability Scoring

The vulnerability analysis tools rank genes by their criticality to the network:

```
vulnerability_score = hub_score × 1.0 + cascade_reach × 0.3 + avg_edge_weight × 10 + isolation_factor × 5
```

Where:
- **hub_score**: Number of direct target genes
- **cascade_reach**: Number of 2nd-order downstream targets
- **avg_edge_weight**: Mean mutual information of outgoing edges
- **isolation_factor**: `1 / (regulator_count + 1)` — genes with fewer upstream regulators are harder to compensate for if knocked out

**Interpretation:**
- High vulnerability = Gene is critical to network (good drug target)
- Master regulators (high hub score, low regulator count) = High-value therapeutic targets
- Downstream effectors (many regulators, few targets) = Lower priority targets

## Running Tests

```bash
pip install pytest pytest-cov pytest-asyncio
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coverage commands and testing guidelines.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and the pull request process.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If you use CASCADE in your research, please cite:

```bibtex
@software{bird2026cascade,
  title     = {CASCADE: An MCP Server for In Silico Gene Perturbation Analysis in Immuno-Oncology},
  author    = {Bird, Jose},
  year      = {2026},
  url       = {https://github.com/jab57/CASCADE},
  version   = {0.1.0},
  license   = {MIT}
}
```

A `CITATION.cff` file is included for GitHub's citation feature.

## License

MIT
