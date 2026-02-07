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
│  └──────────┘         └──────────────┘         │  LINCS)  │   │
│                                                └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Benefits of LangGraph Architecture

- **Intelligent Routing**: Automatically selects analysis strategy based on gene type
- **Parallel Execution**: Independent analyses run concurrently (3-5x faster)
- **Automatic Synthesis**: Generates comprehensive reports with actionable recommendations
- **LLM-Powered Insights**: Optional Ollama integration for biological interpretation of results
- **Graceful Degradation**: Falls back to network-only if embeddings unavailable

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

### Model-Enhanced Analysis
- **Embedding-Enhanced Knockdown**: Combines network topology with learned gene representations
- **Embedding-Enhanced Overexpression**: More accurate predictions using 11M-cell-trained embeddings
- **Gene Similarity**: Compute functional similarity between genes using learned embeddings
- **Similar Gene Discovery**: Find functionally related genes even without direct network connections

### LLM-Powered Biological Insights (Optional)
- **Narrative Synthesis**: Ollama-powered interpretation of analysis results
- **Mechanism Explanation**: Automatic generation of biological mechanism summaries
- **Therapeutic Implications**: AI-generated drug development relevance
- **Pathway Identification**: Automated identification of key affected pathways
- **Follow-up Suggestions**: Intelligent recommendations for further analysis

### Network Vulnerability Analysis (Drug Target Discovery)
- **Hub Gene Identification**: Find genes with the most downstream targets
- **Vulnerability Scoring**: Rank genes by network criticality (cascade impact, connectivity)
- **Drug Target Comparison**: Compare candidate genes to identify best therapeutic targets
- **Master Regulator Detection**: Identify genes that control large portions of the network

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

The server provides three types of analysis:

1. **Perturbation analysis** (`analyze_gene_knockdown`, `analyze_gene_overexpression`): Combines BFS propagation through the regulatory network with gene embeddings learned from 11 million cells. This discovers both direct network effects and indirect functional relationships. Falls back to network-only if model is unavailable.

2. **Vulnerability analysis** (`analyze_network_vulnerability`, `compare_gene_vulnerability`): Identifies critical network nodes (hub genes, master regulators) for drug target discovery. Ranks genes by downstream impact if disrupted.

3. **Protein-protein interactions** (`get_protein_interactions`): Queries STRING database for physical and functional protein interactions. Explains what happens at the protein level after perturbation.

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

# Optional: Install Ollama for LLM-powered insights
# Download from https://ollama.ai and run: ollama pull llama3.1:8b
```

## Usage

### Run the MCP Server

```bash
# LangGraph-based server (recommended)
python cascade_langgraph_mcp_server.py

# With LLM insights enabled (requires Ollama running)
USE_LLM_INSIGHTS=true python cascade_langgraph_mcp_server.py

# Original FastMCP server (deprecated, kept for reference)
python cascade_mcp_server_original.py
```

### LLM Insights Configuration (Optional)

Copy `.env.example` to `.env` and configure:

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

## Available MCP Tools (22 total)

### Workflow Tools (LangGraph Orchestration)
| Tool | Description |
|------|-------------|
| `comprehensive_perturbation_analysis` | **Main entry point** - Full automated workflow with intelligent routing. Supports `include_llm_insights=true` for AI-powered biological interpretation |
| `multi_gene_analysis` | Analyze multiple genes in parallel |

### Perturbation Analysis Tools
| Tool | Description |
|------|-------------|
| `list_cell_types` | List available cell types with networks |
| `get_gene_metadata` | Get gene classification (TF, effector, scaffold) and analysis recommendations |
| `analyze_gene_knockdown` | Simulate gene knockdown using network + embeddings |
| `analyze_gene_overexpression` | Simulate overexpression using network + embeddings |
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

## Project Structure

```
CASCADE/
├── cascade_langgraph_mcp_server.py  # LangGraph MCP server (main entry)
├── cascade_langgraph_workflow.py    # LangGraph StateGraph workflow
├── cascade_mcp_server_original.py   # Original FastMCP server (deprecated)
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
│   └── ppi/
│       └── string_client.py        # STRING database API client
├── data/
│   ├── networks/                   # Pre-computed regulatory networks (10 cell types)
│   ├── lincs/                      # LINCS L1000 knockdown expression data
│   └── super_enhancers/            # dbSUPER super-enhancer annotations
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
| Workflow | Depth | Typical Time |
|----------|-------|--------------|
| `comprehensive_perturbation_analysis` | basic | ~3s |
| `comprehensive_perturbation_analysis` | standard | ~5s |
| `comprehensive_perturbation_analysis` | comprehensive | ~8-10s |
| `comprehensive_perturbation_analysis` | + LLM insights | +5-15s (depends on model) |
| `multi_gene_analysis` (3 genes) | standard | ~10s (parallel) |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- LangGraph (for workflow orchestration)
- MCP (Model Context Protocol SDK)
- pandas, numpy
- scGraphLLM (GREmLN package from CZI)
- requests (for Ensembl API gene ID lookups)
- ollama (optional, for LLM-powered insights)

## Technical Details

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
> analyze_gene_knockdown("APC", cell_type="epithelial_cell")
Result: {
  "total_affected_genes": 0,
  "suggestions": [
    {"action": "get_protein_interactions", "priority": "high"},
    {"action": "analyze_functional_partners",
     "genes": ["CTNNB1", "AXIN1", "GSK3B", "CSNK1A1"],
     "recommended_followup": "Run analyze_gene_overexpression on CTNNB1..."}
  ]
}

Step 3: Follow suggestions - analyze CTNNB1 (the key functional partner)
> analyze_gene_overexpression("CTNNB1", cell_type="epithelial_cell")
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

## Roadmap

### Recently Completed

- **LLM-Powered Biological Insights** (2025-02): Optional Ollama integration for AI-generated interpretation of analysis results
- **LangGraph Orchestration** (2025-02): Intelligent workflow routing with parallel execution and automatic synthesis
- **LINCS L1000 Expression Perturbation** (2025-01): Find regulatory relationships from experimental CRISPR knockdown data
- **Super-Enhancer Annotations** (2025-01): BRD4/BET inhibitor sensitivity using dbSUPER database

### Planned Features

| Feature | Status | Description |
|---------|--------|-------------|
| Raw LINCS Integration | Planned | Full LINCS L1000 data from clue.io for better coverage (e.g., BRD4 → MYC) |
| Expression Data Fetching | Planned | Fetch baseline expression from CellxGene Census, Human Protein Atlas |
| Context-Aware Embeddings | Planned | Cell-type-specific gene embeddings via model forward pass |
| Perturb_GDTransformer | Waiting | Integration when fine-tuned perturbation checkpoint becomes available |
| Unified Bio-Orchestrator | Postponed | Cross-project orchestration (CASCADE + RegNetAgents in single server) |

See `tools/expression_fetcher.py` for expression fetching implementation notes.

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
