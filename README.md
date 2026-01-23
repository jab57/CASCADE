# GREmLN MCP Server

A Model Context Protocol (MCP) server for **in silico gene perturbation analysis** using pre-computed gene regulatory networks and GREmLN model embeddings.

## Features

### Network-Based Analysis
- **Gene Knockdown Simulation**: Predict downstream effects of silencing a gene
- **Gene Overexpression Simulation**: Predict effects of increased gene expression
- **Regulator Discovery**: Find transcription factors controlling a target gene
- **Target Discovery**: Find genes controlled by a regulator
- **Gene ID Mapping**: Convert between gene symbols (MYC) and Ensembl IDs (ENSG...)

### Model-Enhanced Analysis (NEW)
- **Embedding-Enhanced Knockdown**: Combines network topology with learned gene representations
- **Embedding-Enhanced Overexpression**: More accurate predictions using 11M-cell-trained embeddings
- **Gene Similarity**: Compute functional similarity between genes using learned embeddings
- **Similar Gene Discovery**: Find functionally related genes even without direct network connections

## How It Works

The server provides two types of analysis:

1. **Network-only** (`analyze_gene_knockdown`, `analyze_gene_overexpression`): Uses BFS propagation through the regulatory network based on mutual information edge weights.

2. **Model-enhanced** (`analyze_gene_knockdown_model`, `analyze_gene_overexpression_model`): Combines network propagation with gene embeddings learned from 11 million cells. This can discover indirect effects not captured in the static network.

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
```

## Usage

### Run the MCP Server

```bash
python gremln_mcp_server.py
```

### Claude Desktop Configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac):

```json
{
  "mcpServers": {
    "GREmLN": {
      "command": "C:/Dev/GREmLN/env/Scripts/python.exe",
      "args": ["C:/Dev/GREmLN/gremln_mcp_server.py"]
    }
  }
}
```

### Example Prompts

**Basic Network Analysis:**
- "Simulate knocking down MYC in epithelial cells"
- "What genes does TP53 regulate in CD4 T cells?"
- "Find all regulators of BRCA1 in epithelial cells"
- "What happens if we overexpress HNF4A 3-fold?"

**Model-Enhanced Analysis:**
- "Use the model to simulate MYC knockdown in epithelial cells"
- "How similar are MYC and TP53 based on the model embeddings?"
- "Find genes functionally similar to BRCA1"
- "Check the model status and GPU availability"

## Available MCP Tools

### Network-Based Tools
| Tool | Description |
|------|-------------|
| `list_cell_types` | List available cell types with networks |
| `analyze_gene_knockdown` | Simulate gene knockdown (network only) |
| `analyze_gene_overexpression` | Simulate overexpression (network only) |
| `find_gene_regulators` | Find upstream regulators of a gene |
| `find_gene_targets` | Find downstream targets of a regulator |
| `lookup_gene` | Convert between symbol and Ensembl ID |

### Model-Enhanced Tools
| Tool | Description |
|------|-------------|
| `get_model_status` | Check model loading status and GPU |
| `analyze_gene_knockdown_model` | Knockdown with embedding enhancement |
| `analyze_gene_overexpression_model` | Overexpression with embeddings |
| `get_gene_similarity` | Cosine similarity between two genes |
| `find_similar_genes` | Find top-k functionally similar genes |
| `get_embedding_cache_stats` | Check embedding cache performance |

## Project Structure

```
GREmLN/
├── gremln_mcp_server.py      # MCP server entry point
├── tools/
│   ├── loader.py             # Network/model loading utilities
│   ├── perturb.py            # Perturbation analysis (network + embeddings)
│   ├── model_inference.py    # GREmLN model wrapper for embeddings
│   ├── cache.py              # Embedding similarity cache
│   └── gene_id_mapper.py     # Gene symbol/Ensembl ID conversion
├── data/
│   └── networks/             # Pre-computed regulatory networks (10 cell types)
├── models/
│   └── model.ckpt            # GREmLN model checkpoint (120MB)
└── cache/
    └── gene_id_cache.pkl     # Cached gene ID mappings
```

## Performance

| Operation | CPU | GPU |
|-----------|-----|-----|
| Model loading | ~2-5s | ~0.15s |
| Single gene similarity | <1ms | <1ms |
| All-gene similarity (19K genes) | ~500ms | ~30ms |
| Full knockdown analysis | ~3-5s | ~2s |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- FastMCP
- pandas, numpy
- scGraphLLM (GREmLN package from CZI)
- requests (for Ensembl API gene ID lookups)

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

## Roadmap

### Planned Features

- **Automatic Expression Data Fetching**: Fetch baseline expression profiles for each cell type from public databases (CellxGene Census, Human Protein Atlas)
- **Context-Aware Gene Embeddings**: Use expression data to generate cell-type-specific gene embeddings via model forward pass
- **Perturb_GDTransformer Support**: Integration with fine-tuned perturbation model when checkpoint becomes available

See `tools/expression_fetcher.py` for implementation notes.

## License

MIT
