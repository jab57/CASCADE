# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GREmLN MCP Server provides **in silico gene perturbation analysis** for cancer/immuno-oncology research. It exposes tools via Model Context Protocol (MCP) for simulating gene knockdowns/overexpression using pre-computed regulatory networks and GREmLN model embeddings.

## Key Commands

```bash
# Setup
python -m venv env
.\env\Scripts\activate          # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall  # GPU

# Run the MCP server
python gremln_mcp_server.py
```

## Architecture

### Perturbation Analysis

The core tools (`analyze_gene_knockdown`, `analyze_gene_overexpression`) combine two signals:

1. **Network propagation**: BFS through regulatory network using mutual information edge weights
2. **GREmLN embeddings**: 256-dim gene representations learned from 11M cells

Combined scoring formula:
```
combined_effect = α × network_effect + (1-α) × embedding_similarity × network_effect
```

The tools automatically fall back to network-only analysis if the model fails to load.

### Core Modules

- `gremln_mcp_server.py` - FastMCP server entry point with all tool definitions
- `tools/perturb.py` - Network propagation algorithms (`_propagate_effect` BFS, knockdown/overexpression simulation)
- `tools/model_inference.py` - `GREmLNModel` class for embedding extraction and similarity computation
- `tools/cache.py` - Embedding similarity cache for performance
- `tools/gene_id_mapper.py` - Gene symbol ↔ Ensembl ID conversion (uses Ensembl REST API + local cache)
- `tools/ppi/string_client.py` - STRING database API client for protein-protein interactions

### Data Flow

```
Gene Symbol → gene_id_mapper → Ensembl ID → network lookup + model embedding
                                                    ↓
                                          Network BFS propagation
                                                    ↓
                                          Embedding similarity boost
                                                    ↓
                                          Ranked affected genes
```

### Intelligent Suggestions

When perturbation tools return no targets (e.g., scaffold proteins like APC), results include:
- `gene_metadata`: Classification (master_regulator, transcription_factor, effector, isolated)
- `suggestions`: Recommended follow-up tools (e.g., `get_protein_interactions` for effectors)

The `_generate_suggestions()` function in `gremln_mcp_server.py` handles this routing logic.

## Supported Cell Types

Networks in `data/networks/`: epithelial_cell, cd4_t_cells, cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, nk_cells, nkt_cells, erythrocytes, monocyte-derived_dendritic_cells

## Model Checkpoint

Located at `models/model.ckpt` (~120MB). Lazy-loaded via `get_model()` singleton. Contains embeddings for ~19,247 genes.

## Key Design Decisions

- All tools accept gene symbols (MYC) or Ensembl IDs (ENSG...) - resolved internally
- Results include both `ensembl_id` and `symbol` for each gene
- `max_regulators`/`max_targets` parameters limit API calls to avoid timeouts
- Embedding cache persists similarity computations for repeated queries
- Perturbation tools use embeddings by default with graceful fallback to network-only
