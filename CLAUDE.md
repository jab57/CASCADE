# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GREmLN MCP Server provides **in silico gene perturbation analysis** for cancer/immuno-oncology research. It exposes tools via Model Context Protocol (MCP) for simulating gene knockdowns/overexpression using pre-computed regulatory networks and GREmLN model embeddings.

## Architecture

The project now uses a **LangGraph-based orchestration layer** that automatically coordinates multiple analysis tools:

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop / MCP Client              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              GREmLN LangGraph MCP Server                    │
│              gremln_langgraph_mcp_server.py                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         LangGraph StateGraph Workflow               │   │
│  │                                                      │   │
│  │  initialize → resolve_gene → analyze_network_context│   │
│  │                      ↓                               │   │
│  │              decide_next_steps                       │   │
│  │         ┌──────────┼──────────┐                     │   │
│  │         ↓          ↓          ↓                     │   │
│  │   batch_core  batch_external  batch_insights        │   │
│  │   (parallel)   (parallel)     (parallel)            │   │
│  │         └──────────┼──────────┘                     │   │
│  │                    ↓                                 │   │
│  │            generate_report → synthesize_insights → END│  │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│            ┌─────────────┴─────────────┐                   │
│            ▼                           ▼                    │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │  tools/ modules │         │  External APIs  │          │
│   │  (perturb.py,   │         │  (STRING, LINCS,│          │
│   │   model, etc.)  │         │   Ensembl)      │          │
│   └─────────────────┘         └─────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Key Commands

```bash
# Setup
python -m venv env
.\env\Scripts\activate          # Windows
pip install -r requirements.txt
pip install langgraph           # Required for orchestration
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall  # GPU

# Run the LangGraph MCP server (recommended)
python gremln_langgraph_mcp_server.py
```

## Core Files

### LangGraph Orchestration (NEW)
- `gremln_langgraph_mcp_server.py` - MCP server with 22 tools, LangGraph orchestration
- `gremln_langgraph_workflow.py` - Core workflow: state schema, routing logic, batch processing

### Tool Implementations
- `tools/perturb.py` - Network propagation algorithms (`_propagate_effect` BFS, knockdown/overexpression)
- `tools/model_inference.py` - `GREmLNModel` class for embedding extraction and similarity
- `tools/cache.py` - Embedding similarity cache for performance
- `tools/gene_id_mapper.py` - Gene symbol ↔ Ensembl ID conversion
- `tools/lincs.py` - LINCS L1000 experimental knockdown data
- `tools/super_enhancers.py` - BRD4/BET inhibitor sensitivity (dbSUPER)
- `tools/ppi/string_client.py` - STRING database API client

## LangGraph Workflow

### State Schema (`PerturbationAnalysisState`)
```python
- gene, cell_type, perturbation_type, analysis_depth  # inputs
- include_llm_insights                                 # LLM toggle
- ensembl_id, gene_symbol, gene_role                  # resolved
- perturbation_result, regulators_analysis, targets_analysis  # core
- ppi_interactions, lincs_effects, super_enhancer_status      # external
- similar_genes, vulnerability_analysis, cross_cell_comparison # insights
- comprehensive_report, therapeutic_suggestions               # output
- llm_insights                                        # LLM-generated interpretation
```

### Routing Logic
The `_decide_next_steps()` function determines analysis path based on:
- Gene role (master_regulator, transcription_factor, effector, isolated)
- Analysis depth (basic, comprehensive, focused)
- Completed vs pending analyses

### Parallel Batch Processing
Independent analyses run concurrently:
- `batch_core_analysis`: perturbation + regulators + targets
- `batch_external_data`: PPI + LINCS + super-enhancers
- `batch_insights`: similar genes + vulnerability + cross-cell

## Supported Cell Types

Networks in `data/networks/`: epithelial_cell, cd4_t_cells, cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, nk_cells, nkt_cells, erythrocytes, monocyte-derived_dendritic_cells

## Model Checkpoint

Located at `models/model.ckpt` (~120MB). Pre-loaded on server startup. Contains embeddings for ~19,247 genes.

## LLM Insights (Optional)

The workflow supports optional LLM-powered biological interpretation via Ollama:

```bash
# Enable LLM insights
USE_LLM_INSIGHTS=true python gremln_langgraph_mcp_server.py
```

**Environment Variables:**
- `USE_LLM_INSIGHTS` - Enable/disable LLM synthesis (default: false)
- `OLLAMA_HOST` - Local Ollama URL (default: http://localhost:11434)
- `OLLAMA_API_KEY` - If set, uses Ollama Cloud instead of local
- `OLLAMA_MODEL` - Model to use (default: llama3.1:8b)
- `OLLAMA_TEMPERATURE` - Generation temperature (default: 0.3)
- `OLLAMA_TIMEOUT` - Request timeout in seconds (default: 60)

**Workflow Node:** `synthesize_insights` runs after `generate_report` when `include_llm_insights=True`

## Key Design Decisions

- **Eager initialization**: Workflow pre-loads on server startup to avoid lazy-load delays
- **Parallel execution**: Independent analyses batched for concurrent execution
- **Automatic routing**: Gene type determines analysis path (TF → knockdown, effector → PPI)
- **LLM insights optional**: Disabled by default to avoid latency; enable with `include_llm_insights=True`
- **Graceful fallback**: Embedding model falls back to network-only if unavailable; LLM returns structured data if Ollama unavailable
- All tools accept gene symbols (MYC) or Ensembl IDs (ENSG...) - resolved internally
- Results include both `ensembl_id` and `symbol` for each gene
