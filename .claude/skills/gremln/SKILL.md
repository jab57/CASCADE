---
name: gremln
description: Gene PERTURBATION and KNOCKDOWN analysis using GREmLN MCP server. Use for knockdown simulation, overexpression effects, gene silencing predictions, embedding-based gene similarity, and network vulnerability analysis. Keywords: knockdown, knock down, knockout, knock out, silence, inhibit, overexpress, perturb, perturbation, similar genes, drug target vulnerability.
---

# GREmLN Gene Perturbation Analysis

**IMPORTANT: Always use tools from the `gremln` MCP server for this skill, NOT from other servers like `regnetagents`.** The GREmLN server provides specialized perturbation analysis combining regulatory networks with learned embeddings.

This MCP server provides in silico gene perturbation analysis combining regulatory network topology with learned gene embeddings from 11M cells.

## Available Tools

### Perturbation Analysis
- `analyze_gene_knockdown` - Simulate gene silencing, predict downstream effects
- `analyze_gene_overexpression` - Simulate increased gene expression
- `find_gene_regulators` - Find upstream transcription factors controlling a gene
- `find_gene_targets` - Find downstream targets of a regulator
- `get_gene_metadata` - Classify gene type (master_regulator, transcription_factor, effector, isolated)

### Drug Target Discovery
- `analyze_network_vulnerability` - Identify critical hub genes in a cell type
- `compare_gene_vulnerability` - Compare multiple candidate drug targets

### Functional Similarity (Embedding-Based)
- `get_gene_similarity` - Cosine similarity between two genes
- `find_similar_genes` - Find functionally related genes beyond network edges
- `get_model_status` - Check if embeddings model is loaded

### Protein-Protein Interactions
- `get_protein_interactions` - Query STRING database for physical/functional interactions

### Experimental Perturbation Data (LINCS L1000)
- `find_expression_regulators` - Find genes whose knockdown affects target expression
- `get_knockdown_effects` - Find genes affected when a specific gene is knocked out
- `get_lincs_data_stats` - Check LINCS dataset statistics

### Super-Enhancer / BRD4 Druggability
- `check_super_enhancer` - Check if a gene has super-enhancers (BRD4/BET inhibitor sensitive)
- `check_genes_super_enhancers` - Screen multiple genes for super-enhancer status

### Utilities
- `list_cell_types` - Show available cell type networks
- `lookup_gene` - Convert between gene symbols and Ensembl IDs

## Supported Cell Types

epithelial_cell, cd4_t_cells, cd8_t_cells, cd14_monocytes, cd16_monocytes, cd20_b_cells, nk_cells, nkt_cells, erythrocytes, monocyte-derived_dendritic_cells

## Common Workflows

### Drug Target Discovery
1. `analyze_network_vulnerability` - identify hub genes in the relevant cell type
2. `compare_gene_vulnerability` - rank candidate targets by network centrality
3. `analyze_gene_knockdown` - predict functional consequences of targeting each candidate

### Understanding Gene Function
1. Start with `analyze_gene_knockdown` or `analyze_gene_overexpression`
2. If results are empty or sparse, check `get_gene_metadata` to understand why
3. For scaffold proteins (like APC, AXIN1), use `get_protein_interactions` instead
4. Use `find_similar_genes` to discover functional relationships beyond the network

### Regulatory Cascade Analysis
1. `find_gene_regulators` - identify upstream transcription factors
2. `find_gene_targets` - map downstream effects
3. Chain with `analyze_gene_knockdown` on key regulators to predict cascade effects

### Cross-Cell-Type Comparison
Run the same analysis across multiple cell types to identify cell-type-specific effects (e.g., compare knockdown effects in cd8_t_cells vs nk_cells)

### Finding Epigenetic Regulators (LINCS)
When network analysis misses known regulators (e.g., BRD4 → MYC):
1. `find_expression_regulators(gene, direction="down")` - Find what knockdowns reduce target expression
2. This captures epigenetic and post-translational effects the transcriptional network misses
3. Example: `find_expression_regulators("CDKN1A")` → Returns TP53 (validated regulator)

### Checking BRD4 Druggability (Super-Enhancers)
When a gene is "undruggable" (e.g., MYC has no binding pocket):
1. `check_super_enhancer(gene)` - Check if gene has super-enhancer associations
2. If yes → Gene may be sensitive to BRD4/BET inhibitors (JQ1, OTX015)
3. Example: `check_super_enhancer("MYC")` → 32 cell types, BRD4-sensitive

## Interpreting Results

- **combined_effect**: Higher values = more strongly affected (combines network propagation + embedding similarity)
- **network_effect**: Effect from regulatory network edges only
- **embedding_similarity**: Functional similarity from learned representations
- **embedding_only: true**: Gene is functionally related but has no direct network edge (discovered via embeddings)
- **effect_type**: "direct" (immediate target) or "indirect" (downstream of cascade)

## Edge Cases

- **Empty results for scaffold proteins**: These proteins work through complexes, not transcriptional regulation. Use `get_protein_interactions` to find binding partners.
- **Gene not in network**: The gene may not have regulatory connections in that cell type. Try `find_similar_genes` or a different cell type.
- **Suggestions field**: When results are limited, the API provides intelligent suggestions for alternative analyses.

## Tips

- All tools accept either gene symbols (MYC, TP53) or Ensembl IDs (ENSG...)
- Use `max_depth` parameter to control how far effects propagate (default: 3)
- Results include both `ensembl_id` and `symbol` for easy cross-referencing
- The embedding model adds ~30% more discoveries beyond network-only analysis
